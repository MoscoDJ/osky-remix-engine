"""Evaluador: "escucha" un render mediante mediciones y produce hallazgos con severidad.

Mide sobre el master final (lo que se escucha) y sobre las salidas post-fader de cada pista:
- contraste de energía entre secciones según su nivel declarado (intro < voz < drop),
- presencia de la voz frente al resto en secciones con voz,
- balance bajo/bombo en la sección de mayor energía,
- verificaciones de calidad del master (true peak, recortes, sonoridad),
- enmascaramiento por sección y banda, y diferencias con una referencia (``mix_findings``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..composition.arranger import DEFAULT_ENERGY
from ..core.models import Project, TrackKind
from ..mastering.loudness import integrated_lufs
from .policies import DEFAULT, Policy


@dataclass
class Finding:
    code: str
    severity: float
    detail: str
    data: dict = field(default_factory=dict)


@dataclass
class Evaluation:
    score: float                       # suma de severidades (0 = sin problemas)
    findings: list[Finding]
    metrics: dict

    def summary(self) -> str:
        return f"puntaje {self.score:.2f} · " + ("; ".join(f.detail for f in self.findings) or "sin hallazgos")


def _lufs(x: np.ndarray, sr: int) -> float:
    try:
        v = integrated_lufs(x, sr)
    except ValueError:
        return -70.0
    return float(v) if np.isfinite(v) else -70.0


def sections_of(p: Project) -> list[dict]:
    """Secciones (nombre, energía, voz, rango en muestras) a partir de marcadores y meta."""
    tm = p.tempo_map()
    meta = {s["name"]: s for s in p.meta.get("arrangement", {}).get("sections", [])}
    ms = sorted(p.markers, key=lambda m: m.tick)
    out = []
    for i, m in enumerate(ms):
        end_tick = ms[i + 1].tick if i + 1 < len(ms) else None
        sm = meta.get(m.label, {})
        if "vocal" in sm:
            vocal = sm["vocal"]
        else:   # sin meta: ¿algún clip de voz ocupa al menos un compás de la sección? (la anacrusa no cuenta)
            hi = end_tick if end_tick is not None else 1 << 62
            vocal = False
            for t in p.tracks:
                if not t.id.startswith("vox"):
                    continue
                for c in t.clips:
                    c_end = tm.sample_to_tick(tm.tick_to_sample(c.start_tick) + getattr(c, "source_length", 0)) \
                        if hasattr(c, "source_length") else c.start_tick + c.length_ticks
                    if min(c_end, hi) - max(c.start_tick, m.tick) >= 4 * 960:
                        vocal = True
        out.append({"name": m.label, "tick": m.tick, "start": tm.tick_to_sample(m.tick),
                    "end": tm.tick_to_sample(end_tick) if end_tick is not None else None,
                    "energy": sm.get("energy", DEFAULT_ENERGY.get(m.label, 2)), "vocal": vocal})
    return out


def evaluate(p: Project, renderer, master_audio: np.ndarray, quality: dict, policy: Policy = DEFAULT) -> Evaluation:
    sr = p.sample_rate
    secs = sections_of(p)
    for s in secs:
        if s["end"] is None:
            s["end"] = len(master_audio)
    findings: list[Finding] = []
    metrics: dict = {"sections": []}
    for s in secs:
        s["lufs"] = _lufs(master_audio[s["start"]:s["end"]], sr)
        metrics["sections"].append({k: s[k] for k in ("name", "energy", "lufs")})
    # 1) contraste de energía: cada sección respecto de la de mayor energía
    if secs:
        top = max(secs, key=lambda s: (s["energy"], s["lufs"]))
        for s in secs:
            if s is top or s["energy"] >= top["energy"]:
                continue
            want = -policy.min_energy_delta_lu * (top["energy"] - s["energy"])
            have = s["lufs"] - top["lufs"]
            deficit = have - want
            if deficit > 0.25:
                findings.append(Finding("energy_contrast", deficit,
                                        f"'{s['name']}' está {have:+.1f} LU respecto de '{top['name']}' (se busca ≤ {want:+.1f})",
                                        {"section": s["name"], "deficit": deficit}))
    # 2) voz frente al resto en secciones con voz
    post = renderer.post
    vox_ids = [t.id for t in p.tracks if t.kind == TrackKind.audio and t.id in post and t.id.startswith(("vox", "chops"))]
    others = [t.id for t in p.tracks if t.id in post and t.kind in (TrackKind.audio, TrackKind.instrument)
              and t.id not in vox_ids]
    if vox_ids:
        # en la banda donde compiten (300 Hz-5 kHz): el bombo y el bajo dominan la sonoridad total
        # pero no tapan la voz; compararla contra todo daba -12 LU en una mezcla donde se oye bien
        sos = butter(4, [300, 5000], "band", fs=sr, output="sos")
        lo, hi = policy.vocal_window_lu
        rel = []
        for s in secs:
            if not s["vocal"]:
                continue
            a, b = s["start"], s["end"]
            v = sosfiltfilt(sos, sum(post[t][a:b] for t in vox_ids), axis=0)
            o = sosfiltfilt(sos, sum(post[t][a:b] for t in others), axis=0) if others else np.zeros_like(v)
            rel.append(_lufs(v, sr) - _lufs(o, sr))
        if rel:
            r = float(np.median(rel))
            metrics["vocal_minus_music_lu"] = round(r, 2)
            if r < lo or r > hi:
                findings.append(Finding("vocal_level", 0.5 * (lo - r if r < lo else r - hi),
                                        f"voz {r:+.1f} LU frente al resto en 300 Hz-5 kHz (se busca {lo:+.0f}..{hi:+.0f})",
                                        {"rel": r, "target": (lo + hi) / 2, "window": (lo, hi), "tracks": vox_ids}))
    # 3) bajo frente a bombo en la sección de más energía
    if "bass" in post and "kick" in post and secs:
        top = max(secs, key=lambda s: s["energy"])
        a, b = top["start"], top["end"]
        r = _lufs(post["bass"][a:b], sr) - _lufs(post["kick"][a:b], sr)
        metrics["bass_minus_kick_lu"] = round(r, 2)
        lo, hi = policy.bass_kick_window_lu
        if r < lo or r > hi:
            findings.append(Finding("bass_balance", 0.3 * (lo - r if r < lo else r - hi),
                                    f"bajo {r:+.1f} LU frente al bombo en '{top['name']}' (se busca {lo:+.0f}..{hi:+.0f})",
                                    {"rel": r, "target": (lo + hi) / 2, "window": (lo, hi)}))
    # 3b) enmascaramiento por sección y banda, y 3c) referencia comercial (si meta.reference_track)
    from .mix_findings import masking_findings, reference_findings
    whole = secs or [{"name": "todo", "start": 0, "end": len(master_audio), "vocal": True}]
    for fs, ms in (masking_findings(p, post, whole, sr, policy), reference_findings(p, renderer, master_audio, policy)):
        findings += fs
        metrics.update(ms)
    # 4) calidad del master
    for c in quality.get("checks", []):
        if c["status"] == "FAIL":
            findings.append(Finding("quality_" + c["check"], 5.0, f"calidad: {c['check']} {c['detail']}"))
    metrics["lufs"] = quality.get("lufs")
    metrics["true_peak_dbtp"] = quality.get("true_peak_dbtp")
    metrics["loudness_range_lu"] = quality.get("loudness_range_lu")
    pm = quality.get("premaster", {}).get("peak_dbfs")
    # reducción de picos estimada: pico de cresta del premaster normalizado vs. techo del master
    if quality.get("limiting_gain_reduction_db_est") is not None and pm is not None:
        metrics["limiting_db"] = round(_limiting_estimate(renderer), 2)
    return Evaluation(score=round(sum(f.severity for f in findings), 3), findings=findings, metrics=metrics)


def _limiting_estimate(renderer) -> float:
    """dB que el limitador quita a los picos: cresta del premaster − cresta del master."""
    pre = renderer.post.get(renderer.graph.master)
    m = getattr(renderer, "last_master", None)
    if pre is None or m is None:
        return 0.0
    crest = lambda x: 20 * np.log10(np.abs(x).max() / (np.sqrt(np.mean(x ** 2)) + 1e-12) + 1e-12)
    return float(crest(pre) - crest(m))
