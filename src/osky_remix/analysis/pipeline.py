"""Pipeline de análisis: ritmo, armonía, estructura, características, silencios, transitorios
y actividad vocal. Produce un ``AnalysisReport`` serializable y un resumen en Markdown."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import BaseModel

from .features import Features, Transient, analyze_features, detect_silences, detect_transients
from .harmony import Harmony, analyze_harmony
from .structure import Structure, analyze_structure
from .tempo import Rhythm, analyze_rhythm

ANALYSIS_VERSION = "2.1.0"


class AnalysisReport(BaseModel):
    version: str = ANALYSIS_VERSION
    sample_rate: int
    duration_samples: int
    sources: dict[str, str]                 # qué señal se usó para cada análisis
    rhythm: Rhythm
    harmony: Harmony
    structure: Structure
    features: Features
    silences: list[tuple[int, int]]
    transients: list[Transient]
    vocal_activity: list[tuple[int, int]] | None = None
    timings_s: dict[str, float]


def _load(p: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    return x, sr


def vocal_activity(vox: np.ndarray, sr: int, thr_dbfs: float = -42.0, min_gap_s: float = 0.8,
                   min_len_s: float = 0.5) -> list[tuple[int, int]]:
    m = np.abs(vox).max(axis=1) if vox.ndim == 2 else np.abs(vox)
    hop = int(0.05 * sr)
    n = len(m) // hop
    lvl = 20 * np.log10(np.sqrt(np.mean(m[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-12)
    on = lvl > thr_dbfs
    regs, i = [], 0
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            regs.append([i * hop, j * hop])
            i = j
        else:
            i += 1
    merged: list[list[int]] = []
    for a, b in regs:
        if merged and a - merged[-1][1] < min_gap_s * sr:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= min_len_s * sr]


def run_analysis(mix_path: Path, stems: dict[str, Path] | None = None) -> AnalysisReport:
    stems = stems or {}
    t = {}
    mix, sr = _load(mix_path)
    t0 = time.time(); rhythm = analyze_rhythm(mix, sr); t["rhythm"] = time.time() - t0
    src = {"rhythm": "mezcla"}
    harm_sig = mix
    if "bass" in stems and "other" in stems:
        b, _ = _load(stems["bass"]); o, _ = _load(stems["other"])
        n = min(len(b), len(o), len(mix))
        harm_sig = b[:n] + o[:n]
        src["harmony"] = "stems bass+other (sin voz ni batería)"
    elif "instrumental" in stems:
        harm_sig, _ = _load(stems["instrumental"])
        src["harmony"] = "stem instrumental"
    else:
        src["harmony"] = "mezcla"
    t0 = time.time(); harmony = analyze_harmony(harm_sig, sr, rhythm.beats); t["harmony"] = time.time() - t0
    t0 = time.time(); structure = analyze_structure(mix, sr, rhythm.downbeats, rhythm.beats); t["structure"] = time.time() - t0
    src["structure"] = "mezcla"
    t0 = time.time()
    feats = analyze_features(mix, sr)
    sil = detect_silences(mix, sr)
    trans = detect_transients(mix, sr)
    t["features"] = time.time() - t0
    va = None
    vkey = next((k for k in ("vocals", "lead_vocals") if k in stems), None)
    if vkey:
        v, _ = _load(stems[vkey])
        va = vocal_activity(v, sr)
        src["vocal_activity"] = f"stem {vkey}"
    return AnalysisReport(sample_rate=sr, duration_samples=len(mix), sources=src, rhythm=rhythm, harmony=harmony,
                          structure=structure, features=feats, silences=sil, transients=trans, vocal_activity=va,
                          timings_s={k: round(v, 2) for k, v in t.items()})


def _ts(samples: int, sr: int) -> str:
    s = samples / sr
    return f"{int(s // 60)}:{s % 60:04.1f}"


def report_markdown(r: AnalysisReport, title: str = "Análisis") -> str:
    sr = r.sample_rate
    rh, hm, st, f = r.rhythm, r.harmony, r.structure, r.features
    L = [f"# {title}", "",
         f"Duración {_ts(r.duration_samples, sr)} · {sr} Hz · análisis v{r.version}", "",
         "## Resumen", "",
         "| Estimación | Valor | Confianza | Alternativas / notas |", "|---|---|---|---|",
         f"| Tempo | {rh.tempo.value} BPM | {rh.tempo.confidence:.2f} | "
         + ", ".join(f"{a.value}" for a in rh.tempo.alternatives[:2]) + f"; {rh.tempo.notes} |",
         f"| Compás | {rh.meter.value}/4 | {rh.meter.confidence:.2f} | {rh.meter.notes or '-'} |",
         f"| Tiempo fuerte | fase {rh.downbeat.value} | {rh.downbeat.confidence:.2f} | {len(rh.phase_changes)} cambios de fase |",
         f"| Tonalidad | {hm.key.value} | {hm.key.confidence:.2f} | "
         + (", ".join(a.value for a in hm.key.alternatives) or "-") + f"; {hm.key.notes} |", ""]
    if len(rh.segments) > 1 or rh.free_tempo:
        L += ["## Tempo por tramos", ""]
        L += [f"- {_ts(s.start, sr)}–{_ts(s.end, sr)}: {s.bpm} BPM (residuo {s.residual_ms} ms)" for s in rh.segments]
        L += [f"- Tempo libre {_ts(a, sr)}–{_ts(b, sr)}" for a, b in rh.free_tempo] + [""]
    if rh.phase_changes:
        L += ["## Cambios de fase del compás", ""]
        L += [f"- {_ts(p.sample, sr)}: corrimiento de {p.beats_shift} tiempos (compás irregular)" for p in rh.phase_changes] + [""]
    if len(hm.key_segments) > 1:
        L += ["## Tonalidad por tramos", ""] + [f"- {_ts(k.start, sr)}–{_ts(k.end, sr)}: {k.key} ({k.confidence:.2f})"
                                                 for k in hm.key_segments] + [""]
    L += ["## Secciones", "", "| Inicio | Fin | Etiqueta | Compases | Energía | Confianza frontera |", "|---|---|---|---|---|---|"]
    L += [f"| {_ts(s.start, sr)} | {_ts(s.end, sr)} | {s.label} | {s.bars} | {s.energy_db} dB | {s.confidence:.2f} |"
          for s in st.sections] + [""]
    prog = [c.label for c in hm.chords if c.label != "N"]
    L += ["## Acordes (primeros 32 cambios)", "", " → ".join(prog[:32]) or "-", ""]
    if r.vocal_activity is not None:
        L += ["## Voz", ""] + [f"- {_ts(a, sr)}–{_ts(b, sr)}" for a, b in r.vocal_activity] + [""]
    L += ["## Nivel y espectro", "",
          f"- {f.lufs} LUFS, LRA ≈ {f.loudness_range_lu} LU, true peak {f.true_peak_dbtp} dBTP, cresta {f.crest_db} dB",
          f"- Centroide {f.spectral_centroid_hz} Hz, rolloff {f.spectral_rolloff_hz} Hz, planitud {f.spectral_flatness}",
          "- Bandas: " + ", ".join(f"{k} {v:.0%}" for k, v in f.band_energy.items()),
          f"- Estéreo: correlación {f.stereo_correlation}, lado/centro {f.side_to_mid_db} dB",
          f"- Recortes: {f.clipped_samples} muestras; DC {f.dc_offset}",
          f"- Silencios: " + (", ".join(f"{_ts(a, sr)}–{_ts(b, sr)}" for a, b in r.silences) or "ninguno"),
          f"- Transitorios: {len(r.transients)}", "",
          "## Fuentes", ""] + [f"- {k}: {v}" for k, v in r.sources.items()]
    return "\n".join(L) + "\n"
