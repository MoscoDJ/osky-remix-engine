"""Juego de sonidos: presets de Surge de usuario y kits de samples para un arreglo.

Archivo JSON en ``workspace/sounds/<nombre>.json`` (ignorado por git: los packs comerciales no se
pueden redistribuir; el proyecto guarda los assets por SHA-256, así que el render es reproducible
aunque el archivo original cambie de lugar)::

    {"instruments": {"bass": {"fxp": "/ruta/HoHo.fxp"},            # o {"surge_preset": "Basses/X"}
                     "pad":  {"fxp": "...", "params": {...}}},
     "kits": {"kick":  {"36": {"sample": "<sha o prefijo en la biblioteca>", "gain_db": "match",
                               "pitch_st": "key"}},
              "drums": {"39": {"sample": "...", "gain_db": -2}},
              "fx":    {"1": {"sample": "...", "anchor": "end", "gain_db": "match"}}}}

- ``gain_db: "match"``: iguala el sample al instrumento integrado al que sustituye (máximo de RMS en
  ventanas de 50 ms; 400 ms para FX), para que los faders de la plantilla sigan valiendo.
- ``pitch_st: "key"``: transpone el sample (con nota medida) a la tónica o la quinta de la
  tonalidad del arreglo más cercana; si el sample no tiene nota fiable, no se transpone.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.config import workspace
from ..core.models import Asset, AssetKind, Provenance
from .features import NOTE_NAMES
from .kit import kit_bytes, repitch

SOUND_ROLES_KIT = {"kick": "drums", "drums": "drums", "fx": "fx"}
LINEAR_FX = {"highpass", "lowpass", "eq", "gain", "sidechain", "stereo", "reverb", "delay", "chorus"}
STAGING_PEAK_DBFS = -1.0


@dataclass
class ResolvedSounds:
    presets: dict[str, str] = field(default_factory=dict)       # rol -> sha (fxp o kit)
    assets: list[Asset] = field(default_factory=list)
    params: dict[str, dict] = field(default_factory=dict)       # rol -> parámetros estáticos de Surge
    log: list[str] = field(default_factory=list)


def sounds_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        return p
    q = workspace() / "sounds" / f"{name_or_path}.json"
    if not q.exists():
        raise ValueError(f"no existe el juego de sonidos '{name_or_path}' ({q})")
    return q


def _win_peak_rms(x: np.ndarray, sr: int, win_s: float) -> float:
    m = x.mean(1) if x.ndim == 2 else x
    w = max(1, int(win_s * sr))
    if len(m) < w:
        return float(np.sqrt(np.mean(m ** 2)) + 1e-12)
    c = np.cumsum(np.concatenate([[0.0], m.astype(np.float64) ** 2]))
    return float(np.sqrt((c[w:] - c[:-w]).max() / w) + 1e-12)


def builtin_reference(role: str, pitch: int, sr: int) -> tuple[np.ndarray, float]:
    """Golpe del instrumento integrado que el sample sustituye y su ventana de medida."""
    from ..audio import synth
    if SOUND_ROLES_KIT.get(role) == "fx":
        return synth.fx_hit(pitch, 4 * 4 * 60 / 124, 110, sr, 0), 0.4
    return synth.drum_hit(pitch, 112, sr, 0), 0.05


def match_gain(role: str, pitch: int, x: np.ndarray, sr: int, vel: int = 112) -> float:
    ref, win = builtin_reference(role, pitch, sr)
    if not len(ref) or not np.any(ref):
        return 0.0
    vel_g = vel / 127 if SOUND_ROLES_KIT.get(role) != "fx" else 110 / 127
    return float(20 * np.log10(_win_peak_rms(ref, sr, win) / (_win_peak_rms(x, sr, win) * vel_g)))


def key_shift(note: str | None, key: str, max_st: int = 4) -> float:
    """Semitonos hacia la tónica o la quinta más cercanas (|Δ| ≤ max_st); 0 si no aplica."""
    if not note:
        return 0.0
    name = note.rstrip("-0123456789")
    if name not in NOTE_NAMES:
        return 0.0
    import re
    m = re.match(r"\s*([A-G])([#b]?)", key or "")
    if not m:
        return 0.0
    flats = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}
    root = m.group(1) + m.group(2)
    tonic = NOTE_NAMES.index(flats.get(root, root.replace("E#", "F").replace("B#", "C")))
    pc = NOTE_NAMES.index(name)
    best = min(((t - pc + 6) % 12 - 6 for t in (tonic, (tonic + 7) % 12)), key=abs)
    return float(best) if abs(best) <= max_st else 0.0


def resolve(spec_path: Path, repo, key: str, sr: int, lib=None) -> ResolvedSounds:
    """Registra en el almacén del proyecto los presets y samples del juego y construye los kits."""
    import soundfile as sf
    spec = json.loads(Path(spec_path).read_text())
    out = ResolvedSounds()
    for role, s in spec.get("instruments", {}).items():
        if "fxp" in s:
            f = Path(s["fxp"]).expanduser()
            tool, params = "surge-xt-user", {"file": f.name, "pack": f.parent.name}
        else:
            from ..audio.plugin_host import SURGE_FACTORY
            f = SURGE_FACTORY / f"{s['surge_preset']}.fxp"
            tool, params = "surge-xt-factory", {"preset": s["surge_preset"]}
        if not f.exists():
            raise ValueError(f"{role}: no existe el preset {f}")
        a = repo.assets.ingest(f, AssetKind.preset, Provenance(tool=tool, params=params), name=f.name)
        out.presets[role] = a.sha256
        out.assets.append(a)
        if s.get("params"):
            out.params[role] = s["params"]
        out.log.append(f"{role}: {f.stem}")
    if spec.get("kits"):
        if lib is None:
            from .library import SampleLibrary
            lib = SampleLibrary()
        for role, voices in spec["kits"].items():
            if role not in SOUND_ROLES_KIT:
                raise ValueError(f"kit para un rol sin instrumento de samples: {role} (válidos: {sorted(SOUND_ROLES_KIT)})")
            built = {}
            for pitch_s, v in voices.items():
                pitch = int(pitch_s)
                h = lib.resolve(v["sample"])
                src = lib.path(h.sha)
                smp = repo.assets.ingest(src, AssetKind.sample, Provenance(
                    tool="sample-library", params={"library_sha": h.sha, "category": h.category}), name=h.name)
                out.assets.append(smp)
                pst = v.get("pitch_st", 0.0)
                if pst == "key":
                    pst = key_shift(h.features.get("note"), key)
                gain = v.get("gain_db", 0.0)
                if gain == "match":
                    x, xsr = sf.read(str(src), dtype="float32", always_2d=True)
                    if xsr != sr:
                        from math import gcd

                        from scipy.signal import resample_poly
                        g = gcd(xsr, sr)
                        x = resample_poly(x, sr // g, xsr // g, axis=0)
                    gain = round(match_gain(role, pitch, repitch(x, float(pst)), sr), 2) + float(v.get("trim_db", 0.0))
                built[pitch] = {"sample": smp.sha256, "gain_db": float(gain), "name": h.name,
                                "pitch_st": float(pst), "anchor": v.get("anchor", "start")}
                out.log.append(f"{role} {pitch}: {h.name} ({float(gain):+.1f} dB"
                               + (f", {float(pst):+.0f} st" if pst else "") + ")")
            with tempfile.TemporaryDirectory() as td:
                f = Path(td) / f"kit-{role}.json"
                f.write_bytes(kit_bytes(built))
                kit = repo.assets.ingest(f, AssetKind.preset, Provenance(
                    tool="osky-kit", params={"role": role, "sounds": Path(spec_path).stem},
                    parents=[b["sample"] for b in built.values()]), name=f"kit {role}.json")
            out.presets[role] = kit.sha256
            out.assets.append(kit)
    return out


def swap_ops(project, rs: ResolvedSounds, assets, cache, level_match: bool = True,
             log=lambda *_: None) -> tuple[list[dict], dict[str, dict]]:
    """Operaciones que cambian los instrumentos de las pistas existentes por los del juego.

    Igualación de nivel: se renderiza el instrumento de la pista (sus notas reales, sin efectos ni
    fader) con el sonido anterior y con el nuevo, y la diferencia de sonoridad integrada se
    compensa con un trim de entrada (efecto ``gain`` justo después del instrumento, id
    ``<pista>-sound-trim``). El nivel dentro de la cadena queda como antes (compresores,
    sidechain y gain staging no cambian) y los faders, que son decisiones de mezcla del agente o
    del usuario, no se tocan. Excepción de gain staging: si el sonido nuevo tiene más cresta y la
    cadena quedaría por encima de −1 dBFS, el exceso pasa al fader solo cuando la cadena es
    lineal y no hay envíos pre-fader (misma salida exacta). Las pistas bloqueadas no se tocan.
    """
    from ..audio.renderer import Renderer
    from ..core.operations import Context
    from ..core.transactions import apply_ops, parse_ops
    from ..mastering.loudness import integrated_lufs
    ops: list[dict] = [{"op": "register_asset", "asset": a.model_dump(mode="json")}
                       for a in {a.sha256: a for a in rs.assets}.values() if a.sha256 not in project.assets]
    report: dict[str, dict] = {}
    swaps = []
    for role, sha in rs.presets.items():
        t = next((x for x in project.tracks if x.id == role), None)
        if t is None or not t.plugins:
            report[role] = {"skipped": "la pista no existe"}
            continue
        if t.locked:
            report[role] = {"skipped": "pista bloqueada"}
            continue
        old = t.plugins[0]
        if role in SOUND_ROLES_KIT:
            new = old.model_copy(update={"preset": sha})
        else:
            new = old.model_copy(update={"backend": "surge_xt", "uri": "Surge XT.vst3", "preset": sha,
                                         "params": rs.params.get(role, {})})
        swaps.append((t, old, new))
        ops.append({"op": "replace_plugin", "track": role, "plugin_id": old.id, "plugin": new.model_dump(mode="json")})
    if level_match and swaps:
        after = apply_ops(project, parse_ops(ops), Context(author="claude"))
        r0, r1 = Renderer(project, assets, cache), Renderer(after, assets, cache)
        for t, old, new in swaps:
            log(f"midiendo {t.id}…")
            l0 = integrated_lufs(r0._own(t), project.sample_rate)
            own1 = r1._own(next(x for x in after.tracks if x.id == t.id))
            l1 = integrated_lufs(own1, project.sample_rate)
            peak1 = float(20 * np.log10(np.abs(own1).max() + 1e-12))
            if not (np.isfinite(l0) and np.isfinite(l1)):
                report[t.id] = {"lufs_before": l0, "lufs_after": l1, "trim_db": 0.0}
                continue
            tid = f"{t.id}-sound-trim"
            prev = next((x for x in t.plugins if x.id == tid), None)
            prev_db = float(prev.params.get("db", 0.0)) if prev else 0.0
            trim = float(np.clip(prev_db + l0 - l1, -24, 24))
            # gain staging: con una cadena lineal y sin envíos pre-fader, mover ganancia del trim al
            # fader da la misma salida; se deja la cadena con pico ≤ STAGING_PEAK_DBFS
            fader_db = 0.0
            linear = all(x.uri in LINEAR_FX for x in t.plugins[1:] if x.id != tid) and not any(sd.pre_fader for sd in t.sends)
            if linear and peak1 + trim > STAGING_PEAK_DBFS:
                fader_db = peak1 + trim - STAGING_PEAK_DBFS
                trim -= fader_db
            report[t.id] = {"lufs_before": round(l0 + prev_db, 2), "lufs_after": round(l1, 2), "trim_db": round(trim, 2),
                            "fader_db": round(fader_db, 2)}
            if fader_db:
                ops.append({"op": "set_track", "track": t.id, "gain_db": round(t.gain_db + fader_db, 2)})
            slot = {"id": tid, "backend": "builtin", "uri": "gain", "params": {"db": round(trim, 2)}}
            ops.append({"op": "replace_plugin", "track": t.id, "plugin_id": tid, "plugin": slot} if prev else
                       {"op": "add_plugin", "track": t.id, "plugin": slot, "index": 1})
    return ops, report
