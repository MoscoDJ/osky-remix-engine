"""Transposición no destructiva de assets de audio (voz u otro) y tonalidad destino.

- ``pitch_shift``: Rubber Band (motor R3, ``--fine``), duración intacta; con ``formant=True``
  conserva la envolvente de formantes (``-F``), lo natural para una voz.
- ``transpose_asset``: crea un asset NUEVO con procedencia (el original no se toca), en caché
  por (sha del asset, semitonos, opciones, versión de Rubber Band).
- ``transposition_to_key``: la transposición mínima (−6…+5 semitonos) que lleva una
  tonalidad a otra. Si los modos difieren se compara con la relativa (A minor → C major = 0:
  mismas notas). El tritono (±6) se resuelve siempre hacia abajo (−6): bajar una voz suele
  degradar menos que subirla.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..core.cache import cache_key
from ..core.models import Asset, AssetKind, Provenance
from .stretch import rubberband_version

NOTE = "C C# D D# E F F# G G# A A# B".split()
FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}
_KEY_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)\s*(m|min|minor|maj|major|menor|mayor)?\s*$")
MAX_SEMITONES = 24


def parse_key(key: str) -> tuple[int, bool]:
    """'A minor', 'Am', 'F#', 'Bb major' (notación inglesa) -> (clase de altura, es menor)."""
    m = _KEY_RE.match(key or "")
    if not m:
        raise ValueError(f"tonalidad no reconocida: '{key}' (p. ej. 'A minor', 'Am', 'F# major')")
    root = m.group(1).upper() + m.group(2)
    root = FLAT.get(root, root)
    return NOTE.index(root), (m.group(3) or "") in ("m", "min", "minor", "menor")


def key_label(pc: int, minor: bool) -> str:
    return f"{NOTE[pc % 12]} {'minor' if minor else 'major'}"


def transposition_to_key(src_key: str, dst_key: str) -> int:
    """Semitonos (−6…+5) que llevan ``src_key`` a ``dst_key`` con el menor desplazamiento."""
    spc, smin = parse_key(src_key)
    dpc, dmin = parse_key(dst_key)
    if smin != dmin:                        # modos distintos: se compara la relativa mayor
        spc, dpc = (spc + 3 * smin) % 12, (dpc + 3 * dmin) % 12
    d = (dpc - spc) % 12
    return d - 12 if d >= 6 else d


def pitch_shift(src: Path, dst: Path, semitones: float, formant: bool = True) -> Path:
    """Transpone ``semitones`` sin cambiar la duración. Salida float32 a la misma frecuencia."""
    args = ["rubberband", "-q", "--fine", "-T", "1", "-p", f"{semitones:.6f}"]
    if formant:
        args.append("-F")
    p = subprocess.run(args + [str(src), str(dst)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("rubberband falló: " + p.stderr[-300:])
    # la duración debe coincidir exactamente para que los clips sigan siendo válidos
    n = sf.info(str(src)).frames
    y, sr = sf.read(str(dst), dtype="float32", always_2d=True)
    if len(y) != n:
        y = np.pad(y, ((0, max(0, n - len(y))), (0, 0)))[:n]
        sf.write(str(dst), y, sr, subtype="FLOAT")
    return dst


def transpose_asset(repo, sha: str, semitones: float, formant: bool = True) -> tuple[Asset, bool]:
    """Asset transpuesto (nuevo, con procedencia) y si vino de la caché. No registra nada."""
    if not -MAX_SEMITONES <= semitones <= MAX_SEMITONES:
        raise ValueError(f"semitonos fuera de rango (±{MAX_SEMITONES}): {semitones}")
    proj = repo.load()
    src_asset = proj.assets.get(sha)
    src = repo.assets.find(sha)
    if src is None:
        raise ValueError(f"el archivo del asset {sha[:12]} no está en el almacén")
    rb = rubberband_version()
    params = {"semitones": round(float(semitones), 6), "formant": bool(formant), "engine": "R3"}
    key = cache_key("pitch_shift", "1", [sha], {**params, "rubberband": rb})
    hit = repo.cache.get_json(key)
    if hit and repo.assets.exists(hit["asset"]["sha256"]):
        known = proj.assets.get(hit["asset"]["sha256"])
        return (known or Asset.model_validate(hit["asset"])), True
    base = Path(src_asset.name).stem if src_asset else sha[:12]
    sign = f"{semitones:+g}"
    with tempfile.TemporaryDirectory() as td:
        out = pitch_shift(src, Path(td) / "transpuesto.wav", semitones, formant)
        kind = AssetKind.stem if src_asset is None else (
            AssetKind.working if src_asset.kind == AssetKind.original else src_asset.kind)
        asset = repo.assets.ingest(out, kind,
                                   Provenance(tool="rubberband", tool_version=rb, params=params, parents=[sha]),
                                   name=f"{base} ({sign} st).wav")
    repo.cache.put_json(key, {"asset": asset.model_dump(mode="json")})
    return asset, False
