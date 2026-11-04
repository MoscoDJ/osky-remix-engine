"""Cambio de tempo y tono con Rubber Band (CLI), conservando formantes."""
from __future__ import annotations

import subprocess
from pathlib import Path


def rubberband_version() -> str:
    p = subprocess.run(["rubberband", "--version"], capture_output=True, text=True)
    return (p.stdout or p.stderr).strip().splitlines()[0]


def stretch(src: Path, dst: Path, tempo_ratio: float, semitones: float = 0.0) -> Path:
    """``tempo_ratio`` > 1 acelera (bpm_destino / bpm_origen). Salida float32."""
    args = ["rubberband", "-q", "-F", "--fine", "-T", f"{tempo_ratio:.10f}"]
    if semitones:
        args += ["-p", f"{semitones:.6f}"]
    p = subprocess.run(args + [str(src), str(dst)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("rubberband falló: " + p.stderr[-300:])
    return dst
