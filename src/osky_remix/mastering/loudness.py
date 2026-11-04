"""Medición de sonoridad integrada (ITU-R BS.1770-4 / EBU R128) vía pyloudnorm."""
from __future__ import annotations

import numpy as np
import pyloudnorm as pyln


def _as_frames(audio: np.ndarray) -> np.ndarray:
    """Acepta (n,), (n, ch) o (ch, n) con ch<=8 y devuelve (n, ch)."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        return a[:, None]
    if a.shape[0] <= 8 and a.shape[1] > a.shape[0]:
        return a.T
    return a


def integrated_lufs(audio: np.ndarray, sr: int) -> float:
    x = _as_frames(audio)
    if len(x) < int(np.ceil(.4 * sr)):
        x = np.pad(x, ((0, int(np.ceil(.4 * sr)) - len(x)), (0, 0)))
    return float(pyln.Meter(sr).integrated_loudness(x))


def gain_to_target(audio: np.ndarray, sr: int, target_lufs: float) -> float:
    """Ganancia en dB necesaria para llevar ``audio`` a ``target_lufs``."""
    return target_lufs - integrated_lufs(audio, sr)
