"""Medición de true peak por sobremuestreo (ITU-R BS.1770-4, anexo 2)."""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from .loudness import _as_frames

OVERSAMPLE = 4


def sample_peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio)))
    return 20 * np.log10(peak) if peak > 0 else -np.inf


def true_peak_dbtp(audio: np.ndarray, sr: int | None = None, oversample: int = OVERSAMPLE) -> float:
    """True peak en dBTP. Con 4x la norma admite un error de hasta ~0.5 dB a fs/2."""
    frames = _as_frames(audio)
    up = resample_poly(frames, oversample, 1, axis=0)
    peak = float(np.max(np.abs(up)))
    return 20 * np.log10(peak) if peak > 0 else -np.inf
