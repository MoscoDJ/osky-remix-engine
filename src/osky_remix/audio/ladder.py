"""Filtro escalera (Moog) lineal: el ``LadderFilter`` de JUCE/pedalboard sin su saturación.

El ``LadderFilter`` de pedalboard satura (``tanh``) la entrada y la realimentación: con un seno de
200 Hz da THD −45 dB a −12 dBFS, −34 dB a −6 dBFS y −23 dB a 0 dBFS, y comprime ~0.5 dB a 0 dBFS.
En el motor eso hacía que el timbre de cada pista dependiera de su nivel (un trim antes del
filtro lo distorsionaba). Este módulo reproduce la misma topología de JUCE (``juce_LadderFilter``)
con la no linealidad sustituida por la identidad, así que a nivel bajo la respuesta coincide con
la del original y a cualquier nivel es lineal:

- ``a1 = exp(−2π·fc/sr)``, ``g = 1 − a1``; cada etapa ``yₙ = b0·xₙ + b1·xₙ₋₁ + a1·yₙ₋₁`` con
  ``b0 = g·10/13`` y ``b1 = g·3/13`` (cero en −0.3).
- Entrada ``a = dx − 4R·(s₄ − comp·dx)``, ``R = 0.1 + 0.9·resonancia`` (JUCE mapea 0..1 a 0.1..1),
  ``s₄`` la 4.ª etapa de la muestra anterior.
- Salida ``1.2·Σ Aᵢ·sᵢ``: LP24 = s₄, LP12 = s₂, HP24 = s₀ − 4s₁ + 6s₂ − 4s₃ + s₄,
  HP12 = s₀ − 2s₁ + s₂; ``comp`` = 0.5 en paso bajo y 0 en paso alto.

Frecuencia y resonancia pueden ser curvas por muestra (automatización sin saltos de estado).
"""
from __future__ import annotations

import numpy as np
from numba import njit

MODES = {
    "LP24": (np.array([0, 0, 0, 0, 1.0]), 0.5),
    "LP12": (np.array([0, 0, 1.0, 0, 0]), 0.5),
    "HP24": (np.array([1.0, -4, 6, -4, 1]), 0.0),
    "HP12": (np.array([1.0, -2, 1, 0, 0]), 0.0),
}
OUTPUT_GAIN = 1.2
DRIVE_GAIN = 0.6103 + 0.3903          # gain de JUCE con drive = 1 (aplicado a entrada y realimentación)


@njit(cache=True)
def _kernel(x, a1, res, A, comp, gin):
    n, ch = x.shape
    y = np.empty_like(x)
    for c in range(ch):
        s0 = s1 = s2 = s3 = s4 = 0.0
        for i in range(n):
            a = a1[i]
            g = 1.0 - a
            b0 = g * 0.76923076923
            b1 = g * 0.23076923076
            dx = gin * x[i, c]
            v0 = dx + res[i] * -4.0 * (gin * s4 - dx * comp)
            v1 = b1 * s0 + a * s1 + b0 * v0
            v2 = b1 * s1 + a * s2 + b0 * v1
            v3 = b1 * s2 + a * s3 + b0 * v2
            v4 = b1 * s3 + a * s4 + b0 * v3
            s0, s1, s2, s3, s4 = v0, v1, v2, v3, v4
            y[i, c] = OUTPUT_GAIN * (A[0] * v0 + A[1] * v1 + A[2] * v2 + A[3] * v3 + A[4] * v4)
    return y


def _curve(v, n: int) -> np.ndarray:
    if np.isscalar(v):
        return np.full(n, float(v))
    v = np.asarray(v, dtype=np.float64)
    return v[:n] if len(v) >= n else np.concatenate([v, np.full(n - len(v), v[-1])])


def ladder(x: np.ndarray, sr: int, mode: str, fc, resonance=0.0) -> np.ndarray:
    """Filtra ``x`` (n, canales). ``mode``: LP24, LP12, HP24, HP12. ``fc``/``resonance``: escalar
    o curva por muestra."""
    if mode not in MODES:
        raise ValueError(f"modo de filtro desconocido: {mode}")
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if x.ndim == 1:
        return ladder(x[:, None], sr, mode, fc, resonance)[:, 0]
    n = len(x)
    f = np.clip(_curve(fc, n), 1.0, sr * 0.49)
    r = 0.1 + 0.9 * np.clip(_curve(resonance, n), 0.0, 1.0)
    A, comp = MODES[mode]
    return _kernel(x, np.exp(-2 * np.pi * f / sr), r, A, comp, DRIVE_GAIN).astype(np.float32)
