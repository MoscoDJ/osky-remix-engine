"""Instrumentos integrados (deterministas): batería electrónica, FX de transición y un
sintetizador sustractivo de respaldo cuando Surge XT no está disponible."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from . import SEED


def _rng(pitch: int, start: int) -> np.random.Generator:
    return np.random.default_rng((SEED * 1_000_003 + pitch * 7919 + start) % (2 ** 32))


def _bp(x, lo, hi, sr, o=2):
    return sosfilt(butter(o, [lo, min(hi, sr / 2 * 0.95)], "band", fs=sr, output="sos"), x)


def _hp(x, f, sr, o=2):
    return sosfilt(butter(o, f, "high", fs=sr, output="sos"), x)


DRUM_LEVEL = 0.5   # -6 dB: margen antes del fader (gain staging); se compensa con el fader
FX_LEVEL = 0.5


def drum_hit(pitch: int, vel: int, sr: int, start: int) -> np.ndarray:
    g = vel / 127 * DRUM_LEVEL
    rng = _rng(pitch, start)
    if pitch == 36:                                                      # bombo
        n = int(0.45 * sr); t = np.arange(n) / sr
        f = 47 + 175 * np.exp(-t * 34)
        body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 6.0)
        click = _hp(rng.standard_normal(n), 3000, sr) * np.exp(-t * 350) * 0.25
        return np.tanh(2.4 * (body + click)) * g * 0.95
    if pitch in (38, 39, 37):                                            # caja, palmas, rim
        n = int(0.4 * sr); t = np.arange(n) / sr
        if pitch == 39:
            env = sum(np.where(t >= o, np.exp(-(t - o) * 190), 0) for o in (0, 0.010, 0.021)) + 0.6 * np.exp(-t * 13)
            return _bp(rng.standard_normal(n), 900, 3500, sr) * env * g * 0.7
        if pitch == 37:
            x = _bp(rng.standard_normal(n), 1800, 6000, sr) * np.exp(-t * 60) + np.sin(2 * np.pi * 830 * t) * np.exp(-t * 70) * 0.4
            return x * g * 0.5
        x = _bp(rng.standard_normal(n), 1500, 8000, sr) * np.exp(-t * 26) + np.sin(2 * np.pi * 185 * t) * np.exp(-t * 32) * 0.6
        return x * g * 0.6
    if pitch in (42, 46):                                                # hats
        d = 0.22 if pitch == 46 else 0.045
        n = int(d * 3 * sr); t = np.arange(n) / sr
        return _hp(rng.standard_normal(n), 7500, sr, 4) * np.exp(-t / d * 2.4) * g * 0.45
    if pitch == 51:                                                      # ride
        n = int(0.7 * sr); t = np.arange(n) / sr
        m = sum(np.sin(2 * np.pi * f * t) for f in (3520, 4870, 6230, 7710))
        return (_hp(rng.standard_normal(n), 5000, sr) * 0.6 + m * 0.12) * np.exp(-t * 6) * g * 0.35
    if pitch == 49:                                                      # crash
        n = int(2.2 * sr); t = np.arange(n) / sr
        return _hp(rng.standard_normal(n), 3500, sr) * np.exp(-t * 1.8) * g * 0.35
    return np.zeros(1)


def fx_hit(pitch: int, dur_s: float, vel: int, sr: int, start: int) -> np.ndarray:
    """1 = riser (ruido con barrido ascendente), 2 = impacto, 3 = downlifter."""
    g = vel / 127 * FX_LEVEL
    rng = _rng(pitch, start)
    if pitch in (1, 3):
        n = max(1, int(dur_s * sr)); x = rng.standard_normal(n); out = np.zeros(n); zi = None
        for s in range(0, n, 1024):
            prog = s / n if pitch == 1 else 1 - s / n
            fc = 300 * (9000 / 300) ** prog
            sos = butter(2, [fc * 0.7, min(fc * 1.4, sr / 2 * 0.95)], "band", fs=sr, output="sos")
            if zi is None:
                zi = np.zeros((sos.shape[0], 2))
            out[s:s + 1024], zi = sosfilt(sos, x[s:s + 1024], zi=zi)
        env = np.linspace(0, 1, n) ** 2 if pitch == 1 else np.linspace(1, 0, n) ** 2
        return out * env * g * 0.6
    if pitch == 2:
        n = int(3 * sr); t = np.arange(n) / sr
        boom = np.sin(2 * np.pi * np.cumsum(38 + 70 * np.exp(-t * 9)) / sr) * np.exp(-t * 2.0)
        return (boom * 0.9 + _hp(rng.standard_normal(n), 3000, sr) * np.exp(-t * 1.6) * 0.3) * g
    return np.zeros(1)


def subtractive(freq: float, dur_s: float, vel: int, sr: int, bright: float = 0.5, release: float = 0.15) -> np.ndarray:
    """Respaldo: dientes de sierra desafinados + filtro con envolvente."""
    n = int((dur_s + release) * sr); t = np.arange(n) / sr
    x = sum(2 * ((t * freq * 2 ** (d / 1200)) % 1) - 1 for d in (-7, 0, 7)) / 3
    fc = 200 + 6000 * bright * np.exp(-t * 6)
    y = np.zeros(n); zi = None
    for s in range(0, n, 512):
        sos = butter(2, min(float(fc[s]), sr / 2 * 0.9), "low", fs=sr, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        y[s:s + 512], zi = sosfilt(sos, x[s:s + 512], zi=zi)
    env = np.minimum(1, t / 0.005) * np.where(t < dur_s, 1.0, np.exp(-(t - dur_s) / (release / 3)))
    return y * env * vel / 127 * 0.5
