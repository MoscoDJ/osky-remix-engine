"""De-esser de banda dividida (efecto integrado ``deesser``).

Diseño:
- La banda de sibilancia se aísla con un pasabanda Butterworth de fase cero (``sosfiltfilt``)
  centrado en ``freq_hz`` con ``width_oct`` octavas de ancho (por defecto ≈ 5–9 kHz).
- Como el filtro es de fase cero, ``resto = x - banda`` es su complemento exacto: con ganancia
  unitaria la salida es idéntica a la entrada y no hay desfase ni latencia en ninguna banda.
- Detector: RMS de la banda (estéreo enlazado) en ventanas centradas de ~1 ms, evaluado cada
  ``HOP`` muestras. La reducción es ``(nivel - umbral) · (1 - 1/ratio)``, limitada a ``range_db``,
  con ataque y liberación en dB; la ganancia se interpola a resolución de muestra.
- Solo la banda se atenúa: ``salida = resto + banda · g``. El contenido fuera de la banda
  (p. ej. 200–2000 Hz de la voz) queda intacto.

La ventana centrada actúa como un pequeño lookahead del detector (offline), pero el audio no
se retrasa: la latencia reportada es 0 muestras.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

HOP = 32
LATENCY_SAMPLES = 0


def band_edges(sr: int, freq_hz: float = 6700.0, width_oct: float = 0.85) -> tuple[float, float]:
    if freq_hz <= 0 or width_oct <= 0:
        raise ValueError("deesser: freq_hz y width_oct deben ser positivos")
    lo = freq_hz * 2 ** (-width_oct / 2)
    hi = min(freq_hz * 2 ** (width_oct / 2), 0.45 * sr)
    if lo >= hi:
        raise ValueError(f"deesser: banda {lo:.0f}–{hi:.0f} Hz inválida a {sr} Hz")
    return lo, hi


def sibilance_band(x: np.ndarray, sr: int, freq_hz: float = 6700.0, width_oct: float = 0.85) -> np.ndarray:
    lo, hi = band_edges(sr, freq_hz, width_oct)
    sos = butter(4, [lo, hi], "bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _per_frame(v, centers: np.ndarray) -> np.ndarray:
    """Escalar o curva por muestra (automatización) -> valor en cada centro de bloque."""
    if np.isscalar(v):
        return np.full(len(centers), float(v))
    a = np.asarray(v, dtype=np.float64)
    return a[np.minimum(centers, len(a) - 1)]


def deess_gain(band: np.ndarray, sr: int, threshold_db=-30.0, range_db=6.0, ratio: float = 8.0,
               attack_ms: float = 0.5, release_ms: float = 60.0) -> np.ndarray:
    """Ganancia por muestra (lineal, ≤ 1) que se aplica a la banda de sibilancia."""
    n = len(band)
    power = (band ** 2).mean(axis=1) if band.ndim == 2 else band ** 2
    win = max(HOP, int(round(sr * 0.001)))
    pw = uniform_filter1d(power, size=win, mode="constant")          # ventana centrada: sin retardo
    centers = np.minimum(np.arange(0, n, HOP) + HOP // 2, n - 1)
    level = 10 * np.log10(pw[centers] + 1e-20)
    slope = 1.0 if not np.isfinite(ratio) else 1 - 1 / max(float(ratio), 1.0)
    rng = np.maximum(_per_frame(range_db, centers), 0.0)
    target = np.minimum(np.maximum(level - _per_frame(threshold_db, centers), 0.0) * slope, rng)
    a_att = np.exp(-HOP / max(attack_ms / 1000 * sr, 1e-9))
    a_rel = np.exp(-HOP / max(release_ms / 1000 * sr, 1e-9))
    red = np.empty_like(target)
    s = 0.0
    for i, r in enumerate(target):
        a = a_att if r > s else a_rel
        s = a * s + (1 - a) * r
        red[i] = s
    return 10 ** (-np.interp(np.arange(n), centers, red) / 20)


def deess(x: np.ndarray, sr: int, threshold_db=-30.0, range_db=6.0, freq_hz: float = 6700.0,
          width_oct: float = 0.85, ratio: float = 8.0, attack_ms: float = 0.5,
          release_ms: float = 60.0) -> np.ndarray:
    """Reduce la sibilancia de ``x`` (n, canales) sin tocar el resto del espectro ni el tiempo."""
    x = np.asarray(x)
    if len(x) == 0:
        return x.astype(np.float32)
    xd = x.astype(np.float64)
    band = sibilance_band(xd, sr, freq_hz, width_oct)
    g = deess_gain(band, sr, threshold_db, range_db, ratio, attack_ms, release_ms)
    if np.all(g >= 1.0):
        return x.astype(np.float32)                                  # sin sibilancia: intacto
    gg = g[:, None] if xd.ndim == 2 else g
    return (xd - band + band * gg).astype(np.float32)
