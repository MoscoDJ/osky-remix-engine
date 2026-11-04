"""Bandas de octava (ISO 266) y medición de energía por banda, estéreo por banda y por cuadros.

Las usan la comparación con una referencia (``reference_match``) y el detector de enmascaramiento
(``masking``). Bandas de octava centradas en 31.5 Hz … 16 kHz con bordes fc/√2 … fc·√2 (la última
se corta en 20 kHz): diez bandas de igual ancho logarítmico, así el ruido rosa da la misma energía
en cada una y un tono de 1 kHz cae en el centro de su banda, no en un borde.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import csd, welch

CENTERS = (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)
FLOOR_DB = -150.0


def _edges(fc: float) -> tuple[float, float]:
    return fc / np.sqrt(2), min(fc * np.sqrt(2), 20000.0)


BANDS: list[dict] = [{"id": (f"{fc / 1000:g}k" if fc >= 1000 else f"{fc:g}"), "fc": float(fc),
                      "lo": round(_edges(fc)[0], 1), "hi": round(_edges(fc)[1], 1)} for fc in CENTERS]


def hz_label(f: float) -> str:
    return f"{f / 1000:.1f} kHz".replace(".0 kHz", " kHz") if f >= 1000 else f"{f:.0f} Hz"


def band_label(b: dict) -> str:
    """'5.7–11.3 kHz', '710 Hz–1.4 kHz'."""
    lo, hi = b["lo"], b["hi"]
    if lo >= 1000:
        return f"{lo / 1000:.1f}–{hi / 1000:.1f} kHz"
    return f"{hz_label(lo)}–{hz_label(hi)}"


def as_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x[:, :2]


def _db(p: float) -> float:
    return float(10 * np.log10(p)) if p > 0 else FLOOR_DB


def band_spectrum(x: np.ndarray, sr: int, nperseg: int = 8192) -> list[dict]:
    """Espectro promedio a largo plazo (Welch, Hann, 50 %) integrado por banda de octava.

    Por banda: ``db`` = energía media de los canales (10·log10 de la potencia integrada, la misma
    escala que la potencia media de la señal), ``side_to_mid_db`` = lado/centro (M=(L+R)/2,
    S=(L−R)/2) y ``corr`` = correlación L/R de la señal filtrada en la banda (parte real del
    espectro cruzado integrado sobre √(P_LL·P_RR)): 1 en mono, 0 con canales independientes.
    """
    x = as_stereo(x)
    n = min(nperseg, 1 << int(np.floor(np.log2(max(len(x), 16)))))
    f, pll = welch(x[:, 0], sr, window="hann", nperseg=n)
    _, prr = welch(x[:, 1], sr, window="hann", nperseg=n)
    _, plr = csd(x[:, 0], x[:, 1], sr, window="hann", nperseg=n)
    df = f[1] - f[0]
    plr = plr.real
    out = []
    for b in BANDS:
        k = (f >= b["lo"]) & (f < b["hi"])
        ll, rr, lr = pll[k].sum() * df, prr[k].sum() * df, plr[k].sum() * df
        mid, side = (ll + rr + 2 * lr) / 4, (ll + rr - 2 * lr) / 4
        out.append({"band": b["id"], "fc": b["fc"], "db": _db((ll + rr) / 2),
                    "side_to_mid_db": _db(max(side, 0.0)) - _db(mid) if mid > 0 else 0.0,
                    "corr": float(lr / np.sqrt(ll * rr)) if ll > 0 and rr > 0 else 1.0})
    return out


def band_frames(x: np.ndarray, sr: int, n_fft: int = 4096, hop: int = 2048) -> np.ndarray:
    """Energía por cuadro y banda (dB), promedio de potencia de los canales: (cuadros, bandas).

    El cuadro ``i`` cubre las muestras [i·hop, i·hop + n_fft). Se procesa en bloques para no
    materializar el STFT completo de una canción entera.
    """
    x = as_stereo(x)
    if len(x) < n_fft:
        x = np.pad(x, ((0, n_fft - len(x)), (0, 0)))
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    M = np.stack([((f >= b["lo"]) & (f < b["hi"])).astype(np.float64) for b in BANDS], 1)
    win = np.hanning(n_fft)
    scale = 2.0 / (np.sum(win ** 2) * n_fft)       # potencia media por cuadro (Parseval, ventana)
    nf = 1 + (len(x) - n_fft) // hop
    out = np.empty((nf, len(BANDS)))
    view = np.lib.stride_tricks.sliding_window_view(x, n_fft, axis=0)[::hop]   # (nf, 2, n_fft)
    for a in range(0, nf, 128):
        blk = view[a:a + 128] * win
        p = (np.abs(np.fft.rfft(blk, axis=-1)) ** 2).mean(axis=1) * scale
        out[a:a + 128] = 10 * np.log10(p @ M + 1e-20)
    return out


def frame_range(start: int, end: int, n_frames: int, n_fft: int = 4096, hop: int = 2048) -> slice:
    """Cuadros completamente dentro de [start, end)."""
    a = -(-start // hop)
    b = (end - n_fft) // hop + 1
    return slice(max(0, a), max(max(0, a), min(n_frames, b)))
