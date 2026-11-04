"""Datos para visualizar un master: espectrograma, sonoridad a corto plazo y correlación estéreo.

Todo devuelve arreglos compactos (submuestreados) y serializables en JSON:

- espectrograma: STFT (Hann, 2048) de la suma mono, promediada en ≤ ``max_cols`` columnas y
  agrupada en bandas logarítmicas; se entrega como matriz uint8 (base64) con 0 = ``db_max -
  db_range`` y 255 = ``db_max``. Fila 0 = banda más grave.
- LUFS-S (ITU-R BS.1770-4 / EBU Tech 3341): ponderación K, ventana rectangular de 3 s, sin
  compuerta. Cada punto lleva el tiempo del **final** de su ventana (como un medidor en vivo),
  por eso un escalón en ``t`` aparece entre ``t`` y ``t + 3 s``. Por debajo de -70 LUFS: ``None``.
- correlación L/R: Σ(L·R)/√(ΣL²·ΣR²) por ventana (1 = mono, -1 = canal invertido, 0 = sin
  relación); ``None`` en silencio. También por sección.
"""
from __future__ import annotations

import base64

import numpy as np
import pyloudnorm as pyln

VIS_VERSION = "1"
SHORT_TERM_S = 3.0
CORR_WINDOW_S = 0.4
N_FFT = 2048
SILENCE_MS = 1e-7                 # media cuadrática (~-70 dBFS): por debajo no hay correlación


def _frames(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


def _hop_s(duration_s: float, max_points: int) -> float:
    """0.1 s (resolución de un medidor) salvo que la pieza sea tan larga que excedería ``max_points``."""
    return max(0.1, duration_s / max_points)


def _windowed_mean(sq: np.ndarray, win: int, ends: np.ndarray) -> np.ndarray:
    """Media de ``sq`` (n, ch) en ventanas ``[end - win, end)``, vía suma acumulada."""
    c = np.vstack([np.zeros((1, sq.shape[1])), np.cumsum(sq, axis=0)])
    return (c[ends] - c[ends - win]) / win


def short_term_lufs(x: np.ndarray, sr: int, max_points: int = 3000) -> dict:
    x = _frames(x)
    n = len(x)
    win = int(round(SHORT_TERM_S * sr))
    hop = max(1, int(round(_hop_s(n / sr, max_points) * sr)))
    if n < win:
        return {"window_s": SHORT_TERM_S, "hop_s": hop / sr, "t": [], "lufs": []}
    k = x.copy()
    meter = pyln.Meter(sr)
    for stage in meter._filters.values():                 # filtros K (estante + paso alto RLB)
        for ch in range(k.shape[1]):
            k[:, ch] = stage.apply_filter(k[:, ch])
    ends = np.arange(win, n + 1, hop)
    ms = _windowed_mean(k * k, win, ends)
    gains = np.array([1.0, 1.0, 1.0, 1.41, 1.41][: k.shape[1]])
    z = ms @ gains
    with np.errstate(divide="ignore"):
        lufs = -0.691 + 10 * np.log10(z)
    vals = [round(float(v), 2) if np.isfinite(v) and v >= -70 else None for v in lufs]
    return {"window_s": SHORT_TERM_S, "hop_s": round(hop / sr, 4), "t": [round(e / sr, 3) for e in ends],
            "lufs": vals}


def _corr(l: np.ndarray, r: np.ndarray) -> float | None:
    ll, rr = float(np.dot(l, l)), float(np.dot(r, r))
    if min(ll, rr) / max(len(l), 1) < SILENCE_MS:
        return None
    return round(float(np.dot(l, r) / np.sqrt(ll * rr)), 3)


def correlation(x: np.ndarray, sr: int, sections: list[tuple[str, int, int]] | None = None,
                max_points: int = 3000) -> dict:
    x = _frames(x)
    if x.shape[1] == 1:
        x = np.hstack([x, x])
    n = len(x)
    win = max(1, int(round(CORR_WINDOW_S * sr)))
    hop = max(1, int(round(_hop_s(n / sr, max_points) * sr)))
    ends = np.arange(min(win, n), n + 1, hop) if n else np.array([], dtype=int)
    w = min(win, n)
    lr = _windowed_mean((x[:, 0] * x[:, 1])[:, None], w, ends)[:, 0] if n else np.array([])
    ll = _windowed_mean((x[:, 0] ** 2)[:, None], w, ends)[:, 0] if n else np.array([])
    rr = _windowed_mean((x[:, 1] ** 2)[:, None], w, ends)[:, 0] if n else np.array([])
    vals = []
    for a, b, c in zip(lr, ll, rr):
        vals.append(None if min(b, c) < SILENCE_MS else round(float(np.clip(a / np.sqrt(b * c), -1, 1)), 3))
    per = [{"label": lab, "start_s": round(s / sr, 3), "end_s": round(e / sr, 3),
            "correlation": _corr(x[s:e, 0], x[s:e, 1])} for lab, s, e in (sections or [])]
    return {"window_s": CORR_WINDOW_S, "hop_s": round(hop / sr, 4), "t": [round(e / sr, 3) for e in ends],
            "correlation": vals, "sections": per}


def spectrogram(x: np.ndarray, sr: int, max_cols: int = 600, rows: int = 128, fmin: float = 20.0,
                db_range: float = 90.0) -> dict:
    mono = _frames(x).mean(axis=1)
    n = len(mono)
    if n < N_FFT:
        mono = np.pad(mono, (0, N_FFT - n))
    hop = N_FFT // 2
    nframes = 1 + (len(mono) - N_FFT) // hop
    ncols = min(max_cols, nframes)
    edges_f = np.linspace(0, nframes, ncols + 1).astype(int)       # frames STFT agrupados por columna
    win = np.hanning(N_FFT)
    power = np.zeros((ncols, N_FFT // 2 + 1))
    chunk = 512
    col = np.repeat(np.arange(ncols), np.diff(edges_f))
    for i in range(0, nframes, chunk):
        idx = np.arange(i, min(i + chunk, nframes))
        seg = mono[idx[:, None] * hop + np.arange(N_FFT)] * win
        p = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        np.add.at(power, col[idx], p)
    power /= np.maximum(np.diff(edges_f), 1)[:, None]
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    band_edges = np.geomspace(fmin, sr / 2, rows + 1)
    centers = np.sqrt(band_edges[:-1] * band_edges[1:])
    bands = np.zeros((rows, ncols))
    for r in range(rows):
        sel = np.nonzero((freqs >= band_edges[r]) & (freqs < band_edges[r + 1]))[0]
        if sel.size == 0:                                            # banda más angosta que un bin
            sel = [int(np.argmin(np.abs(freqs - centers[r])))]
        bands[r] = power[:, sel].max(axis=1)
    db = 10 * np.log10(bands + 1e-20)
    top = float(db.max()) if db.size else 0.0
    q = np.clip((db - (top - db_range)) / db_range * 255, 0, 255).round().astype(np.uint8)
    return {"rows": rows, "cols": ncols, "data_b64": base64.b64encode(q.tobytes()).decode(),
            "freqs_hz": [round(float(f), 1) for f in centers], "band_edges_hz": [round(float(f), 1) for f in band_edges],
            "times_s": [round(float((edges_f[c] + edges_f[c + 1]) / 2 * hop + N_FFT / 2) / sr, 3) for c in range(ncols)],
            "db_max": round(top, 1), "db_range": db_range}


def decode_spectrogram(spec: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(spec["data_b64"]), dtype=np.uint8).reshape(spec["rows"], spec["cols"])


def visualize(x: np.ndarray, sr: int, sections: list[tuple[str, int, int]], target_lufs: float | None) -> dict:
    x = _frames(x)
    return {"version": VIS_VERSION, "sample_rate": sr, "duration_s": round(len(x) / sr, 3),
            "target_lufs": target_lufs,
            "sections": [{"label": lab, "start_s": round(s / sr, 3), "end_s": round(e / sr, 3)} for lab, s, e in sections],
            "spectrogram": spectrogram(x, sr), "lufs_s": short_term_lufs(x, sr), "correlation": correlation(x, sr, sections)}
