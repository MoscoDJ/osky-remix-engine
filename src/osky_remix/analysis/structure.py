"""Secciones a nivel de compás: novedad sobre una matriz de autosimilitud y etiquetado por
similitud (A, B, A...). Las fronteras caen siempre en tiempos fuertes."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from .common import AN_SR, clip01, resample_mono


class SectionEst(BaseModel):
    start: int
    end: int
    label: str
    confidence: float        # fuerza de la frontera de inicio (0-1)
    bars: int
    energy_db: float


class Structure(BaseModel):
    sections: list[SectionEst]
    novelty: list[float]     # una cifra por compás


def _bar_features(y: np.ndarray, sr: int, downbeats_s: np.ndarray, beats_s: np.ndarray | None = None) -> np.ndarray:
    """Vector por compás = mediana de los vectores de sus beats (robusto a un golpe aislado,
    p. ej. el crash del primer tiempo de una sección)."""
    import librosa
    hop = 512
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    M = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=14)[1:]
    R = librosa.amplitude_to_db(librosa.feature.rms(y=y, hop_length=hop), ref=1.0)
    S = np.abs(librosa.stft(y, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr)
    lo = librosa.amplitude_to_db(S[f < 150].sum(0, keepdims=True) + 1e-6)
    X = np.vstack([C, M, R, lo])
    if beats_s is None or len(beats_s) < len(downbeats_s):
        beats_s = downbeats_s
    bfr = np.unique(np.clip(librosa.time_to_frames(beats_s, sr=sr, hop_length=hop), 0, X.shape[1] - 1))
    Xb = librosa.util.sync(X, bfr, aggregate=np.median)[:, 1:]          # por beat
    bt = librosa.frames_to_time(bfr, sr=sr, hop_length=hop)
    bars = []
    for i, d in enumerate(downbeats_s):
        e = downbeats_s[i + 1] if i + 1 < len(downbeats_s) else np.inf
        idx = np.where((bt >= d - 1e-3) & (bt < e - 1e-3))[0]
        idx = idx[idx < Xb.shape[1]]
        bars.append(np.median(Xb[:, idx], axis=1) if len(idx) else Xb[:, min(Xb.shape[1] - 1, max(0, len(bars)))])
    B = np.array(bars).T
    B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-9)
    w = np.r_[np.full(12, 1 / np.sqrt(12)), np.full(13, 1 / np.sqrt(13)), np.full(2, 1.5 / np.sqrt(2))]
    return (B * w[:, None]).T


def _novelty(F: np.ndarray, half: int = 4) -> np.ndarray:
    n = len(F)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    S = Fn @ Fn.T
    g = np.exp(-0.5 * (np.linspace(-1, 1, 2 * half) / 0.5) ** 2)
    K = np.outer(g, g) * np.outer(np.r_[-np.ones(half), np.ones(half)], np.r_[-np.ones(half), np.ones(half)])
    Sp = np.pad(S, half, mode="edge")
    nov = np.array([np.sum(Sp[i:i + 2 * half, i:i + 2 * half] * K) for i in range(n)])
    nov = np.maximum(nov, 0)
    return nov / (nov.max() + 1e-9)


def analyze_structure(audio: np.ndarray, sr: int, downbeats: list[int], beats: list[int] | None = None, min_bars: int = 4,
                      label_threshold: float = 0.75) -> Structure:
    y = resample_mono(audio, sr)
    db = np.array(downbeats) / sr
    if len(db) < 2 * min_bars:
        return Structure(sections=[], novelty=[])
    F = _bar_features(y, AN_SR, db, None if beats is None else np.array(beats) / sr)
    nb = len(F)
    nov = _novelty(F)
    # picos: máximos locales sobre media+0.5σ, separados al menos min_bars
    thr = nov.mean() + 0.5 * nov.std()
    cand = [i for i in range(1, nb) if nov[i] >= thr and nov[i] == nov[max(0, i - min_bars + 1):i + min_bars].max()]
    bounds = [0]
    for c in sorted(cand, key=lambda i: -nov[i]):
        if all(abs(c - b) >= min_bars for b in bounds) and nb - c >= min_bars // 2:
            bounds.append(c)
    # el primer compás de una sección suele llevar un golpe (crash, fill) que lo vuelve atípico y
    # empuja el pico un compás tarde: ante casi empate con el compás anterior, gana el anterior
    bounds = [0] + [c - 1 if c - 1 > 0 and nov[c - 1] >= 0.8 * nov[c] else c for c in bounds[1:]]
    # prior de frase: las secciones tienden a empezar cada 4 compases; la fase es la más votada
    # por las fronteras candidatas (ponderadas por su novedad), sin los bordes
    votes = np.zeros(4)
    for c in bounds[1:]:
        if 2 <= c < nb - 2:
            votes[c % 4] += nov[c]
    phase = int(np.argmax(votes)) if votes.any() else 0
    snapped = [0]
    for c in sorted(bounds[1:]):
        if c % 4 != phase:
            p = min((q for q in (c - 1, c + 1) if 0 < q < nb and q % 4 == phase), key=lambda q: -nov[q], default=None)
            if p is not None and nov[p] >= 0.5 * nov[c]:
                c = p
        if all(abs(c - b) >= min_bars for b in snapped):
            snapped.append(c)
    bounds = sorted(snapped) + [nb]
    segs = [(a, b) for a, b in zip(bounds, bounds[1:])]
    # etiquetado: perfil medio por sección, se reusa la etiqueta si es similar
    reps: list[tuple[str, np.ndarray]] = []
    out: list[SectionEst] = []
    for a, b in segs:
        v = F[a:b].mean(0)
        v = v / (np.linalg.norm(v) + 1e-9)
        lab = None
        best = max(((float(v @ r), l) for l, r in reps), default=(-1, None))
        if best[0] >= label_threshold:
            lab = best[1]
        if lab is None:
            lab = chr(ord("A") + len(reps))
            reps.append((lab, v))
        end_s = db[b] if b < len(db) else db[-1] + (db[-1] - db[-2])
        seg = y[int(db[a] * AN_SR):int(end_s * AN_SR)]
        e = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else -120.0
        out.append(SectionEst(start=int(downbeats[a]), end=int(round(end_s * sr)), label=lab,
                              confidence=round(clip01(float(nov[a]) if a else 1.0), 3), bars=b - a,
                              energy_db=round(float(e), 2)))
    return Structure(sections=out, novelty=[round(float(x), 3) for x in nov])
