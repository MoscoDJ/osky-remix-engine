"""Características espectrales y de nivel, silencios y transitorios."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ..mastering.loudness import integrated_lufs
from ..mastering.true_peak import sample_peak_dbfs, true_peak_dbtp
from .common import AN_SR, resample_mono

BANDS = {"sub": (20, 60), "low": (60, 250), "mid": (250, 2000), "high": (2000, 8000), "air": (8000, 20000)}


class Features(BaseModel):
    duration_samples: int
    lufs: float
    loudness_range_lu: float
    true_peak_dbtp: float
    sample_peak_dbfs: float
    crest_db: float
    dc_offset: float
    clipped_samples: int
    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    spectral_flatness: float
    band_energy: dict[str, float]          # fracción de energía por banda
    stereo_correlation: float | None
    side_to_mid_db: float | None


class Transient(BaseModel):
    sample: int
    strength: float


def loudness_range(x: np.ndarray, sr: int) -> float:
    """Aproximación de EBU LRA: percentiles 10-95 de la sonoridad a corto plazo (3 s, salto 1 s),
    con compuertas absoluta (-70 LUFS) y relativa (-20 LU)."""
    import pyloudnorm as pyln
    m = pyln.Meter(sr, block_size=0.4)
    w, h = 3 * sr, sr
    vals = []
    for a in range(0, max(1, len(x) - w + 1), h):
        try:
            v = m.integrated_loudness(x[a:a + w])
        except ValueError:
            continue
        if np.isfinite(v) and v > -70:
            vals.append(v)
    if len(vals) < 2:
        return 0.0
    vals = np.array(vals)
    vals = vals[vals > 10 * np.log10(np.mean(10 ** (vals / 10))) - 20]
    return float(np.percentile(vals, 95) - np.percentile(vals, 10)) if len(vals) > 1 else 0.0


def analyze_features(audio: np.ndarray, sr: int) -> Features:
    import librosa
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    y = resample_mono(x, sr)
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) ** 2
    f = librosa.fft_frequencies(sr=AN_SR, n_fft=4096)
    tot = S.sum() + 1e-20
    bands = {k: round(float(S[(f >= lo) & (f < hi)].sum() / tot), 4) for k, (lo, hi) in BANDS.items()}
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.abs(x).max())
    corr = s2m = None
    if x.shape[1] == 2:
        l, r = x[:, 0], x[:, 1]
        corr = float(np.corrcoef(l, r)[0, 1]) if np.std(l) > 0 and np.std(r) > 0 else 1.0
        mid, side = (l + r) / 2, (l - r) / 2
        s2m = float(10 * np.log10((np.mean(side ** 2) + 1e-20) / (np.mean(mid ** 2) + 1e-20)))
    return Features(
        duration_samples=len(x),
        lufs=round(integrated_lufs(x, sr), 2),
        loudness_range_lu=round(loudness_range(x, sr), 2),
        true_peak_dbtp=round(true_peak_dbtp(x, sr), 2),
        sample_peak_dbfs=round(sample_peak_dbfs(x), 2),
        crest_db=round(20 * np.log10(peak / rms), 2) if rms > 0 else 0.0,
        dc_offset=round(float(x.mean()), 6),
        clipped_samples=int(np.sum(np.abs(x) >= 0.999)),
        spectral_centroid_hz=round(float(librosa.feature.spectral_centroid(S=np.sqrt(S), sr=AN_SR).mean()), 1),
        spectral_rolloff_hz=round(float(librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=AN_SR).mean()), 1),
        spectral_flatness=round(float(librosa.feature.spectral_flatness(S=np.sqrt(S)).mean()), 4),
        band_energy=bands,
        stereo_correlation=None if corr is None else round(corr, 4),
        side_to_mid_db=None if s2m is None else round(s2m, 2),
    )


def detect_silences(audio: np.ndarray, sr: int, thr_dbfs: float = -55.0, min_s: float = 0.3) -> list[tuple[int, int]]:
    x = np.asarray(audio, dtype=np.float64)
    m = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    hop = int(0.01 * sr)
    n = len(m) // hop
    lvl = 20 * np.log10(np.sqrt(np.mean(m[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-12)
    quiet = lvl < thr_dbfs
    out, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * hop >= min_s * sr:
                out.append((i * hop, min(j * hop, len(m))))
            i = j
        else:
            i += 1
    return out


def detect_transients(audio: np.ndarray, sr: int, max_n: int = 5000) -> list[Transient]:
    import librosa
    from .tempo import attack_envelope
    y = resample_mono(audio, sr)
    o = librosa.onset.onset_strength(y=y, sr=AN_SR, hop_length=256)
    fr = librosa.onset.onset_detect(onset_envelope=o, sr=AN_SR, hop_length=256, units="frames")
    # refinar al ataque real con la envolvente fina (±30 ms)
    fine = attack_envelope(y, AN_SR, 32)
    out = []
    for f in fr[:max_n]:
        c = int(f * 256 / 32)
        a, b = max(0, c - 30), min(len(fine), c + 8)
        k = a + int(np.argmax(fine[a:b])) if b > a else c
        out.append(Transient(sample=int(round(k * 32 / AN_SR * sr)), strength=round(float(o[f] / (o.max() + 1e-9)), 3)))
    return out
