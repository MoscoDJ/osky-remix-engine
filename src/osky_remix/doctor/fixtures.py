"""Fixtures sintéticos deterministas: se conoce de antemano qué debe detectar cada módulo."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

SR = 44100


def midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def sine(freq: float, dur: float, sr: int = SR, amp: float = 1.0, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(round(dur * sr))) / sr
    return amp * np.sin(2 * np.pi * freq * t + phase)


def harmonic_tone(freq: float, dur: float, sr: int = SR, n_harm: int = 8, decay: float = 3.0) -> np.ndarray:
    """Tono con armónicos y envolvente percusiva suave (parecido a un piano eléctrico)."""
    t = np.arange(int(round(dur * sr))) / sr
    x = sum(np.sin(2 * np.pi * freq * h * t) / h ** 1.2 for h in range(1, n_harm + 1) if freq * h < sr / 2.2)
    env = np.minimum(1, t / 0.005) * np.exp(-t * decay)
    fade = np.minimum(1, (t[-1] - t) / 0.01)
    return x * env * fade


# Melodía de prueba para transcripción: (nota MIDI, inicio s, duración s)
MELODY = [(60, 0.25, 0.40), (64, 0.75, 0.40), (67, 1.25, 0.40), (72, 1.75, 0.40),
          (71, 2.25, 0.40), (67, 2.75, 0.40), (64, 3.25, 0.40), (62, 3.75, 0.70)]


def melody(sr: int = SR) -> np.ndarray:
    out = np.zeros(int(5.0 * sr))
    for m, start, dur in MELODY:
        x = harmonic_tone(midi_to_hz(m), dur, sr)
        i = int(start * sr)
        out[i:i + len(x)] += x
    return 0.5 * out / np.abs(out).max()


def band_mixture(dur: float = 8.0, sr: int = SR, seed: int = 1234) -> dict[str, np.ndarray]:
    """Mezcla de batería + bajo + acorde (Re menor) a 120 BPM, con sus componentes."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    spb = 0.5
    drums = np.zeros(n)
    kl = int(0.3 * sr); kt = np.arange(kl) / sr
    kick = np.sin(2 * np.pi * np.cumsum(50 + 120 * np.exp(-kt * 30)) / sr) * np.exp(-kt * 9)
    hl = int(0.05 * sr)
    hat_sos = butter(4, 7000, "high", fs=sr, output="sos")
    for b in range(int(dur / spb)):
        i = int(b * spb * sr)
        drums[i:i + kl] += kick[: n - i]
        j = int((b + 0.5) * spb * sr)
        if j + hl < n:
            drums[j:j + hl] += sosfilt(hat_sos, rng.standard_normal(hl)) * np.exp(-np.arange(hl) / sr * 80) * 0.4
    f_bass = midi_to_hz(38)  # D2
    saw = 2 * ((t * f_bass) % 1) - 1
    bass = sosfilt(butter(4, 300, "low", fs=sr, output="sos"), saw) * 0.5
    bass *= np.minimum(1, ((t % spb) / 0.01)) * np.exp(-(t % spb) * 2)
    chord = sum(harmonic_tone(midi_to_hz(m), dur, sr, n_harm=6, decay=0.05) for m in (62, 65, 69, 74)) * 0.25
    stems = {"drums": drums * 0.8, "bass": bass, "other": chord}
    mix = sum(stems.values())
    g = 0.7 / np.abs(mix).max()
    stems = {k: v * g for k, v in stems.items()}
    stems["mixture"] = mix * g
    return stems


def f0_median(x: np.ndarray, sr: int, fmin: float = 50, fmax: float = 2000) -> float:
    import librosa
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=0) if x.shape[0] < x.shape[-1] else x.mean(axis=1)
    f = librosa.yin(x, fmin=fmin, fmax=fmax, sr=sr, frame_length=4096)
    rms = librosa.feature.rms(y=x, frame_length=4096, hop_length=512)[0][: len(f)]
    voiced = f[rms > rms.max() * 0.1]
    return float(np.median(voiced)) if len(voiced) else float("nan")


def cents(f: float, ref: float) -> float:
    return 1200 * np.log2(f / ref)
