"""Efectos integrados (backend ``builtin``) aplicados a audio estéreo (n, 2).

uri -> parámetros:
  highpass/lowpass: hz
  eq: low_db, low_hz, mid_db, mid_hz, mid_q, high_db, high_hz
  compressor: threshold_db, ratio, attack_ms, release_ms
  reverb: size, damping, wet, width
  delay: beats (sincronizado al tempo), feedback, mix
  chorus: rate_hz, depth, mix
  gain: db
  sidechain: depth (0-1), release_ms  (fuente: Track.sidechain_from)
"""
from __future__ import annotations

import numpy as np
from pedalboard import (Chorus, Compressor, Delay, Gain, HighpassFilter, HighShelfFilter, LowpassFilter,
                        LowShelfFilter, Pedalboard, PeakFilter, Reverb)

EFFECTS = {"highpass", "lowpass", "eq", "compressor", "reverb", "delay", "chorus", "gain", "sidechain"}


def _pb(board: list, x: np.ndarray, sr: int, tail_s: float = 0.0) -> np.ndarray:
    y = Pedalboard(board)(x.T.astype(np.float32).copy(), sr, reset=True)
    return y.T[: len(x)]


def apply_effect(uri: str, p: dict, x: np.ndarray, sr: int, bpm: float, sidechain_src: np.ndarray | None = None) -> np.ndarray:
    if uri == "highpass":
        return _pb([HighpassFilter(cutoff_frequency_hz=p.get("hz", 120))], x, sr)
    if uri == "lowpass":
        return _pb([LowpassFilter(cutoff_frequency_hz=p.get("hz", 8000))], x, sr)
    if uri == "eq":
        return _pb([LowShelfFilter(cutoff_frequency_hz=p.get("low_hz", 120), gain_db=p.get("low_db", 0)),
                    PeakFilter(cutoff_frequency_hz=p.get("mid_hz", 1000), gain_db=p.get("mid_db", 0), q=p.get("mid_q", 1.0)),
                    HighShelfFilter(cutoff_frequency_hz=p.get("high_hz", 8000), gain_db=p.get("high_db", 0))], x, sr)
    if uri == "compressor":
        return _pb([Compressor(threshold_db=p.get("threshold_db", -18), ratio=p.get("ratio", 3),
                               attack_ms=p.get("attack_ms", 10), release_ms=p.get("release_ms", 120))], x, sr)
    if uri == "reverb":
        wet = p.get("wet", 0.25)
        return _pb([Reverb(room_size=p.get("size", 0.8), damping=p.get("damping", 0.5), wet_level=wet,
                           dry_level=1 - wet * 0.5, width=p.get("width", 1.0))], x, sr)
    if uri == "delay":
        secs = p.get("beats", 0.75) * 60 / bpm
        return _pb([Delay(delay_seconds=secs, feedback=p.get("feedback", 0.35), mix=p.get("mix", 0.2))], x, sr)
    if uri == "chorus":
        return _pb([Chorus(rate_hz=p.get("rate_hz", 0.8), depth=p.get("depth", 0.25), mix=p.get("mix", 0.3))], x, sr)
    if uri == "gain":
        return _pb([Gain(gain_db=p.get("db", 0))], x, sr)
    if uri == "sidechain":
        if sidechain_src is None:
            return x
        return x * duck_envelope(sidechain_src, sr, p.get("depth", 0.6), p.get("release_ms", 140))[:, None]
    raise ValueError(f"efecto integrado desconocido: {uri}")


def duck_envelope(src: np.ndarray, sr: int, depth: float, release_ms: float) -> np.ndarray:
    """Ganancia que baja con cada ataque de ``src`` (detección de picos con ataque rápido)."""
    m = np.abs(src).max(axis=1) if src.ndim == 2 else np.abs(src)
    hop = 64
    n = len(m) // hop + 1
    env = np.array([m[i * hop:(i + 1) * hop].max() if i * hop < len(m) else 0 for i in range(n)])
    peak = env.max() or 1.0
    env /= peak
    rel = np.exp(-hop / (release_ms / 1000 * sr))
    out = np.zeros(n)
    e = 0.0
    for i in range(n):
        e = env[i] if env[i] > e else e * rel
        out[i] = e
    g = 1 - depth * np.clip(out, 0, 1)
    return np.repeat(g, hop)[: len(m)]


def pan_gain(pan: float) -> tuple[float, float]:
    a = (pan + 1) * np.pi / 4                    # panorámica de potencia constante
    return float(np.cos(a) * np.sqrt(2)), float(np.sin(a) * np.sqrt(2))
