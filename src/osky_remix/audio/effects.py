"""Efectos integrados (backend ``builtin``) aplicados a audio estéreo (n, 2).

uri -> parámetros (los marcados con * son automatizables):
  highpass/lowpass: hz*, resonance* (0-1), slope (12 o 24 dB/oct, por defecto 24)
  eq: low_db, low_hz, mid_db, mid_hz, mid_q, high_db, high_hz
  compressor: threshold_db*, ratio*, attack_ms, release_ms
  reverb: size*, damping, wet*, width
  delay: beats (sincronizado al tempo), feedback*, mix*
  chorus: rate_hz*, depth*, mix*
  gain: db*
  sidechain: depth*, release_ms  (fuente: Track.sidechain_from)
  stereo: width* (0 = mono, 1 = igual, 2 = doble), mono_below_hz (graves en mono)
"""
from __future__ import annotations

import numpy as np
from pedalboard import (Chorus, Compressor, Delay, Gain, HighShelfFilter, LadderFilter, LowShelfFilter, Pedalboard,
                        PeakFilter, Reverb)
from scipy.signal import butter, sosfiltfilt

EFFECTS = {"highpass", "lowpass", "eq", "compressor", "reverb", "delay", "chorus", "gain", "sidechain", "stereo"}
BLOCK = 512

# parámetro -> (atributo de pedalboard, conversión)
AUTOMATABLE = {
    "highpass": {"hz": ("cutoff_hz", float), "resonance": ("resonance", float)},
    "lowpass": {"hz": ("cutoff_hz", float), "resonance": ("resonance", float)},
    "compressor": {"threshold_db": ("threshold_db", float), "ratio": ("ratio", float)},
    "reverb": {"wet": ("wet_level", float), "size": ("room_size", float)},
    "delay": {"feedback": ("feedback", float), "mix": ("mix", float)},
    "chorus": {"rate_hz": ("rate_hz", float), "depth": ("depth", float), "mix": ("mix", float)},
    "gain": {"db": ("gain_db", float)},
    "sidechain": {"depth": None},
    "stereo": {"width": None},
}


def _plugins(uri: str, p: dict, bpm: float) -> list:
    # LadderFilter (24 dB/oct): HighpassFilter/LowpassFilter de pedalboard son de un polo
    # (6 dB/oct) y no sirven para barridos de filtro (ver docs/COMPATIBILITY.md)
    if uri in ("highpass", "lowpass"):
        slope = int(p.get("slope", 24))
        mode = {("lowpass", 24): LadderFilter.Mode.LPF24, ("lowpass", 12): LadderFilter.Mode.LPF12,
                ("highpass", 24): LadderFilter.Mode.HPF24, ("highpass", 12): LadderFilter.Mode.HPF12}[(uri, slope)]
        return [LadderFilter(mode=mode, cutoff_hz=p.get("hz", 120 if uri == "highpass" else 8000),
                             resonance=p.get("resonance", 0.0), drive=1.0)]
    if uri == "eq":
        return [LowShelfFilter(cutoff_frequency_hz=p.get("low_hz", 120), gain_db=p.get("low_db", 0)),
                PeakFilter(cutoff_frequency_hz=p.get("mid_hz", 1000), gain_db=p.get("mid_db", 0), q=p.get("mid_q", 1.0)),
                HighShelfFilter(cutoff_frequency_hz=p.get("high_hz", 8000), gain_db=p.get("high_db", 0))]
    if uri == "compressor":
        return [Compressor(threshold_db=p.get("threshold_db", -18), ratio=p.get("ratio", 3),
                           attack_ms=p.get("attack_ms", 10), release_ms=p.get("release_ms", 120))]
    if uri == "reverb":
        wet = p.get("wet", 0.25)
        return [Reverb(room_size=p.get("size", 0.8), damping=p.get("damping", 0.5), wet_level=wet,
                       dry_level=1 - wet * 0.5, width=p.get("width", 1.0))]
    if uri == "delay":
        return [Delay(delay_seconds=p.get("beats", 0.75) * 60 / bpm, feedback=p.get("feedback", 0.35), mix=p.get("mix", 0.2))]
    if uri == "chorus":
        return [Chorus(rate_hz=p.get("rate_hz", 0.8), depth=p.get("depth", 0.25), mix=p.get("mix", 0.3))]
    if uri == "gain":
        return [Gain(gain_db=p.get("db", 0))]
    raise ValueError(f"efecto integrado desconocido: {uri}")


def apply_effect(uri: str, p: dict, x: np.ndarray, sr: int, bpm: float, sidechain_src: np.ndarray | None = None,
                 automation: dict[str, np.ndarray] | None = None) -> np.ndarray:
    automation = automation or {}
    bad = set(automation) - set(AUTOMATABLE.get(uri, {}))
    if bad:
        raise ValueError(f"{uri}: parámetros no automatizables: {', '.join(sorted(bad))}")
    if uri == "sidechain":
        if sidechain_src is None:
            return x
        depth = automation.get("depth", p.get("depth", 0.6))
        return x * duck_envelope(sidechain_src, sr, depth, p.get("release_ms", 140))[:, None]
    if uri == "stereo":
        return stereo(x, sr, automation.get("width", p.get("width", 1.0)), p.get("mono_below_hz", 0))
    plugs = _plugins(uri, p, bpm)
    if not automation:
        return Pedalboard(plugs)(x.T.astype(np.float32).copy(), sr, reset=True).T[: len(x)]
    # automatización por bloques: el estado del plugin se conserva entre bloques
    plug = plugs[0]
    board = Pedalboard(plugs)
    out = np.empty_like(x, dtype=np.float32)
    xt = x.T.astype(np.float32)
    for a in range(0, len(x), BLOCK):
        for name, arr in automation.items():
            attr, conv = AUTOMATABLE[uri][name]
            val = conv(arr[min(a, len(arr) - 1)])
            setattr(plug, attr, val)
            if uri == "reverb" and name == "wet":
                plug.dry_level = 1 - val * 0.5
        out[a:a + BLOCK] = board(xt[:, a:a + BLOCK].copy(), sr, reset=False).T
    return out


def stereo(x: np.ndarray, sr: int, width, mono_below_hz: float = 0.0) -> np.ndarray:
    """M/S: lado × width; por debajo de ``mono_below_hz`` el lado se elimina (graves en mono)."""
    mid = (x[:, 0] + x[:, 1]) / 2
    side = (x[:, 0] - x[:, 1]) / 2
    if mono_below_hz and mono_below_hz > 0:
        side = sosfiltfilt(butter(4, mono_below_hz, "high", fs=sr, output="sos"), side)
    side = side * (width if np.isscalar(width) else np.asarray(width)[: len(side)])
    return np.stack([mid + side, mid - side], 1).astype(np.float32)


def duck_envelope(src: np.ndarray, sr: int, depth, release_ms: float) -> np.ndarray:
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
    d = depth if np.isscalar(depth) else np.asarray(depth)[: len(m)]
    g = 1 - d * np.clip(np.repeat(out, hop)[: len(m)], 0, 1)
    return g


def pan_gain(pan: float) -> tuple[float, float]:
    a = (pan + 1) * np.pi / 4                    # panorámica de potencia constante
    return float(np.cos(a) * np.sqrt(2)), float(np.sin(a) * np.sqrt(2))


def pan_gain_curve(pan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return np.cos(a) * np.sqrt(2), np.sin(a) * np.sqrt(2)
