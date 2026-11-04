"""Etapa final de mastering: ganancia a objetivo LUFS + limitador con techo de true peak.

Los objetivos se pasan como perfil; no hay un volumen único impuesto a todos los géneros.

Por qué un limitador propio: el ``Limiter`` de pedalboard (JUCE dsp::Limiter) aplica
ganancia de compensación y recorta en 0 dBFS, sin lookahead ni detección de true peak;
en el doctor dio +1.4 dBTP con miles de muestras recortadas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

from .loudness import _as_frames, integrated_lufs
from .true_peak import OVERSAMPLE, true_peak_dbtp


@dataclass(frozen=True)
class MasterProfile:
    name: str
    target_lufs: float
    ceiling_dbtp: float = -1.0
    lookahead_ms: float = 5.0
    release_ms: float = 60.0


PROFILES = {
    "club": MasterProfile("club", target_lufs=-9.0, ceiling_dbtp=-1.0),
    "streaming": MasterProfile("streaming", target_lufs=-14.0, ceiling_dbtp=-1.0),
    "dynamic": MasterProfile("dynamic", target_lufs=-16.0, ceiling_dbtp=-1.5, release_ms=120.0),
}


@dataclass
class MasterResult:
    audio: np.ndarray
    lufs: float
    true_peak_dbtp: float
    gain_db: float
    iterations: int
    log: list[str] = field(default_factory=list)


def true_peak_envelope(x: np.ndarray, oversample: int = OVERSAMPLE) -> np.ndarray:
    """Pico absoluto entre canales y entre muestras (sobremuestreado), una cifra por muestra."""
    up = np.abs(resample_poly(x, oversample, 1, axis=0)).max(axis=1)
    n = x.shape[0]
    up = up[: n * oversample]
    if len(up) < n * oversample:
        up = np.pad(up, (0, n * oversample - len(up)))
    env = up.reshape(n, oversample).max(axis=1)
    # el pico entre la muestra n y n+1 afecta a ambas
    return np.maximum(env, np.concatenate([env[1:], env[-1:]]))


def limit(x: np.ndarray, sr: int, ceiling_dbtp: float, lookahead_ms: float = 5.0, release_ms: float = 60.0) -> np.ndarray:
    """Limitador con lookahead. Garantía: ganancia[n] <= techo / pico[n] en cada muestra.

    min-filter de ventana 2*H+1 seguido de media móvil de ventana <= H+1: todo valor
    promediado alrededor de n proviene de un mínimo cuya ventana contiene n.
    """
    x = _as_frames(x).astype(np.float64)
    ceil = 10 ** (ceiling_dbtp / 20)
    req = np.minimum(1.0, ceil / np.maximum(true_peak_envelope(x), 1e-12))
    hold = max(1, int(sr * max(lookahead_ms, release_ms) / 1000))
    g = minimum_filter1d(req, size=2 * hold + 1, mode="nearest")
    g = uniform_filter1d(g, size=hold + 1 if hold % 2 == 0 else hold, mode="nearest")
    g = np.minimum(g, req)  # seguro numérico
    return x * g[:, None]


def master(audio: np.ndarray, sr: int, profile: MasterProfile, max_iter: int = 8) -> MasterResult:
    """Lleva el audio al objetivo del perfil sin superar el techo de true peak."""
    x = _as_frames(audio).astype(np.float64)
    gain_db = profile.target_lufs - integrated_lufs(x, sr)
    margin = 0.1
    log: list[str] = []
    y = x
    for it in range(1, max_iter + 1):
        y = limit(x * 10 ** (gain_db / 20), sr, profile.ceiling_dbtp - margin,
                  profile.lookahead_ms, profile.release_ms)
        tp, lufs = true_peak_dbtp(y, sr), integrated_lufs(y, sr)
        log.append(f"it{it}: gain={gain_db:+.2f}dB margin={margin:.2f} -> {lufs:.2f} LUFS, {tp:.2f} dBTP")
        ok_tp = tp <= profile.ceiling_dbtp + 1e-3
        ok_lufs = abs(lufs - profile.target_lufs) <= 0.3
        if ok_tp and ok_lufs:
            return MasterResult(y.astype(np.float32), lufs, tp, gain_db, it, log)
        if not ok_tp:
            margin += (tp - profile.ceiling_dbtp) + 0.05
        if not ok_lufs:
            gain_db += profile.target_lufs - lufs
    return MasterResult(y.astype(np.float32), integrated_lufs(y, sr), true_peak_dbtp(y, sr), gain_db, max_iter, log)
