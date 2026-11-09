"""Mastering por perfiles: gain staging -> graves en mono -> tilt -> glue -> ganancia a objetivo
LUFS + limitador con techo de true peak.

Los perfiles son configurables (``presets/mastering/*.json`` o una ruta); no hay un volumen único
impuesto a todos los géneros.

Por qué un limitador propio: el ``Limiter`` de pedalboard (JUCE dsp::Limiter) aplica ganancia
de compensación y recorta en 0 dBFS, sin lookahead ni detección de true peak; en el doctor dio
+1.4 dBTP con miles de muestras recortadas.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

from .loudness import _as_frames, integrated_lufs
from .true_peak import OVERSAMPLE, true_peak_dbtp

PRESETS_DIR = Path(__file__).resolve().parents[3] / "presets" / "mastering"


@dataclass(frozen=True)
class MasterProfile:
    name: str
    target_lufs: float
    ceiling_dbtp: float = -1.0
    lookahead_ms: float = 5.0
    release_ms: float = 60.0
    input_peak_dbfs: float | None = None     # gain staging: pico de entrada antes de la cadena
    low_mono_hz: float = 0.0                 # lado (S) eliminado por debajo de esta frecuencia
    tilt_db: float = 0.0                     # + brillo / - calidez, pivote 1 kHz
    glue_threshold_db: float | None = None   # compresor de bus (relativo a 0 dBFS tras el staging)
    glue_ratio: float = 2.0


    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name != "name" and v is not None and not np.isfinite(v):
                raise ValueError(f"perfil: {f.name} debe ser finito")
        if not -70 <= self.target_lufs <= 0 or not -30 <= self.ceiling_dbtp <= 0:
            raise ValueError("perfil: objetivo/techo fuera de rango")
        if not 0 < self.lookahead_ms <= 1000 or not 0 < self.release_ms <= 10000:
            raise ValueError("perfil: tiempos inválidos")
        if self.low_mono_hz < 0 or self.glue_ratio < 1:
            raise ValueError("perfil: filtro/ratio inválido")


PROFILES = {
    "club": MasterProfile("club", target_lufs=-9.0, ceiling_dbtp=-1.0),
    "streaming": MasterProfile("streaming", target_lufs=-14.0, ceiling_dbtp=-1.0),
    "dynamic": MasterProfile("dynamic", target_lufs=-16.0, ceiling_dbtp=-1.5, release_ms=120.0),
}


def load_profile(name_or_path: str) -> MasterProfile:
    """Nombre (``presets/mastering/<nombre>.json`` o integrado) o ruta a un JSON."""
    p = Path(name_or_path)
    if not p.exists():
        p = PRESETS_DIR / f"{name_or_path}.json"
    if p.exists():
        data = json.loads(p.read_text())
        known = {f.name for f in fields(MasterProfile)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"perfil {p.name}: campos desconocidos {', '.join(sorted(unknown))}")
        data.setdefault("name", p.stem)
        return MasterProfile(**data)
    if name_or_path in PROFILES:
        return PROFILES[name_or_path]
    avail = sorted({*PROFILES, *(f.stem for f in PRESETS_DIR.glob("*.json"))}) if PRESETS_DIR.exists() else sorted(PROFILES)
    raise ValueError(f"perfil desconocido '{name_or_path}' (disponibles: {', '.join(avail)})")


@dataclass
class MasterResult:
    audio: np.ndarray
    lufs: float | None
    true_peak_dbtp: float | None
    gain_db: float
    iterations: int
    log: list[str] = field(default_factory=list)
    staging_db: float = 0.0
    profile: dict = field(default_factory=dict)
    converged: bool = True
    status: str = "converged"
    unmet: list[str] = field(default_factory=list)


def true_peak_envelope(x: np.ndarray, oversample: int = OVERSAMPLE) -> np.ndarray:
    """Pico absoluto entre canales y entre muestras (sobremuestreado), una cifra por muestra."""
    up = np.abs(resample_poly(x, oversample, 1, axis=0)).max(axis=1)
    n = x.shape[0]
    up = up[: n * oversample]
    if len(up) < n * oversample:
        up = np.pad(up, (0, n * oversample - len(up)))
    env = up.reshape(n, oversample).max(axis=1)
    return np.maximum(env, np.concatenate([env[1:], env[-1:]]))


def limit(x: np.ndarray, sr: int, ceiling_dbtp: float, lookahead_ms: float = 5.0, release_ms: float = 60.0) -> np.ndarray:
    """Limitador con lookahead y release exponencial.

    Garantía: ganancia[n] <= techo / pico_true[n] en cada muestra.
    - Ataque: min-filter de ventana 2L+1 + media móvil de ventana L+1 (todo valor promediado
      viene de un mínimo cuya ventana contiene n, así que nunca supera lo requerido).
    - Release: recursión exponencial a nivel de bloque (32 muestras) sobre el mínimo del bloque,
      siempre <= ese mínimo; luego suavizado y mínimo con el ataque.
    Una versión anterior usaba un min-filter de ventana = release (retención de ±50 ms en cada
    pico): con bombos cada ~0.5 s ponía un techo de sonoridad (ver docs/COMPATIBILITY.md).
    """
    x = _as_frames(x).astype(np.float64)
    ceil = 10 ** (ceiling_dbtp / 20)
    req = np.minimum(1.0, ceil / np.maximum(true_peak_envelope(x), 1e-12))
    L = max(1, int(sr * lookahead_ms / 1000))
    att = minimum_filter1d(req, size=2 * L + 1, mode="nearest")
    att = np.minimum(uniform_filter1d(att, size=L + 1 if L % 2 == 0 else L, mode="nearest"), req)
    B = 32
    nb = int(np.ceil(len(att) / B))
    pad = np.pad(att, (0, nb * B - len(att)), constant_values=1.0)
    blk = pad.reshape(nb, B).min(axis=1)
    rel = float(np.exp(-B / (max(release_ms, 1.0) / 1000 * sr)))
    env = np.empty(nb)
    e = 1.0
    for k in range(nb):
        e = min(blk[k], 1.0 - (1.0 - e) * rel)
        env[k] = e
    g = np.repeat(env, B)[: len(att)]
    g = minimum_filter1d(g, size=B + 1, mode="nearest")          # suavizar escalones sin subir
    g = np.minimum(uniform_filter1d(g, size=B, mode="nearest"), att)
    return x * g[:, None]


def _pre_chain(x: np.ndarray, sr: int, prof: MasterProfile, log: list[str]) -> tuple[np.ndarray, float]:
    staging = 0.0
    from scipy.signal import butter, sosfilt
    x = sosfilt(butter(2, 15, "high", fs=sr, output="sos"), x, axis=0)      # bloqueo de DC
    if prof.input_peak_dbfs is not None:
        pk = float(np.abs(x).max())
        if pk > 0:
            staging = prof.input_peak_dbfs - 20 * np.log10(pk)
            x = x * 10 ** (staging / 20)
            log.append(f"gain staging {staging:+.2f} dB (pico de entrada {prof.input_peak_dbfs} dBFS)")
    if prof.low_mono_hz and x.shape[1] == 2:
        from ..audio.effects import stereo
        x = stereo(x, sr, 1.0, prof.low_mono_hz).astype(np.float64)
        log.append(f"graves en mono por debajo de {prof.low_mono_hz:g} Hz")
    if prof.tilt_db or prof.glue_threshold_db is not None:
        from pedalboard import Compressor, HighShelfFilter, LowShelfFilter, Pedalboard
        chain = []
        if prof.tilt_db:
            chain += [LowShelfFilter(cutoff_frequency_hz=1000, gain_db=-prof.tilt_db / 2),
                      HighShelfFilter(cutoff_frequency_hz=1000, gain_db=prof.tilt_db / 2)]
            log.append(f"tilt {prof.tilt_db:+g} dB")
        if prof.glue_threshold_db is not None:
            chain.append(Compressor(threshold_db=prof.glue_threshold_db, ratio=prof.glue_ratio, attack_ms=30, release_ms=150))
            log.append(f"glue {prof.glue_threshold_db:g} dB, {prof.glue_ratio:g}:1")
        x = Pedalboard(chain)(x.T.astype(np.float32).copy(), sr).T.astype(np.float64)
    return x, staging


DC_BLOCK_HZ = 5.0


def dc_block(x: np.ndarray, sr: int, hz: float = DC_BLOCK_HZ) -> np.ndarray:
    """Paso alto de 1.er orden en fase cero (ida y vuelta). El limitador atenúa más los picos de un
    bombo asimétrico (más alto hacia +) y eso deja un desplazamiento de DC en el master; el pre-
    master no lo tiene. Va dentro del bucle de mastering, antes de medir el true peak."""
    from scipy.signal import butter, sosfiltfilt
    if not np.any(x):
        return x
    sos = butter(1, hz, "high", fs=sr, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def master(audio: np.ndarray, sr: int, profile: MasterProfile, max_iter: int = 8) -> MasterResult:
    """Lleva el audio al objetivo del perfil sin superar el techo de true peak."""
    if max_iter < 1:
        raise ValueError("max_iter debe ser positivo")
    audio = _as_frames(audio)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("audio vacío o con NaN/Inf")
    if not 0 <= profile.low_mono_hz < sr / 2:
        raise ValueError("frecuencia de mono fuera de Nyquist")
    if not np.any(audio):
        return MasterResult(audio.astype(np.float32), None, None, 0, 0, ["silencio"],
                            profile=asdict(profile), converged=False, status="silent", unmet=["loudness"])
    log: list[str] = []
    x, staging = _pre_chain(_as_frames(audio).astype(np.float64), sr, profile, log)
    initial_lufs = integrated_lufs(x, sr)
    if not np.isfinite(initial_lufs):
        return MasterResult(x.astype(np.float32), None, true_peak_dbtp(x, sr), 0, 0,
                            ["señal bajo el umbral de medición"], profile=asdict(profile),
                            converged=False, status="unmeasurable", unmet=["loudness"])
    gain_db = profile.target_lufs - initial_lufs
    margin = 0.1
    y = x
    for it in range(1, max_iter + 1):
        y = dc_block(limit(x * 10 ** (gain_db / 20), sr, profile.ceiling_dbtp - margin,
                           profile.lookahead_ms, profile.release_ms), sr)
        tp, lufs = true_peak_dbtp(y, sr), integrated_lufs(y, sr)
        log.append(f"it{it}: gain={gain_db:+.2f}dB margin={margin:.2f} -> {lufs:.2f} LUFS, {tp:.2f} dBTP")
        ok_tp = tp <= profile.ceiling_dbtp + 1e-3
        ok_lufs = abs(lufs - profile.target_lufs) <= 0.3
        if ok_tp and ok_lufs:
            return MasterResult(y.astype(np.float32), lufs, tp, gain_db, it, log, staging, asdict(profile))
        if not ok_tp:
            margin += (tp - profile.ceiling_dbtp) + 0.05
        if not ok_lufs and it < max_iter:
            gain_db += profile.target_lufs - lufs
    return MasterResult(y.astype(np.float32), integrated_lufs(y, sr), true_peak_dbtp(y, sr), gain_db, max_iter, log,
                        staging, asdict(profile), False, "not_converged",
                        [k for k, ok in (("true_peak", ok_tp), ("loudness", ok_lufs)) if not ok])
