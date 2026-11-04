"""Preparación y control de calidad de stems: alineación, ganancia, artefactos y regiones a reparar.

Nada se supone: el desfase y la ganancia de cada separador se MIDEN contra la mezcla.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from pydantic import BaseModel
from scipy.signal import fftconvolve, resample_poly

WIN_S = 1.0
ACTIVE_DBFS = -50.0


class RepairRegion(BaseModel):
    stem: str
    start: int            # muestras
    end: int
    reason: str           # reconstruction_error | shared_with_drums | drum_bleed | bass_bleed | highfreq_bleed
    severity: float       # 0..1
    metric: float


class StemQuality(BaseModel):
    name: str
    rms_dbfs: float
    active_ratio: float


class SeparationQuality(BaseModel):
    sample_rate: int
    lag_samples: int
    gain_correction_db: float
    reconstruction_snr_db: float
    stems: list[StemQuality]
    regions: list[RepairRegion]


def _frames(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g, axis=0)


def measure_lag(ref: np.ndarray, est: np.ndarray, sr: int, max_lag_s: float = 1.0, span_s: float = 60.0) -> int:
    """Desfase (muestras) que hay que aplicar a ``est`` para alinearlo con ``ref``."""
    n = min(len(ref), len(est), int(span_s * sr))
    a, b = ref[:n].mean(1), est[:n].mean(1)
    c = fftconvolve(a, b[::-1], mode="full")
    mid = n - 1
    m = int(max_lag_s * sr)
    seg = c[mid - m: mid + m + 1]
    return int(np.argmax(seg) - m)


def _shift(x: np.ndarray, lag: int, n: int) -> np.ndarray:
    out = np.zeros((n, x.shape[1]))
    if lag >= 0:
        k = min(n - lag, len(x))
        if k > 0:
            out[lag:lag + k] = x[:k]
    else:
        k = min(n, len(x) + lag)
        if k > 0:
            out[:k] = x[-lag:-lag + k]
    return out


def prepare_stems(mixture: np.ndarray, sr: int, stems: dict[str, tuple[np.ndarray, int]],
                  reconstructs: bool = True, drums_ref: np.ndarray | None = None
                  ) -> tuple[dict[str, np.ndarray], SeparationQuality]:
    """Remuestrea, iguala canales, alinea, corrige ganancia y mide calidad.

    ``reconstructs``: True si la suma de los stems debe reproducir la mezcla.
    ``drums_ref``: stem de batería de referencia (si no está entre ``stems``) para distinguir
    fugas de batería de consonantes/sibilantes.
    """
    mix = _frames(mixture)
    n, ch = mix.shape
    prepared = {}
    for name, (x, s_sr) in stems.items():
        y = _resample(_frames(x), s_sr, sr)
        if y.shape[1] != ch:
            y = np.repeat(y.mean(1, keepdims=True), ch, axis=1)
        prepared[name] = y
    total = sum(prepared.values())
    lag = measure_lag(mix, total, sr) if reconstructs else 0
    prepared = {k: _shift(v, lag, n) for k, v in prepared.items()}
    total = sum(prepared.values())
    g = 1.0
    if reconstructs:
        den = float(np.sum(total * total))
        g = float(np.sum(mix * total) / den) if den > 0 else 1.0
        prepared = {k: v * g for k, v in prepared.items()}
        total = total * g
    res = mix - total
    snr = 10 * np.log10(np.sum(mix ** 2) / max(np.sum(res ** 2), 1e-20)) if reconstructs else float("nan")
    regions: list[RepairRegion] = []
    if reconstructs:
        regions += _reconstruction_regions(mix, res, sr)
    sq = []
    drums = prepared.get("drums", None if drums_ref is None else _frames(drums_ref))
    for name, y in prepared.items():
        rms = float(np.sqrt(np.mean(y ** 2)))
        act = _activity(y, sr)
        sq.append(StemQuality(name=name, rms_dbfs=round(20 * np.log10(rms + 1e-12), 2), active_ratio=round(float(act.mean()), 3)))
        regions += _bleed_regions(name, y, sr, act, drums)
    q = SeparationQuality(sample_rate=sr, lag_samples=lag, gain_correction_db=round(20 * np.log10(abs(g) + 1e-12), 3),
                          reconstruction_snr_db=round(float(snr), 2), stems=sq,
                          regions=sorted(regions, key=lambda r: (r.start, r.stem)))
    return {k: v.astype(np.float32) for k, v in prepared.items()}, q


def _windows(x: np.ndarray, sr: int):
    w = int(WIN_S * sr)
    for i in range(0, len(x) - w // 2, w):
        yield i, min(i + w, len(x))


def _activity(y: np.ndarray, sr: int) -> np.ndarray:
    return np.array([20 * np.log10(np.sqrt(np.mean(y[a:b] ** 2)) + 1e-12) > ACTIVE_DBFS for a, b in _windows(y, sr)])


def _merge(flags: list[tuple[int, int, float, float]], stem: str, reason: str) -> list[RepairRegion]:
    out: list[RepairRegion] = []
    for a, b, sev, met in flags:
        if out and out[-1].end >= a and out[-1].reason == reason:
            out[-1].end = b
            out[-1].severity = max(out[-1].severity, sev)
            out[-1].metric = min(out[-1].metric, met) if reason == "reconstruction_error" else max(out[-1].metric, met)
        else:
            out.append(RepairRegion(stem=stem, start=a, end=b, reason=reason, severity=round(sev, 3), metric=round(met, 3)))
    return out


def _reconstruction_regions(mix, res, sr, snr_floor_db: float = 12.0):
    flags = []
    for a, b in _windows(mix, sr):
        e_mix = np.mean(mix[a:b] ** 2)
        if 10 * np.log10(e_mix + 1e-20) < ACTIVE_DBFS:
            continue
        snr = 10 * np.log10(e_mix / max(np.mean(res[a:b] ** 2), 1e-20))
        if snr < snr_floor_db:
            flags.append((a, b, float(np.clip((snr_floor_db - snr) / snr_floor_db, 0, 1)), float(snr)))
    return _merge(flags, "mixture", "reconstruction_error")


def _band_ratio(seg: np.ndarray, sr: int, lo: float, hi: float) -> float:
    s = np.abs(np.fft.rfft(seg.mean(1))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    tot = s.sum()
    return float(s[(f >= lo) & (f < hi)].sum() / tot) if tot > 0 else 0.0


BLEED_FLOOR_DB = -20.0     # fuga marcada si su energía supera 1 % de la del stem


def _projection_bleed_db(v: np.ndarray, d: np.ndarray) -> float:
    """Nivel (dB relativo a ``v``) de la componente de ``v`` que es copia escalada de ``d``.

    Mínimos cuadrados g = <v,d>/<d,d>; fuga = g²·E_d / E_v. Solo cuenta g > 0 (misma polaridad).
    """
    ed = float(np.sum(d * d))
    ev = float(np.sum(v * v))
    if ed <= 0 or ev <= 0:
        return -120.0
    g = float(np.sum(v * d)) / ed
    if g <= 0:
        return -120.0
    return 10 * np.log10(g * g * ed / ev + 1e-12)


def _bleed_regions(name: str, y: np.ndarray, sr: int, active: np.ndarray,
                   drums: np.ndarray | None = None) -> list[RepairRegion]:
    """Fugas típicas: batería o bajo en la voz; agudos en el bajo.

    Con stem de batería: la fuga es una copia escalada de la batería dentro de la voz; se mide
    proyectando la voz sobre la batería (sibilantes y consonantes no proyectan). Sin stem de
    batería: proporción percusiva HPSS > 35 % (menos fiable, confunde sibilantes).
    """
    out: list[RepairRegion] = []
    voc = name in ("vocals", "lead_vocals", "backing_vocals")
    if voc:
        drum, bass = [], []
        if drums is None:
            import librosa
            mono = librosa.resample(y.mean(1).astype(np.float32), orig_sr=sr, target_sr=22050)
            S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
            H, Pc = librosa.decompose.hpss(S, margin=2.0)
            hop_s = 512 / 22050
        for k, (a, b) in enumerate(_windows(y, sr)):
            if not active[k]:
                continue
            if drums is not None:
                lvl = _projection_bleed_db(y[a:b].mean(1), drums[a:b].mean(1))
                if lvl > BLEED_FLOOR_DB:
                    drum.append((a, b, float(np.clip((lvl - BLEED_FLOOR_DB) / 20, 0, 1)), float(lvl)))
            else:
                fa, fb = int(a / sr / hop_s), int(b / sr / hop_s)
                h, p = (H[:, fa:fb] ** 2).sum(), (Pc[:, fa:fb] ** 2).sum()
                ratio = p / (h + p + 1e-12)
                if ratio > 0.35:
                    drum.append((a, b, float(np.clip((ratio - 0.35) / 0.4, 0, 1)), float(ratio)))
            low = _band_ratio(y[a:b], sr, 20, 90)
            if low > 0.25:
                bass.append((a, b, float(np.clip((low - 0.25) / 0.5, 0, 1)), low))
        # con referencia se mide contenido compartido (la dirección de la fuga no es observable)
        out += _merge(drum, name, "shared_with_drums" if drums is not None else "drum_bleed") + _merge(bass, name, "bass_bleed")
    if name == "bass":
        hf = []
        for k, (a, b) in enumerate(_windows(y, sr)):
            if active[k]:
                r = _band_ratio(y[a:b], sr, 1500, sr / 2)
                if r > 0.2:
                    hf.append((a, b, float(np.clip((r - 0.2) / 0.5, 0, 1)), r))
        out += _merge(hf, name, "highfreq_bleed")
    return out
