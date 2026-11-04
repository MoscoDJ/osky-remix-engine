"""Rasgos de un sample (one-shot o loop) medidos sobre la señal, sin modelos entrenados.

Todo en segundos/Hz salvo donde se indica; las posiciones internas se calculan en muestras.

- ``attack_s``: tiempo entre el 10 % y el 90 % del pico de la envolvente (Hilbert suavizada 2 ms).
- ``decay_s``: desde el pico hasta la última muestra por encima de pico −40 dB (cola).
- ``centroid_hz`` / ``rolloff_hz``: centroide del espectro de potencia y frecuencia bajo la que
  queda el 85 % de la energía. ``brightness`` = log2(centroide/50 Hz)/log2(16 kHz/50 Hz) en [0, 1].
- ``low_ratio``: fracción de la energía por debajo de 150 Hz.
- ``f0_hz``/``note``: tono por NSDF (McLeod) en los 0.5 s tras el pico; ``tonal_conf`` es la
  mediana de la periodicidad (1 = tono puro). Solo se informa nota si ``tonal_conf`` ≥ 0.8.
- Loop: ≥ 4 ataques repartidos por el archivo y alineados a semicorcheas de un tempo cuyo número de
  tiempos en el archivo es potencia de 2 (1/2, 1, 2, 4… compases de 4/4) entre 85 y 170 BPM. El BPM
  sale de la duración exacta (``60·tiempos/duración``), así que un loop bien cortado da el BPM exacto.
- Categoría: reglas difusas sobre los rasgos; ``category_conf`` = mejor/(mejor+segunda) ∈ [0.5, 1].
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import hilbert

FEATURES_VERSION = "1"
CATEGORIES = ("kick", "snare", "clap", "hat", "perc", "bass", "tonal", "fx", "loop")
NOTE_NAMES = "C C# D D# E F F# G G# A A# B".split()
BRIGHT_LO, BRIGHT_HI = 50.0, 16000.0
LOOP_BPM = (85.0, 170.0)


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def brightness(centroid_hz: float) -> float:
    return float(np.clip(np.log2(max(centroid_hz, 1e-9) / BRIGHT_LO) / np.log2(BRIGHT_HI / BRIGHT_LO), 0, 1))


def _envelope(x: np.ndarray, sr: int) -> np.ndarray:
    env = np.abs(hilbert(np.concatenate([x, np.zeros(len(x))])))[: len(x)]   # sin arrastre circular
    w = max(1, int(0.002 * sr))
    return np.convolve(env, np.ones(w) / w, mode="same")


def _nsdf_f0(seg: np.ndarray, sr: int, fmin: float = 30.0, fmax: float = 2000.0) -> tuple[float | None, float]:
    frame = 4096 if sr >= 32000 else 2048
    if len(seg) < frame:
        frame = 1 << int(np.floor(np.log2(max(len(seg), 2))))
    lag_min = max(2, int(sr / fmax))
    lag_max = min(int(sr / fmin) + 1, frame // 2)
    if frame < 256 or lag_max <= lag_min + 2:
        return None, 0.0
    hop = frame // 4
    starts = list(range(0, len(seg) - frame + 1, hop))
    rms = np.array([np.sqrt(np.mean(seg[i:i + frame] ** 2)) for i in starts])
    f0s, confs = [], []
    k = np.arange(frame)
    for i, r_ in zip(starts, rms):
        if r_ < 0.05 * rms.max():
            continue
        w = seg[i:i + frame] - seg[i:i + frame].mean()
        spec = np.fft.rfft(w, 2 * frame)
        r = np.fft.irfft(spec * np.conj(spec))[:frame]
        e = np.cumsum(w ** 2)
        e1 = e[frame - 1 - k]
        e2 = e[-1] - np.concatenate([[0.0], e[:-1]])
        nsdf = r / np.sqrt(e1 * e2 + 1e-20)
        reg = nsdf[lag_min:lag_max]
        if reg.max() <= 0:
            continue
        thr = 0.9 * reg.max()
        cand = [j for j in range(1, len(reg) - 1) if reg[j] >= thr and reg[j] >= reg[j - 1] and reg[j] >= reg[j + 1]]
        if not cand:
            continue
        j = cand[0]
        a, b, c = reg[j - 1], reg[j], reg[j + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
        f0s.append(sr / (lag_min + j + off))
        confs.append(float(b))
    if not f0s:
        return None, 0.0
    return float(np.median(f0s)), float(np.median(confs))


def _onsets(x: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    hop = 256
    env = librosa.onset.onset_strength(y=x.astype(np.float32), sr=sr, hop_length=hop)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units="samples", backtrack=True)
    return np.asarray(on, dtype=int)


def _loop_bpm(onsets: np.ndarray, n: int, sr: int) -> tuple[float | None, int | None, float]:
    """(bpm, tiempos, ajuste a la rejilla) del candidato con tiempos = potencia de 2 en el rango."""
    dur = n / sr
    best = (None, None, 0.0)
    for beats in (2, 4, 8, 16, 32, 64):
        bpm = 60.0 * beats / dur
        if not (LOOP_BPM[0] <= bpm < LOOP_BPM[1]):
            continue
        step = n / (beats * 4)                       # semicorchea en muestras
        dist = np.abs(onsets / step - np.round(onsets / step)) * step / sr
        fit = float(np.mean(dist <= 0.030)) if len(onsets) else 0.0
        if fit > best[2]:
            best = (bpm, beats, fit)
    return best


def _bursts(env: np.ndarray, onset_i: int, sr: int) -> int:
    """Ataques en los primeros 50 ms (las palmas tienen 2-4 muy juntos): un máximo local de la
    envolvente (Hilbert, suavizada 3 ms más) cuenta si supera la mitad del pico y ×1.4 el mínimo desde el anterior."""
    w = max(1, int(0.003 * sr))
    a = onset_i
    seg = env[a: a + int(0.05 * sr) + w]
    if len(seg) <= w + 2:
        return 1
    e = np.convolve(seg, np.ones(w) / w, mode="valid")
    pk = e.max()
    count, runmin = 0, 0.0
    for i in range(1, len(e) - 1):
        runmin = min(runmin, e[i])
        if e[i] >= e[i - 1] and e[i] > e[i + 1] and e[i] >= 0.5 * pk and e[i] >= 1.4 * runmin:
            count += 1
            runmin = e[i]
    return max(1, count)


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def classify(f: dict) -> tuple[str, float, dict[str, float]]:
    if f["is_loop"]:
        return "loop", round(min(1.0, 0.5 + f["loop_fit"] / 2), 2), {"loop": f["loop_fit"]}
    br, low, dec, ton = f["brightness"], f["low_ratio"], f["decay_s"], f["tonal_conf"]
    f0 = f["f0_hz"] or 0.0
    noisy = 1 - _ramp(ton, 0.6, 0.85)
    sc = {
        "kick": _ramp(low, 0.3, 0.6) * (1 - _ramp(dec, 0.8, 1.5)) * (1 - _ramp(br, 0.4, 0.6))
                * (1.0 if f0 < 150 else 0.3),
        "bass": _ramp(ton, 0.7, 0.9) * (1 - _ramp(f0, 200, 350)) * _ramp(dec, 0.5, 1.0) * _ramp(low, 0.2, 0.5),
        "hat": _ramp(br, 0.8, 0.88) * (1 - _ramp(low, 0.05, 0.15)) * noisy * (1 - _ramp(dec, 1.2, 2.0)),
        "snare": _ramp(br, 0.45, 0.55) * (1 - _ramp(br, 0.84, 0.9)) * noisy * (1 - _ramp(dec, 0.6, 1.0))
                 * (1 - _ramp(low, 0.3, 0.5)),
        "perc": 0.3 * (1 - _ramp(dec, 0.5, 0.9)),
        "tonal": _ramp(ton, 0.75, 0.9) * _ramp(f0, 120, 180),
        "fx": 0.1 + 0.8 * _ramp(f["duration_s"], 1.5, 3.0),
    }
    sc["clap"] = sc["snare"] * (1.0 if f["bursts"] >= 2 else 0.0)
    if f["bursts"] >= 2:
        sc["snare"] *= 0.5
    order = sorted(sc.items(), key=lambda kv: -kv[1])
    (cat, best), (_, second) = order[0], order[1]
    return cat, round(best / (best + second + 1e-9), 2), {k: round(v, 3) for k, v in sc.items()}


def extract(x: np.ndarray, sr: int) -> dict:
    """Rasgos de una señal (mono o (n, canales)) a frecuencia ``sr``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    if n == 0 or not np.isfinite(x).all():
        raise ValueError("sample vacío o con NaN/Inf")
    peak = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(x ** 2)))
    env = _envelope(x, sr)
    pi = int(np.argmax(env))
    pk = env[pi]
    a10 = int(np.argmax(env[: pi + 1] >= 0.1 * pk))
    a90 = int(np.argmax(env[: pi + 1] >= 0.9 * pk))
    above = np.nonzero(env >= pk * 10 ** (-40 / 20))[0]
    tail_end = int(above[-1]) if len(above) else pi
    nfft = 1 << int(np.ceil(np.log2(max(n, 2048))))
    mag = np.abs(np.fft.rfft(x * 1.0, nfft))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    pw = mag ** 2
    centroid = float((freqs * pw).sum() / (pw.sum() + 1e-20))
    cum = np.cumsum(pw)
    rolloff = float(freqs[min(len(freqs) - 1, int(np.searchsorted(cum, 0.85 * cum[-1])))])
    low_ratio = float(pw[freqs < 150].sum() / (pw.sum() + 1e-20))
    on = _onsets(x, sr) if n >= sr else np.array([], dtype=int)
    is_loop, bpm, beats, fit = False, None, None, 0.0
    if n >= sr and len(on) >= 4 and (on.max() - on.min()) >= 0.5 * n:
        bpm, beats, fit = _loop_bpm(on, n, sr)
        is_loop = bpm is not None and fit >= 0.8
    f0, ton = _nsdf_f0(x[pi: pi + int(0.5 * sr)], sr)
    if is_loop:
        f0 = None                                      # el tono solo se informa en one-shots
    midi = int(round(69 + 12 * np.log2(f0 / 440))) if f0 and ton >= 0.8 else None
    f = {
        "duration_s": round(n / sr, 6), "frames": n, "sample_rate": sr,
        "peak_dbfs": round(20 * np.log10(peak + 1e-12), 2), "rms_dbfs": round(20 * np.log10(rms + 1e-12), 2),
        "attack_s": round(max(0, a90 - a10) / sr, 5), "decay_s": round(max(0, tail_end - pi) / sr, 5),
        "centroid_hz": round(centroid, 1), "rolloff_hz": round(rolloff, 1), "brightness": round(brightness(centroid), 4),
        "low_ratio": round(low_ratio, 4), "f0_hz": round(f0, 3) if f0 and ton >= 0.8 else None,
        "tonal_conf": round(ton, 3), "midi": midi, "note": note_name(midi) if midi is not None else None,
        "cents": round(1200 * np.log2(f0 / (440 * 2 ** ((midi - 69) / 12))), 1) if midi is not None else None,
        "onsets": int(len(on)), "is_loop": bool(is_loop), "bpm": round(bpm, 3) if is_loop else None,
        "loop_bars": beats / 4 if is_loop else None, "loop_fit": round(fit, 3), "bursts": _bursts(env, a10, sr),
    }
    f["category"], f["category_conf"], f["category_scores"] = classify(f)
    return f


def extract_file(path) -> dict:
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return extract(x, sr)


VECTOR_KEYS = ("log_duration", "brightness", "rolloff", "low_ratio", "log_attack", "log_decay", "tonal_conf",
               "rms", "loop")


def vector(f: dict) -> np.ndarray:
    """Vector de rasgos crudo (se normaliza por z-score sobre la biblioteca al buscar)."""
    return np.array([
        np.log10(f["duration_s"] + 1e-3), f["brightness"], brightness(f["rolloff_hz"]), f["low_ratio"],
        np.log10(f["attack_s"] + 1e-4), np.log10(f["decay_s"] + 1e-3), f["tonal_conf"], f["rms_dbfs"] / 60.0,
        1.0 if f["is_loop"] else 0.0,
    ], dtype=float)
