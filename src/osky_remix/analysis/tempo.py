"""Tempo, rejilla de beats, compás, tiempos fuertes y cambios de fase.

Método:
1. Claridad de pulso por ventana (tempograma) -> zonas de tempo libre (rubato, intros sin batería).
2. Seguimiento de beats + ajuste lineal por tramos (tempo constante por tramo) -> rejilla estable.
3. Compás y tiempo fuerte: acentos por beat (graves, cambio armónico, ataque) y su periodicidad.
4. Fase del tiempo fuerte por ventanas de 8 compases -> detecta compases cortos/insertados.
Todas las posiciones de salida están en muestras de la frecuencia original.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from .common import AN_SR, HOP, Alternative, Estimate, clip01, resample_mono


RUBATO_REL = 0.04     # residuo > 4 % del periodo: el tramo no tiene rejilla (tempo libre)


class TempoSegment(BaseModel):
    start: int
    end: int
    bpm: float
    residual_ms: float


class PhaseChange(BaseModel):
    sample: int
    beats_shift: int


class Rhythm(BaseModel):
    tempo: Estimate                     # bpm global
    meter: Estimate                     # tiempos por compás
    downbeat: Estimate                  # confianza de la fase de tiempo fuerte
    segments: list[TempoSegment]
    beats: list[int]
    downbeats: list[int]
    phase_changes: list[PhaseChange]
    free_tempo: list[tuple[int, int]]


def onset_envelope(y: np.ndarray, sr: int, hop: int, n_fft: int = 2048) -> np.ndarray:
    """Envolvente de ataques multibanda con más peso en graves (bombo/bajo marcan el pulso).

    La mediana entre bandas (``aggregate=np.median``) ignora el bombo y deja dominar a los
    hi-hats a contratiempo: en los fixtures daba la mitad del tempo.
    """
    import librosa
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    M = librosa.amplitude_to_db(S, ref=np.max)
    out = np.zeros(S.shape[1])
    for lo, hi, w in ((20, 200, 0.5), (200, 2000, 0.3), (2000, sr / 2, 0.2)):
        sel = (f >= lo) & (f < hi)
        if sel.sum() == 0:
            continue
        o = librosa.onset.onset_strength(S=M[sel], sr=sr, hop_length=hop)
        out[: len(o)] += w * o / (o.max() + 1e-9)
    return out


def attack_envelope(y: np.ndarray, sr: int, hop: int = 32) -> np.ndarray:
    """Subida de energía en la forma de onda (derivada log de la energía en 2 ms).

    El flujo espectral marca el pico 20-40 ms después del ataque real; esta función lo marca
    en el ataque (resolución hop/sr ≈ 1.5 ms a 22.05 kHz).
    """
    w = max(1, int(0.002 * sr))
    e = np.sqrt(np.convolve(y.astype(np.float64) ** 2, np.ones(w) / w, mode="same"))[::hop]
    le = np.log(e + 1e-5)
    d = np.maximum(0.0, np.diff(le, prepend=le[0]))
    return np.convolve(d, np.ones(3) / 3, mode="same")


def _refine_shift(grid_s: np.ndarray, fine: np.ndarray, sr: int, hop: int, max_ms: float = 60) -> float:
    """Corrimiento global (s) que alinea la rejilla con la envolvente fina de ataques."""
    best, best_v = 0.0, -1.0
    for sh in np.arange(-max_ms, max_ms + 0.1, 1.0) / 1000:
        fr = np.round((grid_s + sh) * sr / hop).astype(int)
        fr = fr[(fr >= 0) & (fr < len(fine))]
        v = float(fine[fr].sum())
        if v > best_v:
            best, best_v = float(sh), v
    return best


def _free_regions(oenv: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Máscara por frame de pulso débil (tempo libre)."""
    import librosa
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop, win_length=512)
    bpms = librosa.tempo_frequencies(tg.shape[0], sr=sr, hop_length=hop)
    band = (bpms >= 50) & (bpms <= 200)
    clarity = tg[band].max(axis=0)
    energy = librosa.util.normalize(oenv)
    ref = np.median(clarity[energy > 0.05]) if np.any(energy > 0.05) else np.median(clarity)
    weak = clarity < 0.55 * ref
    # mínimo 6 s para considerar una zona libre
    k = int(6 * sr / hop)
    out = np.zeros_like(weak)
    i = 0
    while i < len(weak):
        if weak[i]:
            j = i
            while j < len(weak) and weak[j]:
                j += 1
            if j - i >= k:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _tempo_zones(oenv: np.ndarray, sr: int, hop: int, ref_bpm: float, rel_tol: float = 0.04,
                 min_s: float = 8.0) -> list[tuple[int, int, float]]:
    """Zonas de tempo local estable: (frame_ini, frame_fin, bpm)."""
    import librosa
    from scipy.ndimage import median_filter
    lt = librosa.feature.tempo(onset_envelope=oenv, sr=sr, hop_length=hop, aggregate=None,
                               start_bpm=ref_bpm, std_bpm=2.0)
    # plegar octavas hacia el tempo de referencia
    lt = np.where(lt > ref_bpm * 1.6, lt / 2, np.where(lt < ref_bpm / 1.6, lt * 2, lt))
    k = int(min_s * sr / hop)
    lt = median_filter(lt, size=max(3, k | 1), mode="nearest")
    zones, a = [], 0
    for i in range(1, len(lt) + 1):
        if i == len(lt) or abs(lt[i] - lt[a]) / lt[a] > rel_tol:
            zones.append([a, i, float(np.median(lt[a:i]))])
            a = i
    merged = []
    for z in zones:      # zonas cortas se funden con la anterior
        if merged and (z[1] - z[0] < k or abs(z[2] - merged[-1][2]) / z[2] < rel_tol):
            merged[-1][1] = z[1]
        else:
            merged.append(z)
    return [(a, b, bpm) for a, b, bpm in merged]


def _fit_grid(times: np.ndarray, period: float) -> tuple[float, float, float, np.ndarray]:
    """Ajuste lineal t = t0 + k*p con k entero (robusto a beats perdidos).

    k se cuenta secuencialmente (cada intervalo ≈ entero de periodos): redondear contra el
    primer beat acumula el error del periodo inicial y salta beats en tramos largos.
    """
    ibi = np.diff(times)
    k = np.concatenate([[0], np.cumsum(np.maximum(1, np.round(ibi / period)))])
    A = np.vstack([k, np.ones_like(k)]).T
    (p, t0), *_ = np.linalg.lstsq(A, times, rcond=None)
    res = times - (t0 + k * p)
    return float(p), float(t0), float(np.std(res) * 1000), k


def _segment_beats(times: np.ndarray, rel_tol: float = 0.03, min_beats: int = 8) -> list[tuple[int, int]]:
    """Corta donde el intervalo mediano cambia > rel_tol de forma sostenida."""
    if len(times) < 2 * min_beats:
        return [(0, len(times))]
    ibi = np.diff(times)
    med = np.array([np.median(ibi[max(0, i - 4): i + 4]) for i in range(len(ibi))])
    cuts, start, ref = [], 0, np.median(med[:min_beats])
    run = 0
    for i in range(min_beats, len(med)):
        if abs(med[i] - ref) / ref > rel_tol:
            run += 1
            if run >= min_beats:
                c = i - run + 1
                cuts.append(c)
                start, ref, run = c, np.median(med[c:c + min_beats]), 0
        else:
            run = 0
            ref = np.median(med[start:i + 1])
    bounds = [0] + cuts + [len(times)]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 2]


def _beat_features(y: np.ndarray, sr: int, beats_s: np.ndarray, oenv: np.ndarray, hop: int) -> np.ndarray:
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[(f >= 30) & (f <= 150)].sum(0)
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=sr)
    fr = librosa.time_to_frames(beats_s, sr=sr, hop_length=hop)
    fr = np.clip(fr, 0, S.shape[1] - 1)
    lowf, ons, harm = [], [], []
    for i, b in enumerate(fr):
        a, z = max(0, b - 2), min(S.shape[1], b + 6)
        lowf.append(low[a:z].max())
        ons.append(oenv[a:z].max() if a < len(oenv) else 0)
        nxt = fr[i + 1] if i + 1 < len(fr) else min(S.shape[1], b + (b - fr[i - 1] if i else 20))
        prv = fr[i - 1] if i else max(0, b - 20)
        c_now = chroma[:, b:max(b + 1, nxt)].mean(1)
        c_prev = chroma[:, prv:max(prv + 1, b)].mean(1)
        harm.append(1 - np.dot(c_now, c_prev) / (np.linalg.norm(c_now) * np.linalg.norm(c_prev) + 1e-9))
    z = lambda v: (np.asarray(v) - np.mean(v)) / (np.std(v) + 1e-9)
    return 0.4 * z(harm) + 0.4 * z(lowf) + 0.2 * z(ons)


def _phase_scores(F: np.ndarray, m: int) -> np.ndarray:
    return np.array([F[ph::m].mean() - np.delete(F, np.arange(ph, len(F), m)).mean() if len(F) > m else 0
                     for ph in range(m)])


def _autocorr(F: np.ndarray, lag: int) -> float:
    if len(F) <= lag + 4:
        return 0.0
    return float(np.corrcoef(F[:-lag], F[lag:])[0, 1])


def _downbeats_in(F: np.ndarray, meter: int) -> tuple[list[int], list[tuple[int, int]], list[int], list[float]]:
    """Tiempos fuertes de un tramo: índices, cambios de fase (índice, corrimiento), fases y márgenes."""
    W, H = 8 * meter, 4 * meter
    nb = len(F)
    centers, phases, margins = [], [], []
    for a in range(0, max(1, nb - W + 1), H):
        sc = _phase_scores(F[a:a + W], meter)
        o = np.argsort(sc)[::-1]
        phases.append(int((o[0] + a) % meter))
        margins.append(float(sc[o[0]] - sc[o[1]]))
        centers.append(a + min(W, nb) // 2)
    sm = phases[:]
    for i in range(1, len(sm) - 1):         # un cambio de fase debe sostenerse 2 ventanas
        if sm[i] != sm[i - 1] and sm[i] != sm[i + 1]:
            sm[i] = sm[i - 1]
    step = np.array([sm[int(np.argmin(np.abs(np.array(centers) - i)))] for i in range(nb)])
    downs, changes = [], []
    for i in range(nb):
        if (i - step[i]) % meter == 0:
            downs.append(i)
        if i and step[i] != step[i - 1]:
            changes.append((i, int((step[i] - step[i - 1]) % meter)))
    return downs, changes, phases, margins


def analyze_rhythm(audio: np.ndarray, sr: int) -> Rhythm:
    import librosa
    y = resample_mono(audio, sr)
    a_sr, hop = AN_SR, HOP
    scale = sr / a_sr
    oenv = onset_envelope(y, a_sr, hop)
    fine_hop = 32
    fine = attack_envelope(y, a_sr, fine_hop)
    free = _free_regions(oenv, a_sr, hop)
    clear = ~free
    # candidatos de tempo (tempograma promedio en zonas claras)
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=a_sr, hop_length=hop)
    bpms = librosa.tempo_frequencies(tg.shape[0], sr=a_sr, hop_length=hop)
    prof = tg[:, clear[: tg.shape[1]]].mean(1) if clear.any() else tg.mean(1)
    ok = (bpms >= 60) & (bpms <= 200)
    idx = np.where(ok)[0]
    order = idx[np.argsort(prof[idx])[::-1]]
    cands = []
    for i in order:
        if all(abs(bpms[i] - c) / c > 0.04 for c, _ in cands):
            cands.append((float(bpms[i]), float(prof[i])))
        if len(cands) == 3:
            break
    # Essentia como segunda opinión
    try:
        import essentia.standard as es
        y44 = librosa.resample(y, orig_sr=a_sr, target_sr=44100)
        e_bpm, _, e_conf, *_ = es.RhythmExtractor2013(method="multifeature")(y44.astype(np.float32))
    except Exception:  # noqa: BLE001
        e_bpm, e_conf = float("nan"), 0.0
    start_bpm = float(e_bpm) if np.isfinite(e_bpm) and 60 <= e_bpm <= 200 else cands[0][0]
    # tramos de tempo local antes de rastrear (el rastreador mantiene su tempo inicial)
    bt_list = []
    for fa, fb, seg_bpm in _tempo_zones(oenv, a_sr, hop, start_bpm):
        _, bfr = librosa.beat.beat_track(onset_envelope=oenv[fa:fb], sr=a_sr, hop_length=hop,
                                         start_bpm=seg_bpm, tightness=300)
        bfr = bfr + fa
        bt_list += [librosa.frames_to_time(f, sr=a_sr, hop_length=hop) for f in bfr if clear[min(f, len(clear) - 1)]]
    bt = np.array(sorted(bt_list))
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    loud = 20 * np.log10(rms + 1e-12) > -55

    def is_loud(t: float) -> bool:
        i = int(t * a_sr / hop)
        return 0 <= i < len(loud) and bool(loud[i]) and not bool(free[min(i, len(free) - 1)])
    if len(bt) < 8:
        raise ValueError("no hay suficientes beats para estimar el tempo")
    # tramos de tempo y rejilla
    segs: list[TempoSegment] = []
    grids: list[np.ndarray] = []
    for a, b in _segment_beats(bt):
        tt = bt[a:b]
        p, t0, res_ms, k = _fit_grid(tt, float(np.median(np.diff(tt))))
        ks = np.arange(int(k.min()), int(k.max()) + 1)
        g = t0 + ks * p
        g = g + _refine_shift(g, fine, a_sr, fine_hop)      # compensa la latencia de la envolvente
        if res_ms > RUBATO_REL * p * 1000:
            # los beats no siguen una rejilla: tempo libre (rubato), sin rejilla inventada
            free[int(tt[0] * a_sr / hop):int((tt[-1] + p) * a_sr / hop) + 1] = True
            continue
        seg = TempoSegment(start=int(round(g[0] * sr)), end=int(round((g[-1] + p) * sr)),
                           bpm=round(60 / p, 3), residual_ms=round(res_ms, 2))
        if segs and abs(segs[-1].bpm - seg.bpm) / seg.bpm < 0.01:     # fusionar tramos casi iguales
            segs[-1].end = seg.end
            grids[-1] = np.concatenate([grids[-1], g[g > grids[-1][-1] + 0.5 * p]])
        else:
            segs.append(seg)
            grids.append(g)
    if not segs:
        raise ValueError("no se encontró ningún tramo con pulso estable (todo es tempo libre)")
    # librosa recorta beats débiles al inicio y final (intros/outros sin batería): se extrapola la
    # rejilla del primer y último tramo mientras haya audio y no sea tempo libre
    p_first = 60 / segs[0].bpm
    while is_loud(grids[0][0] - p_first):
        grids[0] = np.r_[grids[0][0] - p_first, grids[0]]
    p_last = 60 / segs[-1].bpm
    while is_loud(grids[-1][-1] + p_last):
        grids[-1] = np.r_[grids[-1], grids[-1][-1] + p_last]
    segs[0].start = int(round(grids[0][0] * sr))
    segs[-1].end = int(round((grids[-1][-1] + p_last) * sr))
    merged = segs
    grid_s = np.concatenate(grids)
    lens = np.array([s.end - s.start for s in merged], float)
    main = merged[int(np.argmax(lens))]
    bpm = main.bpm                                       # tempo del tramo más largo (no un promedio)
    res = float(np.average([s.residual_ms for s in merged], weights=lens))
    if np.isfinite(e_bpm):
        r = e_bpm / bpm
        agree = 1.0 if abs(r - 1) < 0.02 else (0.6 if min(abs(r - 2), abs(r - 0.5)) < 0.04 else 0.2)
    else:
        agree = 0.5
    conf = clip01(0.5 * np.exp(-res / 25) + 0.3 * agree + 0.2 * clip01(e_conf / 3.5))
    alts = [Alternative(value=round(bpm * f, 2), score=round(float(np.interp(bpm * f, bpms[::-1], prof[::-1])), 4))
            for f in (0.5, 2.0, 1.5, 2 / 3)]
    tempo = Estimate(value=round(bpm, 3), confidence=round(conf, 3), alternatives=alts,
                     method="librosa beat_track + ajuste lineal por tramos; Essentia RhythmExtractor2013",
                     notes=f"residuo {res:.1f} ms; Essentia {e_bpm:.2f} BPM (conf {e_conf:.2f})"
                           + (f"; {len(merged)} tramos de tempo: " + ", ".join(f"{s.bpm:g}" for s in merged)
                              if len(merged) > 1 else ""))
    # compás y tiempo fuerte
    F = _beat_features(y, a_sr, grid_s, oenv, hop)
    ac = {m: _autocorr(F, m) for m in (2, 3, 4)}
    meter = 4 if ac[4] >= ac[3] else 3
    if meter == 4 and ac[2] > 1.25 * ac[4] + 0.05:
        notes_m = "periodicidad de 2 tiempos más fuerte que de 4; se asume 4/4 (convención)"
    else:
        notes_m = ""
    m_conf = clip01(0.5 + (ac[meter] - ac[3 if meter == 4 else 4]) * 1.5)
    meter_est = Estimate(value=meter, confidence=round(m_conf, 3),
                         alternatives=[Alternative(value=k, score=round(v, 3)) for k, v in ac.items()],
                         method="autocorrelación de acentos por beat (graves, cambio armónico, ataque)", notes=notes_m)
    # fase por ventanas de 8 compases, por separado en cada tramo de tempo
    downs, changes, all_phases, margins = [], [], [], []
    off = 0
    for g in grids:
        Fs = F[off:off + len(g)]
        d, ch, ph, mg = _downbeats_in(Fs, meter)
        downs += [off + i for i in d]
        changes += [PhaseChange(sample=int(round(grid_s[off + i] * sr)), beats_shift=s) for i, s in ch]
        all_phases += ph
        margins += mg
        off += len(g)
    db_conf = clip01(float(np.median(margins)) / 1.0) if margins else 0.0
    downbeat = Estimate(value=int(all_phases[0]) if all_phases else 0, confidence=round(db_conf, 3),
                        alternatives=[Alternative(value=int(p), score=round(m, 3)) for p, m in zip(all_phases, margins)][:8],
                        method="fase con mayor acento por ventanas de 8 compases",
                        notes="value = fase del primer tramo; margen mediano entre la mejor y la segunda fase")
    free_s = []
    i = 0
    while i < len(free):
        if free[i]:
            j = i
            while j < len(free) and free[j]:
                j += 1
            free_s.append((int(i * hop * scale), int(j * hop * scale)))
            i = j
        else:
            i += 1
    joined: list[tuple[int, int]] = []          # zonas libres separadas por menos de 1 s se unen
    for a, b in free_s:
        if joined and a - joined[-1][1] < sr:
            joined[-1] = (joined[-1][0], b)
        else:
            joined.append((a, b))
    free_s = joined
    return Rhythm(tempo=tempo, meter=meter_est, downbeat=downbeat, segments=merged,
                  beats=[int(round(t * sr)) for t in grid_s], downbeats=[int(round(grid_s[i] * sr)) for i in downs],
                  phase_changes=changes, free_tempo=free_s)
