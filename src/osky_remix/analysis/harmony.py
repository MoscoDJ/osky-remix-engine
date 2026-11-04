"""Tonalidad (global y por tramos) y acordes por beat.

- Tonalidad global: votación de Essentia KeyExtractor con varios perfiles.
- Modulaciones: correlación Krumhansl-Kessler por ventanas, con histéresis.
- Acordes: croma sincronizado a beats contra 24 plantillas (mayor/menor), suavizado
  por programación dinámica con penalización por cambio.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from pydantic import BaseModel

from .common import AN_SR, Alternative, Estimate, clip01, resample_mono

NOTE = "C C# D D# E F F# G G# A A# B".split()
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class KeySegment(BaseModel):
    start: int
    end: int
    key: str
    confidence: float


class ChordSegment(BaseModel):
    start: int
    end: int
    label: str          # "Dm", "F", "N" (sin acorde)
    confidence: float


class Harmony(BaseModel):
    key: Estimate
    key_segments: list[KeySegment]
    chords: list[ChordSegment]


def key_name(pc: int, minor: bool) -> str:
    return f"{NOTE[pc]} {'minor' if minor else 'major'}"


def _kk_scores(chroma_mean: np.ndarray) -> list[tuple[float, str]]:
    out = []
    for pc in range(12):
        out.append((float(np.corrcoef(chroma_mean, np.roll(KK_MAJ, pc))[0, 1]), key_name(pc, False)))
        out.append((float(np.corrcoef(chroma_mean, np.roll(KK_MIN, pc))[0, 1]), key_name(pc, True)))
    return sorted(out, reverse=True)


def global_key(y: np.ndarray, sr: int) -> Estimate:
    import essentia.standard as es
    import librosa
    y44 = librosa.resample(y, orig_sr=sr, target_sr=44100).astype(np.float32)
    votes, strengths = Counter(), {}
    for prof in ("edma", "krumhansl", "temperley", "bgate"):
        try:
            k, s, st = es.KeyExtractor(profileType=prof)(y44)
        except Exception:  # noqa: BLE001
            continue
        name = f"{k.replace('Bb', 'A#').replace('Eb', 'D#').replace('Ab', 'G#').replace('Db', 'C#').replace('Gb', 'F#')} {s}"
        votes[name] += 1
        strengths.setdefault(name, []).append(float(st))
    if not votes:
        raise RuntimeError("Essentia no devolvió tonalidad")
    best, n = votes.most_common(1)[0]
    total = sum(votes.values())
    conf = clip01((n / total) * float(np.mean(strengths[best])) / 0.8)
    alts = [Alternative(value=k, score=round(v / total, 3)) for k, v in votes.most_common() if k != best]
    rel = _relative(best)
    note = f"relativa: {rel}" if rel else ""
    return Estimate(value=best, confidence=round(conf, 3), alternatives=alts,
                    method=f"Essentia KeyExtractor, voto de {total} perfiles", notes=note)


def _relative(key: str) -> str | None:
    root, mode = key.split()
    pc = NOTE.index(root)
    return key_name((pc + 3) % 12, False) if mode == "minor" else key_name((pc - 3) % 12, True)


def key_segments(chroma: np.ndarray, frame_s: float, sr: int, win_s: float = 20.0, hop_s: float = 5.0) -> list[KeySegment]:
    n = chroma.shape[1]
    w, h = max(1, int(win_s / frame_s)), max(1, int(hop_s / frame_s))
    labels, confs, centers = [], [], []
    for a in range(0, max(1, n - w + 1), h):
        c = chroma[:, a:a + w].mean(1)
        if c.sum() < 1e-6:
            continue
        sc = _kk_scores(c)
        labels.append(sc[0][1])
        confs.append(clip01((sc[0][0] - sc[1][0]) * 5 + 0.3 * sc[0][0]))
        centers.append(a + w // 2)
    if not labels:
        return []
    # histéresis: un cambio debe sostenerse 3 ventanas (15 s)
    sm = labels[:]
    i = 1
    while i < len(sm):
        if sm[i] != sm[i - 1]:
            run = 1
            while i + run < len(sm) and labels[i + run] == labels[i]:
                run += 1
            if run < 3:
                for j in range(i, i + run):
                    sm[j] = sm[i - 1]
            i += run
        else:
            i += 1
    segs: list[KeySegment] = []
    for i, lab in enumerate(sm):
        start = 0 if i == 0 else int((centers[i - 1] + centers[i]) / 2 * frame_s * sr)
        if segs and segs[-1].key == lab:
            segs[-1].confidence = max(segs[-1].confidence, confs[i])
            continue
        if segs:
            segs[-1].end = start
        segs.append(KeySegment(start=start, end=0, key=lab, confidence=round(confs[i], 3)))
    segs[-1].end = int(n * frame_s * sr)
    return segs


def _templates() -> tuple[np.ndarray, list[str]]:
    T, names = [], []
    for pc in range(12):
        for q, iv in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            t = np.zeros(12)
            t[[(pc + i) % 12 for i in iv]] = 1
            T.append(t / np.linalg.norm(t))
            names.append(NOTE[pc] + q)
    return np.array(T), names


def chords_by_beat(y: np.ndarray, a_sr: int, beats: list[int], sr: int, change_penalty: float = 0.08) -> list[ChordSegment]:
    import librosa
    if len(beats) < 2:
        return []
    yh = librosa.effects.harmonic(y, margin=2.0)
    hop = 512
    C = librosa.feature.chroma_cqt(y=yh, sr=a_sr, hop_length=hop, bins_per_octave=36)
    bf = librosa.time_to_frames(np.array(beats) / sr, sr=a_sr, hop_length=hop)
    bf = np.unique(np.clip(bf, 0, C.shape[1] - 1))
    Cs = librosa.util.sync(C, bf, aggregate=np.median)          # columnas: [0,b0), [b0,b1), ...
    Cs = Cs[:, 1:len(bf)]                                         # solo entre beats
    T, names = _templates()
    norm = np.linalg.norm(Cs, axis=0) + 1e-9
    sim = (T @ Cs) / norm                                         # (24, n)
    energy = Cs.sum(0)
    silent = energy < 0.05 * np.median(energy[energy > 0]) if np.any(energy > 0) else np.ones_like(energy, bool)
    n = sim.shape[1]
    # Viterbi: puntaje = similitud - penalización por cambio
    score = sim[:, 0].copy()
    back = np.zeros((24, n), int)
    for t in range(1, n):
        stay = score
        best_prev = int(np.argmax(score))
        move = score[best_prev] - change_penalty
        choose_stay = stay >= move
        back[:, t] = np.where(choose_stay, np.arange(24), best_prev)
        score = np.where(choose_stay, stay, move) + sim[:, t]
    path = [int(np.argmax(score))]
    for t in range(n - 1, 0, -1):
        path.append(back[path[-1], t])
    path = path[::-1]
    out: list[ChordSegment] = []
    bsamp = [int(librosa.frames_to_time(f, sr=a_sr, hop_length=hop) * sr) for f in bf]
    for t in range(n):
        lab = "N" if silent[t] else names[path[t]]
        s = np.sort(sim[:, t])
        c = 0.0 if silent[t] else clip01((sim[path[t], t] - s[-2]) * 4 + 0.5 * sim[path[t], t])
        a, b = bsamp[t], bsamp[t + 1]
        if out and out[-1].label == lab:
            out[-1].end = b
            out[-1].confidence = round(max(out[-1].confidence, c), 3)
        else:
            out.append(ChordSegment(start=a, end=b, label=lab, confidence=round(c, 3)))
    return out


def diatonic_chords(key: str) -> set[str]:
    root, mode = key.split()
    pc = NOTE.index(root)
    if mode == "major":
        degs = [(0, ""), (2, "m"), (4, "m"), (5, ""), (7, ""), (9, "m")]
    else:  # menor natural + V mayor (armónica)
        degs = [(0, "m"), (3, ""), (5, "m"), (7, "m"), (7, ""), (8, ""), (10, "")]
    return {NOTE[(pc + d) % 12] + q for d, q in degs}


ALL_KEYS = [key_name(pc, m) for pc in range(12) for m in (False, True)]


def relabel_by_chords(segs: list[KeySegment], chords: list[ChordSegment]) -> list[KeySegment]:
    """Reetiqueta cada tramo con la tonalidad que contiene más duración de sus acordes.

    El perfil de croma (Krumhansl) confunde tonalidades vecinas (p. ej. Dm–Gm–C–F–A daba
    "G major"); los acordes ya estimados son evidencia más directa. Empates: se conserva la
    etiqueta del croma si está entre las mejores.
    """
    out: list[KeySegment] = []
    for s in segs:
        inside = [c for c in chords if c.label != "N" and c.end > s.start and c.start < s.end]
        total = sum(min(c.end, s.end) - max(c.start, s.start) for c in inside)
        if total <= 0:
            out.append(s)
            continue
        scores = {}
        for k in ALL_KEYS:
            d = diatonic_chords(k)
            # tónica presente: desempata relativas (Dm vs F) a favor de la que tiene su tónica
            tonic = k.split()[0] + ("m" if k.endswith("minor") else "")
            fit = sum((min(c.end, s.end) - max(c.start, s.start)) for c in inside if c.label in d) / total
            has_tonic = sum((min(c.end, s.end) - max(c.start, s.start)) for c in inside if c.label == tonic) / total
            scores[k] = fit + 0.25 * has_tonic
        best = max(scores.values())
        top = [k for k, v in scores.items() if v >= best - 1e-9]
        key = s.key if s.key in top else top[0]
        conf = clip01(best / 1.25)
        if out and out[-1].key == key:
            out[-1].end = s.end
            out[-1].confidence = round(max(out[-1].confidence, conf), 3)
        else:
            out.append(KeySegment(start=s.start, end=s.end, key=key, confidence=round(conf, 3)))
    return out


def refine_key_boundaries(segs: list[KeySegment], chords: list[ChordSegment], sr: int,
                          radius_s: float = 15.0) -> list[KeySegment]:
    """Mueve cada frontera de tonalidad al cambio de acorde que mejor separa ambas tonalidades."""
    for i in range(1, len(segs)):
        a, b = segs[i - 1], segs[i]
        lo, hi = b.start - radius_s * sr, b.start + radius_s * sr
        near = [c for c in chords if c.end > lo and c.start < hi and c.label != "N"]
        if len(near) < 2:
            continue
        d1, d2 = diatonic_chords(a.key), diatonic_chords(b.key)
        best, best_s = b.start, -1e9
        for split in [c.start for c in near]:
            s = sum(((c.label in d1) - (c.label in d2)) * (c.end - c.start) for c in near if c.start < split) \
                + sum(((c.label in d2) - (c.label in d1)) * (c.end - c.start) for c in near if c.start >= split)
            if s > best_s:
                best, best_s = split, s
        a.end = b.start = int(best)
    return segs


def analyze_harmony(audio: np.ndarray, sr: int, beats: list[int]) -> Harmony:
    import librosa
    y = resample_mono(audio, sr)
    key = global_key(y, AN_SR)
    hop = 4096
    C = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=AN_SR, hop_length=hop)
    chords = chords_by_beat(y, AN_SR, beats, sr)
    segs = refine_key_boundaries(relabel_by_chords(key_segments(C, hop / AN_SR, sr), chords), chords, sr)
    return Harmony(key=key, key_segments=segs, chords=chords)
