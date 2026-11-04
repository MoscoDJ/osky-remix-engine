"""Melodías: desde una transcripción (p. ej. la voz) o generadas con motivos dentro de la escala."""
from __future__ import annotations

import random

import numpy as np

from ..core.models import MidiNote
from .theory import S16, chord_pcs, snap_to_scale


def from_transcription(notes: list[tuple[float, float, int]], bpm: float, length_ticks: int,
                       scale: list[int] | None = None, register: int = 74, min_dur: int = S16) -> list[MidiNote]:
    """(inicio_s, fin_s, pitch) -> notas cuantizadas a semicorcheas, sin solapes, en el registro pedido."""
    tps = bpm / 60 * 960
    q = []
    for s, e, p in sorted(notes):
        a = int(round(s * tps / S16)) * S16
        b = max(a + min_dur, int(round(e * tps / S16)) * S16)
        if a >= length_ticks:
            continue
        q.append([a, min(b, length_ticks), p])
    if not q:
        return []
    shift = 12 * round((register - float(np.median([p for *_, p in q]))) / 12)
    out: list[MidiNote] = []
    for i, (a, b, p) in enumerate(q):
        if i + 1 < len(q):
            b = min(b, q[i + 1][0])                 # monofónica: corta en la siguiente
        if b - a < min_dur:
            continue
        pitch = p + shift
        if scale:
            pitch = snap_to_scale(pitch, scale)
        out.append(MidiNote(tick=a, duration=b - a, pitch=int(pitch), velocity=100))
    return out


def motif(chords_halfbar: list[str], scale: list[int], seed: int, register: int = 74) -> list[MidiNote]:
    """Motivo de 1 compás repetido, anclado a notas del acorde en los tiempos fuertes."""
    rng = random.Random(seed)
    rhythm = rng.choice([[0, 3, 6, 8, 11, 14], [0, 2, 4, 7, 10, 12], [0, 4, 6, 10, 12, 14]])
    out = []
    for h in range(0, len(chords_halfbar), 2):
        pcs = chord_pcs(chords_halfbar[h]) or scale[:3]
        anchor = min((register + ((pc - register) % 12) for pc in pcs), key=lambda n: abs(n - register))
        steps = [0] + [rng.choice([-2, -1, 1, 2, 3]) for _ in rhythm[1:]]
        pitch = anchor
        for k, (t, st) in enumerate(zip(rhythm, steps)):
            pitch = snap_to_scale(pitch + st, scale) if k else anchor
            nxt = rhythm[k + 1] if k + 1 < len(rhythm) else 16
            out.append(MidiNote(tick=h * 1920 + t * S16, duration=(nxt - t) * S16 - 20, pitch=int(pitch),
                                velocity=96 if t % 4 == 0 else 84))
    return out
