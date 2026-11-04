"""Pads y arpegios a partir de la progresión (un acorde por medio compás)."""
from __future__ import annotations

import random

from ..core.models import MidiNote
from .theory import S16, TICKS_PER_BEAT as Q, chord_pcs, voice_lead


def pads(chords_halfbar: list[str], lo: int = 50, hi: int = 74, velocity: int = 80) -> list[MidiNote]:
    out, prev, i = [], None, 0
    while i < len(chords_halfbar):
        ch = chords_halfbar[i]
        j = i
        while j + 1 < len(chords_halfbar) and chords_halfbar[j + 1] == ch:
            j += 1                                   # acorde repetido: una sola nota larga
        pcs = chord_pcs(ch)
        if pcs:
            v = voice_lead(prev, pcs, lo, hi)
            prev = v
            dur = (j - i + 1) * 2 * Q - 20
            out += [MidiNote(tick=i * 2 * Q, duration=dur, pitch=n, velocity=velocity) for n in v]
        i = j + 1
    return out


ARP_SHAPES = {
    "up": [0, 1, 2, 3, 2, 1, 2, 3],
    "pulse": [0, 2, 1, 2, 0, 3, 1, 2],
    "octaves": [0, 3, 1, 3, 2, 3, 1, 3],
}


def arp(chords_halfbar: list[str], seed: int, lo: int = 62, hi: int = 86, shape: str | None = None) -> list[MidiNote]:
    rng = random.Random(seed)
    shape = shape or rng.choice(sorted(ARP_SHAPES))
    seq = ARP_SHAPES[shape]
    out, prev = [], None
    for h, ch in enumerate(chords_halfbar):
        pcs = chord_pcs(ch)
        if not pcs:
            continue
        v = voice_lead(prev, pcs, lo, hi, size=3)
        prev = v
        tones = v + [v[0] + 12]
        for k in range(8):
            out.append(MidiNote(tick=h * 2 * Q + k * S16, duration=S16 - 20, pitch=tones[seq[k] % len(tones)],
                                velocity=96 if k % 4 == 0 else 74))
    return out
