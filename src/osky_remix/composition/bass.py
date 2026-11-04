"""Líneas de bajo que siguen la armonía (raíz por medio compás)."""
from __future__ import annotations

import random

from ..core.models import MidiNote
from .theory import BAR, S16, TICKS_PER_BEAT as Q, note_in_range, parse_chord


def rolling(chords_halfbar: list[str], seed: int, lo: int = 33, hi: int = 45, octave_jumps: bool = True) -> list[MidiNote]:
    """Bajo rodante (melodic techno): semicorcheas 2-3-4 tras cada bombo, raíz del acorde."""
    rng = random.Random(seed)
    out = []
    pattern = [1, 2, 3]
    jump = rng.choice([None, 2, 3]) if octave_jumps else None
    for h, ch in enumerate(chords_halfbar):
        p = parse_chord(ch)
        if p is None:
            continue
        root = note_in_range(p[0], lo, hi)
        for b in range(2):
            beat_tick = h * 2 * Q + b * Q
            for j in pattern:
                pitch = root + (12 if jump == j and b == 1 else 0)
                out.append(MidiNote(tick=beat_tick + j * S16, duration=S16 - 30, pitch=pitch,
                                    velocity=100 if j == 1 else 86))
    return out


def offbeat(chords_halfbar: list[str], lo: int = 33, hi: int = 45) -> list[MidiNote]:
    """Bajo a contratiempo (house clásico): corchea en el "y" de cada tiempo."""
    out = []
    for h, ch in enumerate(chords_halfbar):
        p = parse_chord(ch)
        if p is None:
            continue
        root = note_in_range(p[0], lo, hi)
        for b in range(2):
            out.append(MidiNote(tick=h * 2 * Q + b * Q + Q // 2, duration=Q // 2 - 30, pitch=root, velocity=100))
    return out


def sustained(chords_halfbar: list[str], lo: int = 33, hi: int = 45) -> list[MidiNote]:
    """Raíces largas (intros y breaks)."""
    out = []
    for h, ch in enumerate(chords_halfbar):
        p = parse_chord(ch)
        if p is not None:
            out.append(MidiNote(tick=h * 2 * Q, duration=2 * Q - 40, pitch=note_in_range(p[0], lo, hi), velocity=90))
    return out
