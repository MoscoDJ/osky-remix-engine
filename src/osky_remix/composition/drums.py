"""Patrones de batería por estilo e intensidad. Mapa GM: 36 bombo, 38 caja, 39 palmas,
42 hat cerrado, 46 hat abierto, 49 crash, 51 ride, 37 rim (percusión)."""
from __future__ import annotations

import random

from ..core.models import MidiNote
from .theory import BAR, S16, TICKS_PER_BEAT as Q

KICK, SNARE, CLAP, CHH, OHH, CRASH, RIDE, RIM = 36, 38, 39, 42, 46, 49, 51, 37


def bar_pattern(intensity: int, rng: random.Random, beats: int = 4, variant: int = 0) -> list[MidiNote]:
    """Un compás. intensity: 0 = solo bombo, 1 = +hats, 2 = +palmas/abierto, 3 = +ride/percusión."""
    n: list[MidiNote] = []
    add = lambda p, t, v, d=S16: n.append(MidiNote(tick=t, duration=d, pitch=p, velocity=max(1, min(127, v))))
    for b in range(beats):
        add(KICK, b * Q, 118 if b == 0 else 112)
    if intensity >= 1:
        accents = [(2, 92), (1, 58), (3, 64), (0, 70)]
        for i in range(beats * 4):
            vel = dict(accents).get(i % 4, 60) + rng.randint(-4, 4)
            if variant == 1 and i % 4 == 1:
                continue                      # variante: hats más abiertos (sin la "e")
            add(CHH, i * S16, vel)
    if intensity >= 2:
        for b in (1, 3):
            if b < beats:
                add(CLAP, b * Q, 104)
        for b in range(beats):
            add(OHH, b * Q + Q // 2, 78, Q // 2)
    if intensity >= 3:
        for b in range(beats):
            add(RIDE, b * Q + Q // 2, 70)
        for t in ((3 * S16), (7 * S16), (14 * S16)) if variant == 0 else ((2 * S16), (10 * S16), (13 * S16)):
            if t < beats * Q:
                add(RIM, t, 66 + rng.randint(-6, 6))
    return n


def fill(beats: int = 4) -> list[MidiNote]:
    """Redoble de caja creciente en el último compás de una sección."""
    out = []
    for i in range(beats * 4):
        if i >= beats * 2 or i % 2 == 0:
            out.append(MidiNote(tick=i * S16, duration=S16, pitch=SNARE, velocity=min(127, 60 + i * 4)))
    return out


def crash_at(tick: int = 0) -> MidiNote:
    return MidiNote(tick=tick, duration=Q * 2, pitch=CRASH, velocity=100)


def section_drums(bars: int, intensity: int, seed: int, with_fill: bool = True, crash: bool = True,
                  no_kick_last_bar: bool = False) -> list[MidiNote]:
    rng = random.Random(seed)
    variant = rng.randint(0, 1)
    out: list[MidiNote] = []
    for b in range(bars):
        notes = bar_pattern(intensity, rng, variant=variant)
        if with_fill and b == bars - 1 and bars >= 4:
            notes = [x for x in notes if x.pitch in (KICK, CHH)] + fill()
        if no_kick_last_bar and b == bars - 1:
            notes = [x for x in notes if x.pitch != KICK]
        out += [x.model_copy(update={"tick": x.tick + b * BAR}) for x in notes]
    if crash and intensity >= 1:
        out.append(crash_at(0))
    return sorted(out, key=lambda x: (x.tick, x.pitch))
