"""Groove y gancho para la forma de canción (melodic techno con voz).

Patrones por compás en semicorcheas (``S16`` ticks), con swing aplicado a las semicorcheas
pares (la "e" y la "a" de cada tiempo se retrasan ``(swing − 0.5)·2·S16``). Todo es determinista
por ``seed``.

- Bombo 4x4; en subidas se quita el último tiempo de cada frase de 8 compases.
- Bajo rodante: no toca en la semicorchea del bombo (deja el golpe limpio con el sidechain) y
  salta a la octava en la última semicorchea del tiempo; variantes por energía.
- Hats: cerrados en semicorcheas con acentos y swing, abierto a contratiempo; clap en 2 y 4
  con un fantasma cada 4 compases; percusión sincopada 3-3-2; ride en los drops.
- Gancho: contorno de la melodía del coro (la voz) re-ritmado en semicorcheas sincopadas; en
  cada tiempo fuerte la nota se ajusta a una nota del acorde que suena. Frases de 8 compases:
  A A A' B (pregunta/respuesta cada 2 compases).
"""
from __future__ import annotations

import random

from ..core.models import MidiNote
from .theory import BAR, S16, TICKS_PER_BEAT as Q, chord_pcs, note_in_range, snap_to_scale

KICK, SNARE, CLAP, CHH, OHH, CRASH, RIDE, RIM = 36, 38, 39, 42, 46, 49, 51, 37


def _swing(slot: int, swing: float) -> int:
    return int(round((swing - 0.5) * 2 * S16)) if slot % 2 == 1 else 0


def _n(tick: int, pitch: int, vel: int, dur: int = S16 - 10) -> MidiNote:
    return MidiNote(tick=max(0, tick), duration=max(10, dur), pitch=pitch, velocity=max(1, min(127, vel)))


# ------------------------------------------------------------------------------ batería
def kicks(bars: int, drop_phrase_end: bool = False, phrase: int = 8) -> list[MidiNote]:
    out = []
    for b in range(bars):
        for beat in range(4):
            if drop_phrase_end and (b + 1) % phrase == 0 and beat == 3:
                continue
            out.append(_n(b * BAR + beat * Q, KICK, 118 if beat == 0 else 112, S16))
    return out


def tops(bars: int, level: int, seed: int, swing: float = 0.56, fills: bool = True, phrase: int = 8) -> list[MidiNote]:
    """level 1: hat abierto a contratiempo + cerrados suaves; 2: + clap y percusión; 3: + ride."""
    rng = random.Random(seed)
    out = []
    perc_bars = [[3, 6, 10], [3, 6, 11, 14], [2, 6, 10, 13]]
    for b in range(bars):
        t0 = b * BAR
        last = fills and (b + 1) % phrase == 0
        for s in range(16):
            beat_pos = s % 4
            if beat_pos == 2:
                out.append(_n(t0 + s * S16, OHH, 92 + rng.randint(-4, 4), Q // 2))
            elif level >= 1 and not (last and s >= 8):
                vel = {0: 54, 1: 66, 3: 72}[beat_pos] + rng.randint(-6, 6)
                out.append(_n(t0 + s * S16 + _swing(s, swing), CHH, vel))
        if level >= 2:
            for beat in (1, 3):
                out.append(_n(t0 + beat * Q, CLAP, 104))
            if b % 4 == 3:
                out.append(_n(t0 + 15 * S16 + _swing(15, swing), CLAP, 58))
            for s in perc_bars[(b // 2 + seed) % len(perc_bars)]:
                out.append(_n(t0 + s * S16 + _swing(s, swing), RIM, 70 + rng.randint(-8, 8)))
        if level >= 3:
            for beat in range(4):
                out.append(_n(t0 + beat * Q + Q // 2, RIDE, 64 + rng.randint(-5, 5)))
        if last and level >= 1:
            for s in range(8, 16):
                out.append(_n(t0 + s * S16, SNARE, 52 + (s - 8) * 8))
    return sorted(out, key=lambda x: (x.tick, x.pitch))


def snare_build(bars: int) -> list[MidiNote]:
    """Redoble que se acelera: negras, corcheas y semicorcheas en los últimos 4 compases."""
    out = []
    for b in range(bars):
        rem = bars - b
        step = Q if rem > 4 else (Q // 2 if rem > 2 else S16)
        for t in range(0, BAR, step):
            prog = (b * BAR + t) / (bars * BAR)
            out.append(_n(b * BAR + t, SNARE, int(48 + 70 * prog), min(step, S16)))
    return out


# ------------------------------------------------------------------------------ bajo
BASS_PATTERNS = {
    # (semicorchea dentro del tiempo, desplazamiento: 0 = raíz, 12 = octava, 7 = quinta)
    "pulse": [(2, 0)],                                   # corchea a contratiempo
    "light": [(1, 0), (3, 0)],
    "rolling": [(1, 0), (2, 0), (3, 12)],
    "rolling_fifth": [(1, 0), (2, 7), (3, 12)],
}


def bass(halfbars: list[str], style: str, lo: int = 33, hi: int = 45, seed: int = 0) -> list[MidiNote]:
    rng = random.Random(seed)
    pat = BASS_PATTERNS[style]
    out = []
    for h, ch in enumerate(halfbars):
        pcs = chord_pcs(ch)
        if not pcs:
            continue
        root = note_in_range(pcs[0], lo, hi)
        for beat in range(2):
            t = h * 2 * Q + beat * Q
            for s, off in pat:
                p = root + off
                if off == 7 and rng.random() < 0.3:
                    p = root
                out.append(_n(t + s * S16, p, (100 if s == 1 else 88) + rng.randint(-5, 5), int(S16 * 0.8)))
    return out


def bass_sustained(halfbars: list[str], lo: int = 33, hi: int = 45) -> list[MidiNote]:
    out, i = [], 0
    while i < len(halfbars):
        j = i
        while j + 1 < len(halfbars) and halfbars[j + 1] == halfbars[i]:
            j += 1
        pcs = chord_pcs(halfbars[i])
        if pcs:
            out.append(_n(i * 2 * Q, note_in_range(pcs[0], lo, hi), 92, (j - i + 1) * 2 * Q - 30))
        i = j + 1
    return out


# ------------------------------------------------------------------------------ gancho
HOOK_RHYTHMS = [
    [0, 3, 6, 8, 11, 14, 16, 19, 22, 24, 26, 28],         # 3-3-2 y respuesta en corcheas
    [0, 2, 3, 6, 8, 11, 14, 16, 19, 20, 22, 27],
    [0, 3, 4, 7, 10, 12, 16, 18, 19, 22, 24, 28],
]


def hook_cell(vocal_notes: list[tuple[float, float, int]], size: int = 5) -> list[int]:
    """Contorno (alturas) de la frase: notas sucesivas distintas, sin adornos de < 90 ms."""
    seq = []
    for s, e, p in sorted(vocal_notes):
        if e - s < 0.09:
            continue
        if not seq or seq[-1] != p:
            seq.append(int(p))
        if len(seq) >= size:
            break
    return seq


def hook(halfbars: list[str], cell: list[int], scale: list[int], seed: int, register: int = 79,
         variation: int = 0) -> list[MidiNote]:
    """Gancho sobre la progresión ``halfbars``; ``cell`` = contorno; frases de 8 compases A A A' B."""
    rng = random.Random(seed)
    rhythm = HOOK_RHYTHMS[(seed + variation) % len(HOOK_RHYTHMS)]
    if not cell:
        cell = [scale[0] + 60, scale[2] + 60, scale[4 % len(scale)] + 60, scale[2] + 60]
    base = 12 * round((register - sum(cell) / len(cell)) / 12)
    cell = [p + base for p in cell]
    bars = len(halfbars) // 2
    out = []
    for b0 in range(0, bars, 2):                         # unidades de 2 compases
        phrase_pos = (b0 // 2) % 4                       # 0,1,2 = A, A, A'; 3 = B
        shift = {0: 0, 1: 0, 2: 0, 3: rng.choice([2, 3, -2])}[phrase_pos]
        for i, slot in enumerate(rhythm):
            tick = b0 * BAR + slot * S16
            if tick >= bars * BAR:
                break
            h = tick // (2 * Q)
            pcs = chord_pcs(halfbars[min(h, len(halfbars) - 1)])
            p = cell[i % len(cell)] + (shift if i >= len(rhythm) // 2 else 0)
            if phrase_pos == 2 and i >= len(rhythm) - 3:
                p += 12 if variation % 2 == 0 else -5           # A': remate hacia arriba / abajo
            if slot % 4 == 0 and pcs:                            # tiempo fuerte: nota del acorde
                p = min((n for n in range(p - 6, p + 7) if n % 12 in pcs), key=lambda n: abs(n - p))
            else:
                p = snap_to_scale(p, scale)
            nxt = rhythm[i + 1] if i + 1 < len(rhythm) else 32
            dur = min(int((nxt - slot) * S16 * 0.9), 3 * S16)
            out.append(_n(tick, int(p), 104 if slot % 4 == 0 else 90, dur))
    return out


def hook_answer(halfbars: list[str], cell: list[int], scale: list[int], seed: int, register: int = 79) -> list[MidiNote]:
    """Respuesta corta (1 compás) para huecos de la voz: las últimas notas del gancho."""
    full = hook(halfbars[:4] if len(halfbars) >= 4 else halfbars, cell, scale, seed, register)
    first_bar = [n for n in full if BAR <= n.tick < 2 * BAR]
    return [n.model_copy(update={"tick": n.tick - BAR, "velocity": n.velocity - 12}) for n in first_bar]


# ------------------------------------------------------------------------------ otros estilos
def halfstep(bars: int, level: int, seed: int, fills: bool = True, phrase: int = 8) -> list[MidiNote]:
    """Dubstep half-time escrito al tempo de la voz (p. ej. 75 = 150/2): cada compás contiene dos
    compases de 150, así que bombo en 1 y 3 y snare+clap en 2 y 4 (el "3" de cada compás a 150).
    level 1: bombo, snare y hats en corcheas; 2: + bombo fantasma y semicorcheas; 3: + abierto."""
    rng = random.Random(seed)
    out = []
    for b in range(bars):
        t0 = b * BAR
        last = fills and (b + 1) % phrase == 0
        out += [_n(t0, KICK, 120, S16), _n(t0 + 2 * Q, KICK, 116, S16)]
        if level >= 2:
            out.append(_n(t0 + 2 * Q + 3 * S16 if b % 2 else t0 + 3 * Q + 2 * S16, KICK, 104, S16))
        for beat in (1, 3):
            out += [_n(t0 + beat * Q, SNARE, 122), _n(t0 + beat * Q, CLAP, 110)]
        step = S16 if level >= 2 else 2 * S16
        for t in range(0, BAR, step):
            if last and t >= 3 * Q:
                break
            out.append(_n(t0 + t, CHH, (84 if t % (2 * S16) == 0 else 58) + rng.randint(-6, 6)))
        if level >= 3:
            for beat in (0, 2):
                out.append(_n(t0 + beat * Q + 2 * S16, OHH, 80, Q // 2))
        if last:
            for k in range(8):                                   # redoble en fusas (a 150)
                out.append(_n(t0 + 3 * Q + k * S16 // 2, SNARE, 70 + 6 * k, S16 // 2))
    return sorted(out, key=lambda x: (x.tick, x.pitch))


def dembow(bars: int, level: int, seed: int, fills: bool = True, phrase: int = 8) -> list[MidiNote]:
    """Dembow: bombo en cada tiempo y caja en la 4.ª semicorchea del tiempo 1 y en el "y" del 2
    (patrón de 2 tiempos repetido). level 2: + hats en corcheas; 3: + rim y timbal de remate."""
    rng = random.Random(seed)
    out = []
    for b in range(bars):
        t0 = b * BAR
        last = fills and (b + 1) % phrase == 0
        for beat in range(4):
            out.append(_n(t0 + beat * Q, KICK, 118 if beat % 2 == 0 else 108, S16))
        for half in (0, 2):
            for s in (3, 6):
                out.append(_n(t0 + half * Q + s * S16, SNARE, 112 if s == 3 else 104))
        if level >= 2:
            for t in range(0, BAR, 2 * S16):
                out.append(_n(t0 + t, CHH, (78 if t % Q == 0 else 64) + rng.randint(-5, 5)))
        if level >= 3:
            for s in (2, 7, 10, 15):
                out.append(_n(t0 + s * S16, RIM, 66 + rng.randint(-8, 8)))
        if last:
            for k in range(4):
                out.append(_n(t0 + 3 * Q + k * S16, SNARE, 80 + 10 * k))
    return sorted(out, key=lambda x: (x.tick, x.pitch))


def bass_808(halfbars: list[str], lo: int = 28, hi: int = 40) -> list[MidiNote]:
    """Tumbao del 808 que sigue al dembow: raíz en el tiempo, en la 4.ª semicorchea y en el "y"."""
    out = []
    for h, ch in enumerate(halfbars):
        pcs = chord_pcs(ch)
        if not pcs:
            continue
        root = note_in_range(pcs[0], lo, hi)
        t = h * 2 * Q
        out += [_n(t, root, 112, 3 * S16 - 20), _n(t + 3 * S16, root, 96, 3 * S16 - 20), _n(t + 6 * S16, root, 104, 2 * S16 - 20)]
    return out


def wobble_points(bars: int, rates: list[float], lo: float, hi: float, seed: int = 0) -> list[tuple[int, int, float, float]]:
    """Rampas (t1, t2, desde, hasta) de un LFO triangular del filtro: ``rates`` ciclos por tiempo,
    cambiando cada 2 compases (1 = negras, 2 = corcheas, 3 = tresillos, 4 = semicorcheas)."""
    rng = random.Random(seed)
    out = []
    for b0 in range(0, bars, 2):
        rate = rates[(b0 // 2) % len(rates)] if b0 % 8 != 6 else rng.choice(rates)
        half = int(round(Q / rate / 2))
        t, up = b0 * BAR, True
        end = min(bars, b0 + 2) * BAR
        while t < end:
            t2 = min(end, t + half)
            out.append((t, t2, lo if up else hi, hi if up else lo))
            t, up = t2, not up
    return out
