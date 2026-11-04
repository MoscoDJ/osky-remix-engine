"""Teoría básica: notas, acordes, escalas y voicings con conducción de voces."""
from __future__ import annotations

NOTE = "C C# D D# E F F# G G# A A# B".split()
FLAT = {"Bb": "A#", "Eb": "D#", "Ab": "G#", "Db": "C#", "Gb": "F#", "Cb": "B", "Fb": "E"}
TICKS_PER_BEAT = 960
BAR = 4 * TICKS_PER_BEAT
S16 = TICKS_PER_BEAT // 4


def pc_of(name: str) -> int:
    return NOTE.index(FLAT.get(name, name))


def parse_chord(label: str) -> tuple[int, str] | None:
    """'Dm' -> (2, 'm'); 'F#' -> (6, ''); 'N' -> None."""
    if not label or label == "N":
        return None
    root = label[:2] if len(label) > 1 and label[1] in "#b" else label[:1]
    return pc_of(root), label[len(root):]


def chord_pcs(label: str) -> list[int]:
    p = parse_chord(label)
    if p is None:
        return []
    r, q = p
    third = 3 if q.startswith("m") else 4
    return [r, (r + third) % 12, (r + 7) % 12]


def key_scale(key: str) -> list[int]:
    """'D minor' -> pcs de la escala (menor natural + sensible armónica)."""
    root, mode = key.split()
    r = pc_of(root)
    iv = [0, 2, 4, 5, 7, 9, 11] if mode == "major" else [0, 2, 3, 5, 7, 8, 10, 11]
    return [(r + i) % 12 for i in iv]


def note_in_range(pc: int, lo: int, hi: int) -> int:
    """Nota MIDI con clase ``pc`` dentro de [lo, hi] (la más grave posible)."""
    n = lo + ((pc - lo) % 12)
    return n if n <= hi else n - 12


def voice_lead(prev: list[int] | None, pcs: list[int], lo: int = 50, hi: int = 74, size: int = 4) -> list[int]:
    """Voicing de ``size`` notas cercano al anterior (mínimo movimiento total)."""
    if not pcs:
        return []
    cands = sorted({n for pc in pcs for n in range(lo, hi + 1) if n % 12 == pc})
    if prev is None:
        base = note_in_range(pcs[0], lo, lo + 11)
        out = [base]
        for pc in (pcs[1:] + pcs[:1])[: size - 1]:
            n = out[-1] + ((pc - out[-1]) % 12 or 12)
            out.append(n if n <= hi else n - 12)
        return sorted(out)
    out = []
    for p in sorted(prev)[:size]:
        best = min(cands, key=lambda n: (abs(n - p), n in out))
        out.append(best)
    missing = [pc for pc in pcs if pc not in {n % 12 for n in out}]
    for pc in missing:                 # garantizar todas las notas del acorde
        i = max(range(len(out)), key=lambda k: sum(o % 12 == out[k] % 12 for o in out))
        out[i] = min((n for n in cands if n % 12 == pc), key=lambda n: abs(n - out[i]))
    return sorted(out)


def snap_to_scale(pitch: int, scale: list[int]) -> int:
    if pitch % 12 in scale:
        return pitch
    for d in (1, -1, 2, -2):
        if (pitch + d) % 12 in scale:
            return pitch + d
    return pitch
