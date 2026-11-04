"""Canciones sintéticas con verdad conocida para probar el análisis.

``make_song`` devuelve el audio y un ``GroundTruth`` con beats, tiempos fuertes, secciones,
acordes, tonalidad, silencios y transitorios exactos (en muestras).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt

from .fixtures import SR, harmonic_tone, midi_to_hz

NOTE = "C C# D D# E F F# G G# A A# B".split()
FLAT = {"Bb": "A#", "Eb": "D#", "Ab": "G#", "Db": "C#", "Gb": "F#"}


def chord_notes(label: str, octave: int = 4) -> tuple[int, list[int]]:
    """'Dm' -> (pc raíz, notas MIDI de la tríada)."""
    minor = label.endswith("m")
    root = label[:-1] if minor else label
    pc = NOTE.index(FLAT.get(root, root))
    base = 12 * (octave + 1) + pc
    return pc, [base, base + (3 if minor else 4), base + 7]


@dataclass
class Section:
    label: str
    bars: int
    chords: list[str]               # uno por compás, se repite
    drums: bool = True
    lead: bool = False
    meter: int = 4                  # tiempos por compás
    bpm: float | None = None        # cambia el tempo desde esta sección


@dataclass
class GroundTruth:
    sr: int
    beats: list[int] = field(default_factory=list)
    downbeats: list[int] = field(default_factory=list)
    sections: list[tuple[int, str]] = field(default_factory=list)       # (muestra, etiqueta)
    chords: list[tuple[int, int, str]] = field(default_factory=list)    # (ini, fin, acorde)
    tempo: list[tuple[int, float]] = field(default_factory=list)        # (muestra, bpm)
    meter: int = 4
    key: str = ""
    silences: list[tuple[int, int]] = field(default_factory=list)
    kicks: list[int] = field(default_factory=list)
    inserted_short_bars: list[int] = field(default_factory=list)        # muestras


def _kick(sr):
    n = int(0.3 * sr); t = np.arange(n) / sr
    return np.sin(2 * np.pi * np.cumsum(48 + 140 * np.exp(-t * 30)) / sr) * np.exp(-t * 9)


def _noise(n, lo, hi, decay, sr, rng):
    t = np.arange(n) / sr
    return sosfilt(butter(2, [lo, hi], "band", fs=sr, output="sos"), rng.standard_normal(n)) * np.exp(-t * decay)


def make_song(sections: list[Section], bpm: float = 100.0, key: str = "D minor", lead_silence: float = 1.0,
              tail_silence: float = 1.0, short_bar_after: int | None = None, sr: int = SR,
              seed: int = 7) -> tuple[np.ndarray, GroundTruth]:
    """``short_bar_after``: inserta un compás de 2 tiempos tras ese compás global (desfase de fase)."""
    rng = np.random.default_rng(seed)
    gt = GroundTruth(sr=sr, key=key, meter=sections[0].meter)
    # 1) línea de tiempo: (inicio_s, tiempos, spb, sección, acorde) por compás
    bars, t, cur_bpm, gbar = [], lead_silence, bpm, 0
    for s in sections:
        if s.bpm:
            cur_bpm = s.bpm
        gt.sections.append((int(round(t * sr)), s.label))
        gt.tempo.append((int(round(t * sr)), cur_bpm))
        for b in range(s.bars):
            spb = 60.0 / cur_bpm
            bars.append((t, s.meter, spb, s, s.chords[b % len(s.chords)]))
            t += s.meter * spb
            if short_bar_after is not None and gbar == short_bar_after:
                bars.append((t, 2, spb, s, s.chords[b % len(s.chords)]))
                gt.inserted_short_bars.append(int(round(t * sr)))
                t += 2 * spb
            gbar += 1
    total = t + tail_silence
    n = int(total * sr)
    out = np.zeros(n)
    kick = _kick(sr)
    # 2) eventos
    for i, (t0, beats, spb, s, ch) in enumerate(bars):
        i0 = int(round(t0 * sr))
        gt.downbeats.append(i0)
        bar_len = beats * spb
        end = int(round((t0 + bar_len) * sr))
        gt.chords.append((i0, end, ch))
        pc, notes = chord_notes(ch, 3)
        # pad del acorde (todo el compás)
        pad = sum(harmonic_tone(midi_to_hz(m), bar_len, sr, n_harm=5, decay=0.4) for m in notes) * 0.18
        out[i0:i0 + len(pad)] += pad[: n - i0]
        # bajo: raíz en el tiempo 1 (acento), corcheas suaves después
        for k in range(beats * 2):
            ib = int(round((t0 + k * spb / 2) * sr))
            note = harmonic_tone(midi_to_hz(36 + pc if pc < 7 else 24 + pc), spb / 2 * 0.9, sr, n_harm=4, decay=6)
            out[ib:ib + len(note)] += note[: n - ib] * (0.55 if k == 0 else 0.25)
        for k in range(beats):
            tb = t0 + k * spb
            ib = int(round(tb * sr))
            gt.beats.append(ib)
            if s.drums:
                acc = 1.0 if k == 0 else 0.7
                out[ib:ib + len(kick)] += kick[: n - ib] * acc
                gt.kicks.append(ib)
                if beats == 4 and k in (1, 3):
                    sn = _noise(int(0.18 * sr), 1200, 6000, 22, sr, rng)
                    out[ib:ib + len(sn)] += sn[: n - ib] * 0.5
                ih = int(round((tb + spb / 2) * sr))
                hh = _noise(int(0.05 * sr), 6000, 15000, 70, sr, rng)
                out[ih:ih + len(hh)] += hh[: n - ih] * 0.25
                if k == 0 and (i == 0 or bars[i - 1][3] is not s):
                    cr = _noise(int(1.2 * sr), 3000, 16000, 3, sr, rng)      # crash al iniciar sección
                    out[ib:ib + len(cr)] += cr[: n - ib] * 0.35
            if s.lead:
                for j, m in enumerate((notes[0] + 12, notes[1] + 12, notes[2] + 12, notes[1] + 24)):
                    il = int(round((tb + j * spb / 4) * sr))
                    ln = harmonic_tone(midi_to_hz(m), spb / 4 * 0.9, sr, n_harm=7, decay=8) * 0.22
                    out[il:il + len(ln)] += ln[: n - il]
    gt.silences = [(0, int(lead_silence * sr))] if lead_silence > 0 else []
    if tail_silence > 0:
        gt.silences.append((int((total - tail_silence + 0.6) * sr), n))   # el pad decae ~0.6 s
    out = 0.8 * out / np.abs(out).max()
    return np.stack([out, out * 0.97], 1).astype(np.float32), gt


def pop_song(**kw) -> tuple[np.ndarray, GroundTruth]:
    """Intro / A / B / A en Re menor a 100 BPM, 4/4."""
    secs = [Section("intro", 4, ["Dm"], drums=False),
            Section("A", 8, ["Dm", "Gm", "C", "F"]),
            Section("B", 8, ["Bb", "C", "Dm", "A"], lead=True),
            Section("A", 8, ["Dm", "Gm", "C", "F"])]
    return make_song(secs, **kw)
