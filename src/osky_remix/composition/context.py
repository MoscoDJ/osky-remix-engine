"""Contexto de composición derivado del análisis de la canción original.

Mapeo: 1 tiempo del original = 1 tiempo del remix (la voz se estira por bpm_destino/bpm_origen),
así que la armonía del original, leída por beats, cae exactamente bajo la voz estirada.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analysis.pipeline import AnalysisReport


@dataclass
class VocalSource:
    asset: str                 # sha del stem de voz YA estirado
    sr: int
    region_start: int          # muestra (en el audio estirado) del tiempo fuerte de inicio
    samples_per_beat: float    # en el audio estirado (= tiempo del remix)
    frames: int


@dataclass
class ComposeContext:
    bpm: float
    key: str
    source_bpm: float
    beats: list[int]           # beats del original (muestras)
    b0: int                    # índice del beat de inicio de la región
    chords_src: list[tuple[int, int, str]]
    vocal: VocalSource | None = None
    melody: list[tuple[float, float, int]] = field(default_factory=list)   # (s, s, pitch) rel. a la región
    seed: int = 1

    @property
    def ratio(self) -> float:
        return self.bpm / self.source_bpm

    def chord_at(self, sample: int) -> str:
        for a, b, lab in self.chords_src:
            if a <= sample < b:
                return lab
        return "N"

    def halfbars(self, from_bar: int, bars: int) -> list[str]:
        """Un acorde por medio compás del remix, leído de la región original."""
        span = len(self.beats) - 1 - self.b0
        out, last = [], "N"
        for h in range(bars * 2):
            labs = []
            for k in (0, 1):
                i = self.b0 + (from_bar * 4 + 2 * h + k) % max(span, 1)
                mid = (self.beats[i] + self.beats[min(i + 1, len(self.beats) - 1)]) // 2
                labs.append(self.chord_at(mid))
            lab = next((x for x in labs if x != "N"), last)
            out.append(lab)
            last = lab
        return out


def build_context(report: AnalysisReport, region_start_s: float, bpm: float, seed: int = 1) -> ComposeContext:
    sr = report.sample_rate
    beats = report.rhythm.beats
    downs = report.rhythm.downbeats
    target = int(region_start_s * sr)
    start = min(downs, key=lambda d: abs(d - target)) if downs else min(beats, key=lambda b: abs(b - target))
    b0 = beats.index(start) if start in beats else int(np.argmin(np.abs(np.array(beats) - start)))
    seg = next((s for s in report.rhythm.segments if s.start <= start < s.end), report.rhythm.segments[0])
    key = next((k.key for k in report.harmony.key_segments if k.start <= start < k.end), report.harmony.key.value)
    chords = [(c.start, c.end, c.label) for c in report.harmony.chords]
    return ComposeContext(bpm=bpm, key=key, source_bpm=seg.bpm, beats=beats, b0=b0, chords_src=chords, seed=seed)
