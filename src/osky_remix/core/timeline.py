"""Tempo map y conversión exacta entre ticks musicales y muestras de audio.

Posiciones musicales: ticks enteros (PPQ 960). Posiciones de audio: muestras enteras.
Los segundos solo existen como fracciones exactas (``Fraction``) dentro de la conversión;
nunca se guardan como referencia de posición.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction

PPQ = 960


def _frac(x: float | int | str | Fraction) -> Fraction:
    # str() evita arrastrar el error binario de un float (124.0 -> 124, 86.1 -> 861/10)
    return x if isinstance(x, Fraction) else Fraction(str(x))


@dataclass(frozen=True)
class TempoPoint:
    tick: int
    bpm: float


@dataclass(frozen=True)
class MeterPoint:
    bar: int            # compás (0-indexado) donde empieza este compás de tiempo
    numerator: int = 4
    denominator: int = 4


class TempoMap:
    """Tempo constante por tramos. ``points`` ordenados, el primero en tick 0."""

    def __init__(self, points: list[TempoPoint], sample_rate: int, ppq: int = PPQ):
        if not points or points[0].tick != 0:
            raise ValueError("el tempo map debe empezar en el tick 0")
        ticks = [p.tick for p in points]
        if ticks != sorted(set(ticks)):
            raise ValueError("los puntos de tempo deben tener ticks únicos y crecientes")
        if any(p.bpm <= 0 for p in points):
            raise ValueError("bpm debe ser positivo")
        self.points = points
        self.sr = sample_rate
        self.ppq = ppq
        self._ticks = ticks
        # segundos exactos acumulados al inicio de cada tramo
        self._sec = [Fraction(0)]
        for a, b in zip(points, points[1:]):
            self._sec.append(self._sec[-1] + self._span(a, b.tick - a.tick))

    def _span(self, p: TempoPoint, dticks: int) -> Fraction:
        return Fraction(dticks, self.ppq) * 60 / _frac(p.bpm)

    def seconds_at(self, tick: int) -> Fraction:
        if tick < 0:
            raise ValueError("tick negativo")
        i = bisect_right(self._ticks, tick) - 1
        return self._sec[i] + self._span(self.points[i], tick - self.points[i].tick)

    def tick_to_sample(self, tick: int) -> int:
        """Muestra entera más cercana (redondeo half-even de Fraction)."""
        return round(self.seconds_at(tick) * self.sr)

    def sample_to_tick(self, sample: int) -> int:
        if sample < 0:
            raise ValueError("muestra negativa")
        sec = Fraction(sample, self.sr)
        i = bisect_right(self._sec, sec) - 1
        p = self.points[i]
        return p.tick + round((sec - self._sec[i]) * _frac(p.bpm) / 60 * self.ppq)

    def bpm_at(self, tick: int) -> float:
        return self.points[bisect_right(self._ticks, tick) - 1].bpm


class MeterMap:
    def __init__(self, points: list[MeterPoint], ppq: int = PPQ):
        if not points or points[0].bar != 0:
            raise ValueError("el mapa de compás debe empezar en el compás 0")
        self.points = sorted(points, key=lambda p: p.bar)
        self.ppq = ppq

    def ticks_per_bar(self, p: MeterPoint) -> int:
        return self.ppq * 4 * p.numerator // p.denominator

    def bar_to_tick(self, bar: int, beat: int = 0) -> int:
        """Tick del ``beat`` (en unidades del denominador) del compás ``bar``."""
        tick = 0
        for i, p in enumerate(self.points):
            end = self.points[i + 1].bar if i + 1 < len(self.points) else None
            if end is not None and bar >= end:
                tick += (end - p.bar) * self.ticks_per_bar(p)
                continue
            return tick + (bar - p.bar) * self.ticks_per_bar(p) + beat * self.ppq * 4 // p.denominator
        raise AssertionError("inalcanzable")
