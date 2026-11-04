"""Contexto de composición derivado del análisis de la canción original.

Mapeo: 1 tiempo del original = 1 tiempo del remix (la voz se estira por bpm_destino/bpm_origen),
así que la armonía del original, leída por beats, cae exactamente bajo la voz estirada.

Cada sección puede tomar su propia región del original (``source``):
- ``{"lyric": "Creí ver", "occurrence": 1}``: tiempo fuerte que marca el MIDI de referencia
  para esa línea; la anacrusa se mide en la energía real del stem de voz.
- ``{"at": "2:52"}``: tiempo fuerte más cercano a ese instante.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..analysis.pipeline import AnalysisReport


@dataclass
class VocalSource:
    asset: str                 # sha del stem de voz YA estirado
    sr: int
    region_start: int          # muestra (en el audio estirado) del tiempo fuerte de inicio por defecto
    samples_per_beat: float    # en el audio estirado (= tiempo del remix)
    frames: int


@dataclass
class Region:
    beat: int                  # índice del beat (tiempo fuerte) del original
    pickup_beats: float        # anacrusa medida antes del tiempo fuerte
    label: str = ""


@dataclass
class ComposeContext:
    bpm: float
    key: str
    source_bpm: float
    beats: list[int]           # beats del original (muestras)
    b0: int                    # índice del beat de inicio de la región por defecto
    chords_src: list[tuple[int, int, str]]
    vocal: VocalSource | None = None
    melody: list[tuple[float, float, int]] = field(default_factory=list)   # (s, s, pitch) rel. a la región por defecto
    seed: int = 1
    downbeats: list[int] = field(default_factory=list)
    sr: int = 44100
    reference: object | None = None               # analysis.midi_ref.Reference
    vocal_env: np.ndarray | None = None           # energía (dB) del stem de voz original, cada 10 ms
    melody_provider: Callable[[int, int], list[tuple[float, float, int]]] | None = None

    @property
    def ratio(self) -> float:
        return self.bpm / self.source_bpm

    @property
    def period(self) -> float:
        return float(np.median(np.diff(self.beats))) / self.sr

    def chord_at(self, sample: int) -> str:
        for a, b, lab in self.chords_src:
            if a <= sample < b:
                return lab
        return "N"

    def halfbars(self, from_bar: int, bars: int, b0: int | None = None) -> list[str]:
        """Un acorde por medio compás del remix, leído de la región original."""
        b0 = self.b0 if b0 is None else b0
        span = len(self.beats) - 1 - b0
        out, last = [], "N"
        for h in range(bars * 2):
            labs = []
            for k in (0, 1):
                i = b0 + (from_bar * 4 + 2 * h + k) % max(span, 1)
                mid = (self.beats[i] + self.beats[min(i + 1, len(self.beats) - 1)]) // 2
                labs.append(self.chord_at(mid))
            lab = next((x for x in labs if x != "N"), last)
            out.append(lab)
            last = lab
        return out

    # ------------------------------------------------------------- regiones
    def _nearest_downbeat_beat(self, t: float) -> int:
        cands = self.downbeats or self.beats
        d = min(cands, key=lambda x: abs(x / self.sr - t))
        return int(np.argmin(np.abs(np.array(self.beats) - d)))

    def vocal_onset(self, t_stress: float, max_beats: float = 4.0) -> float | None:
        """Primer ataque de voz antes del tiempo fuerte (tras ≥0.3 s de silencio relativo)."""
        if self.vocal_env is None:
            return None
        env = self.vocal_env
        a = max(30, int((t_stress - max_beats * self.period) * 100))
        b = min(len(env), int((t_stress + 0.4) * 100))
        for i in range(a, b):
            if env[i] > -30 and env[i - 30:i].max() < -38:
                return i / 100
        return None

    def region(self, spec: dict | None) -> Region:
        if not spec:
            return Region(self.b0, 1.0, "por defecto")
        if "lyric" in spec:
            from ..analysis.midi_ref import find_lyric
            if self.reference is None:
                raise ValueError("la sección pide una línea de letra pero no hay MIDI de referencia (--reference)")
            line = find_lyric(self.reference, spec["lyric"], spec.get("occurrence", 1))
            if line is None:
                raise ValueError(f"no se encontró la letra '{spec['lyric']}' (aparición {spec.get('occurrence', 1)})")
            bi = self._nearest_downbeat_beat(line.start)
            onset = self.vocal_onset(self.beats[bi] / self.sr)
            pick = (self.beats[bi] / self.sr - onset) / self.period if onset is not None else 1.0
            return Region(bi, float(np.clip(pick, 0.0, 4.0)), line.text)
        if "at" in spec:
            from ..cli.produce import _parse_time
            return Region(self._nearest_downbeat_beat(_parse_time(spec["at"])), 1.0, spec["at"])
        raise ValueError(f"fuente de sección no reconocida: {spec}")

    def beat_of(self, t: float) -> float:
        bt = np.array(self.beats) / self.sr
        return float(np.interp(t, bt, np.arange(len(bt))))

    def ref_notes(self, role: str, beat: int, bars: int) -> list[tuple[float, float, int, int]]:
        """Notas de un canal del MIDI de referencia en la región: (inicio, dur) en beats relativos."""
        if self.reference is None:
            return []
        out = []
        for ch in self.reference.channels:
            if ch.role != role:
                continue
            for n in ch.notes:
                b1, b2 = self.beat_of(n.start) - beat, self.beat_of(n.end) - beat
                if 0 <= b1 < bars * 4:
                    out.append((b1, max(0.25, b2 - b1), n.pitch, n.velocity))
        return sorted(out)

    def melody_for(self, beat: int, bars: int) -> list[tuple[float, float, int]]:
        if beat == self.b0 and self.melody:
            return self.melody
        return self.melody_provider(beat, bars) if self.melody_provider else []


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
    return ComposeContext(bpm=bpm, key=key, source_bpm=seg.bpm, beats=beats, b0=b0, chords_src=chords, seed=seed,
                          downbeats=downs, sr=sr)
