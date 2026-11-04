"""Transcripción de audio a MIDI con Basic Pitch (ONNX) + limpieza de armónicos fantasma.

Basic Pitch con parámetros por defecto suele añadir notas 1-3 octavas arriba de una nota
real (armónicos), simultáneas y con ~la mitad de amplitud. Se filtran aquí.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pretty_midi


@dataclass(frozen=True)
class Note:
    start: float
    end: float
    pitch: int
    amplitude: float


def _model_path() -> Path:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    p = Path(ICASSP_2022_MODEL_PATH)
    onnx = p.with_suffix(".onnx")
    return onnx if onnx.exists() else p


def is_harmonic_ghost(n: Note, others: list[Note], amp_ratio: float = 0.6, min_overlap: float = 0.5) -> bool:
    """True si ``n`` está 12/19/24/28/31/36 semitonos arriba de una nota más fuerte simultánea."""
    for o in others:
        if o is n or o.pitch >= n.pitch:
            continue
        if (n.pitch - o.pitch) not in (12, 19, 24, 28, 31, 36):
            continue
        overlap = min(n.end, o.end) - max(n.start, o.start)
        if overlap >= min_overlap * (n.end - n.start) and n.amplitude <= amp_ratio * o.amplitude:
            return True
    return False


def transcribe(audio_path: Path, midi_out: Path | None = None, remove_harmonics: bool = True) -> tuple[list[Note], list[Note]]:
    """Devuelve (notas_limpias, notas_crudas). Si ``midi_out``, escribe el MIDI limpio."""
    from basic_pitch.inference import predict
    _, _, events = predict(str(audio_path), _model_path())
    raw = sorted((Note(float(s), float(e), int(p), float(a)) for s, e, p, a, *_ in events), key=lambda n: (n.start, n.pitch))
    clean = [n for n in raw if not (remove_harmonics and is_harmonic_ghost(n, raw))]
    if midi_out:
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0, name="transcription")
        for n in clean:
            inst.notes.append(pretty_midi.Note(velocity=max(1, min(127, int(n.amplitude * 127))),
                                               pitch=n.pitch, start=n.start, end=n.end))
        pm.instruments.append(inst)
        pm.write(str(midi_out))
    return clean, raw
