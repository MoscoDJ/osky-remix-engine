"""Cue points y beatgrid para software de DJ, derivados EXACTAMENTE del mapa de tempo.

Cada posición sale de ``tempo_map().tick_to_sample(tick) / sample_rate``: la misma muestra
entera en la que el renderizador coloca ese tick en el master. Nunca se acumulan segundos con
aritmética de bpm en coma flotante.

Formatos:
- ``rekordbox``: XML ``DJ_PLAYLISTS`` (importable como "rekordbox xml"). Un ``TEMPO`` por punto
  del mapa de tempo (``Inizio`` en segundos, ``Bpm``, ``Metro``, ``Battito`` = tiempo del compás,
  1-4) y ``POSITION_MARK``: hot cues (``Num`` 0-7, A-H, con color) y memory cues (``Num`` -1).
- ``csv`` / ``json``: lista genérica (tick, compás, muestra, segundos, bpm) para otras
  herramientas o para revisar a mano.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET

from ..composition.arranger import DEFAULT_ENERGY
from ..core.models import Project

FORMATS = ("rekordbox", "csv", "json")
MAX_HOT_CUES = 8
# Colores (RGB) por categoría, de la paleta de hot cues de Rekordbox.
COLORS = {
    "intro": (40, 226, 20),      # verde
    "voz": (48, 90, 255),        # azul
    "break": (180, 50, 255),     # violeta
    "drop": (230, 40, 40),       # rojo
    "outro": (224, 100, 27),     # naranja
    "cue": (195, 175, 4),        # amarillo
}
# Prioridad cuando hay más secciones que hot cues: estas categorías entran primero.
PRIORITY = ("intro", "drop", "voz", "outro", "break")


@dataclass
class GridPoint:
    tick: int
    sample: int
    seconds: float
    bpm: float
    meter: str
    beat: int                 # 1..numerador: tiempo del compás en el que empieza el tramo
    on_beat: bool


@dataclass
class Cue:
    name: str
    category: str
    tick: int
    bar: int                  # compás (1-indexado, como en un DAW)
    sample: int
    seconds: float
    hot: int | None           # 0-7 (A-H) o None (solo memory cue)
    color: tuple[int, int, int]


def _seconds(p: Project, tick: int) -> tuple[int, Fraction]:
    s = p.tempo_map().tick_to_sample(tick)
    return s, Fraction(s, p.sample_rate)


def _bar_beat(p: Project, tick: int) -> tuple[int, int, int, str]:
    """(compás 0-indexado, tiempo 0-indexado, resto en ticks, "num/den") del tick."""
    mm = p.meter_map()
    pts = mm.points
    start, bar = 0, 0
    for i, m in enumerate(pts):
        tpb = mm.ticks_per_bar(m)
        end_bar = pts[i + 1].bar if i + 1 < len(pts) else None
        seg = None if end_bar is None else (end_bar - m.bar) * tpb
        if seg is None or tick < start + seg:
            rel = tick - start
            beat_t = p.ppq * 4 // m.denominator
            return m.bar + rel // tpb, (rel % tpb) // beat_t, (rel % tpb) % beat_t, f"{m.numerator}/{m.denominator}"
        start += seg
    raise AssertionError("inalcanzable")


def beatgrid(p: Project) -> list[GridPoint]:
    out = []
    for tp in p.tempo:
        s, sec = _seconds(p, tp.tick)
        _, beat, rest, meter = _bar_beat(p, tp.tick)
        out.append(GridPoint(tp.tick, s, float(sec), tp.bpm, meter, beat + 1, rest == 0))
    return out


def _categorize(p: Project, label: str, kind: str) -> str:
    if kind == "cue":
        return "cue"
    secs = {s["name"]: s for s in (p.meta.get("arrangement") or {}).get("sections", [])}
    s = secs.get(label, {})
    low = label.lower()
    if low.startswith("intro"):
        return "intro"
    if low.startswith("outro"):
        return "outro"
    energy = s.get("energy", DEFAULT_ENERGY.get(label, 2))
    if energy >= 3 or low.startswith("drop"):
        return "drop"
    if low.startswith(("break", "build")):
        return "break"
    vocal = s.get("vocal", True)
    return "voz" if vocal else "break"


def cue_points(p: Project, max_hot: int = MAX_HOT_CUES) -> list[Cue]:
    """Un memory cue por marcador de sección (o ``cue``); hasta ``max_hot`` de ellos son hot cues."""
    seen, marks = set(), []
    for m in sorted(p.markers, key=lambda m: (m.tick, m.id)):
        if m.kind == "note" or m.tick in seen:
            continue
        seen.add(m.tick)
        marks.append(m)
    cues = []
    for m in marks:
        s, sec = _seconds(p, m.tick)
        bar, _, _, _ = _bar_beat(p, m.tick)
        cat = _categorize(p, m.label, m.kind)
        cues.append(Cue(m.label, cat, m.tick, bar + 1, s, float(sec), None, COLORS[cat]))
    chosen: list[int] = []
    if len(cues) <= max_hot:
        chosen = list(range(len(cues)))
    else:
        for cat in PRIORITY:                         # primera de cada categoría (el outro, la última)
            idx = [i for i, c in enumerate(cues) if c.category == cat]
            if idx and len(chosen) < max_hot:
                chosen.append(idx[-1] if cat == "outro" else idx[0])
        for i in range(len(cues)):
            if len(chosen) >= max_hot:
                break
            if i not in chosen:
                chosen.append(i)
    for n, i in enumerate(sorted(chosen)):
        cues[i].hot = n
    return cues


def _num(x: float, nd: int = 6) -> str:
    return f"{x:.{nd}f}"


def _bpm(x: float) -> str:
    s = f"{x:.6f}".rstrip("0")
    whole, frac = s.split(".")
    return f"{whole}.{frac.ljust(2, '0')}"


def _average_bpm(p: Project, end_tick: int) -> float:
    if end_tick <= 0:
        return p.tempo[0].bpm
    return float(Fraction(end_tick, p.ppq) * 60 / p.tempo_map().seconds_at(end_tick))


def rekordbox_xml(p: Project, audio: Path, grid: list[GridPoint], cues: list[Cue], frames: int, sr: int,
                  title: str | None = None) -> str:
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="osky-remix", Version="1", Company="osky")
    col = ET.SubElement(root, "COLLECTION", Entries="1")
    total = Fraction(frames, sr)
    end_tick = p.tempo_map().sample_to_tick(round(total * p.sample_rate))
    path = audio.resolve()
    tr = ET.SubElement(col, "TRACK", TrackID="1", Name=title or p.name, Artist="", Album="", Genre="",
                       Kind="WAV File", Size=str(path.stat().st_size), TotalTime=str(int(total)),
                       AverageBpm=_bpm(round(_average_bpm(p, end_tick), 2)), SampleRate=str(sr),
                       Tonality=p.key or "", Location="file://localhost" + quote(path.as_posix()))
    for g in grid:
        ET.SubElement(tr, "TEMPO", Inizio=_num(g.seconds), Bpm=_bpm(g.bpm), Metro=g.meter, Battito=str(g.beat))
    for c in cues:                                   # memory cues
        ET.SubElement(tr, "POSITION_MARK", Name=c.name, Type="0", Start=_num(c.seconds), Num="-1")
    for c in cues:                                   # hot cues
        if c.hot is not None:
            r, g_, b = c.color
            ET.SubElement(tr, "POSITION_MARK", Name=c.name, Type="0", Start=_num(c.seconds), Num=str(c.hot),
                          Red=str(r), Green=str(g_), Blue=str(b))
    pl = ET.SubElement(root, "PLAYLISTS")
    node = ET.SubElement(pl, "NODE", Type="0", Name="ROOT", Count="1")
    lst = ET.SubElement(node, "NODE", Name="Osky Remix", Type="1", KeyType="0", Entries="1")
    ET.SubElement(lst, "TRACK", Key="1")
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def cues_csv(grid: list[GridPoint], cues: list[Cue]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["type", "hot_cue", "name", "category", "tick", "bar", "sample", "seconds", "bpm", "meter", "beat", "color"])
    for g in grid:
        w.writerow(["tempo", "", "", "", g.tick, "", g.sample, _num(g.seconds), _bpm(g.bpm), g.meter, g.beat, ""])
    for c in cues:
        w.writerow(["cue", "" if c.hot is None else "ABCDEFGH"[c.hot], c.name, c.category, c.tick, c.bar, c.sample,
                    _num(c.seconds), "", "", "", "#%02x%02x%02x" % c.color])
    return buf.getvalue()


def cues_json(p: Project, grid: list[GridPoint], cues: list[Cue]) -> str:
    return json.dumps({"project": p.name, "sample_rate": p.sample_rate, "ppq": p.ppq,
                       "beatgrid": [asdict(g) for g in grid], "cues": [asdict(c) for c in cues]},
                      indent=2, ensure_ascii=False)


def write_cues(p: Project, out: Path, fmt: str = "rekordbox", audio: Path | None = None) -> dict:
    """Escribe el archivo de cues. ``audio``: el master renderizado de ESTA revisión."""
    import soundfile as sf
    if fmt not in FORMATS:
        raise ValueError(f"formato desconocido '{fmt}' (hay: {', '.join(FORMATS)})")
    grid, cues = beatgrid(p), cue_points(p)
    warnings = [f"el cambio de tempo del tick {g.tick} no cae en un tiempo: el software de DJ lo moverá al tiempo"
                for g in grid if not g.on_beat]
    frames = sr = None
    if audio is not None:
        info = sf.info(str(audio))
        frames, sr = info.frames, info.samplerate
        if sr != p.sample_rate:
            warnings.append(f"el audio está a {sr} Hz y el proyecto a {p.sample_rate} Hz (los segundos no cambian)")
        last = max((c.seconds for c in cues), default=0.0)
        if frames / sr < last:
            raise ValueError(f"el audio dura {frames / sr:.2f} s y el último cue está en {last:.2f} s: "
                             "¿es el master de esta revisión?")
    if fmt == "rekordbox":
        if audio is None:
            raise ValueError("el XML de Rekordbox necesita el audio (--audio master.wav)")
        text = rekordbox_xml(p, audio, grid, cues, frames, sr)
    elif fmt == "csv":
        text = cues_csv(grid, cues)
    else:
        text = cues_json(p, grid, cues)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return {"file": str(out), "format": fmt, "tempo_points": len(grid), "cues": len(cues),
            "hot_cues": sum(c.hot is not None for c in cues), "warnings": warnings}
