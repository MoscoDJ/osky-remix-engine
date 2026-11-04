"""Modelo canónico del proyecto (Pydantic v2).

Reglas:
- Posiciones musicales en ticks enteros (``Project.ppq``, 960 por defecto).
- Posiciones dentro de un audio en muestras enteras.
- Los assets se identifican por SHA-256 de su contenido.
- El proyecto es un valor: las operaciones devuelven una copia nueva (ver operations.py).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .timeline import PPQ, MeterMap, MeterPoint, TempoMap, TempoPoint

SCHEMA_VERSION = 1
Id = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_\-]{0,63}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ------------------------------------------------------------------ assets
class AssetKind(str, Enum):
    original = "original"
    working = "working"          # copia PCM float de trabajo derivada de un original
    stem = "stem"
    sample = "sample"
    render = "render"
    midi = "midi"
    preset = "preset"
    plugin_state = "plugin_state"
    analysis = "analysis"


class Provenance(Strict):
    """De dónde salió un asset: herramienta, versión, parámetros, semilla y padres."""
    tool: str
    tool_version: str | None = None
    model: str | None = None
    params: dict = Field(default_factory=dict)
    seed: int | None = None
    parents: list[Sha256] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Asset(Strict):
    sha256: Sha256
    kind: AssetKind
    name: str
    ext: str = Field(pattern=r"^\.[a-z0-9]{1,8}$")
    size_bytes: int = Field(ge=0)
    sample_rate: int | None = None
    channels: int | None = None
    frames: int | None = None
    provenance: Provenance


# ------------------------------------------------------------------ clips
class MidiNote(Strict):
    tick: int = Field(ge=0)            # relativo al inicio del clip
    duration: int = Field(gt=0)
    pitch: int = Field(ge=0, le=127)
    velocity: int = Field(ge=1, le=127)


class AudioClip(Strict):
    type: Literal["audio"] = "audio"
    id: Id
    asset: Sha256
    start_tick: int = Field(ge=0)
    source_offset: int = Field(0, ge=0, description="muestra de inicio dentro del asset")
    source_length: int = Field(gt=0, description="muestras del asset que suenan")
    gain_db: float = 0.0
    fade_in: int = Field(0, ge=0, description="muestras")
    fade_out: int = Field(0, ge=0, description="muestras")


class MidiClip(Strict):
    type: Literal["midi"] = "midi"
    id: Id
    start_tick: int = Field(ge=0)
    length_ticks: int = Field(gt=0)
    notes: list[MidiNote] = Field(default_factory=list)

    @model_validator(mode="after")
    def _notes_inside(self):
        for n in self.notes:
            if n.tick >= self.length_ticks:
                raise ValueError(f"nota en tick {n.tick} fuera del clip ({self.length_ticks} ticks)")
        return self


Clip = Annotated[Union[AudioClip, MidiClip], Field(discriminator="type")]


# ------------------------------------------------------------------ procesamiento
class PluginSlot(Strict):
    id: Id
    backend: Literal["surge_xt", "vst3", "builtin", "faust"]
    uri: str = Field(description="ruta del plugin, nombre builtin o código faust")
    preset: Sha256 | None = None
    state: Sha256 | None = None
    params: dict[str, float] = Field(default_factory=dict)
    bypass: bool = False


class Send(Strict):
    target: Id
    level_db: float = 0.0
    pre_fader: bool = False


class TrackKind(str, Enum):
    audio = "audio"
    instrument = "instrument"
    bus = "bus"
    ret = "return"
    master = "master"


class Track(Strict):
    id: Id
    name: str
    kind: TrackKind
    clips: list[Clip] = Field(default_factory=list)
    plugins: list[PluginSlot] = Field(default_factory=list)
    sends: list[Send] = Field(default_factory=list)
    sidechain_from: Id | None = None
    output: Id | None = "master"
    gain_db: float = 0.0
    pan: float = Field(0.0, ge=-1.0, le=1.0)
    mute: bool = False
    solo: bool = False
    locked: bool = False


class AutomationPoint(Strict):
    tick: int = Field(ge=0)
    value: float
    curve: Literal["linear", "hold"] = "linear"


class AutomationLane(Strict):
    id: Id
    track: Id
    target: str = Field(description="'gain_db', 'pan' o '<plugin_id>.<parámetro>'")
    points: list[AutomationPoint]

    @field_validator("points")
    @classmethod
    def _sorted(cls, v):
        ticks = [p.tick for p in v]
        if ticks != sorted(ticks):
            raise ValueError("los puntos de automatización deben estar ordenados por tick")
        return v


class Marker(Strict):
    id: Id
    tick: int = Field(ge=0)
    label: str
    kind: Literal["section", "cue", "note"] = "section"


class TempoPointM(Strict):
    tick: int = Field(ge=0)
    bpm: float = Field(gt=20, lt=400)


class MeterPointM(Strict):
    bar: int = Field(ge=0)
    numerator: int = Field(ge=1, le=32)
    denominator: Literal[1, 2, 4, 8, 16, 32] = 4


def routing_edges(tracks: list[Track]) -> dict[str, set[str]]:
    """Dependencias de audio: pista -> pistas cuya salida necesita antes de procesarse."""
    deps: dict[str, set[str]] = {t.id: set() for t in tracks}
    for t in tracks:
        if t.output and t.output in deps and t.kind != TrackKind.master:
            deps[t.output].add(t.id)
        for s in t.sends:
            if s.target in deps:
                deps[s.target].add(t.id)
        if t.sidechain_from and t.sidechain_from in deps:
            deps[t.id].add(t.sidechain_from)
    return deps


def routing_cycle(tracks: list[Track]) -> list[str] | None:
    deps = routing_edges(tracks)
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(n: str) -> list[str] | None:
        state[n] = 1
        stack.append(n)
        for m in sorted(deps[n]):
            if state.get(m) == 1:
                return stack[stack.index(m):] + [m]
            if m not in state:
                c = visit(m)
                if c:
                    return c
        state[n] = 2
        stack.pop()
        return None

    for n in deps:
        if n not in state:
            c = visit(n)
            if c:
                return c
    return None


# ------------------------------------------------------------------ proyecto
class Project(Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    name: Id
    sample_rate: Literal[44100, 48000, 88200, 96000] = 48000
    ppq: Literal[960] = PPQ
    key: str | None = None
    master_profile: str = Field("club", pattern=r"^[a-z0-9_\-]{1,64}$", description="perfil de mastering por defecto")
    tempo: list[TempoPointM] = Field(default_factory=lambda: [TempoPointM(tick=0, bpm=124.0)])
    meter: list[MeterPointM] = Field(default_factory=lambda: [MeterPointM(bar=0, numerator=4)])
    assets: dict[Sha256, Asset] = Field(default_factory=dict)
    tracks: list[Track] = Field(default_factory=lambda: [Track(id="master", name="Master", kind=TrackKind.master, output=None)])
    automation: list[AutomationLane] = Field(default_factory=list)
    markers: list[Marker] = Field(default_factory=list)

    @model_validator(mode="after")
    def _integrity(self):
        ids = [t.id for t in self.tracks]
        if len(ids) != len(set(ids)):
            raise ValueError("ids de pista duplicados")
        if sum(t.kind == TrackKind.master for t in self.tracks) != 1:
            raise ValueError("debe existir exactamente una pista master")
        known = set(ids)
        clip_ids: set[str] = set()
        for t in self.tracks:
            for c in t.clips:
                if c.id in clip_ids:
                    raise ValueError(f"id de clip duplicado: {c.id}")
                clip_ids.add(c.id)
                if isinstance(c, AudioClip):
                    if t.kind != TrackKind.audio:
                        raise ValueError(f"clip de audio {c.id} en pista {t.kind.value}")
                    a = self.assets.get(c.asset)
                    if a is None:
                        raise ValueError(f"clip {c.id} referencia un asset inexistente")
                    if a.frames is not None and c.source_offset + c.source_length > a.frames:
                        raise ValueError(f"clip {c.id} excede la duración del asset")
                elif t.kind != TrackKind.instrument:
                    raise ValueError(f"clip MIDI {c.id} en pista {t.kind.value}")
            for ref in [t.output, t.sidechain_from, *(s.target for s in t.sends)]:
                if ref is not None and ref not in known:
                    raise ValueError(f"pista {t.id} referencia una pista inexistente: {ref}")
            for p in t.plugins:
                for h in (p.preset, p.state):
                    if h and h not in self.assets:
                        raise ValueError(f"plugin {p.id} referencia un asset inexistente")
        for lane in self.automation:
            if lane.track not in known:
                raise ValueError(f"automatización {lane.id} sobre pista inexistente")
        cycle = routing_cycle(self.tracks)
        if cycle:
            raise ValueError("ruteo circular: " + " -> ".join(cycle))
        if self.tempo[0].tick != 0 or self.meter[0].bar != 0:
            raise ValueError("tempo y compás deben definirse desde el inicio")
        return self

    # utilidades
    def track(self, track_id: str) -> Track:
        for t in self.tracks:
            if t.id == track_id:
                return t
        raise KeyError(track_id)

    def find_clip(self, clip_id: str) -> tuple[Track, Clip]:
        for t in self.tracks:
            for c in t.clips:
                if c.id == clip_id:
                    return t, c
        raise KeyError(clip_id)

    def tempo_map(self) -> TempoMap:
        return TempoMap([TempoPoint(p.tick, p.bpm) for p in self.tempo], self.sample_rate, self.ppq)

    def meter_map(self) -> MeterMap:
        return MeterMap([MeterPoint(p.bar, p.numerator, p.denominator) for p in self.meter], self.ppq)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()
