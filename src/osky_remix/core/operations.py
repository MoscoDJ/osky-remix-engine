"""Contrato de operaciones sobre el proyecto.

Cada operación declara:
- ``check``: precondiciones sobre el proyecto actual (lanza ``OperationError``).
- ``apply``: muta una COPIA del proyecto.
- ``verify``: postcondiciones específicas comparando antes/después.
La revalidación completa del modelo se hace siempre en ``transactions.apply_ops``.

Política de pistas bloqueadas: ninguna operación puede tocar una pista con ``locked=True``
(ni sus clips, plugins ni automatizaciones). Solo un autor ``user`` puede desbloquearla.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

from pydantic import Field

from .models import (Asset, AudioClip, AutomationLane, AutomationPoint, Id, Marker, MeterPointM, MidiClip, MidiNote,
                     PluginSlot, Project, Send, Strict, TempoPointM, Track, TrackKind)

Author = Literal["user", "claude"]


class OperationError(Exception):
    pass


@dataclass
class Context:
    author: Author
    asset_exists: callable = lambda sha: True     # el AssetStore lo reemplaza


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise OperationError(msg)


def _track(p: Project, track_id: str) -> Track:
    try:
        return p.track(track_id)
    except KeyError:
        raise OperationError(f"no existe la pista '{track_id}'") from None


def _unlocked(p: Project, track_id: str) -> Track:
    t = _track(p, track_id)
    _require(not t.locked, f"la pista '{track_id}' está bloqueada (edición manual protegida)")
    return t


def _clip(p: Project, clip_id: str):
    try:
        return p.find_clip(clip_id)
    except KeyError:
        raise OperationError(f"no existe el clip '{clip_id}'") from None


def _plugin(t: Track, plugin_id: str) -> int:
    for i, s in enumerate(t.plugins):
        if s.id == plugin_id:
            return i
    raise OperationError(f"la pista '{t.id}' no tiene el plugin '{plugin_id}'")


class Op(Strict):
    def check(self, p: Project, ctx: Context) -> None: ...
    def apply(self, p: Project) -> None: ...
    def verify(self, before: Project, after: Project) -> None: ...
    def touches(self, p: Project) -> set[str]:
        """Pistas afectadas (para la política de bloqueo)."""
        return set()


# ------------------------------------------------------------------ pistas
class AddTrack(Op):
    op: Literal["add_track"] = "add_track"
    id: Id
    name: str
    kind: Literal["audio", "instrument", "bus", "return"]
    output: Id | None = "master"
    index: int | None = None

    def check(self, p, ctx):
        _require(all(t.id != self.id for t in p.tracks), f"ya existe la pista '{self.id}'")
        if self.output:
            _track(p, self.output)

    def apply(self, p):
        t = Track(id=self.id, name=self.name, kind=TrackKind(self.kind), output=self.output)
        p.tracks.insert(len(p.tracks) if self.index is None else self.index, t)

    def verify(self, before, after):
        _require(len(after.tracks) == len(before.tracks) + 1, "no se agregó la pista")


class RemoveTrack(Op):
    op: Literal["remove_track"] = "remove_track"
    track: Id

    def touches(self, p):
        return {self.track}

    def check(self, p, ctx):
        t = _track(p, self.track)
        _require(t.kind != TrackKind.master, "no se puede eliminar el master")
        refs = [o.id for o in p.tracks if self.track in (o.output, o.sidechain_from) or any(s.target == self.track for s in o.sends)]
        _require(not refs, f"la pista '{self.track}' es destino de: {', '.join(refs)}")

    def apply(self, p):
        p.tracks = [t for t in p.tracks if t.id != self.track]
        p.automation = [a for a in p.automation if a.track != self.track]

    def verify(self, before, after):
        _require(all(t.id != self.track for t in after.tracks), "la pista sigue existiendo")


class SetTrack(Op):
    op: Literal["set_track"] = "set_track"
    track: Id
    name: str | None = None
    gain_db: float | None = Field(None, ge=-96, le=24)
    pan: float | None = Field(None, ge=-1, le=1)
    mute: bool | None = None
    solo: bool | None = None
    output: Id | None = None
    sidechain_from: Id | None = None

    def touches(self, p):
        return {self.track}

    def check(self, p, ctx):
        _track(p, self.track)
        for ref in (self.output, self.sidechain_from):
            if ref:
                _track(p, ref)
                _require(ref != self.track, "una pista no puede enviarse a sí misma")

    def apply(self, p):
        t = p.track(self.track)
        for k in ("name", "gain_db", "pan", "mute", "solo", "output", "sidechain_from"):
            v = getattr(self, k)
            if v is not None:
                setattr(t, k, v)


class LockTrack(Op):
    op: Literal["lock_track"] = "lock_track"
    track: Id

    def check(self, p, ctx):
        _require(not _track(p, self.track).locked, f"la pista '{self.track}' ya está bloqueada")

    def apply(self, p):
        p.track(self.track).locked = True

    def verify(self, before, after):
        _require(after.track(self.track).locked, "la pista no quedó bloqueada")


class UnlockTrack(Op):
    op: Literal["unlock_track"] = "unlock_track"
    track: Id

    def check(self, p, ctx):
        _require(_track(p, self.track).locked, f"la pista '{self.track}' no está bloqueada")
        _require(ctx.author == "user", "solo el usuario puede desbloquear una pista")

    def apply(self, p):
        p.track(self.track).locked = False


class AddSend(Op):
    op: Literal["add_send"] = "add_send"
    track: Id
    send: Send

    def touches(self, p):
        return {self.track}

    def check(self, p, ctx):
        t = _track(p, self.track)
        dst = _track(p, self.send.target)
        _require(dst.kind in (TrackKind.ret, TrackKind.bus), "un envío va a una pista return o bus")
        _require(all(s.target != self.send.target for s in t.sends), f"ya hay un envío de '{self.track}' a '{self.send.target}'")

    def apply(self, p):
        p.track(self.track).sends.append(self.send.model_copy())


class RemoveSend(Op):
    op: Literal["remove_send"] = "remove_send"
    track: Id
    target: Id

    def touches(self, p):
        return {self.track}

    def check(self, p, ctx):
        _require(any(s.target == self.target for s in _track(p, self.track).sends), "no existe ese envío")

    def apply(self, p):
        t = p.track(self.track)
        t.sends = [s for s in t.sends if s.target != self.target]


# ------------------------------------------------------------------ assets
class RegisterAsset(Op):
    op: Literal["register_asset"] = "register_asset"
    asset: Asset

    def check(self, p, ctx):
        _require(self.asset.sha256 not in p.assets, "el asset ya está registrado")
        _require(ctx.asset_exists(self.asset.sha256), "el archivo del asset no está en el almacén")

    def apply(self, p):
        p.assets[self.asset.sha256] = self.asset


# ------------------------------------------------------------------ clips
class AddClip(Op):
    op: Literal["add_clip"] = "add_clip"
    track: Id
    clip: Annotated[Union[AudioClip, MidiClip], Field(discriminator="type")]

    def touches(self, p):
        return {self.track}

    def check(self, p, ctx):
        t = _track(p, self.track)
        want = TrackKind.audio if isinstance(self.clip, AudioClip) else TrackKind.instrument
        _require(t.kind == want, f"un clip {self.clip.type} va en una pista {want.value}, no {t.kind.value}")
        try:
            p.find_clip(self.clip.id)
            raise OperationError(f"ya existe el clip '{self.clip.id}'")
        except KeyError:
            pass
        if isinstance(self.clip, AudioClip):
            _require(self.clip.asset in p.assets, "el clip referencia un asset no registrado")

    def apply(self, p):
        p.track(self.track).clips.append(self.clip.model_copy(deep=True))


class MoveClip(Op):
    op: Literal["move_clip"] = "move_clip"
    clip: Id
    start_tick: int = Field(ge=0)
    to_track: Id | None = None

    def touches(self, p):
        t, _ = _clip(p, self.clip)
        return {t.id} | ({self.to_track} if self.to_track else set())

    def check(self, p, ctx):
        t, c = _clip(p, self.clip)
        if self.to_track:
            dst = _track(p, self.to_track)
            _require(dst.kind == t.kind, f"no se puede mover un clip de {t.kind.value} a {dst.kind.value}")

    def apply(self, p):
        t, c = p.find_clip(self.clip)
        c.start_tick = self.start_tick
        if self.to_track and self.to_track != t.id:
            t.clips = [x for x in t.clips if x.id != self.clip]
            p.track(self.to_track).clips.append(c)

    def verify(self, before, after):
        t, c = after.find_clip(self.clip)
        _require(c.start_tick == self.start_tick, "el clip no quedó en la posición pedida")
        _require(self.to_track is None or t.id == self.to_track, "el clip no quedó en la pista destino")


class DeleteClip(Op):
    op: Literal["delete_clip"] = "delete_clip"
    clip: Id

    def touches(self, p):
        return {_clip(p, self.clip)[0].id}

    def check(self, p, ctx):
        _clip(p, self.clip)

    def apply(self, p):
        t, _ = p.find_clip(self.clip)
        t.clips = [c for c in t.clips if c.id != self.clip]


class TransposeMidi(Op):
    op: Literal["transpose_midi"] = "transpose_midi"
    clip: Id
    semitones: int = Field(ge=-48, le=48)

    def touches(self, p):
        return {_clip(p, self.clip)[0].id}

    def check(self, p, ctx):
        _, c = _clip(p, self.clip)
        _require(isinstance(c, MidiClip), "transpose_midi solo aplica a clips MIDI")
        bad = [n.pitch for n in c.notes if not 0 <= n.pitch + self.semitones <= 127]
        _require(not bad, f"{len(bad)} notas saldrían del rango MIDI 0-127")

    def apply(self, p):
        _, c = p.find_clip(self.clip)
        for n in c.notes:
            n.pitch += self.semitones

    def verify(self, before, after):
        old = [n.pitch for n in before.find_clip(self.clip)[1].notes]
        new = [n.pitch for n in after.find_clip(self.clip)[1].notes]
        _require(new == [x + self.semitones for x in old], "la transposición no es exacta")


class SetMidiNotes(Op):
    op: Literal["set_midi_notes"] = "set_midi_notes"
    clip: Id
    notes: list[MidiNote]
    length_ticks: int | None = Field(None, gt=0)

    def touches(self, p):
        return {_clip(p, self.clip)[0].id}

    def check(self, p, ctx):
        _, c = _clip(p, self.clip)
        _require(isinstance(c, MidiClip), "set_midi_notes solo aplica a clips MIDI")

    def apply(self, p):
        _, c = p.find_clip(self.clip)
        if self.length_ticks:
            c.length_ticks = self.length_ticks
        c.notes = [n.model_copy() for n in self.notes]
        MidiClip.model_validate(c.model_dump())


# ------------------------------------------------------------------ plugins
class AddPlugin(Op):
    op: Literal["add_plugin"] = "add_plugin"
    track: Id
    plugin: PluginSlot
    index: int | None = None

    def touches(self, p):
        return {self.track}

    def check(self, p, ctx):
        t = _track(p, self.track)
        _require(all(s.id != self.plugin.id for s in t.plugins), f"ya existe el plugin '{self.plugin.id}' en '{self.track}'")

    def apply(self, p):
        t = p.track(self.track)
        t.plugins.insert(len(t.plugins) if self.index is None else self.index, self.plugin.model_copy(deep=True))


class ReplacePlugin(Op):
    op: Literal["replace_plugin"] = "replace_plugin"
    track: Id
    plugin_id: Id
    plugin: PluginSlot

    def touches(self, p):
        return {self.track}

    def check(self, p, ctx):
        _plugin(_track(p, self.track), self.plugin_id)

    def apply(self, p):
        t = p.track(self.track)
        t.plugins[_plugin(t, self.plugin_id)] = self.plugin.model_copy(deep=True)

    def verify(self, before, after):
        i = _plugin(before.track(self.track), self.plugin_id)
        _require(after.track(self.track).plugins[i] == self.plugin, "el plugin no quedó en la misma posición")


class RemovePlugin(Op):
    op: Literal["remove_plugin"] = "remove_plugin"
    track: Id
    plugin_id: Id

    def touches(self, p):
        return {self.track}

    def check(self, p, ctx):
        _plugin(_track(p, self.track), self.plugin_id)
        prefix = f"{self.plugin_id}."
        used = [a.id for a in p.automation if a.track == self.track and a.target.startswith(prefix)]
        _require(not used, f"el plugin tiene automatizaciones: {', '.join(used)}")

    def apply(self, p):
        t = p.track(self.track)
        t.plugins = [s for s in t.plugins if s.id != self.plugin_id]


class SetPluginParam(Op):
    op: Literal["set_plugin_param"] = "set_plugin_param"
    track: Id
    plugin_id: Id
    param: str
    value: float

    def touches(self, p):
        return {self.track}

    def check(self, p, ctx):
        _plugin(_track(p, self.track), self.plugin_id)

    def apply(self, p):
        t = p.track(self.track)
        t.plugins[_plugin(t, self.plugin_id)].params[self.param] = self.value


# ------------------------------------------------------------------ automatización y tiempo
class SetAutomation(Op):
    op: Literal["set_automation"] = "set_automation"
    lane: AutomationLane

    def touches(self, p):
        return {self.lane.track}

    def check(self, p, ctx):
        t = _track(p, self.lane.track)
        tgt = self.lane.target
        if tgt not in ("gain_db", "pan"):
            _require("." in tgt, "target debe ser 'gain_db', 'pan' o '<plugin>.<parámetro>'")
            _plugin(t, tgt.split(".", 1)[0])
        other = [a for a in p.automation if a.id == self.lane.id]
        _require(not other or other[0].track == self.lane.track, "ese id de automatización pertenece a otra pista")

    def apply(self, p):
        p.automation = [a for a in p.automation if a.id != self.lane.id] + [self.lane.model_copy(deep=True)]


class DeleteAutomation(Op):
    op: Literal["delete_automation"] = "delete_automation"
    lane: Id

    def touches(self, p):
        return {a.track for a in p.automation if a.id == self.lane}

    def check(self, p, ctx):
        _require(any(a.id == self.lane for a in p.automation), f"no existe la automatización '{self.lane}'")

    def apply(self, p):
        p.automation = [a for a in p.automation if a.id != self.lane]


class SetTempo(Op):
    op: Literal["set_tempo"] = "set_tempo"
    points: list[TempoPointM] = Field(min_length=1)

    def check(self, p, ctx):
        _require(self.points[0].tick == 0, "el primer punto de tempo debe estar en el tick 0")
        locked = [t.id for t in p.tracks if t.locked and t.clips]
        _require(not locked, f"cambiar el tempo movería pistas bloqueadas: {', '.join(locked)}")

    def apply(self, p):
        p.tempo = [x.model_copy() for x in self.points]


class InsertTime(Op):
    """Inserta compases vacíos en ``at_tick``: todo lo que empieza ahí o después se desplaza.

    Es "insertar tiempo" de un DAW: clips, automatización, marcadores, tempo y compás se mueven
    en bloque y su sonido relativo no cambia. Por eso es la única operación que desplaza pistas
    bloqueadas: las mueve rígidamente sin alterar su contenido (``verify`` lo comprueba). Se
    rechaza si cortaría un clip o si no cae en un límite de compás.
    """
    op: Literal["insert_time"] = "insert_time"
    at_tick: int = Field(ge=0)
    ticks: int = Field(gt=0)

    def _bar(self, p: Project) -> tuple[int, int]:
        """(índice de compás en ``at_tick``, ticks por compás del tramo de compás vigente)."""
        mm = p.meter_map()
        bar, tick = 0, 0
        while tick < self.at_tick:
            pt = [m for m in mm.points if m.bar <= bar][-1]
            tick += mm.ticks_per_bar(pt)
            bar += 1
        _require(tick == self.at_tick, f"insert_time: el tick {self.at_tick} no es inicio de compás")
        return bar, mm.ticks_per_bar([m for m in mm.points if m.bar <= bar][-1])

    def check(self, p, ctx):
        _, tpb = self._bar(p)
        _require(self.ticks % tpb == 0, f"insert_time: {self.ticks} ticks no son compases enteros ({tpb} por compás)")
        tm = p.tempo_map()
        for t in p.tracks:
            for c in t.clips:
                end = (tm.sample_to_tick(tm.tick_to_sample(c.start_tick) + p.audio_length(c))
                       if isinstance(c, AudioClip) else c.start_tick + c.length_ticks)
                _require(not c.start_tick < self.at_tick < end, f"insert_time cortaría el clip '{c.id}'")

    def apply(self, p):
        at, n = self.at_tick, self.ticks
        bar, tpb = self._bar(p)
        for t in p.tracks:
            for c in t.clips:
                if c.start_tick >= at:
                    c.start_tick += n
        for lane in p.automation:
            pts = lane.points
            before = [x for x in pts if x.tick < at]
            after = [x.model_copy(update={"tick": x.tick + n}) for x in pts if x.tick >= at]
            if before and after and pts[len(before)].tick > at:
                a, b = before[-1], pts[len(before)]       # valor en ``at`` según la curva del tramo
                v = a.value if a.curve == "hold" else a.value + (b.value - a.value) * (at - a.tick) / (b.tick - a.tick)
                after = [AutomationPoint(tick=at, value=v, curve="hold"),
                         AutomationPoint(tick=at + n, value=v, curve=a.curve)] + after
            elif before and after:
                after = [AutomationPoint(tick=at, value=after[0].value, curve="hold")] + after
            lane.points = before + after
        for m in p.markers:
            if m.tick >= at:
                m.tick += n
        for tp in p.tempo:
            if tp.tick > at:
                tp.tick += n
        for mp in p.meter:
            if mp.bar > bar:
                mp.bar += n // tpb

    def verify(self, before, after):
        for t0 in before.tracks:
            t1 = after.track(t0.id)
            _require([c.id for c in t0.clips] == [c.id for c in t1.clips], f"insert_time alteró los clips de '{t0.id}'")
            for c0, c1 in zip(t0.clips, t1.clips):
                moved = c0.model_copy(update={"start_tick": c0.start_tick + (self.ticks if c0.start_tick >= self.at_tick else 0)})
                _require(moved == c1, f"insert_time alteró el contenido del clip '{c0.id}'")
            _require(t0.model_dump(exclude={"clips"}) == t1.model_dump(exclude={"clips"}), f"insert_time alteró '{t0.id}'")


class SetMeter(Op):
    op: Literal["set_meter"] = "set_meter"
    points: list[MeterPointM] = Field(min_length=1)

    def check(self, p, ctx):
        _require(self.points[0].bar == 0, "el primer compás debe ser el 0")

    def apply(self, p):
        p.meter = [x.model_copy() for x in self.points]


class SetKey(Op):
    op: Literal["set_key"] = "set_key"
    key: str = Field(pattern=r"^[A-G](#|b)?(m| minor| major)?$")

    def apply(self, p):
        p.key = self.key


class SetMeta(Op):
    op: Literal["set_meta"] = "set_meta"
    key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    value: Any

    def check(self, p, ctx):
        import json
        _require(len(json.dumps(self.value, default=str)) <= 65536, "valor de meta demasiado grande (máx. 64 KB)")

    def apply(self, p):
        p.meta[self.key] = self.value


class SetMasterProfile(Op):
    op: Literal["set_master_profile"] = "set_master_profile"
    profile: str = Field(pattern=r"^[a-z0-9_\-]{1,64}$")

    def check(self, p, ctx):
        from ..mastering.dynamics import load_profile
        try:
            load_profile(self.profile)
        except ValueError as e:
            raise OperationError(str(e)) from None

    def apply(self, p):
        p.master_profile = self.profile


class AddMarker(Op):
    op: Literal["add_marker"] = "add_marker"
    marker: Marker

    def check(self, p, ctx):
        _require(all(m.id != self.marker.id for m in p.markers), f"ya existe el marcador '{self.marker.id}'")

    def apply(self, p):
        p.markers.append(self.marker.model_copy())
        p.markers.sort(key=lambda m: m.tick)


class RemoveMarker(Op):
    op: Literal["remove_marker"] = "remove_marker"
    marker: Id

    def check(self, p, ctx):
        _require(any(m.id == self.marker for m in p.markers), f"no existe el marcador '{self.marker}'")

    def apply(self, p):
        p.markers = [m for m in p.markers if m.id != self.marker]


Operation = Annotated[
    Union[AddTrack, RemoveTrack, SetTrack, LockTrack, UnlockTrack, AddSend, RemoveSend, RegisterAsset, AddClip, MoveClip,
          DeleteClip, TransposeMidi, SetMidiNotes, AddPlugin, ReplacePlugin, RemovePlugin, SetPluginParam,
          SetAutomation, DeleteAutomation, SetTempo, InsertTime, SetMeter, SetKey, SetMeta, SetMasterProfile, AddMarker, RemoveMarker],
    Field(discriminator="op"),
]
