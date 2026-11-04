"""Exportación del proyecto a DAWproject 1.0 (Bitwig Studio, Studio One, Cubase).

DAWproject (https://github.com/bitwig/dawproject) es un ZIP con ``project.xml`` y
``metadata.xml`` más los archivos que referencian. Contenido del ZIP que se genera:

  project.xml            estructura (pistas, canales, envíos), arreglo, transporte
  metadata.xml           título y comentario
  audio/<nombre>-<sha12>.<ext>   assets de los clips de audio (bytes idénticos, mismo SHA-256)
  audio/bounce/<pista>.wav       instrumentos renderizados (pre-fader, float, opcional)
  plugins/<pista>-<slot>.vstpreset  estado VST3 de Surge XT (desde el .fxp del proyecto)
  presets/<pista>-<slot>.fxp        el preset original de Surge XT
  osky/manifest.json     lo que no tiene equivalente en DAWproject (efectos integrados, sidechain,
                         ganancia de clip, automatización de plugins, mastering) y SHA-256 de cada archivo
  osky/project.json      proyecto canónico de la revisión exportada
  LEEME.md               resumen legible

Tiempo: el arreglo usa la unidad ``beats`` (negras): ``beats = tick / PPQ``. El audio no se estira
en el motor, así que cada clip de audio es un ``Warps`` (contenido en segundos del archivo fuente,
a su propia frecuencia de muestreo) con un punto en cada cambio de tempo: la conversión
segundos → beats es exacta (el tempo es constante por tramos). Ver docs/DAWPROJECT.md.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from bisect import bisect_right
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import soundfile as sf

from ..core.models import AudioClip, AutomationLane, MidiClip, PluginSlot, Project, Track, TrackKind
from .automation import lanes_for

DAWPROJECT_VERSION = "1.0"
APP_NAME = "Osky Remix Engine"
# Surge XT 1.3 (VST3): clase "Audio Module Class" leída de la fábrica del plugin instalado
# (IPluginFactory::getClassInfo). Texto canónico 8-4-4-4-12 como pide DAWproject para VST3.
SURGE_XT_VST3_CID = "ABCDEF01-9182-FAEB-566D-624153675854"
SURGE_XT_VENDOR = "Surge Synth Team"
GAIN_STEP_DB = 1.0          # paso máximo al densificar rampas de ganancia (dB lineal → amplitud lineal)
GAIN_MAX_STEPS = 64

ROLE = {TrackKind.audio: "regular", TrackKind.instrument: "regular", TrackKind.bus: "submix",
        TrackKind.ret: "effect", TrackKind.master: "master"}
CONTENT = {TrackKind.audio: "audio", TrackKind.instrument: "notes", TrackKind.bus: "tracks",
           TrackKind.ret: "audio", TrackKind.master: "audio notes"}


# ------------------------------------------------------------------ tiempo exacto
def _frac(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(str(x))


class Clock:
    """Conversión exacta ticks / segundos / beats con el mapa de tempo del proyecto.

    Antes del segundo 0 (un clip cuyo archivo empieza antes del inicio) se extrapola con el
    primer tempo, igual que lo haría el DAW con el contenido fuera del arreglo.
    """

    def __init__(self, p: Project):
        self.ppq = p.ppq
        self.tm = p.tempo_map()
        self.points = [(pt.tick, self.tm.seconds_at(pt.tick), _frac(pt.bpm)) for pt in p.tempo]
        self._secs = [s for _, s, _ in self.points]

    def beats(self, tick: int) -> Fraction:
        return Fraction(tick, self.ppq)

    def seconds(self, tick: int) -> Fraction:
        return self.tm.seconds_at(tick)

    def beats_at(self, sec: Fraction) -> Fraction:
        if sec < 0:
            return sec * self.points[0][2] / 60
        i = bisect_right(self._secs, sec) - 1
        tick, s0, bpm = self.points[i]
        return Fraction(tick, self.ppq) + (sec - s0) * bpm / 60

    def changes(self) -> list[Fraction]:
        """Segundos donde cambia el tempo (sin el 0)."""
        return self._secs[1:]


def _n(x) -> str:
    """Número para XML: repr de float (ida y vuelta exacta al parsearlo)."""
    return repr(float(x))


def db_to_linear(db: float) -> float:
    return 10 ** (db / 20)


def pan_to_normalized(pan: float) -> float:
    return (pan + 1) / 2


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(s).stem).strip("._")
    return s[:48] or "audio"


# ------------------------------------------------------------------ VST3 preset
def vst3_preset(class_id: str, component_state: bytes) -> bytes:
    """Archivo .vstpreset (formato PresetFile del SDK VST3) con solo el estado del componente.

    Cabecera: 'VST3', versión 1 (int32 LE), class ID (32 caracteres ASCII), offset de la lista
    de chunks (int64 LE); luego los datos y la lista 'List' con una entrada 'Comp'.
    """
    cid = class_id.replace("-", "").upper().encode("ascii")
    if len(cid) != 32:
        raise ValueError(f"class ID VST3 inválido: {class_id}")
    header = 48
    out = b"VST3" + struct.pack("<i", 1) + cid + struct.pack("<q", header + len(component_state))
    out += component_state
    out += b"List" + struct.pack("<i", 1) + b"Comp" + struct.pack("<qq", header, len(component_state))
    return out


def vst3_preset_component(data: bytes) -> tuple[str, bytes]:
    """Inverso de ``vst3_preset``: (class ID de 32 caracteres, estado del componente)."""
    if data[:4] != b"VST3":
        raise ValueError("no es un .vstpreset")
    cid = data[8:40].decode("ascii")
    (list_off,) = struct.unpack("<q", data[40:48])
    if data[list_off:list_off + 4] != b"List":
        raise ValueError("lista de chunks ausente")
    (count,) = struct.unpack("<i", data[list_off + 4:list_off + 8])
    for i in range(count):
        e = list_off + 8 + i * 20
        if data[e:e + 4] == b"Comp":
            off, size = struct.unpack("<qq", data[e + 4:e + 20])
            return cid, data[off:off + size]
    raise ValueError("chunk 'Comp' ausente")


# ------------------------------------------------------------------ plan de exportación
@dataclass
class _Plan:
    files: dict[str, Path] = field(default_factory=dict)          # ruta en el zip -> archivo local
    audio_path: dict[str, str] = field(default_factory=dict)      # sha -> ruta en el zip
    audio_info: dict[str, tuple[int, int, int]] = field(default_factory=dict)  # sha -> (sr, canales, frames)
    bounce: dict[str, tuple[str, int]] = field(default_factory=dict)          # pista -> (ruta, frames)
    states: dict[tuple[str, str], str] = field(default_factory=dict)          # (pista, slot) -> .vstpreset
    manifest_tracks: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _effects(t: Track) -> list[PluginSlot]:
    return t.plugins[1:] if t.kind == TrackKind.instrument else list(t.plugins)


def _bounce(p: Project, assets, cache, tracks: list[str], tmp: Path) -> dict[str, tuple[Path, int]]:
    """Render pre-fader (instrumento + efectos) de cada pista, sin mute ni solo.

    Pre-fader porque la pista bounce del DAW lleva el mismo volumen, paneo, envíos y
    automatización que la original; un stem post-fader los aplicaría dos veces.
    """
    from .renderer import Renderer
    q = p.model_copy(deep=True)
    for t in q.tracks:
        t.mute = False
        t.solo = False
    r = Renderer(q, assets, cache)
    out = {}
    for tid in tracks:
        r.output(tid)
        y = r.pre[tid]
        f = tmp / f"{tid}.wav"
        sf.write(f, y, q.sample_rate, subtype="FLOAT")
        out[tid] = (f, len(y))
    return out


def _plan(p: Project, assets, cache, bounce_instruments: bool, tmp: Path) -> _Plan:
    from .plugin_host import fxp_chunk
    plan = _Plan()
    for t in p.tracks:
        for c in t.clips:
            if isinstance(c, AudioClip) and c.asset not in plan.audio_path:
                src = assets.find(c.asset)
                if src is None:
                    raise FileNotFoundError(f"asset {c.asset[:12]} no está en el almacén")
                a = p.assets[c.asset]
                rel = f"audio/{_slug(a.name)}-{c.asset[:12]}{a.ext}"
                plan.audio_path[c.asset] = rel
                plan.files[rel] = src
                if a.sample_rate and a.channels and a.frames is not None:
                    plan.audio_info[c.asset] = (a.sample_rate, a.channels, a.frames)
                else:
                    info = sf.info(str(src))
                    plan.audio_info[c.asset] = (info.samplerate, info.channels, info.frames)
    lanes_all = {t.id: lanes_for(p, t.id) for t in p.tracks}
    to_bounce = []
    for t in p.tracks:
        m: dict = {"kind": t.kind.value, "dawproject_track": f"trk-{t.id}", "output": t.output,
                   "mute": t.mute, "solo": t.solo}
        fx = _effects(t)
        if fx:
            m["effects_not_mapped"] = [{"id": s.id, "backend": s.backend, "uri": s.uri, "params": s.params,
                                        "bypass": s.bypass} for s in fx]
        if t.sidechain_from:
            m["sidechain_from_not_mapped"] = t.sidechain_from
        gains = {c.id: c.gain_db for c in t.clips if isinstance(c, AudioClip) and c.gain_db != 0}
        if gains:
            m["clip_gain_db_not_mapped"] = gains
        other = sorted(k for k in lanes_all[t.id] if k not in ("gain_db", "pan"))
        if other:
            m["automation_not_mapped"] = other
        if t.kind == TrackKind.instrument and t.plugins:
            inst = t.plugins[0]
            m["instrument"] = {"id": inst.id, "backend": inst.backend, "uri": inst.uri, "params": inst.params,
                               "bypass": inst.bypass, "preset_sha256": inst.preset}
            if inst.backend == "surge_xt":
                m["instrument"]["dawproject"] = "Vst3Plugin Surge XT"
                if inst.preset:
                    src = assets.find(inst.preset)
                    if src is None:
                        raise FileNotFoundError(f"preset {inst.preset[:12]} no está en el almacén")
                    fx_rel = f"presets/{t.id}-{inst.id}.fxp"
                    st_rel = f"plugins/{t.id}-{inst.id}.vstpreset"
                    plan.files[fx_rel] = src
                    vp = tmp / f"{t.id}-{inst.id}.vstpreset"
                    vp.write_bytes(vst3_preset(SURGE_XT_VST3_CID, fxp_chunk(src)))
                    plan.files[st_rel] = vp
                    plan.states[(t.id, inst.id)] = st_rel
                    m["instrument"]["state"] = st_rel
                    m["instrument"]["fxp"] = fx_rel
                if inst.params:
                    plan.warnings.append(f"{t.id}: parámetros fijos de Surge XT no incluidos en el estado "
                                         f"({', '.join(sorted(inst.params))}); ver manifiesto")
            else:
                m["instrument"]["dawproject"] = "no representable (instrumento del motor)"
            has_notes = any(isinstance(c, MidiClip) and c.notes for c in t.clips)
            if bounce_instruments and has_notes and not inst.bypass:
                to_bounce.append(t.id)
            elif inst.backend != "surge_xt" and has_notes:
                plan.warnings.append(f"{t.id}: instrumento integrado '{inst.uri}' sin equivalente; "
                                     "sin bounce la pista no sonará en el DAW")
        plan.manifest_tracks[t.id] = m
    if to_bounce:
        rendered = _bounce(p, assets, cache, to_bounce, tmp)
        for tid, (f, frames) in rendered.items():
            rel = f"audio/bounce/{tid}.wav"
            plan.files[rel] = f
            plan.bounce[tid] = (rel, frames)
            plan.manifest_tracks[tid]["bounce"] = {"file": rel, "dawproject_track": f"bnc-{tid}",
                                                    "point": "pre-fader (instrumento + efectos)"}
    return plan


# ------------------------------------------------------------------ XML
def _el(parent, tag: str, **attrs) -> ET.Element:
    return ET.SubElement(parent, tag, {k.rstrip("_"): v for k, v in attrs.items() if v is not None})


def _real(parent, tag: str, pid: str, unit: str, value: float, lo=None, hi=None, name=None) -> ET.Element:
    return _el(parent, tag, id=pid, name=name, unit=unit, value=_n(value),
               min=None if lo is None else _n(lo), max=None if hi is None else _n(hi))


def _channel(parent, p: Project, t: Track, key: str, plan: _Plan, devices: bool, mute: bool, comment=None):
    out = t.output if t.kind != TrackKind.master and t.output else None
    ch = _el(parent, "Channel", id=f"ch-{key}", name=t.name, comment=comment, audioChannels="2",
             destination=f"ch-{out}" if out else None, role=ROLE[t.kind], solo="true" if t.solo else "false")
    if devices and t.kind == TrackKind.instrument and t.plugins and t.plugins[0].backend == "surge_xt":
        inst = t.plugins[0]
        devs = _el(ch, "Devices")
        d = _el(devs, "Vst3Plugin", id=f"dev-{key}-{inst.id}", name="Surge XT", deviceID=SURGE_XT_VST3_CID,
                deviceName="Surge XT", deviceRole="instrument", deviceVendor=SURGE_XT_VENDOR, loaded="true")
        _el(d, "Enabled", id=f"en-{key}-{inst.id}", name="On/Off", value="false" if inst.bypass else "true")
        st = plan.states.get((t.id, inst.id))
        if st:
            _el(d, "State", path=st)
    _el(ch, "Mute", id=f"mute-{key}", name="Mute", value="true" if mute else "false")
    _real(ch, "Pan", f"pan-{key}", "normalized", pan_to_normalized(t.pan), 0, 1, "Pan")
    if t.sends:
        sends = _el(ch, "Sends")
        for i, s in enumerate(t.sends):
            se = _el(sends, "Send", id=f"send-{key}-{i}", name=s.target, destination=f"ch-{s.target}",
                     type="pre" if s.pre_fader else "post")
            _real(se, "Volume", f"sendvol-{key}-{i}", "linear", db_to_linear(s.level_db), 0,
                  max(2.0, db_to_linear(s.level_db)), "Volume")
    vol = db_to_linear(t.gain_db)
    _real(ch, "Volume", f"vol-{key}", "linear", vol, 0, max(2.0, vol), "Volume")
    return ch


def _track_comment(t: Track, plan: _Plan, bounced: bool) -> str | None:
    notes = []
    m = plan.manifest_tracks[t.id]
    if "effects_not_mapped" in m:
        notes.append("efectos del motor sin equivalente: " + ", ".join(e["uri"] for e in m["effects_not_mapped"]))
    if t.sidechain_from:
        notes.append(f"sidechain desde {t.sidechain_from} (no representable)")
    if t.kind == TrackKind.instrument and t.plugins and t.plugins[0].backend != "surge_xt":
        notes.append(f"instrumento integrado '{t.plugins[0].uri}' sin equivalente")
    if bounced:
        notes.append(f"silenciada: suena la pista bounce '{t.name} (bounce)'")
    return "; ".join(notes) or None


def _structure(root, p: Project, plan: _Plan):
    st = _el(root, "Structure")
    kinds = {t.id: t.kind for t in p.tracks}
    children: dict[str | None, list[Track]] = {}
    for t in p.tracks:
        if t.kind == TrackKind.master:
            continue
        parent = t.output if t.output and kinds.get(t.output) == TrackKind.bus else None
        children.setdefault(parent, []).append(t)

    def emit(parent_el, t: Track):
        bounced = t.id in plan.bounce
        tr = _el(parent_el, "Track", id=f"trk-{t.id}", name=t.name, comment=_track_comment(t, plan, bounced),
                 contentType=CONTENT[t.kind], loaded="true")
        _channel(tr, p, t, t.id, plan, devices=True, mute=t.mute or bounced)
        for c in children.get(t.id, []):
            emit(tr, c)
        if bounced:
            b = _el(parent_el, "Track", id=f"bnc-{t.id}", name=f"{t.name} (bounce)",
                    comment=f"render pre-fader de '{t.name}' (instrumento + efectos del motor)",
                    contentType="audio", loaded="true")
            bt = t.model_copy(update={"kind": TrackKind.audio, "name": f"{t.name} (bounce)"})
            _channel(b, p, bt, f"{t.id}-bounce", plan, devices=False, mute=t.mute)

    for t in children.get(None, []):
        emit(st, t)
    master = next(t for t in p.tracks if t.kind == TrackKind.master)
    emit(st, master)


def _audio_clip(parent, p: Project, clock: Clock, c: AudioClip, plan: _Plan, *, path: str | None = None,
                info: tuple[int, int, int] | None = None, name: str | None = None, comment: str | None = None):
    sr, chans, frames = info or plan.audio_info[c.asset]
    S = clock.seconds(c.start_tick)
    off = Fraction(c.source_offset, sr)
    length = Fraction(c.source_length, sr)
    file_dur = Fraction(frames, sr)
    F = S - off                                   # segundo del arreglo donde empezaría el archivo
    b0 = clock.beats_at(F)
    play_start = clock.beats(c.start_tick) - b0
    duration = clock.beats_at(S + length) - clock.beats(c.start_tick)
    clip = _el(parent, "Clip", name=name or c.id, comment=comment, time=_n(clock.beats(c.start_tick)),
               duration=_n(duration), contentTimeUnit="beats", playStart=_n(play_start),
               playStop=_n(play_start + duration), fadeTimeUnit="seconds",
               fadeInTime=_n(Fraction(c.fade_in, sr)), fadeOutTime=_n(Fraction(c.fade_out, sr)))
    w = _el(clip, "Warps", id=f"warps-{c.id}", timeUnit="beats", contentTimeUnit="seconds")
    au = _el(w, "Audio", id=f"audio-{c.id}", timeUnit="seconds", channels=str(chans), sampleRate=str(sr),
             duration=_n(file_dur))
    _el(au, "File", path=path or plan.audio_path[c.asset])
    cuts = sorted({Fraction(0), file_dur} | {s - F for s in clock.changes() if 0 < s - F < file_dur})
    for cs in cuts:
        _el(w, "Warp", time=_n(clock.beats_at(F + cs) - b0), contentTime=_n(cs))
    return clip


def _midi_clip(parent, clock: Clock, c: MidiClip):
    clip = _el(parent, "Clip", name=c.id, time=_n(clock.beats(c.start_tick)), duration=_n(clock.beats(c.length_ticks)),
               playStart="0.0")
    notes = _el(clip, "Notes", id=f"notes-{c.id}")
    for nt in sorted(c.notes, key=lambda n: (n.tick, n.pitch)):
        _el(notes, "Note", time=_n(clock.beats(nt.tick)), duration=_n(clock.beats(nt.duration)), channel="0",
            key=str(nt.pitch), vel=_n(nt.velocity / 127))
    return clip


def gain_points(lane: AutomationLane, clock: Clock) -> list[tuple[Fraction, float, str]]:
    """(beats, amplitud lineal, interpolación). El motor interpola la ganancia en dB; el DAW, en
    la unidad del parámetro (lineal): las rampas se subdividen en pasos de ≤ ``GAIN_STEP_DB``."""
    out = []
    pts = lane.points
    for i, a in enumerate(pts):
        out.append((clock.beats(a.tick), db_to_linear(a.value), a.curve))
        if a.curve == "linear" and i + 1 < len(pts):
            b = pts[i + 1]
            k = min(GAIN_MAX_STEPS, math.ceil(abs(b.value - a.value) / GAIN_STEP_DB))
            if k > 1 and b.tick > a.tick:
                for j in range(1, k):
                    tick = a.tick + Fraction(b.tick - a.tick) * j / k
                    out.append((tick / clock.ppq, db_to_linear(a.value + (b.value - a.value) * j / k), "linear"))
    return out


def _automation(parent, lanes: dict[str, AutomationLane], key: str, clock: Clock):
    if "gain_db" in lanes:
        pe = _el(parent, "Points", id=f"auto-vol-{key}", unit="linear")
        _el(pe, "Target", parameter=f"vol-{key}")
        for b, v, kind in gain_points(lanes["gain_db"], clock):
            _el(pe, "RealPoint", time=_n(b), value=_n(v), interpolation=kind)
    if "pan" in lanes:
        pe = _el(parent, "Points", id=f"auto-pan-{key}", unit="normalized")
        _el(pe, "Target", parameter=f"pan-{key}")
        for pt in lanes["pan"].points:
            _el(pe, "RealPoint", time=_n(clock.beats(pt.tick)), value=_n(pan_to_normalized(pt.value)),
                interpolation=pt.curve)


def _arrangement(root, p: Project, plan: _Plan, clock: Clock):
    arr = _el(root, "Arrangement", id="arrangement")
    lanes = _el(arr, "Lanes", id="lanes", timeUnit="beats")
    for t in p.tracks:
        tl = lanes_for(p, t.id)
        lane = _el(lanes, "Lanes", id=f"lanes-{t.id}", track=f"trk-{t.id}")
        clips = _el(lane, "Clips", id=f"clips-{t.id}")
        for c in sorted(t.clips, key=lambda c: c.start_tick):
            if isinstance(c, AudioClip):
                cm = f"ganancia de clip {c.gain_db:+g} dB (no representable)" if c.gain_db else None
                _audio_clip(clips, p, clock, c, plan, comment=cm)
            else:
                _midi_clip(clips, clock, c)
        _automation(lane, tl, t.id, clock)
        if t.id in plan.bounce:
            rel, frames = plan.bounce[t.id]
            bl = _el(lanes, "Lanes", id=f"lanes-{t.id}-bounce", track=f"bnc-{t.id}")
            bc = _el(bl, "Clips", id=f"clips-{t.id}-bounce")
            fake = AudioClip(id=f"{t.id}-bounce", asset="0" * 64, start_tick=0, source_length=frames)
            _audio_clip(bc, p, clock, fake, plan, path=rel, info=(p.sample_rate, 2, frames), name=f"{t.name} (bounce)")
            _automation(bl, tl, f"{t.id}-bounce", clock)
    if p.markers:
        mk = _el(arr, "Markers", id="markers", timeUnit="beats")
        for m in sorted(p.markers, key=lambda m: m.tick):
            _el(mk, "Marker", name=m.label, comment=f"{m.kind}:{m.id}", time=_n(clock.beats(m.tick)))
    if len(p.tempo) > 1:
        ta = _el(arr, "TempoAutomation", id="tempo-automation", timeUnit="beats", unit="bpm")
        _el(ta, "Target", parameter="tempo")
        for tp in p.tempo:
            _el(ta, "RealPoint", time=_n(clock.beats(tp.tick)), value=_n(tp.bpm), interpolation="hold")
    if len(p.meter) > 1:
        mm = p.meter_map()
        sa = _el(arr, "TimeSignatureAutomation", id="timesig-automation", timeUnit="beats")
        _el(sa, "Target", parameter="timesig")
        for mp in p.meter:
            _el(sa, "TimeSignaturePoint", time=_n(clock.beats(mm.bar_to_tick(mp.bar))),
                numerator=str(mp.numerator), denominator=str(mp.denominator))


def project_xml(p: Project, plan: _Plan) -> bytes:
    clock = Clock(p)
    root = ET.Element("Project", version=DAWPROJECT_VERSION)
    from .renderer import RENDER_VERSION
    _el(root, "Application", name=APP_NAME, version=RENDER_VERSION)
    tr = _el(root, "Transport")
    _real(tr, "Tempo", "tempo", "bpm", p.tempo[0].bpm, 20, 400, "Tempo")
    _el(tr, "TimeSignature", id="timesig", numerator=str(p.meter[0].numerator), denominator=str(p.meter[0].denominator))
    _structure(root, p, plan)
    _arrangement(root, p, plan, clock)
    _el(root, "Scenes")
    return _to_bytes(root)


def metadata_xml(p: Project, revision: str | None) -> bytes:
    root = ET.Element("MetaData")
    ET.SubElement(root, "Title").text = p.name
    ET.SubElement(root, "Comment").text = (f"Exportado por {APP_NAME}"
                                           + (f", revisión {revision}" if revision else "")
                                           + ". Ver osky/manifest.json para lo no representable.")
    return _to_bytes(root)


def _to_bytes(root: ET.Element) -> bytes:
    ET.indent(root)
    return b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + ET.tostring(root, encoding="utf-8") + b"\n"


def _readme(p: Project, plan: _Plan, revision: str | None, bounce: bool) -> str:
    lines = [f"# {p.name} (DAWproject)", "",
             f"Revisión `{revision or '-'}` · {p.sample_rate} Hz · tempo {p.tempo[0].bpm} BPM · exportado por {APP_NAME}.",
             "", "Abrir el archivo .dawproject en Bitwig Studio, Studio One o Cubase.", ""]
    if bounce and plan.bounce:
        lines += ["Las pistas de instrumento tienen al lado una pista `(bounce)` con su render pre-fader",
                  "(instrumento + efectos del motor); la pista MIDI original está silenciada.", ""]
    if plan.states:
        lines += ["Surge XT se carga como VST3 con su estado en `plugins/`; el preset original está en `presets/`.", ""]
    lines += ["No tienen equivalente en DAWproject (detalle en `osky/manifest.json`):", ""]
    for tid, m in plan.manifest_tracks.items():
        bits = []
        if "effects_not_mapped" in m:
            bits.append("efectos " + ", ".join(e["uri"] + (" (bypass)" if e["bypass"] else "")
                                               for e in m["effects_not_mapped"]))
        if "sidechain_from_not_mapped" in m:
            bits.append(f"sidechain desde {m['sidechain_from_not_mapped']}")
        if "clip_gain_db_not_mapped" in m:
            bits.append("ganancia de clip " + ", ".join(f"{k} {v:+g} dB" for k, v in m["clip_gain_db_not_mapped"].items()))
        if "automation_not_mapped" in m:
            bits.append("automatización " + ", ".join(m["automation_not_mapped"]))
        inst = m.get("instrument")
        if inst and inst["backend"] != "surge_xt":
            bits.append(f"instrumento integrado {inst['uri']}")
        if bits:
            lines.append(f"- `{tid}`: " + "; ".join(bits))
    lines += [f"- master: perfil de mastering `{p.master_profile}` (el master del DAW queda sin procesar)", ""]
    lines += [f"Aviso: {w}" for w in plan.warnings]
    return "\n".join(lines) + "\n"


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(f: Path) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def export_dawproject(project: Project, assets, cache, out_path: Path, bounce_instruments: bool = True,
                      revision: str | None = None) -> dict:
    """Escribe ``out_path`` (.dawproject). Devuelve un resumen (pistas, clips, archivos, avisos)."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} ya existe")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = project
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        plan = _plan(p, assets, cache, bounce_instruments, tmp)
        texts = {"project.xml": project_xml(p, plan), "metadata.xml": metadata_xml(p, revision),
                 "osky/project.json": json.dumps(json.loads(p.canonical_json()), indent=2, ensure_ascii=False).encode(),
                 "LEEME.md": _readme(p, plan, revision, bounce_instruments).encode()}
        from .renderer import RENDER_VERSION
        manifest = {"format": f"DAWproject {DAWPROJECT_VERSION}", "engine": APP_NAME, "render_version": RENDER_VERSION,
                    "project": p.name, "revision": revision, "sample_rate": p.sample_rate, "ppq": p.ppq,
                    "time_unit": "beats (negras) = tick / ppq", "bounce_instruments": bounce_instruments,
                    "master_profile_not_mapped": p.master_profile, "tracks": plan.manifest_tracks,
                    "warnings": plan.warnings,
                    "files": {**{k: _sha_bytes(v) for k, v in texts.items()},
                              **{k: _sha_file(f) for k, f in plan.files.items()}}}
        texts["osky/manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode()
        part = out_path.with_name(out_path.name + ".part")
        try:
            with zipfile.ZipFile(part, "w") as z:
                for name, data in texts.items():
                    z.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
                for name, f in sorted(plan.files.items()):
                    z.write(f, name, compress_type=zipfile.ZIP_STORED)
            os.replace(part, out_path)
        finally:
            if part.exists():
                part.unlink()
    return {"path": str(out_path), "tracks": len(p.tracks) + len(plan.bounce), "bounced": sorted(plan.bounce),
            "audio_files": len(plan.audio_path), "files": len(texts) + len(plan.files), "warnings": plan.warnings}
