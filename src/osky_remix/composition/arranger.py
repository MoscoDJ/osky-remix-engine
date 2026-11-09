"""Arreglador: plantilla de género + contexto -> lote de operaciones del contrato.

No escribe el proyecto: devuelve operaciones que ``op apply`` / ``ProjectRepo.commit``
aplican de forma atómica (todo o nada) en una revisión nueva.

- ``arrange``: proyecto nuevo (crea pistas, plugins, mezcla, clips, automatización, marcadores).
- ``rearrange``: proyecto existente; regenera clips y automatización de las pistas generadas que
  NO estén bloqueadas, con una variante. Las pistas bloqueadas (ediciones manuales) no se tocan.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field

from ..core.config import REPO_ROOT
from ..core.models import Project, AudioClip, MidiClip, AutomationLane, Marker
from . import bass as B
from . import drums as D
from . import harmony as H
from . import melody as M
from .context import ComposeContext
from .theory import BAR, TICKS_PER_BEAT, key_scale

TEMPLATES = REPO_ROOT / "templates"
TRACK_ORDER = ["kick", "drums", "perc", "bass", "sub", "pad", "arp", "lead", "fx", "vox", "chops"]
AUDIO_TRACKS = ("vox", "chops", "perc")
DEFAULT_ENERGY = {"intro": 1, "break": 1, "outro": 1, "voz": 2, "verso": 2, "pre": 2, "puente": 2, "build": 2,
                  "drop": 3, "drop2": 3, "coro2": 3}

# Variantes de arreglo: mismas secciones (la voz no se mueve), distinto contenido.
VARIANTS: dict[str, dict] = {
    "A": {"label": "rodante (base)"},
    "B": {"label": "contratiempo + motivo", "seed_offset": 11, "arp_shape": "octaves", "lead": "motif",
          "lead_register": 79, "bass": {"voz": "offbeat"}, "drums_delta": {"intro": 1}},
    "C": {"label": "minimal + lead alto", "seed_offset": 23, "arp_shape": "pulse", "lead_register": 86,
          "bass": {"intro": "sustained", "voz": "sustained"}, "drums_delta": {"voz": -1}, "pad_sections": ["drop"]},
}


def load_template(name: str) -> dict:
    p = TEMPLATES / name / "template.json"
    if not p.exists():
        avail = sorted(d.name for d in TEMPLATES.iterdir() if (d / "template.json").exists())
        raise ValueError(f"plantilla '{name}' no existe (disponibles: {', '.join(avail) or 'ninguna'})")
    return json.loads(p.read_text())


@dataclass
class Plan:
    ops: list[dict]
    sections: list[dict]
    tracks: list[str]
    skipped: list[str] = field(default_factory=list)


@dataclass
class Parts:
    clips: list[tuple[str, dict]]
    autom: list[dict]              # {"track", "fx"|None, "target"|None, "param", "points"}
    sections: list[dict]
    used: set[str]


def _clip(cid: str, start: int, length: int, notes) -> dict:
    return {"type": "midi", "id": cid, "start_tick": start, "length_ticks": length,
            "notes": [n.model_dump() for n in notes if n.tick < length]}


def sections_for(tpl: dict, bars: int, variant: str = "A") -> tuple[list[dict], dict]:
    key = str(bars)
    if key not in tpl["arrangements"]:
        raise ValueError(f"la plantilla no tiene arreglo de {bars} compases (hay: {', '.join(tpl['arrangements'])})")
    if variant not in VARIANTS:
        raise ValueError(f"variante desconocida '{variant}' (hay: {', '.join(VARIANTS)})")
    v = VARIANTS[variant]
    secs = copy.deepcopy(tpl["arrangements"][key])
    for s in secs:
        s.setdefault("energy", DEFAULT_ENERGY.get(s["name"], 2))
        if s["name"] in v.get("bass", {}) and s.get("bass"):
            s["bass"] = v["bass"][s["name"]]
        d = v.get("drums_delta", {}).get(s["name"])
        if d and s.get("drums") is not None:
            s["drums"] = max(0, min(3, s["drums"] + d))
        if s["name"] in v.get("pad_sections", []):
            s["pad"] = True
        if s.get("lead") and v.get("lead"):
            s["lead"] = v["lead"]
    return secs, v


def _ref_lead(ctx: ComposeContext, role: str, beat: int, nb: int, register: int) -> list:
    """Notas de un canal del MIDI de referencia como lead: cuantizadas, monofónicas, en registro."""
    raw = ctx.ref_notes(role, beat, nb)
    secs = [(b1 * 60 / ctx.bpm, (b1 + d) * 60 / ctx.bpm, p) for b1, d, p, _ in raw]
    return M.from_transcription(secs, ctx.bpm, nb * BAR, None, register)


def generate_parts(tpl: dict, ctx: ComposeContext, bars: int, prefix: str = "", variant: str = "A") -> Parts:
    secs, v = sections_for(tpl, bars, variant)
    scale = key_scale(ctx.key)
    clips: list[tuple[str, dict]] = []
    autom: list[dict] = []
    t0, vocal_bar = 0, 0
    seed0 = ctx.seed * 100 + v.get("seed_offset", 0)
    chop_n = 0
    for si, s in enumerate(secs):
        nb, seed = s["bars"], seed0 + si
        L = nb * BAR
        name = s["name"]
        if s.get("source"):
            reg = ctx.region(s["source"])
        else:
            reg = ctx.region(None)
            reg.beat = ctx.b0 + (vocal_bar * 4 if s.get("vocal") else 0)
        s["_region"] = {"beat": reg.beat, "pickup_beats": round(reg.pickup_beats, 2), "label": reg.label}
        hb = ctx.halfbars(0, nb, b0=reg.beat)
        if s.get("kick"):
            k = [n for n in D.section_drums(nb, 0, seed, with_fill=False, crash=False,
                                            no_kick_last_bar=s.get("no_kick_last_bar", False)) if n.pitch == D.KICK]
            if s.get("kick_bars"):
                k = [n for n in k if n.tick < s["kick_bars"] * BAR]
            clips.append(("kick", _clip(f"{prefix}kick-{name}", t0, L, k)))
        if s.get("drums") is not None:
            d = [n for n in D.section_drums(nb, s["drums"], seed, with_fill=s.get("fill", False)) if n.pitch != D.KICK]
            clips.append(("drums", _clip(f"{prefix}drums-{name}", t0, L, d)))
        if s.get("bass"):
            lo, hi = tpl["instruments"]["bass"].get("range", [33, 45])
            fn = {"rolling": lambda: B.rolling(hb, seed, lo, hi), "offbeat": lambda: B.offbeat(hb, lo, hi),
                  "sustained": lambda: B.sustained(hb, lo, hi)}[s["bass"]]
            clips.append(("bass", _clip(f"{prefix}bass-{name}", t0, L, fn())))
        if s.get("pad"):
            lo, hi = tpl["instruments"]["pad"].get("range", [50, 74])
            clips.append(("pad", _clip(f"{prefix}pad-{name}", t0, L, H.pads(hb, lo, hi))))
        if s.get("arp"):
            lo, hi = tpl["instruments"]["arp"].get("range", [62, 86])
            clips.append(("arp", _clip(f"{prefix}arp-{name}", t0, L, H.arp(hb, seed, lo, hi, v.get("arp_shape")))))
        if s.get("lead"):
            reg_hi = v.get("lead_register", tpl["instruments"]["lead"].get("register", 74))
            notes = []
            mode = s["lead"]
            if mode == "vocal_melody":
                mel = ctx.melody_for(reg.beat, nb)
                notes = M.from_transcription([x for x in mel if x[0] < nb * 4 * 60 / ctx.bpm], ctx.bpm, L, None, reg_hi)
            elif mode.startswith("midi:"):
                notes = _ref_lead(ctx, mode.split(":", 1)[1], reg.beat, nb, reg_hi)
            if not notes:
                notes = M.motif(hb, scale, seed, reg_hi)
            clips.append(("lead", _clip(f"{prefix}lead-{name}", t0, L, notes)))
        fxn = []
        for f in s.get("fx", []):
            if f == "riser_end" and nb >= 4:
                fxn.append(D.MidiNote(tick=(nb - 4) * BAR, duration=4 * BAR, pitch=1, velocity=110))
            if f == "impact_start":
                fxn.append(D.MidiNote(tick=0, duration=BAR, pitch=2, velocity=110))
        if fxn:
            clips.append(("fx", _clip(f"{prefix}fx-{name}", t0, L, fxn)))
        vs = ctx.vocal
        if s.get("vocal") == "phrase" and vs is not None:
            vbars = s.get("vocal_bars", nb)
            pick_ticks = min(int(round(reg.pickup_beats * TICKS_PER_BEAT)), t0)    # anacrusa medida
            pick = int(round(pick_ticks / TICKS_PER_BEAT * vs.samples_per_beat))
            start = int(round(ctx.beats[reg.beat] / ctx.ratio))
            off = start - pick
            length = int(round(vbars * 4 * vs.samples_per_beat)) + pick
            if off >= 0 and off + length <= vs.frames:
                clips.append(("vox", {"type": "audio", "id": f"{prefix}vox-{name}", "asset": vs.asset,
                                      "start_tick": t0 - pick_ticks, "source_offset": off,
                                      "source_length": length, "fade_in": int(0.01 * vs.sr), "fade_out": int(0.25 * vs.sr)}))
            if not s.get("source"):
                vocal_bar += nb
        for ch in s.get("chops", []):
            if vs is None or ctx.reference is None:
                break
            creg = ctx.region({"lyric": ch["lyric"], "occurrence": ch.get("occurrence", 1)})
            from ..analysis.midi_ref import find_lyric
            line = find_lyric(ctx.reference, ch["lyric"], ch.get("occurrence", 1))
            start = int(round(ctx.beats[creg.beat] / ctx.ratio))
            pick = int(round(creg.pickup_beats * vs.samples_per_beat))
            end_s = (line.end + 0.35) / ctx.ratio * vs.sr                       # final de la línea + cola
            length = int(min(end_s - (start - pick), ch.get("max_beats", 8) * vs.samples_per_beat + pick))
            for at in ch.get("at_bars", [0]):
                tick = t0 + at * BAR - int(round(creg.pickup_beats * TICKS_PER_BEAT))
                if tick < 0 or start - pick < 0 or length <= 0:
                    continue
                chop_n += 1
                clips.append(("chops", {"type": "audio", "id": f"{prefix}chop-{name}-{chop_n}", "asset": vs.asset,
                                        "start_tick": tick, "source_offset": start - pick, "source_length": length,
                                        "fade_in": int(0.005 * vs.sr), "fade_out": int(0.15 * vs.sr)}))
        for a in s.get("automation", []):
            autom.append({**a, "t1": t0 + a.get("from_bar", 0) * BAR, "t2": t0 + a.get("to_bar", nb) * BAR})
        t0 += L
    return Parts(clips=clips, autom=autom, sections=secs, used={t for t, _ in clips})


def _lane_ops(autom: list[dict], resolve, prefix: str, allowed: set[str]) -> list[dict]:
    lanes: dict[tuple[str, str], list[dict]] = {}
    for a in autom:
        if a["track"] not in allowed:
            continue
        target = a.get("target") or resolve(a["track"], a["fx"], a["param"])
        if target is None:
            continue
        lanes.setdefault((a["track"], target), []).extend([
            {"tick": a["t1"], "value": a["from"], "curve": a.get("curve", "linear")},
            {"tick": a["t2"], "value": a["to"], "curve": a.get("curve", "linear")}])
    ops = []
    for (tr, target), pts in lanes.items():
        pts.sort(key=lambda p: p["tick"])
        lid = f"{prefix}auto-{tr}-{target.replace('.', '-')}"[:64]
        ops.append({"op": "set_automation", "lane": {"id": lid, "track": tr, "target": target, "points": pts}})
    return ops


def _marker_ops(secs: list[dict], prefix: str) -> list[dict]:
    ops, tick = [], 0
    for s in secs:
        ops.append({"op": "add_marker", "marker": {"id": f"{prefix}m-{s['name']}", "tick": tick, "label": s["name"]}})
        tick += s.get("ticks", s["bars"] * BAR)          # piezas con compases irregulares: ticks exactos
    return ops


def section_meta(secs: list[dict]) -> list[dict]:
    return [{"name": s["name"], "bars": s["bars"], "energy": s.get("energy", 2), "vocal": bool(s.get("vocal")),
             **({"type": s["type"]} if "type" in s else {}),
             **({"region": s["_region"]} if "_region" in s else {})} for s in secs]


def _fingerprint(obj):
    return hashlib.sha256(json.dumps(obj.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _ownership(ops: list[dict], previous: dict | None = None) -> dict:
    """Contenido generado exacto: una edición posterior deja de ser reemplazable."""
    registry = copy.deepcopy(previous or {"clips": {}, "automation": {}, "markers": {}})
    registry.setdefault("clip_tracks", {})
    registry.setdefault("automation_tracks", {})
    for op in ops:
        if op["op"] == "add_clip":
            raw = op["clip"]
            obj = (AudioClip if raw["type"] == "audio" else MidiClip).model_validate(raw)
            registry["clips"][obj.id] = _fingerprint(obj)
            registry["clip_tracks"][obj.id] = op["track"]
        elif op["op"] == "set_automation":
            obj = AutomationLane.model_validate(op["lane"])
            registry["automation"][obj.id] = _fingerprint(obj)
            registry["automation_tracks"][obj.id] = obj.track
        elif op["op"] == "add_marker":
            obj = Marker.model_validate(op["marker"])
            registry["markers"][obj.id] = _fingerprint(obj)
        elif op["op"] == "delete_clip":
            registry["clips"].pop(op["clip"], None)
            registry["clip_tracks"].pop(op["clip"], None)
        elif op["op"] == "delete_automation":
            registry["automation"].pop(op["lane"], None)
            registry["automation_tracks"].pop(op["lane"], None)
        elif op["op"] == "remove_marker":
            registry["markers"].pop(op["marker"], None)
    return registry


def _record_ownership(ops, previous=None):
    ops.append({"op": "set_meta", "key": "generated_content", "value": _ownership(ops, previous)})


def arrange(tpl: dict, ctx: ComposeContext, bars: int, presets: dict[str, str], prefix: str = "",
            variant: str = "A", compose_meta: dict | None = None,
            preset_params: dict[str, dict] | None = None) -> Plan:
    """``presets``: rol -> sha ya registrado: .fxp para los sintes, kit de samples para kick, drums
    y fx (ausente: instrumento integrado). ``preset_params``: parámetros estáticos de Surge por rol."""
    if tpl.get("form") == "song":
        from .song_form import generate_song_parts
        parts = generate_song_parts(tpl, ctx, prefix, variant)
    else:
        parts = generate_parts(tpl, ctx, bars, prefix, variant)
    tracks = [t for t in TRACK_ORDER if t in parts.used]
    tempo = getattr(parts, "tempo", None) or [(0, ctx.bpm)]
    ops: list[dict] = [{"op": "set_tempo", "points": [{"tick": t, "bpm": b} for t, b in tempo]},
                       {"op": "set_key", "key": _short_key(ctx.key)}]
    for t in tracks:
        kind = "audio" if t in AUDIO_TRACKS else "instrument"
        ops.append({"op": "add_track", "id": t, "name": t.capitalize(), "kind": kind})
        if kind == "instrument":
            if t in ("kick", "drums", "fx"):
                inst = {"id": f"{t}-inst", "backend": "builtin", "uri": "fx" if t == "fx" else "drums"}
                if t in presets:
                    inst["preset"] = presets[t]                      # kit de samples (samples.kit)
            elif t in presets:
                inst = {"id": f"{t}-inst", "backend": "surge_xt", "uri": "Surge XT.vst3", "preset": presets[t]}
                if (preset_params or {}).get(t):
                    inst["params"] = preset_params[t]
            else:
                inst = {"id": f"{t}-inst", "backend": "builtin", "uri": "synth"}
            ops.append({"op": "add_plugin", "track": t, "plugin": inst})
    mix = tpl.get("mix", {})
    groups = {**{k: ("bus", v) for k, v in tpl.get("buses", {}).items()},
              **{k: ("return", v) for k, v in tpl.get("returns", {}).items()}}
    needed = []
    for t in tracks:
        m = mix.get(t, {})
        for g in [m.get("output")] + [snd["target"] for snd in m.get("sends", [])]:
            if g in groups and g not in needed:
                needed.append(g)
    for g in needed:
        kind, spec = groups[g]
        ops.append({"op": "add_track", "id": g, "name": g.capitalize(), "kind": kind})
        for i, fx in enumerate(spec.get("fx", [])):
            ops.append({"op": "add_plugin", "track": g, "plugin": {"id": f"{g}-{fx['uri']}-{i}", "backend": "builtin",
                                                                  "uri": fx["uri"], "params": fx.get("params", {})}})
        if spec.get("gain_db") is not None:
            ops.append({"op": "set_track", "track": g, "gain_db": spec["gain_db"]})
    fx_ids: dict[tuple[str, str], str] = {}
    for t in tracks:                                     # mezcla (después: sidechain necesita la fuente)
        m = mix.get(t, {})
        upd = {k: m[k] for k in ("gain_db", "pan") if k in m}
        if m.get("sidechain_from") in tracks:
            upd["sidechain_from"] = m["sidechain_from"]
        if m.get("output") in needed:
            upd["output"] = m["output"]
        if upd:
            ops.append({"op": "set_track", "track": t, **upd})
        for snd in m.get("sends", []):
            if snd["target"] in needed:
                ops.append({"op": "add_send", "track": t, "send": {"target": snd["target"], "level_db": snd.get("level_db", 0.0),
                                                                    "pre_fader": snd.get("pre_fader", False)}})
        for i, fx in enumerate(m.get("fx", [])):
            if fx["uri"] == "sidechain" and "sidechain_from" not in upd:
                continue
            pid = f"{t}-{fx['uri']}-{i}"
            fx_ids.setdefault((t, fx["uri"]), pid)
            ops.append({"op": "add_plugin", "track": t,
                        "plugin": {"id": pid, "backend": "builtin", "uri": fx["uri"], "params": fx.get("params", {})}})
    for i, fx in enumerate(tpl.get("master", {}).get("fx", [])):
        ops.append({"op": "add_plugin", "track": "master", "plugin": {"id": f"master-{fx['uri']}-{i}", "backend": "builtin",
                                                                       "uri": fx["uri"], "params": fx.get("params", {})}})
    if tpl.get("master_profile"):
        ops.append({"op": "set_master_profile", "profile": tpl["master_profile"]})
    for tr, c in parts.clips:
        ops.append({"op": "add_clip", "track": tr, "clip": c})

    def resolve(track, fx, param):
        pid = fx_ids.get((track, fx))
        if pid is None:
            raise ValueError(f"automatización: la pista '{track}' no tiene el efecto '{fx}'")
        return f"{pid}.{param}"
    ops += _lane_ops(parts.autom, resolve, prefix, set(tracks))
    ops += _marker_ops(parts.sections, prefix)
    ops.append({"op": "set_meta", "key": "arrangement", "value": {
        "variant": variant, "label": VARIANTS[variant]["label"], "prefix": prefix, "sections": section_meta(parts.sections)}})
    if compose_meta:
        ops.append({"op": "set_meta", "key": "compose", "value": compose_meta})
    _record_ownership(ops)
    return Plan(ops=ops, sections=parts.sections, tracks=tracks)


def legacy_ownership(p: Project) -> dict:
    """Registro para proyectos anteriores al seguimiento de procedencia.

    Reproduce el criterio previo: todo clip de una pista generada, las lanes ``auto-*``
    y todos los marcadores se consideran contenido generado (reemplazable).
    """
    registry = {"clips": {}, "automation": {}, "markers": {}, "clip_tracks": {}, "automation_tracks": {}}
    for t in p.tracks:
        if t.id in TRACK_ORDER:
            for c in t.clips:
                registry["clips"][c.id] = _fingerprint(c)
                registry["clip_tracks"][c.id] = t.id
    for a in p.automation:
        if "auto-" in a.id:
            registry["automation"][a.id] = _fingerprint(a)
            registry["automation_tracks"][a.id] = a.track
    for m in p.markers:
        registry["markers"][m.id] = _fingerprint(m)
    return registry


def rearrange(p: Project, tpl: dict, ctx: ComposeContext, bars: int, variant: str, prefix: str,
              adopt_legacy: bool = False) -> Plan:
    """Nuevo arreglo sobre un proyecto existente sin tocar pistas bloqueadas.

    Un proyecto sin ``generated_content`` (creado antes del registro de procedencia) se
    conserva entero salvo que ``adopt_legacy`` indique tratar su contenido como generado.
    """
    if p.meta.get("dj_extended"):
        raise ValueError("la revisión es una versión extendida DJ: varía la revisión original y vuelve a extenderla")
    parts = generate_parts(tpl, ctx, bars, prefix, variant)
    existing = {t.id: t for t in p.tracks}
    free = {t for t in TRACK_ORDER if t in existing and not existing[t].locked}
    legacy = "generated_content" not in p.meta
    if legacy and adopt_legacy:
        registry = legacy_ownership(p)
    else:
        registry = p.meta.get("generated_content", {"clips": {}, "automation": {}, "markers": {}})
    def owned(kind, obj):
        return registry.get(kind, {}).get(obj.id) == _fingerprint(obj)
    # Cualquier clip/lane manual o editado conserva la pista completa, aun sin bloqueo.
    manual = {t for t in free if any(not owned("clips", c) for c in existing[t].clips)
              or any(not owned("automation", a) for a in p.automation if a.track == t)}
    clip_ids = {c.id for track in p.tracks for c in track.clips}
    lane_ids = {a.id for a in p.automation}
    manual |= {track for cid, track in registry.get("clip_tracks", {}).items() if cid not in clip_ids}
    manual |= {track for aid, track in registry.get("automation_tracks", {}).items() if aid not in lane_ids}
    free -= manual
    locked = [t for t in TRACK_ORDER if t in existing and existing[t].locked]
    ops: list[dict] = []
    for t in sorted(free):
        ops += [{"op": "delete_clip", "clip": c.id} for c in existing[t].clips]
    ops += [{"op": "delete_automation", "lane": a.id} for a in p.automation if a.track in free and owned("automation", a)]
    ops += [{"op": "remove_marker", "marker": m.id} for m in p.markers if owned("markers", m)]
    skipped = [f"pista bloqueada '{t}': se conserva tal cual" for t in locked]
    if legacy and not adopt_legacy and manual:
        skipped.append("proyecto anterior al registro de procedencia: no se sabe qué contenido es generado, "
                       "se conserva todo (usa --adopt-legacy para regenerarlo como antes)")
    skipped += [f"pista manual o sin procedencia {t}: se conserva" for t in sorted(manual)]
    for tr, c in parts.clips:
        if tr in free:
            ops.append({"op": "add_clip", "track": tr, "clip": c})
        elif tr not in existing:
            skipped.append(f"la variante usa '{tr}' pero el proyecto no tiene esa pista")

    def resolve(track, fx, param):
        slot = next((s for s in existing[track].plugins if s.uri == fx and s.backend == "builtin"), None)
        return None if slot is None else f"{slot.id}.{param}"
    ops += _lane_ops(parts.autom, resolve, prefix, free)
    preserved_markers = {m.id for m in p.markers if not owned("markers", m)}
    ops += [op for op in _marker_ops(parts.sections, prefix) if op["marker"]["id"] not in preserved_markers]
    ops.append({"op": "set_meta", "key": "arrangement", "value": {
        "variant": variant, "label": VARIANTS[variant]["label"], "prefix": prefix, "sections": section_meta(parts.sections)}})
    _record_ownership(ops, registry)
    return Plan(ops=ops, sections=parts.sections, tracks=sorted(free), skipped=skipped)


def _short_key(key: str) -> str:
    root, mode = key.split()
    return root + ("m" if mode == "minor" else "")
