"""Arreglador: plantilla de género + contexto -> lote de operaciones del contrato.

No escribe el proyecto: devuelve operaciones que ``op apply`` / ``ProjectRepo.commit``
aplican de forma atómica (todo o nada) en una revisión nueva.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..core.config import REPO_ROOT
from . import bass as B
from . import drums as D
from . import harmony as H
from . import melody as M
from .context import ComposeContext
from .theory import BAR, TICKS_PER_BEAT, key_scale

TEMPLATES = REPO_ROOT / "templates"
TRACK_ORDER = ["kick", "drums", "bass", "pad", "arp", "lead", "fx", "vox"]


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


def _clip(cid: str, start: int, length: int, notes) -> dict:
    return {"type": "midi", "id": cid, "start_tick": start, "length_ticks": length,
            "notes": [n.model_dump() for n in notes if n.tick < length]}


def arrange(tpl: dict, ctx: ComposeContext, bars: int, presets: dict[str, str], prefix: str = "") -> Plan:
    """``presets``: rol -> sha del .fxp ya registrado (o ausente para usar el integrado)."""
    key = str(bars)
    if key not in tpl["arrangements"]:
        raise ValueError(f"la plantilla no tiene arreglo de {bars} compases (hay: {', '.join(tpl['arrangements'])})")
    secs = tpl["arrangements"][key]
    scale = key_scale(ctx.key)
    used: set[str] = set()
    clips: list[tuple[str, dict]] = []
    t0, vocal_bar = 0, 0
    for si, s in enumerate(secs):
        nb, seed = s["bars"], ctx.seed * 100 + si
        L = nb * BAR
        name = s["name"]
        hb_from = s.get("harmony_from", vocal_bar if s.get("vocal") else 0)
        hb = ctx.halfbars(hb_from, nb)
        if s.get("kick"):
            k = [n for n in D.section_drums(nb, 0, seed, with_fill=False, crash=False,
                                            no_kick_last_bar=s.get("no_kick_last_bar", False)) if n.pitch == D.KICK]
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
            clips.append(("arp", _clip(f"{prefix}arp-{name}", t0, L, H.arp(hb, seed, lo, hi))))
        if s.get("lead"):
            reg = tpl["instruments"]["lead"].get("register", 74)
            notes = []
            if s["lead"] == "vocal_melody" and ctx.melody:
                spb = 60 / ctx.bpm
                lo_s, hi_s = hb_from * 4 * spb, (hb_from + nb) * 4 * spb
                sel = [(a - lo_s, b - lo_s, p) for a, b, p in ctx.melody if lo_s <= a < hi_s]
                notes = M.from_transcription(sel, ctx.bpm, L, None, reg)
            if not notes:
                notes = M.motif(hb, scale, seed, reg)
            clips.append(("lead", _clip(f"{prefix}lead-{name}", t0, L, notes)))
        fxn = []
        for f in s.get("fx", []):
            if f == "riser_end" and nb >= 4:
                fxn.append(D.MidiNote(tick=(nb - 4) * BAR, duration=4 * BAR, pitch=1, velocity=110))
            if f == "impact_start":
                fxn.append(D.MidiNote(tick=0, duration=BAR, pitch=2, velocity=110))
        if fxn:
            clips.append(("fx", _clip(f"{prefix}fx-{name}", t0, L, fxn)))
        if s.get("vocal") == "phrase" and ctx.vocal is not None:
            v = ctx.vocal
            pick_ticks = min(TICKS_PER_BEAT, t0)                     # anacrusa: hasta 1 tiempo antes
            pick = int(round(pick_ticks / TICKS_PER_BEAT * v.samples_per_beat))
            off = v.region_start + int(round(vocal_bar * 4 * v.samples_per_beat)) - pick
            length = int(round(nb * 4 * v.samples_per_beat)) + pick
            if off >= 0 and off + length <= v.frames:
                clips.append(("vox", {"type": "audio", "id": f"{prefix}vox-{name}", "asset": v.asset,
                                      "start_tick": t0 - pick_ticks, "source_offset": off,
                                      "source_length": length, "fade_in": int(0.01 * v.sr), "fade_out": int(0.25 * v.sr)}))
            vocal_bar += nb
        for tr, _ in clips:
            used.add(tr)
        t0 += L
    tracks = [t for t in TRACK_ORDER if t in used]
    ops: list[dict] = [{"op": "set_tempo", "points": [{"tick": 0, "bpm": ctx.bpm}]}, {"op": "set_key", "key": _short_key(ctx.key)}]
    for t in tracks:
        kind = "audio" if t == "vox" else "instrument"
        ops.append({"op": "add_track", "id": t, "name": t.capitalize(), "kind": kind})
        if kind == "instrument":
            role = t
            if role in ("kick", "drums"):
                inst = {"id": f"{t}-inst", "backend": "builtin", "uri": "drums"}
            elif role == "fx":
                inst = {"id": "fx-inst", "backend": "builtin", "uri": "fx"}
            elif role in presets:
                inst = {"id": f"{t}-inst", "backend": "surge_xt", "uri": "Surge XT.vst3", "preset": presets[role]}
            else:
                inst = {"id": f"{t}-inst", "backend": "builtin", "uri": "synth"}
            ops.append({"op": "add_plugin", "track": t, "plugin": inst})
    mix = tpl.get("mix", {})
    # buses y retornos que alguna pista usa
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
    for tr, c in clips:
        ops.append({"op": "add_clip", "track": tr, "clip": c})
    # automatización declarada por sección: un carril por destino con todos sus puntos
    lanes: dict[tuple[str, str], list[dict]] = {}
    tick = 0
    for s in secs:
        for a in s.get("automation", []):
            if a["track"] not in tracks:
                continue
            if "target" in a:
                target = a["target"]
            else:
                pid = fx_ids.get((a["track"], a["fx"]))
                if pid is None:
                    raise ValueError(f"automatización: la pista '{a['track']}' no tiene el efecto '{a['fx']}'")
                target = f"{pid}.{a['param']}"
            t1 = tick + a.get("from_bar", 0) * BAR
            t2 = tick + a.get("to_bar", s["bars"]) * BAR
            lanes.setdefault((a["track"], target), []).extend([
                {"tick": t1, "value": a["from"], "curve": a.get("curve", "linear")},
                {"tick": t2, "value": a["to"], "curve": a.get("curve", "linear")}])
        tick += s["bars"] * BAR
    for (tr, target), pts in lanes.items():
        pts.sort(key=lambda p: p["tick"])
        lid = f"{prefix}auto-{tr}-{target.replace('.', '-')}"[:64]
        ops.append({"op": "set_automation", "lane": {"id": lid, "track": tr, "target": target, "points": pts}})
    tick = 0
    for s in secs:
        ops.append({"op": "add_marker", "marker": {"id": f"{prefix}m-{s['name']}", "tick": tick, "label": s["name"]}})
        tick += s["bars"] * BAR
    return Plan(ops=ops, sections=secs, tracks=tracks)


def _short_key(key: str) -> str:
    root, mode = key.split()
    return root + ("m" if mode == "minor" else "")
