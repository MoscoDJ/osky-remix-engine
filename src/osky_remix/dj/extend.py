"""Extended mix para DJ: intro y outro de solo ritmo, alineadas a frases.

A partir de un proyecto arreglado produce un lote de operaciones del contrato:

1. ``insert_time`` en el tick 0 por la duración de la intro: TODO lo existente (clips, también
   de pistas bloqueadas o editadas a mano, automatización, marcadores, tempo y compás) se
   desplaza en bloque, sin regenerarse.
2. Intro (``intro_dj``): primera mitad bombo + hats; en la segunda mitad entra el bajo (y la
   batería sube a palmas/hat abierto, con redoble en el último compás). Outro (``outro_dj``),
   en espejo: bombo + batería + bajo, luego sale el bajo y quedan bombo y hats.
3. Sin voz ni cambios de tempo dentro de la intro y el outro: tempo del primer tramo en la intro
   y del último en el outro, para que el DJ pueda igualar el tempo.

El contenido nuevo y el desplazado se registran en ``meta.generated_content`` con su huella
nueva; lo manual (o editado después de generarse) conserva su huella antigua, así que sigue
sin ser reemplazable.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..composition import bass as B
from ..composition import drums as D
from ..composition.arranger import _clip, _fingerprint, load_template
from ..composition.theory import BAR, NOTE, parse_chord
from ..core.models import AudioClip, MidiClip, Project, TrackKind
from ..core.operations import Context
from ..core.transactions import apply_ops, parse_ops

PHRASE_BARS = (16, 32, 64)
INTRO, OUTRO = "intro_dj", "outro_dj"
ROOT_WINDOW_BARS = 8          # compases de bajo de los que se toma la progresión de raíces


@dataclass
class ExtendPlan:
    ops: list[dict]
    intro_ticks: int
    outro_start: int          # tick del outro en el proyecto extendido
    outro_ticks: int
    tracks: dict[str, str]    # rol (kick, drums, bass) -> pista usada
    notes: list[str] = field(default_factory=list)


def _clip_end(p: Project, c) -> int:
    if isinstance(c, AudioClip):
        tm = p.tempo_map()
        return tm.sample_to_tick(tm.tick_to_sample(c.start_tick) + p.audio_length(c))
    return c.start_tick + c.length_ticks


def _abs_notes(p: Project, track_id: str) -> list[tuple[int, int, int]]:
    """(tick absoluto, fin, altura) de las notas de una pista."""
    try:
        t = p.track(track_id)
    except KeyError:
        return []
    return sorted((c.start_tick + n.tick, c.start_tick + n.tick + n.duration, n.pitch)
                  for c in t.clips if isinstance(c, MidiClip) for n in c.notes)


def _roots(notes: list[tuple[int, int, int]], start: int, halfbars: int) -> list[str | None]:
    """Clase de altura más grave por medio compás en [start, start + halfbars/2 compases)."""
    hb = BAR // 2
    out: list[str | None] = []
    for h in range(halfbars):
        a = start + h * hb
        inside = [pt for t, _, pt in notes if a <= t < a + hb]
        out.append(NOTE[min(inside) % 12] if inside else None)
    last = next((x for x in out if x), None)
    filled = []
    for x in out:                                  # huecos: la raíz anterior (o la primera)
        last = x or last
        filled.append(last)
    return filled


def _key_root(key: str | None) -> str | None:
    if not key:
        return None
    root = key.split()[0].rstrip("m") if " " in key else key.rstrip("m")
    try:
        return NOTE[parse_chord(root)[0]]
    except (ValueError, TypeError):
        return None


def _value_at(points, tick: int) -> float:
    """Valor de un carril en un tick (misma semántica que ``audio.automation.curve``)."""
    if tick <= points[0].tick:
        return points[0].value
    for a, b in zip(points, points[1:]):
        if a.tick <= tick < b.tick:
            if a.curve == "hold":
                return a.value
            return a.value + (b.value - a.value) * (tick - a.tick) / (b.tick - a.tick)
    return points[-1].value


def extend(p: Project, intro_bars: int = 32, outro_bars: int = 32, prefix: str = "dj-",
           author: str = "claude") -> ExtendPlan:
    for name, n in (("intro", intro_bars), ("outro", outro_bars)):
        if n not in PHRASE_BARS:
            raise ValueError(f"--{name}-bars debe ser {', '.join(map(str, PHRASE_BARS))} (frases enteras)")
    if p.meta.get("dj_extended"):
        raise ValueError("el proyecto ya es una versión extendida (meta 'dj_extended'); extiende la revisión original")
    bad = [m for m in p.meter if (m.numerator, m.denominator) != (4, 4)]
    if bad:
        raise ValueError("la versión DJ requiere compás 4/4 en todo el tema")
    notes: list[str] = []
    N, M = intro_bars * BAR, outro_bars * BAR

    # --- estructura existente (en el tiempo original)
    arr = copy.deepcopy(p.meta.get("arrangement") or {})
    secs = arr.get("sections") or []
    content_end = max((_clip_end(p, c) for t in p.tracks for c in t.clips), default=0)
    if secs:
        body = sum(s["bars"] for s in secs) * BAR
    else:
        body = -(-content_end // BAR) * BAR
        notes.append("sin meta 'arrangement': el final del tema se toma del último clip (redondeado al compás)")
    if body <= 0:
        raise ValueError("el proyecto no tiene contenido que extender")
    late = [t.tick for t in p.tempo if t.tick >= body]
    if late:
        raise ValueError(f"hay cambios de tempo en o después del final del tema (tick {late[0]}): el outro no tendría tempo estable")
    outro_start = N + body

    # --- pistas para cada rol (nunca se escribe en una bloqueada)
    existing = {t.id: t for t in p.tracks}

    def free(tid: str, uri: str | None = None) -> bool:
        t = existing.get(tid)
        return (t is not None and not t.locked and t.kind == TrackKind.instrument
                and (uri is None or (bool(t.plugins) and t.plugins[0].backend == "builtin" and t.plugins[0].uri == uri)))

    ops: list[dict] = [{"op": "insert_time", "at_tick": 0, "ticks": N}]
    roles: dict[str, str] = {}
    for tid in ("kick", "drums", "bass"):
        if tid in existing and existing[tid].locked:
            notes.append(f"pista bloqueada '{tid}': se desplaza sin cambios y no recibe la intro/outro")
    if free("kick"):
        roles["kick"] = "kick"
    elif free("drums", "drums"):
        roles["kick"] = "drums"
    if free("drums", "drums"):
        roles["drums"] = "drums"
    elif roles.get("kick") and free(roles["kick"], "drums"):
        roles["drums"] = roles["kick"]
    if "kick" not in roles or "drums" not in roles:
        tid = f"{prefix}drums"
        ops += [{"op": "add_track", "id": tid, "name": "DJ Drums", "kind": "instrument"},
                {"op": "add_plugin", "track": tid, "plugin": {"id": f"{tid}-inst", "backend": "builtin", "uri": "drums"}}]
        roles.setdefault("kick", tid)
        roles.setdefault("drums", tid)
        notes.append(f"pista nueva '{tid}' (batería integrada) para la intro/outro")

    # --- raíces del bajo: del bajo existente (principio para la intro, final para el outro)
    bass_notes = _abs_notes(p, "bass")
    n_root = ROOT_WINDOW_BARS * 2
    if bass_notes:
        first = bass_notes[0][0] // BAR * BAR
        last = -(-max(e for _, e, _ in bass_notes) // BAR) * BAR
        intro_roots = _roots(bass_notes, first, n_root)
        outro_roots = _roots(bass_notes, max(first, last - ROOT_WINDOW_BARS * BAR), n_root)
    else:
        r = _key_root(p.key)
        intro_roots = outro_roots = [r] * n_root if r else []
        notes.append(f"sin bajo existente: raíz de la tonalidad ({r})" if r else
                     "sin bajo existente ni tonalidad: la intro/outro no lleva bajo")
    compose = p.meta.get("compose") or {}
    lo, hi, style = 33, 45, "rolling"
    try:
        tpl = load_template(compose["template"]) if compose.get("template") else None
    except ValueError:
        tpl = None
    if tpl:
        lo, hi = tpl["instruments"].get("bass", {}).get("range", [lo, hi])
        tsecs = tpl["arrangements"].get(str(compose.get("bars")), [])
        top = max((s for s in tsecs if s.get("bass")), key=lambda s: s.get("energy", 2), default=None)
        style = top["bass"] if top else style
    if intro_roots and roles.get("bass") is None:
        if free("bass"):
            roles["bass"] = "bass"
        else:
            tid = f"{prefix}bass"
            ops += [{"op": "add_track", "id": tid, "name": "DJ Bass", "kind": "instrument"},
                    {"op": "add_plugin", "track": tid, "plugin": {"id": f"{tid}-inst", "backend": "builtin", "uri": "synth"}}]
            roles["bass"] = tid
            notes.append(f"pista nueva '{tid}' (sintetizador integrado) para el bajo de la intro/outro")

    def bassline(roots: list[str], bars: int, seed: int):
        hb = (roots * (bars * 2 // len(roots) + 1))[: bars * 2]
        fn = {"offbeat": lambda: B.offbeat(hb, lo, hi), "sustained": lambda: B.sustained(hb, lo, hi)}
        return fn.get(style, lambda: B.rolling(hb, seed, lo, hi))()

    def kicks(bars, seed):
        return [n for n in D.section_drums(bars, 0, seed, with_fill=False, crash=False) if n.pitch == D.KICK]

    def kit(bars, intensity, seed, fill=False):
        return [n for n in D.section_drums(bars, intensity, seed, with_fill=fill, crash=False) if n.pitch != D.KICK]

    # --- contenido nuevo (tiempo extendido)
    seed = int(compose.get("seed", 1)) * 100 + 71
    hi_, ho = intro_bars // 2, outro_bars // 2
    new: list[tuple[str, dict]] = [
        (roles["kick"], _clip(f"{prefix}kick-intro", 0, N, kicks(intro_bars, seed))),
        (roles["drums"], _clip(f"{prefix}drums-intro-1", 0, hi_ * BAR, kit(hi_, 1, seed + 1))),
        (roles["drums"], _clip(f"{prefix}drums-intro-2", hi_ * BAR, hi_ * BAR, kit(hi_, 2, seed + 2, fill=True))),
        (roles["kick"], _clip(f"{prefix}kick-outro", outro_start, M, kicks(outro_bars, seed + 4))),
        (roles["drums"], _clip(f"{prefix}drums-outro-1", outro_start, ho * BAR, kit(ho, 2, seed + 5))),
        (roles["drums"], _clip(f"{prefix}drums-outro-2", outro_start + ho * BAR, ho * BAR, kit(ho, 1, seed + 6))),
    ]
    if roles.get("bass"):
        new += [(roles["bass"], _clip(f"{prefix}bass-intro", hi_ * BAR, hi_ * BAR, bassline(intro_roots, hi_, seed + 3))),
                (roles["bass"], _clip(f"{prefix}bass-outro", outro_start, ho * BAR, bassline(outro_roots, ho, seed + 7)))]
    taken = {c.id for t in p.tracks for c in t.clips}
    clash = [c["id"] for _, c in new if c["id"] in taken]
    if clash:
        raise ValueError(f"ya existen clips con los ids de la versión DJ: {', '.join(clash)} (usa otro prefijo)")
    ops += [{"op": "add_clip", "track": tr, "clip": c} for tr, c in new]

    # --- automatización generada de las pistas de la intro: valor de la sección de más energía
    registry = copy.deepcopy(p.meta.get("generated_content") or {"clips": {}, "automation": {}, "markers": {}})
    for k in ("clips", "automation", "markers", "clip_tracks", "automation_tracks"):
        registry.setdefault(k, {})
    if "generated_content" not in p.meta:
        notes.append("proyecto sin registro de procedencia: todo lo existente se trata como manual (solo se desplaza)")

    def owned(kind, obj) -> bool:
        return registry[kind].get(obj.id) == _fingerprint(obj)

    if secs:
        top_i = max(range(len(secs)), key=lambda i: (secs[i].get("energy", 2), -i))
        top_tick = sum(s["bars"] for s in secs[:top_i]) * BAR
        used = set(roles.values())
        for lane in p.automation:
            if lane.track not in used or existing[lane.track].locked or not owned("automation", lane):
                continue
            ref = _value_at(lane.points, top_tick)
            pts = [{**pt.model_dump(), "tick": pt.tick + N} for pt in lane.points]
            if pts[0]["value"] != ref:
                pts.insert(0, {"tick": 0, "value": ref, "curve": "hold"})
            if pts[-1]["value"] != ref:
                pts += [{"tick": outro_start, "value": pts[-1]["value"], "curve": "linear"},
                        {"tick": outro_start, "value": ref, "curve": "hold"}]
            ops.append({"op": "set_automation", "lane": {"id": lane.id, "track": lane.track,
                                                          "target": lane.target, "points": pts}})

    # --- marcadores y metadatos
    marker_ids = {m.id for m in p.markers}
    mk = [{"id": f"{prefix}m-intro", "tick": 0, "label": INTRO}, {"id": f"{prefix}m-outro", "tick": outro_start, "label": OUTRO}]
    if any(m["id"] in marker_ids for m in mk):
        raise ValueError("ya existen marcadores de la versión DJ (usa otro prefijo)")
    ops += [{"op": "add_marker", "marker": m} for m in mk]
    body_secs = secs or [{"name": "tema", "bars": body // BAR, "energy": 2, "vocal": True}]
    arr["sections"] = ([{"name": INTRO, "bars": intro_bars, "energy": 1, "vocal": False, "dj": True}] + body_secs
                       + [{"name": OUTRO, "bars": outro_bars, "energy": 1, "vocal": False, "dj": True}])
    ops.append({"op": "set_meta", "key": "arrangement", "value": arr})
    ops.append({"op": "set_meta", "key": "dj_extended", "value": {
        "intro_bars": intro_bars, "outro_bars": outro_bars, "intro_ticks": N, "outro_start_tick": outro_start,
        "outro_ticks": M, "bpm_intro": p.tempo[0].bpm, "bpm_outro": p.tempo[-1].bpm, "tracks": roles,
        "bass_style": style if roles.get("bass") else None}})

    # --- procedencia: huellas tras el desplazamiento
    after = apply_ops(p, parse_ops(ops), Context(author=author))
    for t in p.tracks:
        for c in t.clips:
            if owned("clips", c):
                registry["clips"][c.id] = _fingerprint(after.find_clip(c.id)[1])
    lanes_after = {a.id: a for a in after.automation}
    for a in p.automation:
        if owned("automation", a):
            registry["automation"][a.id] = _fingerprint(lanes_after[a.id])
    markers_after = {m.id: m for m in after.markers}
    for m in p.markers:
        if owned("markers", m):
            registry["markers"][m.id] = _fingerprint(markers_after[m.id])
    for tr, c in new:
        registry["clips"][c["id"]] = _fingerprint(after.find_clip(c["id"])[1])
        registry["clip_tracks"][c["id"]] = tr
    for m in mk:
        registry["markers"][m["id"]] = _fingerprint(markers_after[m["id"]])
    ops.append({"op": "set_meta", "key": "generated_content", "value": registry})

    for t in after.tracks:
        if t.kind == TrackKind.audio:
            for c in t.clips:
                if _clip_end(after, c) > outro_start:
                    notes.append(f"aviso: el clip de audio '{c.id}' ({t.id}) entra en el outro; revísalo a mano")
    return ExtendPlan(ops=ops, intro_ticks=N, outro_start=outro_start, outro_ticks=M, tracks=roles, notes=notes)
