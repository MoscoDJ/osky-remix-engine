"""Acciones: traducen hallazgos en operaciones del contrato (nunca sobre pistas bloqueadas)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.models import Project
from .evaluator import Evaluation, sections_of
from .policies import DEFAULT, Policy

ENERGY_LANE = "auto-master-energy"


@dataclass
class Action:
    name: str
    rationale: str
    ops: list[dict]


def _clamp(x: float, m: float) -> float:
    return float(np.clip(x, -m, m))


def energy_action(p: Project, ev: Evaluation, policy: Policy) -> Action | None:
    """Curva de ganancia por sección en el master (antes del mastering): baja las secciones de
    menor energía para que la de mayor energía destaque. Solo recorta (no empuja el limitador)."""
    defs = {f.data["section"]: f.data["deficit"] for f in ev.findings if f.code == "energy_contrast"}
    if not defs:
        return None
    master = next(t for t in p.tracks if t.kind.value == "master")
    if master.locked:
        return None
    secs = sections_of(p)
    current = {}
    lane = next((a for a in p.automation if a.id == ENERGY_LANE), None)
    if lane:
        for s in secs:
            pt = [x for x in lane.points if x.tick <= s["tick"]]
            current[s["name"]] = pt[-1].value if pt else 0.0
    pts, changes = [], []
    for s in secs:
        cur = current.get(s["name"], 0.0)
        new = cur
        if s["name"] in defs:
            step = min(policy.max_gain_step_db, defs[s["name"]] * 1.2 + 0.3)
            new = max(-policy.max_total_section_cut_db, cur - step)
        pts.append({"tick": s["tick"], "value": round(new, 2), "curve": "hold"})
        if abs(new - cur) > 1e-6:
            changes.append(f"{s['name']} {cur:+.1f}→{new:+.1f} dB")
    if not changes:
        return None
    op = {"op": "set_automation", "lane": {"id": ENERGY_LANE, "track": master.id, "target": "gain_db", "points": pts}}
    return Action("energy_contrast", "contraste de energía: " + ", ".join(changes), [op])


def headroom_action(p: Project, ev: Evaluation, policy: Policy) -> Action | None:
    """Si falta contraste y el master limita mucho, pasa al siguiente perfil (menos fuerte)."""
    if not policy.allow_loudness_tradeoff or not any(f.code == "energy_contrast" for f in ev.findings):
        return None
    limiting = ev.metrics.get("limiting_db")
    if limiting is None or limiting < policy.limiting_threshold_db:
        return None
    ladder = list(policy.profile_ladder)
    if p.master_profile not in ladder or ladder.index(p.master_profile) + 1 >= len(ladder):
        return None
    nxt = ladder[ladder.index(p.master_profile) + 1]
    return Action("loudness_tradeoff",
                  f"perfil {p.master_profile}→{nxt}: el limitador reduce ~{limiting:.1f} dB y se come el contraste",
                  [{"op": "set_master_profile", "profile": nxt}])


DRUM_CLIP_ID = "auto-drum-clip"


def drum_clip_action(p: Project, ev: Evaluation, policy: Policy) -> Action | None:
    """Falta contraste y el limitador del master trabaja mucho: recortar primero los picos del bus
    de batería (clipper suave antes de su compresor) en vez de ceder volumen."""
    if not any(f.code == "energy_contrast" for f in ev.findings):
        return None
    limiting = ev.metrics.get("limiting_db")
    if limiting is None or limiting < policy.limiting_threshold_db:
        return None
    bus = next((t for t in p.tracks if t.id == policy.drum_bus), None)
    peak = (ev.metrics.get("bus_input_peak_dbfs") or {}).get(policy.drum_bus)
    if bus is None or bus.locked or peak is None:
        return None
    cur = next((x for x in bus.plugins if x.id == DRUM_CLIP_ID), None)
    shave = float(cur.params.get("shave_db", 0.0)) if cur else 0.0
    new = min(policy.max_drum_clip_db, shave + policy.drum_clip_step_db)
    if new <= shave + 1e-6:
        return None
    slot = {"id": DRUM_CLIP_ID, "backend": "builtin", "uri": "clipper",
            "params": {"ceiling_db": round(peak - new, 2), "knee": 0.5, "shave_db": new}}
    op = ({"op": "replace_plugin", "track": bus.id, "plugin_id": DRUM_CLIP_ID, "plugin": slot} if cur else
          {"op": "add_plugin", "track": bus.id, "plugin": slot, "index": 0})
    return Action("drum_clip", f"{bus.id}: recorte suave de picos {shave:.0f}→{new:.0f} dB (el limitador reduce "
                               f"~{limiting:.1f} dB y se come el contraste)", [op])


def track_gain_action(p: Project, ev: Evaluation, policy: Policy, code: str, track: str) -> Action | None:
    f = next((x for x in ev.findings if x.code == code), None)
    if f is None:
        return None
    t = next((x for x in p.tracks if x.id == track), None)
    if t is None or t.locked:
        return None
    # entrar al rango con 1 LU de margen (no al centro): cambiar lo mínimo necesario
    lo, hi = f.data.get("window", (f.data["target"] - 1, f.data["target"] + 1))
    goal = min(hi - 1.0, max(lo + 1.0, f.data["rel"]))
    delta = _clamp(goal - f.data["rel"], policy.max_gain_step_db)
    new = round(t.gain_db + delta, 2)
    return Action(code, f"{track}: ganancia {t.gain_db:+.1f}→{new:+.1f} dB ({f.detail})",
                  [{"op": "set_track", "track": track, "gain_db": new}])


ENERGY_FAMILY = ("energy_contrast", "drum_clip", "loudness_tradeoff")


def _section_cuts(p: Project) -> dict[str, float]:
    lane = next((x for x in p.automation if x.id == ENERGY_LANE), None)
    out = {}
    if lane:
        for s in sections_of(p):
            pt = [x for x in lane.points if x.tick <= s["tick"]]
            out[s["name"]] = -pt[-1].value if pt else 0.0
    return out


def plan_actions(p: Project, ev: Evaluation, policy: Policy = DEFAULT,
                 exclude: frozenset[str] = frozenset()) -> tuple[list[Action], list[str]]:
    """Acciones propuestas y hallazgos que requieren al usuario (pistas bloqueadas).

    Falta de contraste (familia de energía, una acción por iteración, en este orden):
    1. recortar las secciones con déficit mientras ninguna llegue a ``cut_before_tradeoff_db``;
    2. si ya llegaron y el limitador trabaja mucho: recortar picos del bus de batería;
    3. ceder volumen (perfil siguiente);
    4. seguir recortando secciones hasta ``max_total_section_cut_db``.
    ``exclude``: acciones de la familia ya probadas sin mejora en esta iteración (el planificador
    pasa a la siguiente alternativa).
    """
    acts, blocked = [], []
    a = energy_action(p, ev, policy)
    deficient = [f.data["section"] for f in ev.findings if f.code == "energy_contrast"]
    cuts = _section_cuts(p)
    early = a is not None and min((cuts.get(x, 0.0) for x in deficient), default=0.0) < policy.cut_before_tradeoff_db
    h = headroom_action(p, ev, policy)
    c = drum_clip_action(p, ev, policy)
    order = [a, c, h] if early else [c, h, a]
    pick = next((x for x in order if x is not None and x.name not in exclude), None)
    if pick:
        acts.append(pick)
    for code, track in (("vocal_level", "vox"), ("bass_balance", "bass")):
        f = next((x for x in ev.findings if x.code == code), None)
        if f is None:
            continue
        t = next((x for x in p.tracks if x.id == track), None)
        if t is not None and t.locked:
            blocked.append(f"{f.detail} — la pista '{track}' está bloqueada: decisión del usuario")
            continue
        a = track_gain_action(p, ev, policy, code, track)
        if a:
            acts.append(a)
    from .mix_actions import mix_actions          # enmascaramiento y referencia (EQ / ducking)
    more, why = mix_actions(p, ev, policy)
    return acts + more, blocked + why
