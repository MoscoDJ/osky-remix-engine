"""Acciones del doctor de mezcla: EQ contra el enmascaramiento, ducking del bajo y EQ del master
hacia la referencia. Todas acotadas por iteración y nunca sobre pistas bloqueadas.

- ``unmask_eq``: campana (EQ integrado ``eq``, Q √2 ≈ una octava) que recorta la banda en conflicto
  en el enmascarador; slot ``auto-unmask-<banda>``, ``max_eq_step_db`` por iteración y
  ``max_eq_cut_db`` acumulados.
- ``duck_bass``: sidechain del bajo desde el bombo (efecto ``sidechain`` + ``sidechain_from``); sube
  la profundidad ``duck_step`` por iteración hasta ``max_duck_depth``; agotado, recorta con EQ.
- ``reference_eq``: campana en el master (antes del mastering), slot ``auto-ref-eq-<banda>``,
  ``max_eq_step_db`` por iteración y ``max_reference_eq_db`` acumulados.
"""
from __future__ import annotations

import numpy as np

from ..core.models import Project, Track
from .policies import Policy

EQ_Q = float(np.sqrt(2))


def _slot_id(prefix: str, band: str) -> str:
    return f"{prefix}-{band.replace('.', 'p')}"


def _eq_move(t: Track, slot_id: str, fc: float, delta: float, limit: float) -> tuple[list[dict], float, float] | None:
    """Ops para mover la campana ``slot_id`` de ``t`` en ``delta`` dB, con |ganancia| ≤ ``limit``."""
    slot = next((s for s in t.plugins if s.id == slot_id), None)
    cur = float(slot.params.get("mid_db", 0.0)) if slot else 0.0
    new = round(float(np.clip(cur + delta, -limit, limit)), 2)
    if abs(new - cur) < 0.05:
        return None
    if slot is None:
        ops = [{"op": "add_plugin", "track": t.id, "plugin": {
            "id": slot_id, "backend": "builtin", "uri": "eq",
            "params": {"mid_hz": float(fc), "mid_db": new, "mid_q": EQ_Q}}}]
    else:
        ops = [{"op": "set_plugin_param", "track": t.id, "plugin_id": slot_id, "param": "mid_db", "value": new}]
    return ops, cur, new


def _track(p: Project, tid: str) -> Track | None:
    return next((x for x in p.tracks if x.id == tid), None)


def mix_actions(p: Project, ev, policy: Policy) -> tuple[list, list[str]]:
    """Acciones (``actions.Action``) y hallazgos que requieren al usuario."""
    from .actions import Action
    acts, blocked = [], []
    # 1) enmascaramiento: peor primero, una acción por (enmascarador, banda)
    done: set[tuple[str, str]] = set()
    n = 0
    for f in sorted((x for x in ev.findings if x.code == "masking"), key=lambda x: -x.severity):
        d = f.data
        key = (d["masker"], d["band"])
        if key in done or n >= policy.max_eq_actions:
            continue
        done.add(key)
        t = _track(p, d["masker"])
        if t is None:
            continue
        if t.locked:
            blocked.append(f"{f.detail} — la pista '{t.id}' está bloqueada: decisión del usuario")
            continue
        excess = d["masking_db"] - policy.masking_threshold_db
        if d["target"] == "kick" and d["masker"] == "bass":
            a = _duck(p, t, f, policy, Action)
            if a:
                acts.append(a)
                n += 1
                continue
        mv = _eq_move(t, _slot_id("auto-unmask", d["band"]), d["fc"], -min(policy.max_eq_step_db, excess + 0.5),
                      policy.max_eq_cut_db)
        if mv is None:
            blocked.append(f"{f.detail} — '{t.id}' ya tiene el recorte máximo ({policy.max_eq_cut_db:.0f} dB): "
                           "decisión del usuario")
            continue
        ops, cur, new = mv
        acts.append(Action("unmask_eq", f"{t.id}: EQ {d['fc']:g} Hz {cur:+.1f}→{new:+.1f} dB ({f.detail})", ops))
        n += 1
    # 2) referencia: bandas con mayor diferencia primero, EQ en el master antes del mastering
    tonal = sorted((x for x in ev.findings if x.code == "reference_tonal"), key=lambda x: -abs(x.data["delta_db"]))
    if tonal:
        m = next(t for t in p.tracks if t.kind.value == "master")
        if m.locked:
            blocked += [f"{f.detail} — el master está bloqueado: decisión del usuario" for f in tonal]
        else:
            k = 0
            for f in tonal:
                if k >= policy.max_eq_actions:
                    break
                d = f.data
                step = float(np.clip(d["delta_db"], -policy.max_eq_step_db, policy.max_eq_step_db))
                mv = _eq_move(m, _slot_id("auto-ref-eq", d["band"]), d["fc"], step, policy.max_reference_eq_db)
                if mv is None:
                    blocked.append(f"{f.detail} — el EQ del master ya está en el límite "
                                   f"(±{policy.max_reference_eq_db:.0f} dB): decisión del usuario")
                    continue
                ops, cur, new = mv
                acts.append(Action("reference_eq", f"master: EQ {d['fc']:g} Hz {cur:+.1f}→{new:+.1f} dB ({f.detail})", ops))
                k += 1
    return acts, blocked


def _duck(p: Project, bass: Track, f, policy: Policy, Action):
    """Más sidechain del bajo desde el bombo; None si ya está al máximo (entonces se usa EQ)."""
    slot = next((s for s in bass.plugins if s.uri == "sidechain" and s.backend == "builtin" and not s.bypass), None)
    if bass.sidechain_from not in (None, "kick"):
        return None                       # el sidechain escucha otra fuente: no se cambia
    if slot is None:
        depth = round(min(policy.max_duck_depth, 0.3 + policy.duck_step), 2)
        ops = [{"op": "set_track", "track": bass.id, "sidechain_from": "kick"},
               {"op": "add_plugin", "track": bass.id, "plugin": {
                   "id": "auto-duck", "backend": "builtin", "uri": "sidechain",
                   "params": {"depth": depth, "release_ms": 120.0}}}]
        return Action("duck_bass", f"{bass.id}: sidechain desde 'kick' profundidad {depth:.2f} ({f.detail})", ops)
    cur = float(slot.params.get("depth", 0.6))
    new = round(min(policy.max_duck_depth, cur + policy.duck_step), 2)
    if new - cur < 0.01:
        return None
    ops = ([] if bass.sidechain_from == "kick" else [{"op": "set_track", "track": bass.id, "sidechain_from": "kick"}])
    ops.append({"op": "set_plugin_param", "track": bass.id, "plugin_id": slot.id, "param": "depth", "value": new})
    return Action("duck_bass", f"{bass.id}: sidechain {cur:.2f}→{new:.2f} ({f.detail})", ops)
