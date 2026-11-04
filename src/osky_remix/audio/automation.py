"""Evaluación de carriles de automatización a curvas por muestra."""
from __future__ import annotations

import numpy as np

from ..core.models import AutomationLane, Project
from ..core.timeline import TempoMap


def curve(lane: AutomationLane, tm: TempoMap, n: int) -> np.ndarray:
    """Valor por muestra (n). Antes del primer punto: su valor; después del último: el último.

    ``linear`` interpola hacia el punto siguiente; ``hold`` mantiene el valor hasta él.
    """
    pts = [(tm.tick_to_sample(p.tick), p.value, p.curve) for p in lane.points]
    out = np.empty(n, np.float64)
    first = pts[0]
    out[: min(n, first[0])] = first[1]
    for (a, va, kind), (b, vb, _) in zip(pts, pts[1:]):
        a2, b2 = min(a, n), min(b, n)
        if b2 <= a2:
            continue
        if kind == "hold" or b == a:
            out[a2:b2] = va
        else:
            out[a2:b2] = va + (vb - va) * (np.arange(a2, b2) - a) / (b - a)
    last = pts[-1]
    if last[0] < n:
        out[max(0, last[0]):] = last[1]
    return out


def lanes_for(p: Project, track_id: str) -> dict[str, AutomationLane]:
    """target -> carril, para una pista."""
    return {a.target: a for a in p.automation if a.track == track_id}
