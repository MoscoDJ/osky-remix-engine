"""Grafo DSP del proyecto: orden de procesamiento (topológico) a partir del ruteo."""
from __future__ import annotations

from dataclasses import dataclass

from ..core.models import Project, TrackKind, routing_cycle, routing_edges


@dataclass(frozen=True)
class Graph:
    order: list[str]                 # cada pista aparece después de todo lo que necesita
    deps: dict[str, set[str]]
    master: str


def build_graph(p: Project) -> Graph:
    cyc = routing_cycle(p.tracks)
    if cyc:
        raise ValueError("ruteo circular: " + " -> ".join(cyc))
    deps = routing_edges(p.tracks)
    master = next(t.id for t in p.tracks if t.kind == TrackKind.master)
    order: list[str] = []
    seen: set[str] = set()

    def visit(n: str) -> None:
        if n in seen:
            return
        for m in sorted(deps[n]):
            visit(m)
        seen.add(n)
        order.append(n)

    for t in p.tracks:
        if t.id != master:
            visit(t.id)
    visit(master)
    return Graph(order=order, deps=deps, master=master)


def describe(p: Project) -> list[str]:
    """Líneas legibles del grafo (para reportes y CLI)."""
    lines = []
    for t in p.tracks:
        if t.kind == TrackKind.master:
            continue
        chain = " → ".join(s.uri for s in t.plugins) or "(sin plugins)"
        extra = []
        if t.sidechain_from:
            extra.append(f"sidechain←{t.sidechain_from}")
        extra += [f"envío→{s.target} {s.level_db:+g} dB{' pre' if s.pre_fader else ''}" for s in t.sends]
        lines.append(f"{t.id} [{t.kind.value}]: {chain} → {t.output}" + (f"  ({', '.join(extra)})" if extra else ""))
    return lines
