"""Diferencias legibles entre dos versiones del proyecto (para A/B y revisiones)."""
from __future__ import annotations

from .models import Project


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def diff_projects(a: Project, b: Project) -> list[str]:
    out: list[str] = []
    for f in ("sample_rate", "key", "master_profile"):
        if getattr(a, f) != getattr(b, f):
            out.append(f"{f}: {_fmt(getattr(a, f))} → {_fmt(getattr(b, f))}")
    if [p.model_dump() for p in a.tempo] != [p.model_dump() for p in b.tempo]:
        out.append("tempo: " + ", ".join(f"{p.bpm:g}@{p.tick}" for p in a.tempo) + " → "
                   + ", ".join(f"{p.bpm:g}@{p.tick}" for p in b.tempo))
    ta, tb = {t.id: t for t in a.tracks}, {t.id: t for t in b.tracks}
    for tid in sorted(set(tb) - set(ta)):
        out.append(f"+ pista {tid} ({tb[tid].kind.value})")
    for tid in sorted(set(ta) - set(tb)):
        out.append(f"- pista {tid}")
    for tid in sorted(set(ta) & set(tb)):
        x, y = ta[tid], tb[tid]
        for f in ("gain_db", "pan", "mute", "solo", "output", "sidechain_from", "locked", "name"):
            if getattr(x, f) != getattr(y, f):
                out.append(f"{tid}.{f}: {_fmt(getattr(x, f))} → {_fmt(getattr(y, f))}")
        sx, sy = {s.target: s for s in x.sends}, {s.target: s for s in y.sends}
        for k in sorted(set(sx) | set(sy)):
            if k not in sx:
                out.append(f"{tid}: + envío → {k} {sy[k].level_db:+g} dB")
            elif k not in sy:
                out.append(f"{tid}: - envío → {k}")
            elif sx[k] != sy[k]:
                out.append(f"{tid}: envío → {k} {sx[k].level_db:+g} → {sy[k].level_db:+g} dB")
        px, py = {s.id: s for s in x.plugins}, {s.id: s for s in y.plugins}
        for k in sorted(set(py) - set(px)):
            out.append(f"{tid}: + plugin {k} ({py[k].uri})")
        for k in sorted(set(px) - set(py)):
            out.append(f"{tid}: - plugin {k}")
        for k in sorted(set(px) & set(py)):
            if px[k] != py[k]:
                ch = [f"{p} {_fmt(px[k].params.get(p))}→{_fmt(py[k].params.get(p))}"
                      for p in sorted(set(px[k].params) | set(py[k].params)) if px[k].params.get(p) != py[k].params.get(p)]
                extra = [f for f in ("uri", "preset", "bypass", "backend") if getattr(px[k], f) != getattr(py[k], f)]
                out.append(f"{tid}: plugin {k} " + ", ".join(ch + [f"{f} cambiado" for f in extra]))
        cx, cy = {c.id: c for c in x.clips}, {c.id: c for c in y.clips}
        for k in sorted(set(cy) - set(cx)):
            out.append(f"{tid}: + clip {k} @ {cy[k].start_tick}")
        for k in sorted(set(cx) - set(cy)):
            out.append(f"{tid}: - clip {k}")
        for k in sorted(set(cx) & set(cy)):
            if cx[k] != cy[k]:
                what = []
                if cx[k].start_tick != cy[k].start_tick:
                    what.append(f"movido {cx[k].start_tick}→{cy[k].start_tick}")
                if getattr(cx[k], "notes", None) != getattr(cy[k], "notes", None):
                    what.append("notas cambiadas")
                out.append(f"{tid}: clip {k} " + (", ".join(what) or "modificado"))
    la, lb = {l.id: l for l in a.automation}, {l.id: l for l in b.automation}
    for k in sorted(set(lb) - set(la)):
        out.append(f"+ automatización {k} ({lb[k].track}.{lb[k].target}, {len(lb[k].points)} puntos)")
    for k in sorted(set(la) - set(lb)):
        out.append(f"- automatización {k}")
    for k in sorted(set(la) & set(lb)):
        if la[k] != lb[k]:
            out.append(f"automatización {k} modificada")
    ma, mb = {m.id for m in a.markers}, {m.id for m in b.markers}
    out += [f"+ marcador {m}" for m in sorted(mb - ma)] + [f"- marcador {m}" for m in sorted(ma - mb)]
    return out
