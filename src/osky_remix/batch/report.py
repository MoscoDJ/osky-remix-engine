"""Reporte del lote: JSON y Markdown por canción y comparación entre canciones.

Los valores salen tal cual de los resultados de los pasos en el manifiesto (sin recalcular):
el puntaje de cada variante es el de ``osky-remix evaluate --rev var-x`` y la sonoridad la del
render en caché (``render_cached``).
"""
from __future__ import annotations

import json
from pathlib import Path

from .manifest import Manifest


def song_report(slug: str, s: dict, variants: list[str]) -> dict:
    st = s.get("steps", {})
    res = lambda k: st.get(k, {}).get("result", {}) if st.get(k, {}).get("status") == "done" else {}
    an, seg, comp, vary, ren, ev = (res(k) for k in ("analyze", "vocal_segment", "compose", "vary", "render", "evaluate"))
    rows, warnings = {}, []
    for v in variants:
        r, e = ren.get(v, {}), ev.get(v, {})
        rows[v] = {"branch": vary.get(v, {}).get("branch"), "label": vary.get(v, {}).get("label"),
                   "score": e.get("score"), "lufs": r.get("lufs"), "true_peak_dbtp": r.get("true_peak_dbtp"),
                   "quality": r.get("quality"), "findings": e.get("findings", []), "master": r.get("master")}
        warnings += [f"{v}: {w}" for w in r.get("warnings", [])]
    if comp and not comp.get("vocals"):
        warnings.append("sin stem de voz: arreglo instrumental")
    scored = [(x["score"], v) for v, x in rows.items() if x["score"] is not None]
    failed = next((k for k, x in st.items() if x.get("status") not in ("done", None)), None)
    return {"song": slug, "file": s.get("file"), "project": s.get("project"), "status": s.get("status"),
            "failed_step": failed, "error": s.get("error"),
            "tempo": an.get("tempo"), "key": an.get("key"), "meter": an.get("meter"),
            "vocal_segment": seg or None, "target_bpm": comp.get("bpm"), "variants": rows,
            "best_variant": min(scored)[1] if scored else None, "warnings": warnings,
            "seconds": round(sum(x.get("seconds", 0) or 0 for x in st.values()), 1)}


def _f(x, fmt="{:.2f}"):
    return "–" if x is None else fmt.format(x)


def song_markdown(r: dict) -> str:
    L = [f"# {r['song']}", "", f"- Archivo: `{r['file']}`", f"- Proyecto: `{r['project']}`",
         f"- Estado: **{r['status']}**" + (f" (falló en `{r['failed_step']}`: {r['error']})" if r["failed_step"] else "")]
    if r["tempo"] is not None:
        L.append(f"- Tempo {_f(r['tempo'], '{:.2f}')} BPM · tonalidad {r['key']} · compás {r['meter']}/4")
    seg = r.get("vocal_segment")
    if seg:
        L.append(f"- Segmento vocal: {seg['at']} ({seg['start_s']:.1f}–{seg['end_s']:.1f} s) — {seg['reason']}")
    L += ["", "| variante | rama | puntaje | LUFS | dBTP | calidad |", "|---|---|---|---|---|---|"]
    for v, x in r["variants"].items():
        best = " ★" if v == r["best_variant"] else ""
        L.append(f"| {v}{best} | {x['branch'] or '–'} | {_f(x['score'])} | {_f(x['lufs'])} | "
                 f"{_f(x['true_peak_dbtp'])} | {x['quality'] or '–'} |")
    finds = [(v, f) for v, x in r["variants"].items() for f in x["findings"]]
    if finds:
        L += ["", "## Hallazgos del evaluador", ""] + [f"- {v}: {f}" for v, f in finds]
    if r["warnings"]:
        L += ["", "## Avisos", ""] + [f"- {w}" for w in r["warnings"]]
    return "\n".join(L) + "\n"


def comparison(reports: list[dict], variants: list[str]) -> dict:
    ok = [r for r in reports if r["status"] == "done"]
    mean = {}
    for v in variants:
        xs = [r["variants"][v]["score"] for r in ok if r["variants"][v]["score"] is not None]
        mean[v] = round(sum(xs) / len(xs), 4) if xs else None
    wins = {v: sum(1 for r in ok if r["best_variant"] == v) for v in variants}
    return {"songs": len(reports), "done": len(ok), "failed": [r["song"] for r in reports if r["status"] != "done"],
            "mean_score": mean, "wins": wins,
            "best_overall": min((s, v) for v, s in mean.items() if s is not None)[1] if any(
                s is not None for s in mean.values()) else None}


def batch_markdown(data: dict, reports: list[dict], cmp: dict) -> str:
    vs = data["config"]["variants"]
    L = [f"# Lote {data['id']}", "", f"Plantilla `{data['config']['template']}` · variantes {', '.join(vs)} · "
         f"perfil {data['config']['profile'] or '(del proyecto)'} · {cmp['done']}/{cmp['songs']} canciones completas", "",
         "| canción | estado | tempo | tonalidad | voz desde | " + " | ".join(f"{v} puntaje / LUFS" for v in vs) + " | mejor |",
         "|---|---|---|---|---|" + "---|" * len(vs) + "---|"]
    for r in reports:
        seg = r.get("vocal_segment") or {}
        cells = [f"{_f(r['variants'][v]['score'])} / {_f(r['variants'][v]['lufs'])}" for v in vs]
        L.append(f"| {r['song']} | {r['status']}{' (' + r['failed_step'] + ')' if r['failed_step'] else ''} | "
                 f"{_f(r['tempo'])} | {r['key'] or '–'} | {seg.get('at', '–')} | " + " | ".join(cells)
                 + f" | {r['best_variant'] or '–'} |")
    L += ["", "## Comparación", "", "| variante | puntaje medio (menor es mejor) | canciones donde gana |", "|---|---|---|"]
    L += [f"| {v} | {_f(cmp['mean_score'][v], '{:.3f}')} | {cmp['wins'][v]} |" for v in vs]
    if cmp["best_overall"]:
        L += ["", f"Mejor variante en promedio: **{cmp['best_overall']}**"]
    if cmp["failed"]:
        L += ["", "## Fallidas", ""] + [f"- {r['song']}: {r['error']}" for r in reports if r["status"] != "done"]
    return "\n".join(L) + "\n"


def write_reports(root: Path) -> dict:
    data = Manifest(root).read()
    vs = data["config"]["variants"]
    reports = [song_report(slug, s, vs) for slug, s in data["songs"].items()]
    cmp = comparison(reports, vs)
    sd = Path(root) / "songs"
    sd.mkdir(exist_ok=True)
    for r in reports:
        (sd / f"{r['song']}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
        (sd / f"{r['song']}.md").write_text(song_markdown(r))
    out = {"id": data["id"], "config": data["config"], "songs": reports, "comparison": cmp}
    (Path(root) / "report.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (Path(root) / "report.md").write_text(batch_markdown(data, reports, cmp))
    return out
