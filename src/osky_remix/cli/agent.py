"""Comandos de la etapa 5: evaluate, improve, vary."""
from __future__ import annotations

import json
from typing import Optional

import typer

from ..core.revisions import RepoError
from ..core.transactions import TransactionError
from .project import _fail, _out, _repo, console


def evaluate_cmd(name: str, rev: Optional[str] = typer.Option(None, "--rev"),
                 profile: Optional[str] = typer.Option(None, "--profile"), as_json: bool = typer.Option(False, "--json")):
    """Renderiza en memoria y evalúa: energía por sección, voz, bajo/bombo y calidad."""
    from ..agents.planner import assess
    repo = _repo(name)
    try:
        p = repo.load(rev)
    except RepoError as e:
        _fail(str(e))
    ev = assess(p, repo, profile=profile)
    data = {"score": ev.score, "findings": [f.__dict__ for f in ev.findings], "metrics": ev.metrics}
    if as_json:
        print(json.dumps(data, ensure_ascii=False, default=str))
        return
    for s in ev.metrics["sections"]:
        console.print(f"  {s['name']:8s} energía {s['energy']}  {s['lufs']:6.2f} LUFS")
    for k in ("vocal_minus_music_lu", "bass_minus_kick_lu", "lufs", "true_peak_dbtp", "loudness_range_lu"):
        if k in ev.metrics:
            console.print(f"  {k}: {ev.metrics[k]}")
    console.print(f"[bold]Puntaje {ev.score:.2f}[/]")
    for f in ev.findings:
        console.print(f"  [yellow]•[/] {f.detail} (severidad {f.severity:.2f})")


def improve_cmd(name: str, branch: str = typer.Option(..., "--branch"),
                max_iter: int = typer.Option(4, "--max-iter"), profile: Optional[str] = typer.Option(None, "--profile"),
                as_json: bool = typer.Option(False, "--json")):
    """Bucle del agente: evalúa, propone, prueba en memoria y confirma solo lo que mejora."""
    from dataclasses import replace
    from ..agents.planner import improve
    from ..agents.policies import DEFAULT
    repo = _repo(name)
    if branch not in repo.branches():
        _fail(f"no existe la rama '{branch}'")
    res = improve(repo, branch, replace(DEFAULT, max_iterations=max_iter), profile,
                  log=(lambda *_: None) if as_json else (lambda m: console.print(m)))
    data = {"initial": res.initial.score, "final": res.final.score, "metrics": res.final.metrics,
            "steps": [s.__dict__ for s in res.steps], "needs_user": res.needs_user}
    if as_json:
        print(json.dumps(data, ensure_ascii=False, default=str))
    else:
        for n in res.needs_user:
            console.print(f"[magenta]requiere al usuario:[/] {n}")


def vary_cmd(name: str, base: str = typer.Option(..., "--from", help="revisión o rama base"),
             variants: str = typer.Option("A,B,C", "--variants"),
             prefix: str = typer.Option("var", "--prefix", help="prefijo de las ramas creadas"),
             evaluate_: bool = typer.Option(True, "--evaluate/--no-evaluate"),
             as_json: bool = typer.Option(False, "--json")):
    """Hito 2: varios arreglos del mismo material en ramas, sin tocar pistas bloqueadas."""
    from ..agents.planner import assess
    from ..composition.arranger import VARIANTS, load_template, rearrange
    from .produce import build_compose_context
    repo = _repo(name)
    try:
        p = repo.load(base)
        base_id = repo.resolve(base)
    except RepoError as e:
        _fail(str(e))
    cm = p.meta.get("compose")
    if not cm:
        _fail("la revisión base no tiene meta 'compose' (créala con 'compose')")
    tpl = load_template(cm.get("template", "melodic_techno"))
    ctx, assets, _ = build_compose_context(repo, cm["analysis"], cm["vocals"], cm["at"], cm["bars"], cm["bpm"], cm["seed"],
                                           cm.get("reference"), cm.get("harmony", "auto"))
    rows = []
    for v in [x.strip() for x in variants.split(",") if x.strip()]:
        if v not in VARIANTS:
            _fail(f"variante desconocida '{v}' (hay: {', '.join(VARIANTS)})")
        br = f"{prefix}-{v.lower()}"
        if br in repo.branches():
            _fail(f"ya existe la rama '{br}'")
        plan = rearrange(p, tpl, ctx, cm["bars"], v, prefix=f"{v.lower()}-")
        try:
            repo.create_branch(br, base_id)
            rev = repo.commit(plan.ops, "claude", f"Variante {v}: {VARIANTS[v]['label']}", br)
        except (TransactionError, RepoError) as e:
            _fail(f"variante {v}: {e}")
        row = {"variant": v, "label": VARIANTS[v]["label"], "branch": br, "revision": rev.id,
               "regenerated": plan.tracks, "notes": plan.skipped}
        if evaluate_:
            ev = assess(repo.load(br), repo)
            row["score"] = ev.score
            row["sections"] = ev.metrics["sections"]
            row["findings"] = [f.detail for f in ev.findings]
        rows.append(row)
        if not as_json:
            console.print(f"[green]{br}[/] ({VARIANTS[v]['label']}) revisión {rev.id[:12]} · regeneradas: "
                          f"{', '.join(plan.tracks)}" + (f" · puntaje {row['score']:.2f}" if evaluate_ else ""))
            for n in plan.skipped:
                console.print(f"   [dim]{n}[/]")
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, default=str))
