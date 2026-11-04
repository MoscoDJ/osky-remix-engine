"""``osky-remix listening``: resumen del registro de escucha ciega."""
from __future__ import annotations

import json

import typer
from rich.table import Table

from ..storage.listening import ListeningStore, summarize
from .project import JsonOpt, _repo, console


def listening_cmd(name: str, as_json: bool = JsonOpt):
    """Veredictos de escucha ciega y acuerdo entre la preferencia y el evaluador (menor puntaje = mejor)."""
    repo = _repo(name)
    vs = ListeningStore.for_project(repo.root).verdicts()
    summ = summarize(vs)
    if as_json:
        print(json.dumps({"verdicts": vs, "summary": summ}, ensure_ascii=False))
        return
    if not vs:
        console.print("Sin veredictos todavía (pestaña «Escucha ciega» de la interfaz).")
        return
    t = Table("fecha", "A", "B", "perfil", "prefiere", "nota A/B", "puntaje A/B", "comentario")
    fmt = (lambda s: "—" if s is None else f"{s:.2f}")
    for v in vs:
        pref = {"a": f"A ({v['label_a']})", "b": f"B ({v['label_b']})"}.get(v["preferred"], "sin preferencia")
        t.add_row(v["created"][:19].replace("T", " "), f"{v['label_a']} {v['rev_a'][:8]}", f"{v['label_b']} {v['rev_b'][:8]}",
                  v["profile"], pref, f"{v['rating_a']}/{v['rating_b']}", f"{fmt(v['score_a'])}/{fmt(v['score_b'])}",
                  v["note"])
    console.print(t)
    console.print(f"{summ['verdicts']} veredictos · {summ['no_preference']} sin preferencia · "
                  f"{summ['score_ties']} con puntajes empatados")
    if summ["agreement_pct"] is None:
        console.print("Acuerdo con el evaluador: sin pares comparables.")
    else:
        console.print(f"[bold]Acuerdo con el evaluador: {summ['agreement_pct']:.1f} %[/] "
                      f"({summ['agree']} de {summ['pairs']} pares: la preferida también tiene el menor puntaje)")
        for prof, d in sorted(summ["by_profile"].items()):
            console.print(f"  {prof}: {d['agreement_pct']:.1f} % ({d['agree']}/{d['pairs']})")
    if summ["mean_rating_preferred"] is not None:
        console.print(f"Calificación media: preferida {summ['mean_rating_preferred']} · otra {summ['mean_rating_other']}")
