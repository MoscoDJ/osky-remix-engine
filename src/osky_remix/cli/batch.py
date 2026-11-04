"""Comando ``batch``: la cadena completa sobre una carpeta de canciones, reanudable."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from .project import _author, _fail, console


def batch(folder: Path = typer.Argument(..., help="carpeta con las canciones (WAV/FLAC/AIFF/OGG/MP3)"),
          template: str = typer.Option("melodic_techno", "--template", "-t"),
          variants: str = typer.Option("A,B,C", "--variants"),
          profile: Optional[str] = typer.Option(None, "--profile", help="perfil de mastering (por defecto, el del proyecto)"),
          bars: int = typer.Option(32, "--bars"),
          seed: int = typer.Option(1, "--seed"),
          separator: str = typer.Option("demucs", "--separator", help="backend de separación"),
          sample_rate: int = typer.Option(48000, "--sr"),
          jobs: int = typer.Option(1, "--jobs", "-j", help="canciones en paralelo (procesos); la separación va de a una"),
          resume: bool = typer.Option(False, "--resume", help="continuar un lote existente desde el último paso hecho"),
          batch_id: Optional[str] = typer.Option(None, "--id", help="id del lote (por defecto, derivado de la configuración)"),
          author: str = typer.Option("claude", "--author", "-a"),
          as_json: bool = typer.Option(False, "--json")):
    """Por canción: proyecto, ingesta, separación, análisis, segmento vocal, arreglo, variantes,
    render y evaluación. Estado en workspace/batches/<id>/; reporte JSON y Markdown al final."""
    from ..batch.pipeline import BatchError, make_config, run_batch
    from ..batch.report import write_reports
    from ..composition.arranger import VARIANTS, load_template
    vs = [v.strip().upper() for v in variants.split(",") if v.strip()]
    bad = [v for v in vs if v not in VARIANTS]
    if bad:
        _fail(f"variantes desconocidas: {', '.join(bad)} (hay: {', '.join(VARIANTS)})")
    try:
        load_template(template)
    except ValueError as e:
        _fail(str(e))
    if jobs < 1:
        _fail("--jobs debe ser ≥ 1")
    try:
        cfg = make_config(folder, template, vs, profile, bars, seed, separator, sample_rate, _author(author))
        res = run_batch(cfg, resume=resume, jobs=jobs, bid=batch_id,
                        log=(lambda *_: None) if as_json else (lambda m: console.print(f"[dim]{m}[/]")))
    except BatchError as e:
        _fail(str(e))
    except KeyboardInterrupt:
        console.print("[yellow]Interrumpido: el manifiesto guarda lo hecho; continúa con --resume[/]")
        raise typer.Exit(130)
    rep = write_reports(res.dir)
    if as_json:
        print(json.dumps({"id": res.id, "dir": str(res.dir), "songs": res.songs, "ran": res.ran,
                          "skipped": res.skipped, "report": str(res.dir / "report.json")}, ensure_ascii=False))
    else:
        t = Table(show_header=True, header_style="bold")
        for c in ["canción", "estado", "tempo", "tonalidad", "voz desde"] + [f"{v} puntaje" for v in vs] + [
                f"{v} LUFS" for v in vs] + ["mejor"]:
            t.add_column(c)
        for r in rep["songs"]:
            seg = r.get("vocal_segment") or {}
            fmt = lambda x: "–" if x is None else f"{x:.2f}"
            t.add_row(r["song"], r["status"] + (f" ({r['failed_step']})" if r["failed_step"] else ""), fmt(r["tempo"]),
                      r["key"] or "–", seg.get("at", "–"), *[fmt(r["variants"][v]["score"]) for v in vs],
                      *[fmt(r["variants"][v]["lufs"]) for v in vs], r["best_variant"] or "–")
        console.print(t)
        cmp = rep["comparison"]
        if cmp["best_overall"]:
            console.print(f"Mejor variante en promedio: [bold]{cmp['best_overall']}[/] · puntajes medios "
                          + ", ".join(f"{v} {s:.3f}" for v, s in cmp["mean_score"].items() if s is not None))
        for r in rep["songs"]:
            if r["status"] != "done":
                console.print(f"[red]✗[/] {r['song']}: {r['error']}")
        console.print(f"Reporte: {res.dir / 'report.md'}")
    if any(s != "done" for s in res.songs.values()):
        raise typer.Exit(1)
