from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..doctor.checks import CHECKS
from ..doctor.runner import run_all

console = Console()
STYLE = {"PASS": "bold green", "FAIL": "bold red", "WARN": "yellow", "SKIP": "dim"}
ROOT = Path(__file__).resolve().parents[3]


def doctor(
    only: Optional[list[str]] = typer.Option(None, "--only", "-o", help="Ejecutar solo estas pruebas"),
    keep: bool = typer.Option(False, "--keep", help="Conservar los audios generados por las pruebas"),
    report_dir: Path = typer.Option(ROOT / "workspace" / "reports", help="Dónde guardar el reporte JSON"),
    list_: bool = typer.Option(False, "--list", help="Listar las pruebas disponibles"),
) -> None:
    """Detecta las capacidades reales del equipo verificando audio, no solo instalaciones."""
    if list_:
        for c in CHECKS:
            console.print(f"[bold]{c.name:16s}[/] {'crítica ' if c.critical else 'opcional'}  {c.title}")
        raise typer.Exit()
    valid = {c.name for c in CHECKS}
    if only and (bad := set(only) - valid):
        console.print(f"[red]Pruebas desconocidas: {', '.join(sorted(bad))}[/]")
        raise typer.Exit(2)

    def show(r: dict) -> None:
        console.print(f"[{STYLE[r['status']]}]{r['status']:4s}[/] {r['title']} "
                      f"[dim]({r['seconds']}s)[/]\n      [dim]{r['detail']}[/]")

    console.rule("osky-remix doctor")
    rep = run_all(only, report_dir, keep, on_result=show)
    t = Table(title="Resumen", show_lines=False)
    for col in ("Prueba", "Tipo", "Estado", "Tiempo"):
        t.add_column(col)
    for r in rep["results"]:
        t.add_row(r["title"], "crítica" if r["critical"] else "opcional",
                  f"[{STYLE[r['status']]}]{r['status']}[/]", f"{r['seconds']}s")
    console.print(t)
    gate = rep["stage0_gate"]
    console.print(f"Compuerta etapa 0: [{STYLE[gate]}]{gate}[/]"
                  + (f"  (fallan: {', '.join(rep['critical_failures'])})" if rep["critical_failures"] else ""))
    console.print(f"[dim]Reporte: {rep['report_path']}[/]")
    if rep.get("artifacts_dir"):
        console.print(f"[dim]Audios de prueba: {rep['artifacts_dir']}[/]")
    raise typer.Exit(0 if gate == "PASS" else 1)
