import typer

from .agent import evaluate_cmd, improve_cmd, vary_cmd
from .analyze import analyze, ingest, reference, separate
from .batch import batch
from .doctor import doctor
from .produce import compare, compose, export, graph, package_app, render
from .project import asset_app, op_app, project_app, schema_app
from .samples import samples_app

app = typer.Typer(help="Osky Remix Engine: producción de remixes headless operada por Claude.",
                  no_args_is_help=True, add_completion=False)
app.command("doctor")(doctor)
app.command("ingest")(ingest)
app.command("separate")(separate)
app.command("analyze")(analyze)
app.command("reference")(reference)
app.command("compose")(compose)
app.command("render")(render)
app.command("graph")(graph)
app.command("compare")(compare)
app.command("export")(export)
app.add_typer(package_app, name="package")
app.command("evaluate")(evaluate_cmd)
app.command("improve")(improve_cmd)
app.command("vary")(vary_cmd)
app.command("batch")(batch)
app.add_typer(project_app, name="project")
app.add_typer(op_app, name="op")
app.add_typer(asset_app, name="asset")
app.add_typer(schema_app, name="schema")
app.add_typer(samples_app, name="samples")


@app.command("serve")
def serve(host: str = typer.Option("127.0.0.1", help="solo local por defecto"), port: int = typer.Option(8765)):
    """Interfaz web local: timeline, mezclador, editor MIDI, A/B e historial."""
    import uvicorn
    from ..api.app import FRONTEND_DIST, create_app
    if not FRONTEND_DIST.exists():
        typer.echo("Aviso: no hay frontend compilado (cd frontend && npm install && npm run build); solo API.")
    typer.echo(f"Osky Remix Engine en http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


@app.command("gc")
def gc(older_than: float = typer.Option(30, "--older-than", help="días sin uso"),
       include_cache: bool = typer.Option(False, "--cache", help="incluir la caché de cada proyecto"),
       apply: bool = typer.Option(False, "--apply", help="borrar de verdad (por defecto solo informa)")):
    """Borra renders, comparaciones y temporales sin uso. Todo es regenerable; no toca proyectos."""
    from ..storage.gc import collect
    rep = collect(older_than, include_cache, dry_run=not apply)
    for r in rep.removed:
        typer.echo(("borrado  " if apply else "borraría ") + r)
    verb = "liberados" if apply else "se liberarían"
    typer.echo(f"{len(rep.removed)} entradas, {rep.freed_bytes / 2**20:.1f} MiB {verb}"
               + ("" if apply else " (usa --apply para borrar)"))


@app.callback()
def _root() -> None:
    """Osky Remix Engine."""


if __name__ == "__main__":
    app()
