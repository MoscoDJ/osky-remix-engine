import typer

from .analyze import analyze, ingest, separate
from .doctor import doctor
from .produce import compose, render
from .project import asset_app, op_app, project_app, schema_app

app = typer.Typer(help="Osky Remix Engine: producción de remixes headless operada por Claude.",
                  no_args_is_help=True, add_completion=False)
app.command("doctor")(doctor)
app.command("ingest")(ingest)
app.command("separate")(separate)
app.command("analyze")(analyze)
app.command("compose")(compose)
app.command("render")(render)
app.add_typer(project_app, name="project")
app.add_typer(op_app, name="op")
app.add_typer(asset_app, name="asset")
app.add_typer(schema_app, name="schema")


@app.callback()
def _root() -> None:
    """Osky Remix Engine."""


if __name__ == "__main__":
    app()
