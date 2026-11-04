import typer

from .doctor import doctor

app = typer.Typer(help="Osky Remix Engine: producción de remixes headless operada por Claude.",
                  no_args_is_help=True, add_completion=False)
app.command("doctor")(doctor)


@app.callback()
def _root() -> None:
    """Osky Remix Engine."""


if __name__ == "__main__":
    app()
