"""Comando ``dawproject``: exporta una revisión para abrirla en Bitwig, Studio One o Cubase."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..core.revisions import RepoError
from .project import _fail, _out, _repo


def dawproject_cmd(name: str, out: Path = typer.Argument(..., help="archivo de salida .dawproject (no debe existir)"),
                   rev: Optional[str] = typer.Option(None, "--rev"),
                   bounce: bool = typer.Option(True, "--bounce/--no-bounce",
                                               help="renderizar cada pista de instrumento a audio (pre-fader)"),
                   as_json: bool = typer.Option(False, "--json")):
    """Exporta la revisión como DAWproject 1.0 (pistas, ruteo, envíos, clips, notas, tempo, marcadores, automatización)."""
    from ..audio.dawproject import export_dawproject
    if out.suffix.lower() != ".dawproject":
        _fail("la salida debe terminar en .dawproject")
    repo = _repo(name)
    try:
        rid = repo.resolve(rev) if rev else repo.head()
        p = repo.load(rid)
        res = export_dawproject(p, repo.assets, repo.cache, out, bounce_instruments=bounce, revision=rid)
    except (RepoError, FileExistsError, FileNotFoundError, ValueError) as e:
        _fail(str(e))
    human = (f"DAWproject en {out}: {res['tracks']} pistas, {res['audio_files']} archivos de audio"
             + (f", bounce de {', '.join(res['bounced'])}" if res["bounced"] else "")
             + "".join(f"\n[yellow]aviso:[/] {w}" for w in res["warnings"]))
    _out(as_json, {"revision": rid, **res}, human)
