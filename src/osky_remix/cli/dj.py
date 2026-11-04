"""Comandos de la versión DJ: extend (extended mix) y cues (beatgrid + cue points)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..core.revisions import RepoError
from ..core.transactions import TransactionError
from .project import _author, _fail, _out, _repo


def extend_cmd(name: str, base: str = typer.Option(..., "--from", help="revisión o rama base (arreglada)"),
               branch: str = typer.Option(..., "--branch", help="rama nueva para la versión extendida"),
               intro_bars: int = typer.Option(32, "--intro-bars", help="16, 32 o 64"),
               outro_bars: int = typer.Option(32, "--outro-bars", help="16, 32 o 64"),
               prefix: str = typer.Option("dj-", "--prefix", help="prefijo de los ids generados"),
               author: str = typer.Option("claude", "--author", "-a"),
               message: Optional[str] = typer.Option(None, "-m"),
               as_json: bool = typer.Option(False, "--json")):
    """Extended mix para DJ: intro y outro de solo ritmo (frases enteras) en una rama nueva."""
    from ..dj.extend import extend
    repo = _repo(name)
    try:
        p = repo.load(base)
        base_id = repo.resolve(base)
    except RepoError as e:
        _fail(str(e))
    if branch in repo.branches():
        _fail(f"ya existe la rama '{branch}'")
    try:
        plan = extend(p, intro_bars, outro_bars, prefix, _author(author))
    except (ValueError, TransactionError) as e:
        _fail(str(e))
    try:
        repo.create_branch(branch, base_id)
        rev = repo.commit(plan.ops, _author(author),
                          message or f"Versión extendida DJ: intro {intro_bars} y outro {outro_bars} compases", branch)
    except (TransactionError, RepoError) as e:
        _fail(str(e))
    info = {"branch": branch, "revision": rev.id, "ops": len(plan.ops), "intro_ticks": plan.intro_ticks,
            "outro_start_tick": plan.outro_start, "outro_ticks": plan.outro_ticks, "tracks": plan.tracks,
            "notes": plan.notes}
    _out(as_json, info,
         f"[green]{branch}[/] revisión {rev.id[:12]} · intro {intro_bars} compases, outro {outro_bars} "
         f"(desde el compás {plan.outro_start // 3840 + 1}) · pistas {', '.join(sorted(set(plan.tracks.values())))}"
         + "".join(f"\n   [dim]{n}[/]" for n in plan.notes))


def cues_cmd(name: str, rev: Optional[str] = typer.Option(None, "--rev"),
             audio: Optional[Path] = typer.Option(None, "--audio", help="master renderizado de esta revisión"),
             out: Path = typer.Option(..., "--out", help="archivo de salida (cues.xml, cues.csv, cues.json)"),
             fmt: str = typer.Option("rekordbox", "--format", help="rekordbox | csv | json"),
             as_json: bool = typer.Option(False, "--json")):
    """Beatgrid y cue points (secciones) para software de DJ, exactos desde el mapa de tempo."""
    from ..dj.cues import write_cues
    repo = _repo(name)
    try:
        p = repo.load(rev)
    except RepoError as e:
        _fail(str(e))
    if audio is not None and not audio.exists():
        _fail(f"no existe el audio {audio}")
    try:
        res = write_cues(p, out, fmt, audio)
    except (ValueError, RuntimeError) as e:
        _fail(str(e))
    _out(as_json, res, f"[green]OK[/] {res['file']} ({fmt}): {res['tempo_points']} puntos de tempo, "
                       f"{res['cues']} memory cues, {res['hot_cues']} hot cues"
                       + "".join(f"\n[yellow]aviso:[/] {w}" for w in res["warnings"]))
