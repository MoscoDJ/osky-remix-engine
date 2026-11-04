"""Comando ``transpose``: transposición no destructiva de una voz (u otro audio) a la tonalidad del proyecto."""
from __future__ import annotations

from typing import Optional

import typer

from ..core.operations import RegisterAsset, SetClipAsset
from ..core.transactions import TransactionError
from ..core.revisions import RepoError
from .analyze import _asset_path
from .project import _author, _fail, _out, _repo


def _source_key(repo, from_key: str | None, analysis: str | None) -> str:
    if from_key:
        return from_key
    if analysis:
        from ..analysis.pipeline import AnalysisReport
        _, apath = _asset_path(repo, analysis)
        return AnalysisReport.model_validate_json(apath.read_text()).harmony.key.value
    _fail("con --to-key hace falta la tonalidad de origen: --from-key o --analysis <sha>")


def transpose(name: str, sha: str,
              semitones: Optional[float] = typer.Option(None, "--semitones", "-s", help="semitonos (±24)"),
              to_key: Optional[str] = typer.Option(None, "--to-key",
                                                   help="tonalidad destino, o 'proyecto' para la del proyecto"),
              from_key: Optional[str] = typer.Option(None, "--from-key", help="tonalidad de origen (p. ej. 'A minor')"),
              analysis: Optional[str] = typer.Option(None, "--analysis", help="análisis del que leer la tonalidad de origen"),
              formant: bool = typer.Option(True, "--formant/--no-formant", help="conservar formantes (voz)"),
              clip: Optional[str] = typer.Option(None, "--clip", help="clip de audio que pasa a usar el asset transpuesto"),
              author: str = typer.Option("claude", "--author", "-a"), as_json: bool = typer.Option(False, "--json")):
    """Transpone un asset sin tocar el original: registra un asset nuevo y, con --clip, lo usa en ese clip."""
    from ..audio.pitch import parse_key, transpose_asset, transposition_to_key
    if (semitones is None) == (to_key is None):
        _fail("indica --semitones o --to-key (uno de los dos)")
    repo = _repo(name)
    sha, _ = _asset_path(repo, sha)
    src_key = dst_key = None
    if to_key is not None:
        dst_key = repo.load().key if to_key == "proyecto" else to_key
        if not dst_key:
            _fail("el proyecto no tiene tonalidad (set_key); indica --to-key explícita")
        src_key = _source_key(repo, from_key, analysis)
        try:
            semitones = float(transposition_to_key(src_key, dst_key))
        except ValueError as e:
            _fail(str(e))
    elif from_key:
        try:
            parse_key(from_key)
        except ValueError as e:
            _fail(str(e))
    info = {"source": sha, "semitones": semitones, "from_key": src_key, "to_key": dst_key, "formant": formant}
    if semitones == 0:
        _out(as_json, {**info, "asset": sha, "cached": None, "revision": None},
             f"{src_key} ya encaja con {dst_key}: no hace falta transponer")
        return
    try:
        asset, cached = transpose_asset(repo, sha, semitones, formant)
    except (ValueError, RuntimeError) as e:
        _fail(str(e))
    ops = []
    if asset.sha256 not in repo.load().assets:
        ops.append(RegisterAsset(asset=asset))
    if clip:
        ops.append(SetClipAsset(clip=clip, asset=asset.sha256))
    rev = None
    if ops:
        try:
            rev = repo.commit(ops, _author(author), f"Transposición {semitones:+g} st de {sha[:12]}"
                              + (f" en el clip {clip}" if clip else ""))
        except (TransactionError, RepoError) as e:
            _fail(str(e))
    _out(as_json, {**info, "asset": asset.sha256, "cached": cached, "clip": clip, "revision": rev.id if rev else None},
         f"Asset {asset.sha256[:12]} ({semitones:+g} st{', caché' if cached else ''})"
         + (f" · {src_key} → {dst_key}" if dst_key else "") + (f" · clip {clip}" if clip else ""))
