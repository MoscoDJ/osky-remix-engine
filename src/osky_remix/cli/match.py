"""Comando ``match``: compara el master renderizado con una referencia comercial."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ..core.models import AssetKind, Provenance
from ..core.operations import RegisterAsset, SetMeta
from ..core.revisions import RepoError
from ..core.transactions import TransactionError
from .project import _author, _fail, _repo, console


def _resolve_reference(repo, p, reference: Optional[str]):
    """(asset, nuevo) a partir de un archivo, un sha (o prefijo) o meta.reference_track."""
    from ..agents.mix_findings import reference_of
    if reference and Path(reference).is_file():
        f = Path(reference)
        a = repo.assets.ingest(f, AssetKind.reference, Provenance(tool="import", params={"source_path": str(f.resolve())}))
        if a.sample_rate is None:
            _fail(f"{f.name} no es un archivo de audio legible")
        return p.assets.get(a.sha256, a), a.sha256 not in p.assets
    key = reference or reference_of(p)
    if not key:
        _fail("indica --reference <archivo|sha> o registra una con 'match --set'")
    hits = [a for s, a in p.assets.items() if s.startswith(key)]
    if len(hits) != 1:
        _fail(f"referencia '{key[:12]}' no encontrada en el proyecto" if not hits else f"prefijo ambiguo '{key}'")
    return hits[0], False


def match_cmd(name: str, rev: Optional[str] = typer.Option(None, "--rev"),
              reference: Optional[str] = typer.Option(None, "--reference",
                                                      help="archivo de audio (cualquier sr) o sha de un asset; "
                                                           "por defecto meta.reference_track"),
              profile: Optional[str] = typer.Option(None, "--profile"),
              set_: bool = typer.Option(False, "--set", help="registrar la referencia en el proyecto (asset + "
                                                             "meta.reference_track): evaluate/improve la usarán"),
              branch: Optional[str] = typer.Option(None, "--branch", help="rama donde registrar (con --set)"),
              author: str = typer.Option("user", "--author", "-a"), as_json: bool = typer.Option(False, "--json")):
    """Compara el master con una referencia igualada en sonoridad: bandas, estéreo, dinámica y graves."""
    import soundfile as sf
    from ..agents.evaluator import sections_of
    from ..agents.mix_findings import reference_profile, tolerances
    from ..agents.policies import DEFAULT
    from ..analysis import reference_match as rm
    from ..audio.render_service import render_cached
    repo = _repo(name)
    try:
        rid = repo.resolve(rev) if rev else repo.head()
        p = repo.load(rid)
    except RepoError as e:
        _fail(str(e))
    asset, new = _resolve_reference(repo, p, reference)
    if not as_json:
        console.print(f"Renderizando {name}@{rid[:12]} (caché)…")
    rid, _, out = render_cached(repo, name, rid, profile)
    mix, sr = sf.read(str(out / "mix_master.wav"), dtype="float64", always_2d=True)
    secs = [(s["name"], s["start"], s["end"] if s["end"] is not None else len(mix)) for s in sections_of(p)]
    pr = reference_profile(asset.sha256, repo.assets, repo.cache, sr)
    rep = rm.match(mix, None, sr, secs, tolerances(DEFAULT), ref_profile=pr)
    rep.update({"revision": rid, "reference_sha256": asset.sha256, "reference_name": asset.name})
    f = out / f"match_{asset.sha256[:12]}.json"
    f.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    rep["report_file"] = str(f)
    if set_:
        ops = ([RegisterAsset(asset=asset)] if new else []) + \
              [SetMeta(key="reference_track", value={"sha256": asset.sha256, "name": asset.name})]
        try:
            r = repo.commit(ops, _author(author), f"Referencia de mezcla {asset.name}", branch)
        except (TransactionError, RepoError) as e:
            _fail(str(e))
        rep["registered_revision"] = r.id
    if as_json:
        print(json.dumps(rep, ensure_ascii=False, default=str))
        return
    console.print(f"Referencia [bold]{asset.name}[/] ({asset.sha256[:12]})")
    console.print(rm.report_text(rep), markup=False)
    if set_:
        console.print(f"[green]Registrada[/] en la revisión {rep['registered_revision'][:12]} (meta.reference_track)")
    console.print(f"[dim]{f}[/]")
