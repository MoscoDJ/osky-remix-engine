"""Comandos de proyecto, operaciones, assets y esquemas."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from pydantic import TypeAdapter
from rich.console import Console
from rich.table import Table

from ..core.config import REPO_ROOT, project_dir, workspace
from ..core.models import AssetKind, Project, Provenance, TempoPointM
from ..core.operations import Operation, RegisterAsset
from ..core.revisions import MergeConflict, ProjectRepo, RepoError
from ..core.transactions import TransactionError

console = Console()
err = Console(stderr=True)

project_app = typer.Typer(help="Proyectos y revisiones", no_args_is_help=True)
op_app = typer.Typer(help="Contrato de operaciones (única vía para modificar un proyecto)", no_args_is_help=True)
asset_app = typer.Typer(help="Assets direccionados por SHA-256", no_args_is_help=True)
schema_app = typer.Typer(help="Esquemas JSON del contrato", no_args_is_help=True)

AuthorOpt = typer.Option("user", "--author", "-a", help="user | claude")
JsonOpt = typer.Option(False, "--json", help="Salida JSON (para agentes)")


def _repo(name: str) -> ProjectRepo:
    try:
        return ProjectRepo(project_dir(name))
    except (RepoError, ValueError) as e:
        _fail(str(e))


def _fail(msg: str, code: int = 1):
    err.print(f"[red]error:[/] {msg}")
    raise typer.Exit(code)


def _out(as_json: bool, data: dict, human: str) -> None:
    if as_json:
        print(json.dumps(data, ensure_ascii=False, default=str))
    else:
        console.print(human)


def _author(a: str) -> str:
    if a not in ("user", "claude"):
        _fail("--author debe ser 'user' o 'claude'")
    return a


# ------------------------------------------------------------------ proyecto
@project_app.command("new")
def project_new(name: str, bpm: float = typer.Option(124.0), sample_rate: int = typer.Option(48000, "--sr"),
                key: Optional[str] = None, author: str = AuthorOpt, as_json: bool = JsonOpt):
    """Crea un proyecto vacío (con pista master) y su primera revisión."""
    try:
        p = Project(name=name, sample_rate=sample_rate, key=key, tempo=[TempoPointM(tick=0, bpm=bpm)])
        repo = ProjectRepo.create(project_dir(name), p, _author(author))
    except (RepoError, ValueError) as e:
        _fail(str(e))
    _out(as_json, {"project": name, "head": repo.head(), "path": str(repo.root)},
         f"Proyecto [bold]{name}[/] creado en {repo.root} (revisión {repo.head()[:12]})")


@project_app.command("list")
def project_list(as_json: bool = JsonOpt):
    root = workspace() / "projects"
    names = sorted(d.name for d in root.glob("*") if (d / "project.db").exists()) if root.exists() else []
    _out(as_json, {"projects": names}, "\n".join(names) or "(sin proyectos)")


@project_app.command("show")
def project_show(name: str, rev: Optional[str] = None, as_json: bool = JsonOpt):
    """Resumen del proyecto (o JSON completo con --json)."""
    repo = _repo(name)
    try:
        p = repo.load(rev)
    except RepoError as e:
        _fail(str(e))
    if as_json:
        print(p.canonical_json())
        return
    tm = p.tempo_map()
    console.print(f"[bold]{p.name}[/]  rama {repo.current_branch}  revisión {(rev and repo.resolve(rev) or repo.head())[:12]}")
    console.print(f"{p.sample_rate} Hz, PPQ {p.ppq}, tempo {', '.join(f'{x.bpm:g}@{x.tick}' for x in p.tempo)}, "
                  f"compás {p.meter[0].numerator}/{p.meter[0].denominator}, tonalidad {p.key or '-'}")
    t = Table()
    for c in ("Pista", "Tipo", "Clips", "Plugins", "Ganancia", "Salida", "Bloq."):
        t.add_column(c)
    for tr in p.tracks:
        t.add_row(tr.id, tr.kind.value, str(len(tr.clips)), ", ".join(s.id for s in tr.plugins) or "-",
                  f"{tr.gain_db:+.1f} dB", tr.output or "-", "🔒" if tr.locked else "")
    console.print(t)
    if p.assets:
        console.print(f"Assets: {len(p.assets)}  " + ", ".join(f"{a.name} ({a.kind.value}, {a.sha256[:10]})" for a in p.assets.values()))
    if p.markers:
        console.print("Marcadores: " + ", ".join(f"{m.label}@{tm.tick_to_sample(m.tick) / p.sample_rate:.2f}s" for m in p.markers))


@project_app.command("log")
def project_log(name: str, branch: Optional[str] = None, limit: int = 30, as_json: bool = JsonOpt):
    repo = _repo(name)
    revs = repo.log(branch, limit)
    if as_json:
        print(json.dumps([r.__dict__ for r in revs], ensure_ascii=False))
        return
    for r in revs:
        extra = f" + {r.merge_parent[:8]}" if r.merge_parent else ""
        console.print(f"[yellow]{r.id[:12]}[/]{extra} [dim]{r.created_at[:19]}[/] [cyan]{r.author:6s}[/] "
                      f"[magenta]{r.kind:6s}[/] {r.message} [dim]({len(r.ops)} ops)[/]")


@project_app.command("branch")
def project_branch(name: str, branch: Optional[str] = typer.Argument(None), from_rev: Optional[str] = typer.Option(None, "--from")):
    """Lista ramas, o crea una nueva."""
    repo = _repo(name)
    if branch is None:
        cur = repo.current_branch
        for b, h in repo.branches().items():
            console.print(f"{'*' if b == cur else ' '} {b:20s} {h[:12]}")
        return
    try:
        h = repo.create_branch(branch, from_rev)
    except RepoError as e:
        _fail(str(e))
    console.print(f"Rama [bold]{branch}[/] creada en {h[:12]}")


@project_app.command("checkout")
def project_checkout(name: str, branch: str):
    repo = _repo(name)
    try:
        repo.checkout(branch)
    except RepoError as e:
        _fail(str(e))
    console.print(f"Rama actual: [bold]{branch}[/]")


@project_app.command("revert")
def project_revert(name: str, rev: str, author: str = AuthorOpt, message: Optional[str] = typer.Option(None, "-m"),
                   as_json: bool = JsonOpt):
    """Crea una revisión nueva con el contenido de REV (no reescribe historial)."""
    repo = _repo(name)
    try:
        r = repo.revert(rev, _author(author), message)
    except RepoError as e:
        _fail(str(e))
    _out(as_json, {"revision": r.id}, f"Revertido: nueva revisión {r.id[:12]}")


@project_app.command("merge")
def project_merge(name: str, source: str, author: str = AuthorOpt, message: Optional[str] = typer.Option(None, "-m"),
                  as_json: bool = JsonOpt):
    """Fusiona la rama SOURCE en la actual. Aborta completo ante conflicto."""
    repo = _repo(name)
    try:
        r = repo.merge(source, _author(author), message)
    except MergeConflict as e:
        _fail(f"conflicto: {e}", 3)
    except RepoError as e:
        _fail(str(e))
    if r is None:
        _out(as_json, {"revision": None, "status": "up_to_date"}, "Nada que fusionar")
    else:
        _out(as_json, {"revision": r.id, "status": r.kind}, f"Fusión: {r.id[:12]} ({r.kind})")


@project_app.command("export")
def project_export(name: str, out: Path, rev: Optional[str] = None):
    """Exporta el proyecto como JSON canónico."""
    repo = _repo(name)
    out.write_text(json.dumps(json.loads(repo.load(rev).canonical_json()), indent=2, ensure_ascii=False))
    console.print(f"Exportado a {out}")


# ------------------------------------------------------------------ operaciones
@op_app.command("apply")
def op_apply(name: str, ops_file: Path = typer.Argument(..., help="JSON: lista de operaciones, o '-' para stdin"),
             author: str = AuthorOpt, message: str = typer.Option(..., "-m", "--message"),
             branch: Optional[str] = None, as_json: bool = JsonOpt):
    """Aplica un lote de operaciones de forma atómica y crea una revisión."""
    raw = json.loads(sys.stdin.read() if str(ops_file) == "-" else ops_file.read_text())
    if isinstance(raw, dict):
        raw = raw.get("ops", [raw])
    repo = _repo(name)
    try:
        r = repo.commit(raw, _author(author), message, branch)
    except TransactionError as e:
        if as_json:
            print(json.dumps({"ok": False, "index": e.index, "reason": e.reason}, ensure_ascii=False))
            raise typer.Exit(2)
        _fail(f"{e} — no se aplicó ningún cambio", 2)
    except RepoError as e:
        _fail(str(e))
    _out(as_json, {"ok": True, "revision": r.id, "ops": len(r.ops)},
         f"[green]OK[/] {len(r.ops)} operaciones → revisión {r.id[:12]}")


# ------------------------------------------------------------------ assets
@asset_app.command("import")
def asset_import(name: str, file: Path, kind: AssetKind = typer.Option(AssetKind.original),
                 tool: str = typer.Option("import"), author: str = AuthorOpt, as_json: bool = JsonOpt):
    """Copia un archivo al almacén (SHA-256, solo lectura) y lo registra con una operación."""
    if not file.exists():
        _fail(f"no existe {file}")
    repo = _repo(name)
    asset = repo.assets.ingest(file, kind, Provenance(tool=tool, params={"source_path": str(file.resolve())}))
    if asset.sha256 in repo.load().assets:
        _out(as_json, {"sha256": asset.sha256, "status": "already_registered"}, f"Ya registrado: {asset.sha256[:12]}")
        return
    try:
        r = repo.commit([RegisterAsset(asset=asset)], _author(author), f"Importar {file.name}")
    except TransactionError as e:
        _fail(str(e))
    _out(as_json, {"sha256": asset.sha256, "revision": r.id, "frames": asset.frames, "sample_rate": asset.sample_rate},
         f"Asset {asset.sha256[:12]} ({asset.frames or '-'} muestras @ {asset.sample_rate or '-'} Hz) → revisión {r.id[:12]}")


@asset_app.command("verify")
def asset_verify(name: str):
    """Recalcula el SHA-256 de todos los assets del proyecto."""
    repo = _repo(name)
    bad = [a.name for a in repo.load().assets.values() if not repo.assets.verify(a)]
    if bad:
        _fail(f"assets alterados o faltantes: {', '.join(bad)}")
    console.print("[green]Todos los assets íntegros[/]")


# ------------------------------------------------------------------ esquemas
@schema_app.command("export")
def schema_export(out_dir: Path = typer.Option(REPO_ROOT / "schemas")):
    """Escribe project.schema.json, operation.schema.json y render.schema.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "project.schema.json").write_text(json.dumps(Project.model_json_schema(), indent=2, ensure_ascii=False))
    (out_dir / "operation.schema.json").write_text(
        json.dumps(TypeAdapter(list[Operation]).json_schema(), indent=2, ensure_ascii=False))
    from ..audio.renderer import RenderReport
    (out_dir / "render.schema.json").write_text(json.dumps(TypeAdapter(RenderReport).json_schema(), indent=2, ensure_ascii=False))
    console.print(f"Esquemas escritos en {out_dir}")
