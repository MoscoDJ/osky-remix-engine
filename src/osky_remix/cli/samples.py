"""Comandos de la biblioteca de samples: import, search, show y use (kit en un proyecto)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from ..core.models import AssetKind, PluginSlot, Provenance
from ..core.operations import AddPlugin, AddTrack, RegisterAsset, ReplacePlugin
from ..core.revisions import RepoError
from ..core.transactions import TransactionError
from .project import _author, _fail, _out, _repo, console

samples_app = typer.Typer(help="Biblioteca de samples (one-shots y loops) con búsqueda por rasgos",
                          no_args_is_help=True)


def _lib():
    from ..samples.library import SampleLibrary
    return SampleLibrary()


def _row(h) -> dict:
    f = h.features
    return {"sha": h.sha, "name": h.name, "category": h.category, "category_conf": f["category_conf"],
            "duration_s": f["duration_s"], "attack_s": f["attack_s"], "decay_s": f["decay_s"],
            "brightness": f["brightness"], "centroid_hz": f["centroid_hz"], "low_ratio": f["low_ratio"],
            "note": f["note"], "f0_hz": f["f0_hz"], "bpm": f["bpm"], "distance": h.distance}


def _table(rows: list[dict]) -> Table:
    t = Table(show_header=True, header_style="bold")
    for c in ("sha", "nombre", "cat.", "dur s", "ataque ms", "cola s", "brillo", "nota", "bpm", "dist."):
        t.add_column(c)
    for r in rows:
        t.add_row(r["sha"][:12], r["name"], f"{r['category']} ({r['category_conf']:.2f})", f"{r['duration_s']:.2f}",
                  f"{r['attack_s'] * 1000:.1f}", f"{r['decay_s']:.2f}", f"{r['brightness']:.2f}", r["note"] or "-",
                  f"{r['bpm']:.1f}" if r["bpm"] else "-", "" if r["distance"] is None else f"{r['distance']:.2f}")
    return t


@samples_app.command("import")
def samples_import(folder: Path, as_json: bool = typer.Option(False, "--json")):
    """Importa WAV/FLAC/AIFF de una carpeta (recursivo): almacén por SHA-256 y rasgos en caché."""
    lib = _lib()
    try:
        rep = lib.import_dir(folder, log=(lambda *_: None) if as_json else (lambda m: console.print(f"  {m}")))
    except ValueError as e:
        _fail(str(e))
    data = {"added": rep.added, "known": rep.known, "cache_hits": rep.cache_hits, "extracted": rep.extracted,
            "failed": [{"path": p, "error": e} for p, e in rep.failed], "total": len(lib.all())}
    _out(as_json, data, f"[green]{len(rep.added)}[/] nuevos, {len(rep.known)} ya estaban · rasgos: "
         f"{rep.extracted} calculados, {rep.cache_hits} de caché · {len(rep.failed)} errores · "
         f"{data['total']} samples en la biblioteca")
    if rep.failed and not rep.added and not rep.known:
        raise typer.Exit(1)


@samples_app.command("search")
def samples_search(cat: Optional[str] = typer.Option(None, "--cat", help="kick, snare, clap, hat, perc, bass, tonal, fx, loop"),
                   bright: Optional[str] = typer.Option(None, "--bright", help="rango de brillo 0-1, p. ej. 0.2-0.5"),
                   max_decay: Optional[float] = typer.Option(None, "--max-decay", help="cola máxima en s (−40 dB)"),
                   min_decay: Optional[float] = typer.Option(None, "--min-decay"),
                   note: Optional[str] = typer.Option(None, "--note", help="C (cualquier octava) o A4"),
                   loop: Optional[bool] = typer.Option(None, "--loop/--one-shot"),
                   bpm: Optional[str] = typer.Option(None, "--bpm", help="rango, p. ej. 120-126"),
                   like: Optional[str] = typer.Option(None, "--like", help="sha: ordenar por parecido"),
                   limit: int = typer.Option(20, "--limit"), as_json: bool = typer.Option(False, "--json")):
    """Busca por rasgos; con --like, vecinos más cercanos (vectores normalizados)."""
    try:
        hits = _lib().search(cat, bright, max_decay, min_decay, note, loop, bpm, like, limit)
    except ValueError as e:
        _fail(str(e))
    rows = [_row(h) for h in hits]
    if as_json:
        print(json.dumps({"results": rows}, ensure_ascii=False))
    elif rows:
        console.print(_table(rows))
    else:
        console.print("(sin resultados)")


@samples_app.command("show")
def samples_show(sha: str, as_json: bool = typer.Option(False, "--json")):
    """Todos los rasgos de un sample."""
    try:
        h = _lib().resolve(sha)
    except ValueError as e:
        _fail(str(e))
    data = {"sha": h.sha, "name": h.name, "features": h.features}
    _out(as_json, data, "\n".join([f"[bold]{h.name}[/] {h.sha}"] + [f"  {k}: {v}" for k, v in h.features.items()]))


@samples_app.command("use")
def samples_use(name: str, sha: str, track: str = typer.Option(..., "--track", help="pista con instrumento drums o sampler"),
                pitch: Optional[int] = typer.Option(None, "--pitch", help="nota MIDI (por defecto, según la categoría)"),
                gain_db: float = typer.Option(0.0, "--gain-db"),
                create: bool = typer.Option(False, "--create", help="crear la pista con un sampler si no existe"),
                branch: Optional[str] = typer.Option(None, "--branch"),
                author: str = typer.Option("claude", "--author", "-a"), as_json: bool = typer.Option(False, "--json")):
    """Asigna un sample de la biblioteca a una nota del kit de una pista (operaciones validadas)."""
    from ..samples.kit import DEFAULT_PITCH, KIT_INSTRUMENTS, kit_bytes, load_kit
    lib = _lib()
    try:
        h = lib.resolve(sha)
    except ValueError as e:
        _fail(str(e))
    repo = _repo(name)
    try:
        p = repo.load(branch)
    except RepoError as e:
        _fail(str(e))
    pitch = pitch if pitch is not None else DEFAULT_PITCH.get(h.category, 60)
    if not 0 <= pitch <= 127:
        _fail("--pitch debe estar entre 0 y 127")
    ops = []
    t = next((x for x in p.tracks if x.id == track), None)
    if t is None:
        if not create:
            _fail(f"no existe la pista '{track}' (usa --create para crearla con un sampler)")
        ops.append(AddTrack(id=track, name=track.capitalize(), kind="instrument"))
        slot, existing = PluginSlot(id=f"{track}-sampler", backend="builtin", uri="sampler"), {}
        add_slot = True
    else:
        inst = t.plugins[0] if t.plugins else None
        if t.kind.value != "instrument" or inst is None or inst.backend != "builtin" or inst.uri not in KIT_INSTRUMENTS:
            _fail(f"la pista '{track}' no tiene un instrumento integrado drums o sampler")
        slot, add_slot = inst, False
        existing = {}
        if inst.preset:
            kp = repo.assets.find(inst.preset)
            if kp is None:
                _fail(f"el kit {inst.preset[:12]} no está en el almacén del proyecto")
            existing = load_kit(kp)
    sample = repo.assets.ingest(lib.path(h.sha), AssetKind.sample, Provenance(
        tool="sample-library", params={"library_sha": h.sha, "category": h.category}), name=h.name)
    voices = {k: dict(v) for k, v in existing.items()}
    voices[pitch] = {"sample": sample.sha256, "gain_db": gain_db, "name": h.name}
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "kit.json"
        f.write_bytes(kit_bytes(voices))
        kit = repo.assets.ingest(f, AssetKind.preset, Provenance(
            tool="osky-kit", params={"pitch": pitch, "sample": sample.sha256},
            parents=[sample.sha256] + ([slot.preset] if slot.preset else [])), name=f"kit {track}.json")
    ops = [RegisterAsset(asset=a) for a in (sample, kit) if a.sha256 not in p.assets] + ops
    new_slot = slot.model_copy(update={"preset": kit.sha256})
    ops.append(AddPlugin(track=track, plugin=new_slot) if add_slot else
               ReplacePlugin(track=track, plugin_id=slot.id, plugin=new_slot))
    try:
        rev = repo.commit(ops, _author(author), f"Sample {h.name} en {track} (nota {pitch})", branch)
    except (TransactionError, RepoError) as e:
        _fail(str(e))
    _out(as_json, {"revision": rev.id, "track": track, "pitch": pitch, "sample": sample.sha256, "kit": kit.sha256},
         f"[green]OK[/] {h.name} → {track} nota {pitch} · kit {kit.sha256[:12]} · revisión {rev.id[:12]}")
