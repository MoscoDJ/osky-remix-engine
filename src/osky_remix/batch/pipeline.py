"""Lote: muchas canciones por la misma cadena (proyecto → ingesta → separación → análisis →
segmento vocal → composición → variantes → render → evaluación).

- Estado en ``workspace/batches/<id>/manifest.json`` por canción y paso; ``--resume`` salta los pasos
  ya hechos (sus resultados se leen del manifiesto) y repite el que se cortó o falló.
- Un fallo en una canción se registra y el lote sigue con las demás.
- ``jobs > 1``: una canción por proceso (``spawn``); la separación se serializa con un bloqueo
  global y los renders comparten la caché con sus bloqueos por clave.
"""
from __future__ import annotations

import fcntl
import hashlib
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

from ..core.assets import AUDIO_EXT, sha256_file
from ..core.config import workspace
from .manifest import DONE, FAILED, INTERRUPTED, Manifest
from .steps import DEFAULT_STEPS, SongContext, Step, StepError

CONFIG_KEYS = ("folder", "template", "variants", "profile", "bars", "seed", "separator", "sample_rate", "author")


class BatchError(RuntimeError):
    pass


@dataclass
class BatchResult:
    id: str
    dir: Path
    songs: dict[str, str] = field(default_factory=dict)          # slug -> estado final
    ran: dict[str, list[str]] = field(default_factory=dict)      # slug -> pasos ejecutados
    skipped: dict[str, list[str]] = field(default_factory=dict)  # slug -> pasos saltados (ya hechos)


def slugify(s: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9_\-]+", "-", s.lower()).strip("-_")
    return (s or "cancion")[:n].strip("-_") or "cancion"


def make_config(folder: Path, template: str, variants: list[str], profile: str | None = None, bars: int = 32,
                seed: int = 1, separator: str = "demucs", sample_rate: int = 48000, author: str = "claude") -> dict:
    if not variants:
        raise BatchError("indica al menos una variante")
    return {"folder": str(Path(folder).resolve()), "template": template, "variants": list(variants),
            "profile": profile, "bars": bars, "seed": seed, "separator": separator, "sample_rate": sample_rate,
            "author": author}


def batch_id(config: dict) -> str:
    h = hashlib.sha256(repr(sorted((k, str(config[k])) for k in CONFIG_KEYS)).encode()).hexdigest()[:8]
    return f"{slugify(Path(config['folder']).name, 24)}-{h}"


def batch_dir(bid: str) -> Path:
    return workspace() / "batches" / bid


def find_songs(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise BatchError(f"no es una carpeta: {folder}")
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXT)


@contextmanager
def heavy_slot():
    """Un único paso pesado (separación) a la vez en todo el workspace, entre procesos."""
    lock = workspace() / "batches" / "heavy.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a") as h:
        fcntl.flock(h, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(h, fcntl.LOCK_UN)


def run_song(root: Path, slug: str, steps: list[Step], log=lambda *_: None) -> tuple[str, list[str], list[str]]:
    """Ejecuta los pasos pendientes de una canción. Devuelve (estado, ejecutados, saltados)."""
    m = Manifest(root)
    data = m.read()
    song = data["songs"][slug]
    ctx = SongContext(slug=slug, file=Path(song["file"]), project=song["project"], config=data["config"], log=log)
    ran, skipped = [], []
    if not ctx.file.exists():
        m.song_end(slug, FAILED, "el archivo ya no existe")
        return FAILED, ran, skipped
    if sha256_file(ctx.file) != song["sha256"]:
        m.song_end(slug, FAILED, "el archivo cambió desde que empezó el lote (usa un lote nuevo)")
        return FAILED, ran, skipped
    for step in steps:
        prev = song["steps"].get(step.name, {})
        if prev.get("status") == DONE:
            ctx.results[step.name] = prev.get("result", {})
            skipped.append(step.name)
            continue
        m.step_start(slug, step.name)
        log(f"{slug}: {step.name}…")
        t0 = time.time()
        try:
            with heavy_slot() if step.heavy else nullcontext():
                res = step.fn(ctx) or {}
        except KeyboardInterrupt:
            m.step_end(slug, step.name, INTERRUPTED, error="interrumpido", seconds=time.time() - t0)
            raise
        except Exception as e:                         # el fallo queda en el manifiesto; el lote sigue
            msg = str(e).strip() if isinstance(e, StepError) else f"{type(e).__name__}: {e}"
            m.step_end(slug, step.name, FAILED, error=msg[:2000], seconds=time.time() - t0)
            with m.edit() as d:
                d["songs"][slug]["steps"][step.name]["traceback"] = traceback.format_exc()[-4000:]
            log(f"{slug}: {step.name} FALLÓ: {msg}")
            return FAILED, ran, skipped
        m.step_end(slug, step.name, DONE, result=res, seconds=time.time() - t0)
        ctx.results[step.name] = res
        ran.append(step.name)
    m.song_end(slug, DONE)
    return DONE, ran, skipped


def _worker(root: str, slug: str, steps: list[Step]):
    return run_song(Path(root), slug, steps)


def prepare(config: dict, resume: bool = False, bid: str | None = None) -> tuple[str, Path]:
    """Crea (o reabre con ``resume``) el manifiesto y agrega las canciones de la carpeta."""
    bid = bid or batch_id(config)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_\-]{0,63}", bid):
        raise BatchError("id de lote inválido: minúsculas, dígitos, '-' y '_' (máx. 64)")
    root = batch_dir(bid)
    m = Manifest(root)
    if m.exists():
        if not resume:
            raise BatchError(f"el lote '{bid}' ya existe ({root}); usa --resume para continuarlo")
        old = m.read()["config"]
        diff = [k for k in CONFIG_KEYS if old.get(k) != config.get(k)]
        if diff:
            raise BatchError(f"el lote '{bid}' se creó con otra configuración ({', '.join(diff)})")
    else:
        m.create(bid, config)
    tag = bid.rsplit("-", 1)[-1][:8]
    taken = set(m.read()["songs"])
    for f in find_songs(Path(config["folder"])):
        sha = sha256_file(f)
        known = next((s for s, e in m.read()["songs"].items() if e["file"] == str(f.resolve())), None)
        if known:
            continue
        slug = slugify(f.stem)
        if slug in taken:
            slug = f"{slug}-{sha[:6]}"
        taken.add(slug)
        m.add_song(slug, {"file": str(f.resolve()), "sha256": sha, "project": f"{slug[:40]}-{tag}"})
    return bid, root


def run_batch(config: dict, steps: list[Step] | None = None, resume: bool = False, jobs: int = 1,
              bid: str | None = None, log=lambda *_: None) -> BatchResult:
    steps = steps or DEFAULT_STEPS
    bid, root = prepare(config, resume, bid)
    slugs = list(Manifest(root).read()["songs"])
    out = BatchResult(bid, root)
    if jobs <= 1 or len(slugs) <= 1:
        for slug in slugs:
            st, ran, sk = run_song(root, slug, steps, log)
            out.songs[slug], out.ran[slug], out.skipped[slug] = st, ran, sk
            log(f"{slug}: {st}")
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(_worker, str(root), slug, steps): slug for slug in slugs}
            for fu in as_completed(futs):
                slug = futs[fu]
                try:
                    st, ran, sk = fu.result()
                except Exception as e:                 # el proceso murió: queda registrado
                    Manifest(root).song_end(slug, FAILED, f"proceso: {e}")
                    st, ran, sk = FAILED, [], []
                out.songs[slug], out.ran[slug], out.skipped[slug] = st, ran, sk
                log(f"{slug}: {st}")
    return out
