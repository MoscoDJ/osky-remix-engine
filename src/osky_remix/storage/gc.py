"""Limpieza de resultados derivados que ya no se reutilizan.

Los renders y comparaciones se guardan por clave de contenido (revisión, perfil, entorno), así
que cada cambio de versión o de entorno deja entradas que nadie volverá a pedir. Todo lo que se
borra aquí se puede regenerar; los proyectos y sus assets nunca se tocan.
"""
from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import workspace

_TMP = re.compile(r"^[0-9a-f]{64}-")          # temporal de mkdtemp(prefix=key + "-")
ORPHAN_AGE_S = 3600                            # un temporal más viejo ya no tiene dueño vivo


@dataclass
class GcReport:
    removed: list[str] = field(default_factory=list)
    freed_bytes: int = 0


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _age(p: Path, now: float) -> float:
    return now - p.stat().st_mtime


def collect(older_than_days: float = 30, include_cache: bool = False, dry_run: bool = True,
            root: Path | None = None) -> GcReport:
    """Elimina renders/comparaciones sin uso en ``older_than_days``, temporales huérfanos y locks sueltos.

    Con ``include_cache`` también vacía entradas viejas de la caché de cada proyecto (stems
    estirados, pistas de Surge, calibraciones). ``dry_run`` solo informa.
    """
    root = root or workspace()
    now = time.time()
    limit = older_than_days * 86400
    rep = GcReport()

    def drop(p: Path):
        rep.removed.append(str(p.relative_to(root)))
        rep.freed_bytes += _size(p)
        if not dry_run:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)

    def sweep(parent: Path):
        if not parent.is_dir():
            return
        for entry in sorted(parent.iterdir()):
            if entry.suffix == ".lock":
                # el lock de una entrada que ya no existe no protege nada
                if not entry.with_suffix("").exists() and _age(entry, now) > ORPHAN_AGE_S:
                    drop(entry)
            elif entry.is_dir() and _TMP.match(entry.name):
                if _age(entry, now) > ORPHAN_AGE_S:
                    drop(entry)
            elif entry.is_dir() and _age(entry, now) > limit:
                drop(entry)

    for kind in ("renders", "compare", "listening"):
        base = root / kind
        if base.is_dir():
            for project in sorted(p for p in base.iterdir() if p.is_dir()):
                sweep(project)
    if include_cache:
        for cache in sorted((root / "projects").glob("*/cache")):
            for shard in sorted(p for p in cache.iterdir() if p.is_dir()):
                sweep(shard)
    return rep
