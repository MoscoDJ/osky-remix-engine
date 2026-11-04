"""Caché de resultados derivados (análisis, stems, renders) direccionada por contenido.

La clave combina: nombre y versión del cálculo, hashes de los assets de entrada y parámetros
canónicos. Si cambia cualquiera, la clave cambia; no hay invalidación manual.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import fcntl
from contextlib import contextmanager
from pathlib import Path


def cache_key(task: str, version: str, inputs: list[str], params: dict | None = None) -> str:
    payload = json.dumps({"task": task, "version": version, "inputs": sorted(inputs), "params": params or {}},
                         sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def touch(path: Path) -> None:
    """Marca un resultado como usado: ``osky-remix gc`` borra por antigüedad de último uso."""
    try:
        os.utime(path)
    except OSError:
        pass


def sweep_orphans(parent: Path, key: str) -> None:
    """Borra temporales ``<key>-*`` de escrituras interrumpidas (p. ej. un trabajo cancelado).

    Debe llamarse con el bloqueo de ``key`` tomado: solo su dueño crea esos temporales, así que
    cualquiera que exista quedó de un proceso que murió antes de publicar.
    """
    for orphan in parent.glob(key + "-*"):
        if orphan.is_dir():
            shutil.rmtree(orphan, ignore_errors=True)


class CacheStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, key: str) -> Path:
        return self.root / key[:2] / key

    def get(self, key: str) -> Path | None:
        d = self._dir(key)
        if not (d / ".complete").exists():
            return None
        touch(d)
        return d

    @contextmanager
    def lock(self, key: str):
        d = self._dir(key)
        d.parent.mkdir(parents=True, exist_ok=True)
        with open(d.with_suffix(".lock"), "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _publish(self, key, writer):
        with self.lock(key):
            d = self._dir(key)
            if self.get(key):
                return d
            sweep_orphans(d.parent, key)
            tmp = Path(tempfile.mkdtemp(prefix=key + "-", dir=d.parent))
            try:
                writer(tmp)
                (tmp / ".complete").write_text("")
                if d.exists():
                    shutil.rmtree(d)  # solo una entrada incompleta, bajo el bloqueo
                os.replace(tmp, d)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            return d

    def put_files(self, key: str, files: dict[str, Path], meta: dict | None = None) -> Path:
        def write(tmp):
            for name, src in files.items():
                if Path(name).name != name:
                    raise ValueError("nombre de caché inválido")
                shutil.copyfile(src, tmp / name)
            (tmp / "meta.json").write_text(json.dumps(meta or {}, indent=2, default=str))
        return self._publish(key, write)

    def put_json(self, key: str, data: dict) -> Path:
        return self._publish(key, lambda tmp: (tmp / "result.json").write_text(
            json.dumps(data, sort_keys=True, default=str, allow_nan=False)))

    def get_json(self, key: str) -> dict | None:
        d = self.get(key)
        return json.loads((d / "result.json").read_text()) if d and (d / "result.json").exists() else None
