"""Caché de resultados derivados (análisis, stems, renders) direccionada por contenido.

La clave combina: nombre y versión del cálculo, hashes de los assets de entrada y parámetros
canónicos. Si cambia cualquiera, la clave cambia; no hay invalidación manual.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def cache_key(task: str, version: str, inputs: list[str], params: dict | None = None) -> str:
    payload = json.dumps({"task": task, "version": version, "inputs": sorted(inputs), "params": params or {}},
                         sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class CacheStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, key: str) -> Path:
        return self.root / key[:2] / key

    def get(self, key: str) -> Path | None:
        d = self._dir(key)
        return d if (d / ".complete").exists() else None

    def put_files(self, key: str, files: dict[str, Path], meta: dict | None = None) -> Path:
        """Guarda archivos bajo la clave. Escritura atómica: o queda completa o no existe."""
        d = self._dir(key)
        if (d / ".complete").exists():
            return d
        tmp = d.with_name(key + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for name, src in files.items():
            shutil.copyfile(src, tmp / name)
        (tmp / "meta.json").write_text(json.dumps(meta or {}, indent=2, default=str))
        (tmp / ".complete").write_text("")
        shutil.rmtree(d, ignore_errors=True)
        os.replace(tmp, d)
        return d

    def put_json(self, key: str, data: dict) -> Path:
        d = self._dir(key)
        tmp = d.with_name(key + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        (tmp / "result.json").write_text(json.dumps(data, sort_keys=True, default=str))
        (tmp / ".complete").write_text("")
        shutil.rmtree(d, ignore_errors=True)
        os.replace(tmp, d)
        return d

    def get_json(self, key: str) -> dict | None:
        d = self.get(key)
        return json.loads((d / "result.json").read_text()) if d and (d / "result.json").exists() else None
