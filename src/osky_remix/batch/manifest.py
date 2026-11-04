"""Manifiesto de un lote (``workspace/batches/<id>/manifest.json``).

Un único JSON legible con el estado de cada canción y de cada paso. Toda modificación es
leer-modificar-escribir bajo un bloqueo ``fcntl`` (varios procesos con ``--jobs``) y se publica con
``os.replace`` (atómico): un corte a mitad nunca deja un manifiesto a medio escribir.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_VERSION = 1
PENDING, RUNNING, DONE, FAILED, INTERRUPTED = "pending", "running", "done", "failed", "interrupted"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "manifest.json"
        self._lock = self.root / "manifest.lock"

    def exists(self) -> bool:
        return self.path.exists()

    def read(self) -> dict:
        return json.loads(self.path.read_text())

    def _write(self, data: dict) -> None:
        data["updated_at"] = now()
        fd, tmp = tempfile.mkstemp(prefix=".manifest-", dir=self.root)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str, allow_nan=False)
        os.replace(tmp, self.path)

    @contextmanager
    def edit(self):
        """``with m.edit() as data: ...`` — bloqueo exclusivo, escritura atómica al salir sin error."""
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self._lock, "a") as h:
            fcntl.flock(h, fcntl.LOCK_EX)
            try:
                data = self.read() if self.path.exists() else {}
                yield data
                self._write(data)
            finally:
                fcntl.flock(h, fcntl.LOCK_UN)

    def create(self, batch_id: str, config: dict) -> None:
        with self.edit() as d:
            d.update({"version": MANIFEST_VERSION, "id": batch_id, "created_at": now(), "config": config,
                      "songs": d.get("songs", {})})

    def add_song(self, slug: str, entry: dict) -> None:
        with self.edit() as d:
            d["songs"].setdefault(slug, {**entry, "status": PENDING, "error": None, "steps": {}})

    def song(self, slug: str) -> dict:
        return self.read()["songs"][slug]

    def step_start(self, slug: str, step: str) -> None:
        with self.edit() as d:
            s = d["songs"][slug]
            s["status"] = RUNNING
            prev = s["steps"].get(step, {})
            s["steps"][step] = {"status": RUNNING, "started_at": now(), "attempts": prev.get("attempts", 0) + 1}

    def step_end(self, slug: str, step: str, status: str, result: dict | None = None, error: str | None = None,
                 seconds: float | None = None) -> None:
        with self.edit() as d:
            s = d["songs"][slug]
            st = s["steps"].setdefault(step, {})
            st.update({"status": status, "finished_at": now(), "seconds": round(seconds or 0.0, 3)})
            if result is not None:
                st["result"] = result
            st["error"] = error
            if status != DONE:
                s["status"] = status
                s["error"] = f"{step}: {error}" if error else step

    def song_end(self, slug: str, status: str, error: str | None = None) -> None:
        with self.edit() as d:
            d["songs"][slug]["status"] = status
            d["songs"][slug]["error"] = error
