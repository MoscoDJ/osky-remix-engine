"""Cola de trabajos local en SQLite (render, compare, evaluate) con un worker en segundo plano."""
from __future__ import annotations

import json
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sqlalchemy import String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class _Base(DeclarativeBase):
    pass


class JobRow(_Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    project: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(16))
    params: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(12))          # queued | running | done | failed
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created: Mapped[str] = mapped_column(String(40))
    updated: Mapped[str] = mapped_column(String(40))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JobQueue:
    def __init__(self, db_path: Path, handlers: dict[str, Callable[[str, dict], dict]],
                 notify: Callable[[dict], None] | None = None):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        _Base.metadata.create_all(self.engine)
        self.handlers = handlers
        self.notify = notify or (lambda e: None)
        self._wake = threading.Event()
        self._stop = threading.Event()
        with Session(self.engine) as s, s.begin():               # trabajos interrumpidos al reiniciar
            for r in s.scalars(select(JobRow).where(JobRow.status == "running")):
                r.status, r.error, r.updated = "failed", "interrumpido por reinicio del servidor", _now()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="osky-worker")
        self._thread.start()

    def submit(self, project: str, kind: str, params: dict) -> str:
        if kind not in self.handlers:
            raise ValueError(f"tipo de trabajo desconocido: {kind}")
        jid = uuid.uuid4().hex[:16]
        with Session(self.engine) as s, s.begin():
            s.add(JobRow(id=jid, project=project, kind=kind, params=json.dumps(params), status="queued",
                         created=_now(), updated=_now()))
        self.notify({"type": "job", "id": jid, "project": project, "kind": kind, "status": "queued"})
        self._wake.set()
        return jid

    def get(self, jid: str) -> dict | None:
        with Session(self.engine) as s:
            r = s.get(JobRow, jid)
            if r is None:
                return None
            return {"id": r.id, "project": r.project, "kind": r.kind, "params": json.loads(r.params),
                    "status": r.status, "result": json.loads(r.result) if r.result else None,
                    "error": r.error, "created": r.created, "updated": r.updated}

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=5)

    def _next(self) -> JobRow | None:
        with Session(self.engine) as s, s.begin():
            r = s.scalars(select(JobRow).where(JobRow.status == "queued").order_by(JobRow.created)).first()
            if r is None:
                return None
            r.status, r.updated = "running", _now()
            s.flush()
            s.expunge(r)
            return r

    def _finish(self, jid: str, status: str, result: dict | None, error: str | None) -> None:
        with Session(self.engine) as s, s.begin():
            r = s.get(JobRow, jid)
            r.status, r.updated = status, _now()
            r.result = json.dumps(result, default=str) if result is not None else None
            r.error = error

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self._next()
            if job is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            self.notify({"type": "job", "id": job.id, "project": job.project, "kind": job.kind, "status": "running"})
            try:
                res = self.handlers[job.kind](job.project, json.loads(job.params))
                self._finish(job.id, "done", res, None)
                self.notify({"type": "job", "id": job.id, "project": job.project, "kind": job.kind, "status": "done", "result": res})
            except Exception as e:  # noqa: BLE001
                self._finish(job.id, "failed", None, f"{type(e).__name__}: {e}")
                self.notify({"type": "job", "id": job.id, "project": job.project, "kind": job.kind, "status": "failed",
                             "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]})
