"""Cola de trabajos local en SQLite (render, compare, evaluate) con un worker en segundo plano."""
from __future__ import annotations

import json
import multiprocessing
import time
import fcntl
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


def _execute(handler, project, params, connection):
    try:
        connection.send((True, handler(project, params)))
    except BaseException as e:
        connection.send((False, f"{type(e).__name__}: {e}"))
    finally:
        connection.close()


class JobQueue:
    def __init__(self, db_path: Path, handlers: dict[str, Callable[[str, dict], dict]],
                 notify: Callable[[dict], None] | None = None, timeout_s: float = 1800):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._owner = open(db_path.with_suffix(".worker.lock"), "a")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._owner.close()
            raise RuntimeError("ya existe un trabajador para esta cola") from None
        self.timeout_s = timeout_s
        self.engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        _Base.metadata.create_all(self.engine)
        self.handlers = handlers
        self.notify = notify or (lambda e: None)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._cancel = threading.Event()      # cancelación del trabajo en curso (mismo proceso)
        self._running: str | None = None
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
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("el trabajador no pudo detenerse")
        self.engine.dispose()
        self._owner.close()

    def cancel(self, jid: str) -> bool:
        with Session(self.engine) as s, s.begin():
            r = s.get(JobRow, jid)
            if r is None or r.status not in ("queued", "running"):
                return False
            r.status, r.error, r.updated = "cancelled", "cancelado por el usuario", _now()
        if jid == self._running:
            self._cancel.set()
        self.notify({"type": "job", "id": jid, "status": "cancelled"})
        self._wake.set()
        return True

    def _run_isolated(self, job):
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_execute, args=(self.handlers[job.kind], job.project,
                              json.loads(job.params), child), daemon=True)
        try:
            process.start()
            child.close()
            deadline = time.monotonic() + self.timeout_s
            next_db_check = time.monotonic() + 1.0
            while True:
                if self._stop.is_set():
                    raise RuntimeError("interrumpido por cierre del servidor")
                # El evento cubre cancel() de este proceso; la consulta periódica, otros procesos.
                if self._cancel.is_set():
                    raise RuntimeError("cancelado por el usuario")
                now = time.monotonic()
                if now >= next_db_check:
                    next_db_check = now + 1.0
                    if self.get(job.id)["status"] == "cancelled":
                        raise RuntimeError("cancelado por el usuario")
                if parent.poll(.1):
                    try:
                        ok, result = parent.recv()
                    except EOFError:
                        raise RuntimeError("el proceso de audio terminó sin resultado") from None
                    if not ok:
                        raise RuntimeError(result)
                    return result
                if not process.is_alive():
                    raise RuntimeError(f"el proceso de audio terminó con código {process.exitcode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("el trabajo excedió el tiempo máximo")
        finally:
            child.close()
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
                process.close()
            parent.close()

    def _next(self) -> JobRow | None:
        with Session(self.engine) as s, s.begin():
            r = s.scalars(select(JobRow).where(JobRow.status == "queued").order_by(JobRow.created)).first()
            if r is None:
                return None
            r.status, r.updated = "running", _now()
            s.flush()
            s.expunge(r)
            return r

    def _finish(self, jid: str, status: str, result: dict | None, error: str | None) -> str:
        """Guarda el desenlace y devuelve el estado final (una cancelación previa prevalece)."""
        with Session(self.engine) as s, s.begin():
            r = s.get(JobRow, jid)
            if r.status == "cancelled":
                return "cancelled"
            r.status, r.updated = status, _now()
            r.result = json.dumps(result, default=str) if result is not None else None
            r.error = error
            return status

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self._next()
            if job is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            self._cancel.clear()
            self._running = job.id
            base = {"type": "job", "id": job.id, "project": job.project, "kind": job.kind}
            self.notify({**base, "status": "running"})
            try:
                res = self._run_isolated(job)
                if self._finish(job.id, "done", res, None) == "done":
                    self.notify({**base, "status": "done", "result": res})
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                if self._finish(job.id, "failed", None, error) == "failed":
                    self.notify({**base, "status": "failed", "error": error, "trace": traceback.format_exc()[-800:]})
                # cancelado: cancel() ya publicó el evento y el estado guardado no cambia
            finally:
                self._running = None
