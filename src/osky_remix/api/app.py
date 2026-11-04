"""API local (FastAPI + WebSocket) para la interfaz híbrida.

- Mismo contrato que la CLI: toda modificación es un lote de operaciones atómico.
- En la interfaz el autor es ``user`` (control manual): puede bloquear y desbloquear pistas.
- Trabajos pesados (render, compare, evaluate) van a una cola SQLite con un worker local.
- Eventos en ``/ws``: revisiones nuevas y estado de trabajos.
Solo debe escuchar en 127.0.0.1.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.cache import cache_key
from ..core.config import REPO_ROOT, project_dir, workspace
from ..core.models import AudioClip, TrackKind
from ..core.revisions import MergeConflict, ProjectRepo, RepoError
from ..core.transactions import TransactionError
from ..storage.jobs import JobQueue

FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


class OpsIn(BaseModel):
    ops: list[dict]
    message: str
    branch: str | None = None
    expected_head: str | None = None


class RevertIn(BaseModel):
    ref: str
    branch: str
    expected_head: str


class BranchIn(BaseModel):
    name: str
    from_rev: str | None = None


class RefIn(BaseModel):
    ref: str


class JobIn(BaseModel):
    kind: str
    rev: str | None = None
    a: str | None = None
    b: str | None = None
    profile: str | None = None


class Hub:
    """Difusión de eventos a los WebSocket conectados (seguro desde hilos)."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish(self, event: dict) -> None:
        if self.loop is None:
            return
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(self._send(ws, event), self.loop)

    async def _send(self, ws: WebSocket, event: dict) -> None:
        try:
            await ws.send_text(json.dumps(event, default=str))
        except Exception:  # noqa: BLE001
            self.clients.discard(ws)


def _repo(name: str) -> ProjectRepo:
    try:
        return ProjectRepo(project_dir(name))
    except (RepoError, ValueError) as e:
        raise HTTPException(404, str(e)) from None


def _rel_url(path: Path) -> str:
    return "/files/" + str(path.resolve().relative_to(workspace().resolve()))


# ------------------------------------------------------------------ trabajos
def job_render(project: str, params: dict) -> dict:
    from ..audio.render_service import render_cached
    from ..mastering.dynamics import MasterProfile
    repo = _repo(project)
    prof = MasterProfile(**params["resolved_profile"]) if "resolved_profile" in params else params.get("profile")
    rid, rep, out = render_cached(repo, project, params.get("rev") or repo.head(), prof)
    return {"revision": rid, "profile": rep["profile"], "mastering_status": rep["mastering_status"],
            "mastering_converged": rep["mastering_converged"], "master_url": _rel_url(out / "mix_master.wav"),
            "lufs": rep["lufs"], "true_peak_dbtp": rep["true_peak_dbtp"], "quality": rep["quality"],
            "warnings": rep["warnings"]}


def job_compare(project: str, params: dict) -> dict:
    from ..core.diff import diff_projects
    from ..mastering.compare import compare
    repo = _repo(project)
    ra, rb = job_render(project, {**params, "rev": params["a"]}), job_render(project, {**params, "rev": params["b"]})
    if ra["lufs"] is None or rb["lufs"] is None:
        raise ValueError("A/B requiere señales con sonoridad medible")
    root = workspace().resolve()
    xa, sr = sf.read(str(root / ra["master_url"][len("/files/"):]), dtype="float32", always_2d=True)
    xb, _ = sf.read(str(root / rb["master_url"][len("/files/"):]), dtype="float32", always_2d=True)
    pa = repo.load(ra["revision"])
    tm = pa.tempo_map()
    ms = sorted(pa.markers, key=lambda m: m.tick)
    secs = [(m.label, tm.tick_to_sample(m.tick), tm.tick_to_sample(ms[i + 1].tick) if i + 1 < len(ms) else len(xa))
            for i, m in enumerate(ms)]
    rep, bm = compare(xa, xb, sr, secs)
    out = workspace() / "compare" / project / cache_key("compare", "2", [ra["revision"]], {"b": rb["revision"], "profile": ra["profile"]})
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "B_igualado.wav", bm, sr, subtype="FLOAT")
    return {"a": ra, "b": rb, "a_url": ra["master_url"], "b_url": _rel_url(out / "B_igualado.wav"),
            "report": rep, "diff": diff_projects(pa, repo.load(rb["revision"]))}


def job_evaluate(project: str, params: dict) -> dict:
    from ..agents.planner import assess
    repo = _repo(project)
    rid = repo.resolve(params.get("rev") or repo.head())
    from ..mastering.dynamics import MasterProfile
    prof = MasterProfile(**params["resolved_profile"]) if "resolved_profile" in params else params.get("profile")
    ev = assess(repo.load(rid), repo, profile=prof)
    return {"revision": rid, "score": ev.score, "metrics": ev.metrics,
            "findings": [{"code": f.code, "severity": f.severity, "detail": f.detail} for f in ev.findings]}


# ------------------------------------------------------------------ app
def create_app() -> FastAPI:
    from contextlib import asynccontextmanager
    hub = Hub()
    queue = JobQueue(workspace() / "jobs.db", {"render": job_render, "compare": job_compare, "evaluate": job_evaluate},
                     hub.publish)

    @asynccontextmanager
    async def lifespan(_app):
        hub.loop = asyncio.get_running_loop()
        yield
        queue.stop()

    app = FastAPI(title="Osky Remix Engine", version="0.6", lifespan=lifespan)
    app.state.hub, app.state.queue = hub, queue

    @app.get("/api/projects")
    def projects():
        root = workspace() / "projects"
        return sorted(d.name for d in root.glob("*") if (d / "project.db").exists()) if root.exists() else []

    @app.get("/api/projects/{name}")
    def project_info(name: str):
        repo = _repo(name)
        return {"name": name, "branches": repo.branches(), "current": repo.current_branch, "head": repo.head()}

    @app.get("/api/projects/{name}/state")
    def state(name: str, rev: str | None = None):
        repo = _repo(name)
        try:
            rid = repo.resolve(rev) if rev else repo.head()
            p = repo.load(rid)
        except RepoError as e:
            raise HTTPException(404, str(e)) from None
        tm = p.tempo_map()
        end = 0
        for t in p.tracks:
            for c in t.clips:
                if isinstance(c, AudioClip):
                    end = max(end, tm.sample_to_tick(tm.tick_to_sample(c.start_tick) + p.audio_length(c)))
                else:
                    end = max(end, c.start_tick + c.length_ticks)
        audio = {sha: {"name": a.name, "frames": a.frames, "sample_rate": a.sample_rate}
                 for sha, a in p.assets.items() if a.frames}
        clip_ticks = {}
        for t in p.tracks:
            for c in t.clips:
                if isinstance(c, AudioClip):
                    clip_ticks[c.id] = tm.sample_to_tick(tm.tick_to_sample(c.start_tick) + p.audio_length(c)) - c.start_tick
        return {"revision": rid, "project": json.loads(p.canonical_json()), "total_ticks": end,
                "bar_ticks": p.ppq * 4 * p.meter[0].numerator // p.meter[0].denominator, "audio_assets": audio,
                "audio_clip_ticks": clip_ticks}

    @app.get("/api/projects/{name}/log")
    def log(name: str, branch: str | None = None, limit: int = 50):
        repo = _repo(name)
        try:
            return [r.__dict__ for r in repo.log(branch, limit)]
        except (RepoError, AttributeError) as e:
            raise HTTPException(404, str(e)) from None

    @app.post("/api/projects/{name}/ops")
    def apply_ops_ep(name: str, body: OpsIn):
        repo = _repo(name)
        try:
            r = repo.commit(body.ops, "user", body.message, body.branch, expected_head=body.expected_head)
        except TransactionError as e:
            return JSONResponse({"ok": False, "index": e.index, "reason": e.reason}, status_code=409)
        except RepoError as e:
            raise HTTPException(400, str(e)) from None
        hub.publish({"type": "revision", "project": name, "branch": r.branch, "revision": r.id, "message": r.message})
        return {"ok": True, "revision": r.id, "branch": r.branch}

    @app.post("/api/projects/{name}/branches")
    def new_branch(name: str, body: BranchIn):
        repo = _repo(name)
        try:
            h = repo.create_branch(body.name, body.from_rev)
        except RepoError as e:
            raise HTTPException(400, str(e)) from None
        hub.publish({"type": "branch", "project": name, "branch": body.name, "revision": h})
        return {"branch": body.name, "head": h}

    @app.post("/api/projects/{name}/checkout")
    def checkout(name: str, body: RefIn):
        repo = _repo(name)
        try:
            repo.checkout(body.ref)
        except RepoError as e:
            raise HTTPException(400, str(e)) from None
        return {"current": body.ref}

    @app.post("/api/projects/{name}/revert")
    def revert(name: str, body: RevertIn):
        repo = _repo(name)
        try:
            r = repo.revert(body.ref, "user", branch=body.branch, expected_head=body.expected_head)
        except RepoError as e:
            raise HTTPException(400, str(e)) from None
        hub.publish({"type": "revision", "project": name, "branch": r.branch, "revision": r.id, "message": r.message})
        return {"revision": r.id}

    @app.post("/api/projects/{name}/merge")
    def merge(name: str, body: RefIn):
        repo = _repo(name)
        try:
            r = repo.merge(body.ref, "user")
        except MergeConflict as e:
            return JSONResponse({"ok": False, "reason": str(e)}, status_code=409)
        except RepoError as e:
            raise HTTPException(400, str(e)) from None
        if r:
            hub.publish({"type": "revision", "project": name, "branch": r.branch, "revision": r.id, "message": r.message})
        return {"ok": True, "revision": r.id if r else None}

    @app.get("/api/projects/{name}/assets/{sha}/peaks")
    def peaks(name: str, sha: str, n: int = 1200):
        repo = _repo(name)
        n = max(16, min(n, 8000))
        key = cache_key("peaks", "1", [sha], {"n": n})
        hit = repo.cache.get_json(key)
        if hit:
            return hit
        f = repo.assets.find(sha)
        if f is None:
            raise HTTPException(404, "asset no encontrado")
        x, sr = sf.read(str(f), dtype="float32", always_2d=True)
        m = np.abs(x).max(axis=1)
        hop = max(1, len(m) // n)
        k = len(m) // hop
        pk = m[: k * hop].reshape(k, hop).max(axis=1)
        data = {"sha": sha, "frames": len(m), "sample_rate": sr, "hop": hop, "peaks": [round(float(v), 4) for v in pk]}
        repo.cache.put_json(key, data)
        return data

    @app.post("/api/projects/{name}/jobs")
    def submit(name: str, body: JobIn):
        repo = _repo(name)
        if body.kind == "compare" and not (body.a and body.b):
            raise HTTPException(400, "compare necesita 'a' y 'b'")
        try:
            from dataclasses import asdict
            from ..mastering.dynamics import load_profile
            params = body.model_dump(exclude_none=True)
            if body.kind == "compare":
                params["a"], params["b"] = repo.resolve(body.a), repo.resolve(body.b)
                prof = body.profile or "club"
            else:
                params["rev"] = repo.resolve(body.rev or repo.head())
                prof = body.profile or repo.load(params["rev"]).master_profile
            params["resolved_profile"] = asdict(load_profile(prof))
            jid = queue.submit(name, body.kind, params)
        except (ValueError, RepoError) as e:
            raise HTTPException(400, str(e)) from None
        return {"job": jid}

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        if not queue.cancel(jid):
            raise HTTPException(409, "trabajo ausente o ya terminado")
        return {"status": "cancelled"}

    @app.get("/api/jobs/{jid}")
    def job(jid: str):
        j = queue.get(jid)
        if j is None:
            raise HTTPException(404, "trabajo no encontrado")
        return j

    @app.get("/api/profiles")
    def profiles():
        from ..mastering.dynamics import PRESETS_DIR, PROFILES
        return sorted({*PROFILES, *(f.stem for f in PRESETS_DIR.glob("*.json"))})

    @app.get("/files/{path:path}")
    def files(path: str):
        root = workspace().resolve()
        f = (root / path).resolve()
        if root not in f.parents or not f.is_file():         # evita salir del workspace
            raise HTTPException(404, "archivo no encontrado")
        if f.suffix not in (".wav", ".json", ".md", ".mid"):
            raise HTTPException(403, "tipo de archivo no servido")
        return FileResponse(f)

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        hub.clients.add(socket)
        try:
            while True:
                await socket.receive_text()                   # el cliente solo mantiene viva la conexión
        except WebSocketDisconnect:
            hub.clients.discard(socket)

    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")
    return app
