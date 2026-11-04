"""API local (FastAPI): contrato, trabajos, eventos y archivos."""
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from osky_remix.core.models import AssetKind, Provenance
from osky_remix.core.operations import Context, RegisterAsset
from osky_remix.core.revisions import ProjectRepo
from osky_remix.core.transactions import apply_ops, parse_ops

pytestmark = pytest.mark.audio
SR = 44100


def make_ws_project(root):
    from tests.audio.test_agent import flat_project
    p = flat_project()
    repo = ProjectRepo.create(root / "projects" / "demo", p)
    t = np.arange(4 * SR) / SR
    x = (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    f = root / "tono.wav"
    sf.write(f, np.stack([x, x], 1), SR)
    a = repo.assets.ingest(f, AssetKind.stem, Provenance(tool="test"))
    repo.commit([RegisterAsset(asset=a)] + parse_ops([
        {"op": "add_track", "id": "vox", "name": "Voz", "kind": "audio"},
        {"op": "add_clip", "track": "vox", "clip": {"type": "audio", "id": "v1", "asset": a.sha256, "start_tick": 3840,
                                                    "source_length": 4 * SR}}]), "user", "voz")
    return repo, a.sha256


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path))
    repo, sha = make_ws_project(tmp_path)
    from osky_remix.api.app import create_app
    with TestClient(create_app()) as c:
        c.sha = sha
        yield c


def wait_job(c, jid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.2)
    raise TimeoutError(jid)


def test_state_and_ops_as_user(client):
    assert client.get("/api/projects").json() == ["demo"]
    st = client.get("/api/projects/demo/state").json()
    ids = [t["id"] for t in st["project"]["tracks"]]
    assert ids[:3] == ["master", "kick", "pad"] and st["bar_ticks"] == 3840
    assert st["audio_clip_ticks"]["v1"] > 0 and st["total_ticks"] >= 12 * 3840
    pad = next(t for t in st["project"]["tracks"] if t["id"] == "pad")
    assert pad["locked"]
    r = client.post("/api/projects/demo/ops", json={"ops": [{"op": "unlock_track", "track": "pad"},
                                                             {"op": "set_track", "track": "pad", "gain_db": -3}],
                                                     "message": "desde la interfaz"})
    assert r.status_code == 200 and r.json()["ok"]                  # en la interfaz el autor es el usuario
    log = client.get("/api/projects/demo/log").json()
    assert log[0]["author"] == "user" and log[0]["message"] == "desde la interfaz"


def test_atomic_rejection_reports_index(client):
    head = client.get("/api/projects/demo").json()["head"]
    r = client.post("/api/projects/demo/ops", json={"ops": [{"op": "set_track", "track": "kick", "gain_db": -1},
                                                             {"op": "move_clip", "clip": "nope", "start_tick": 0}],
                                                     "message": "malo"})
    assert r.status_code == 409 and r.json()["index"] == 1
    assert client.get("/api/projects/demo").json()["head"] == head


def test_websocket_revision_event(client):
    with client.websocket_connect("/ws") as ws:
        client.post("/api/projects/demo/ops", json={"ops": [{"op": "set_track", "track": "kick", "gain_db": -2}],
                                                    "message": "evento"})
        ev = ws.receive_json()
        assert ev["type"] == "revision" and ev["message"] == "evento"


def test_render_evaluate_compare_jobs(client):
    j = wait_job(client, client.post("/api/projects/demo/jobs", json={"kind": "render"}).json()["job"])
    assert j["status"] == "done", j
    wav = client.get(j["result"]["master_url"])
    assert wav.status_code == 200 and wav.content[:4] == b"RIFF"
    ev = wait_job(client, client.post("/api/projects/demo/jobs", json={"kind": "evaluate"}).json()["job"])
    assert ev["status"] == "done" and any(f["code"] == "energy_contrast" for f in ev["result"]["findings"])
    client.post("/api/projects/demo/branches", json={"name": "alt", "from_rev": "main"})
    client.post("/api/projects/demo/ops", json={"ops": [{"op": "set_track", "track": "kick", "gain_db": -6}],
                                                "message": "alt", "branch": "alt"})
    cmp = wait_job(client, client.post("/api/projects/demo/jobs", json={"kind": "compare", "a": "main", "b": "alt"}).json()["job"])
    assert cmp["status"] == "done", cmp
    assert cmp["result"]["diff"] == ["kick.gain_db: 0 → -6"]
    assert client.get(cmp["result"]["b_url"]).status_code == 200


def test_peaks_and_file_guard(client):
    pk = client.get(f"/api/projects/demo/assets/{client.sha}/peaks?n=100").json()
    assert 90 <= len(pk["peaks"]) <= 110 and abs(max(pk["peaks"]) - 0.2) < 0.01
    assert client.get("/files/../../etc/passwd").status_code == 404
    assert client.get("/files/projects/demo/project.db").status_code in (403, 404)
