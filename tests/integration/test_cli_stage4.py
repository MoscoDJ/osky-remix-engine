"""Etapa 4 por la CLI: grafo, render con perfil del proyecto, A/B, paquete y verificación."""
import json

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from osky_remix.cli.main import app

run = CliRunner().invoke
pytestmark = pytest.mark.audio
SR = 44100


def J(res):
    assert res.exit_code == 0, res.output
    return json.loads(res.stdout.strip().splitlines()[-1])


def apply(ops, msg, branch=None):
    args = ["op", "apply", "p", "-", "-a", "claude", "-m", msg, "--json"] + (["--branch", branch] if branch else [])
    return J(run(app, args, input=json.dumps(ops)))


def test_stage4_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    t = np.arange(6 * SR) / SR
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    f = tmp_path / "t.wav"
    sf.write(f, np.stack([x, x], 1), SR)
    J(run(app, ["project", "new", "p", "--sr", str(SR), "--bpm", "120", "--json"]))
    sha = J(run(app, ["asset", "import", "p", str(f), "--kind", "stem", "--json"]))["sha256"]
    apply([{"op": "add_track", "id": "v", "name": "V", "kind": "audio"},
           {"op": "add_track", "id": "verb", "name": "Verb", "kind": "return"},
           {"op": "add_plugin", "track": "verb", "plugin": {"id": "r", "backend": "builtin", "uri": "reverb", "params": {"wet": 1.0}}},
           {"op": "add_send", "track": "v", "send": {"target": "verb", "level_db": -6}},
           {"op": "add_clip", "track": "v", "clip": {"type": "audio", "id": "c", "asset": sha, "start_tick": 0, "source_length": 6 * SR}},
           {"op": "set_master_profile", "profile": "streaming"}], "base")
    g = run(app, ["graph", "p"])
    assert "envío→verb -6 dB" in g.stdout and "Orden:" in g.stdout
    r = J(run(app, ["render", "p", "--json"]))
    assert r["profile"]["name"] == "streaming" and abs(r["lufs"] + 14) <= 0.5
    bad = run(app, ["op", "apply", "p", "-", "-a", "claude", "-m", "x", "--json"],
              input=json.dumps([{"op": "set_master_profile", "profile": "inexistente"}]))
    assert bad.exit_code == 2
    assert run(app, ["project", "branch", "p", "alt", "--from", "main"]).exit_code == 0
    apply([{"op": "set_track", "track": "v", "gain_db": -3}], "más bajo", "alt")
    ab = J(run(app, ["compare", "p", "main", "alt", "--profile", "streaming", "--json"]))
    assert ab["diff"] == ["v.gain_db: 0 → -3"]
    pkg = tmp_path / "pkg"
    e = J(run(app, ["export", "p", str(pkg), "--rev", "alt", "--json"]))
    assert e["quality"] in ("PASS", "WARN")
    assert run(app, ["package", "verify", str(pkg)]).exit_code == 0
    (pkg / "quality.md").write_text("alterado")
    assert run(app, ["package", "verify", str(pkg)]).exit_code == 1
