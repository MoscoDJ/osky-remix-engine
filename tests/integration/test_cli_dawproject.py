"""Comando ``dawproject`` y su inclusión en el paquete de entrega."""
import hashlib
import io
import json
import xml.etree.ElementTree as ET
import zipfile

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


def apply(ops, msg):
    return J(run(app, ["op", "apply", "p", "-", "-a", "claude", "-m", msg, "--json"], input=json.dumps(ops)))


def test_dawproject_cli_and_package(tmp_path, monkeypatch):
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    t = np.arange(4 * SR) / SR
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    f = tmp_path / "t.wav"
    sf.write(f, np.stack([x, x], 1), SR)
    J(run(app, ["project", "new", "p", "--sr", str(SR), "--bpm", "120", "--json"]))
    sha = J(run(app, ["asset", "import", "p", str(f), "--kind", "stem", "--json"]))["sha256"]
    r1 = apply([{"op": "add_track", "id": "v", "name": "V", "kind": "audio"},
                {"op": "add_clip", "track": "v", "clip": {"type": "audio", "id": "c", "asset": sha, "start_tick": 1920,
                                                          "source_offset": SR, "source_length": 2 * SR}},
                {"op": "add_track", "id": "k", "name": "K", "kind": "instrument"},
                {"op": "add_plugin", "track": "k", "plugin": {"id": "d", "backend": "builtin", "uri": "drums"}},
                {"op": "add_clip", "track": "k", "clip": {"type": "midi", "id": "kc", "start_tick": 0, "length_ticks": 3840,
                                                          "notes": [{"tick": i * 960, "duration": 240, "pitch": 36,
                                                                     "velocity": 110} for i in range(4)]}}], "base")
    apply([{"op": "set_track", "track": "v", "gain_db": -3}], "más bajo")

    out = tmp_path / "s.dawproject"
    d = J(run(app, ["dawproject", "p", str(out), "--rev", r1["revision"], "--no-bounce", "--json"]))
    assert d["revision"] == r1["revision"] and d["bounced"] == [] and d["audio_files"] == 1
    z = zipfile.ZipFile(out)
    root = ET.fromstring(z.read("project.xml"))
    ids = {e.get("id"): e for e in root.iter() if e.get("id")}
    assert float(ids["vol-v"].get("value")) == 1.0                       # la revisión pedida, no la cabeza
    clip = next(c for c in root.iter("Clip") if c.get("name") == "c")
    assert [float(clip.get(k)) for k in ("time", "duration", "playStart", "playStop")] == [2.0, 4.0, 2.0, 6.0]
    path = clip.find("Warps/Audio/File").get("path")
    assert hashlib.sha256(z.read(path)).hexdigest() == sha
    assert [n.get("key") for n in root.iter("Note")] == ["36"] * 4
    assert "audio/bounce/k.wav" not in z.namelist()

    again = run(app, ["dawproject", "p", str(out)])
    assert again.exit_code == 1 and "ya existe" in " ".join(again.output.split())
    assert run(app, ["dawproject", "p", str(tmp_path / "x.zip")]).exit_code == 1

    b = J(run(app, ["dawproject", "p", str(tmp_path / "b.dawproject"), "--json"]))      # cabeza, con bounce
    assert b["bounced"] == ["k"]
    zb = zipfile.ZipFile(tmp_path / "b.dawproject")
    rb = ET.fromstring(zb.read("project.xml"))
    assert float(next(e for e in rb.iter("Volume") if e.get("id") == "vol-v").get("value")) == pytest.approx(10 ** (-3 / 20))
    y, sr = sf.read(io.BytesIO(zb.read("audio/bounce/k.wav")))
    assert sr == SR and np.abs(y[:200]).max() > 0.05                     # bombo en el tick 0

    pkg = tmp_path / "pkg"
    J(run(app, ["export", "p", str(pkg), "--json"]))
    dp = pkg / "daw" / "p.dawproject"
    man = json.loads((pkg / "manifest.json").read_text())
    assert man["files"]["daw/p.dawproject"] == hashlib.sha256(dp.read_bytes()).hexdigest()
    assert "audio/bounce/k.wav" in zipfile.ZipFile(dp).namelist()
    assert run(app, ["package", "verify", str(pkg)]).exit_code == 0
