"""Hito 2 por la CLI: compose -> ediciones manuales bloqueadas -> vary -> improve."""
import json

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from osky_remix.cli.main import app
from osky_remix.doctor.fixtures import harmonic_tone, midi_to_hz
from osky_remix.doctor.songs import pop_song

run = CliRunner().invoke
pytestmark = pytest.mark.compat
SR = 44100


def J(res):
    assert res.exit_code == 0, res.output
    return json.loads(res.stdout.strip().splitlines()[-1])


def show(rev):
    return json.loads(run(app, ["project", "show", "h2", "--rev", rev, "--json"]).stdout)


def test_milestone2_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    song, gt = pop_song()
    src = tmp_path / "song.wav"
    sf.write(src, song, SR, subtype="FLOAT")
    voice = np.zeros(len(song))
    a0 = gt.sections[1][0]
    for i, m in enumerate([62, 65, 69, 67] * 32):
        t = a0 + i * SR * 60 // 100
        x = harmonic_tone(midi_to_hz(m), 0.55, SR, n_harm=6, decay=1.5) * 0.4
        voice[t:t + len(x)] += x[: max(0, len(voice) - t)]
    vf = tmp_path / "voz.wav"
    sf.write(vf, np.stack([voice, voice], 1).astype(np.float32), SR, subtype="FLOAT")
    J(run(app, ["project", "new", "h2", "--sr", str(SR), "--json"]))
    ing = J(run(app, ["ingest", "h2", str(src), "--json"]))
    vsha = J(run(app, ["asset", "import", "h2", str(vf), "--kind", "stem", "--json"]))["sha256"]
    an = J(run(app, ["analyze", "h2", ing["working"][:12], "--json"]))
    J(run(app, ["compose", "h2", "--analysis", an["asset"][:12], "--vocals", vsha[:12],
                "--at", f"{a0 / SR:.2f}", "--bars", "32", "--json"]))
    J(run(app, ["op", "apply", "h2", "-", "-a", "user", "-m", "mi voz", "--json"],
          input=json.dumps([{"op": "set_track", "track": "vox", "gain_db": 1.5}, {"op": "lock_track", "track": "vox"}])))
    base = show("main")
    rows = J(run(app, ["vary", "h2", "--from", "main", "--variants", "A,B", "--no-evaluate", "--json"]))
    assert [r["branch"] for r in rows] == ["var-a", "var-b"]
    for b in ("var-a", "var-b"):
        p = show(b)
        assert next(t for t in p["tracks"] if t["id"] == "vox") == next(t for t in base["tracks"] if t["id"] == "vox")
    arp = lambda p: [n for t in p["tracks"] if t["id"] == "arp" for c in t["clips"] for n in c["notes"]]
    assert arp(show("var-a")) != arp(show("var-b"))
    res = J(run(app, ["improve", "h2", "--branch", "var-a", "--max-iter", "2", "--json"]))
    assert res["final"] <= res["initial"]
    ev = J(run(app, ["evaluate", "h2", "--rev", "var-a", "--json"]))
    assert ev["score"] == res["final"]
