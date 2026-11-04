"""Hito 1 de punta a punta con material sintético: ingesta -> análisis -> composición -> render."""
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


def test_milestone1_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    song, gt = pop_song()
    src = tmp_path / "song.wav"
    sf.write(src, song, SR, subtype="FLOAT")
    # "voz": melodía sobre las raíces de la sección A (Dm Gm C F), una nota por tiempo
    voice = np.zeros(len(song))
    a_start = gt.sections[1][0]
    for i, m in enumerate([62, 65, 69, 67] * 8 * 4):
        t = a_start + i * SR * 60 // 100
        x = harmonic_tone(midi_to_hz(m), 0.55, SR, n_harm=6, decay=1.5) * 0.4
        voice[t:t + len(x)] += x[: max(0, len(voice) - t)]
    vf = tmp_path / "voz.wav"
    sf.write(vf, np.stack([voice, voice], 1).astype(np.float32), SR, subtype="FLOAT")

    J(run(app, ["project", "new", "h1", "--sr", str(SR), "--json"]))
    ing = J(run(app, ["ingest", "h1", str(src), "--json"]))
    vsha = J(run(app, ["asset", "import", "h1", str(vf), "--kind", "stem", "--json"]))["sha256"]
    an = J(run(app, ["analyze", "h1", ing["working"][:12], "--json"]))
    comp = J(run(app, ["compose", "h1", "--analysis", an["asset"][:12], "--vocals", vsha[:12],
                       "--at", f"{a_start / SR:.2f}", "--bars", "32", "--json"]))
    assert comp["key"] == "D minor" and comp["harmony"][:4] == ["Dm", "Gm", "C", "F"]
    assert comp["melody_notes"] > 20
    show = json.loads(run(app, ["project", "show", "h1", "--json"]).stdout)
    vox = next(t for t in show["tracks"] if t["id"] == "vox")
    assert vox["clips"][0]["start_tick"] == 8 * 3840 - 960          # entra con anacrusa de 1 tiempo
    rep = J(run(app, ["render", "h1", "--json"]))
    m, sr = sf.read(rep["files"]["master"])
    bars_s = 32 * 4 * 60 / 124
    assert abs(len(m) / sr - (bars_s + 4.0)) < 0.5
    assert abs(rep["lufs"] + 9) <= 0.5 and rep["true_peak_dbtp"] <= -1.0 + 1e-3
    assert not [w for w in rep["warnings"] if "silencio" in w]
