"""Versión DJ de punta a punta por la CLI (instrumentos integrados): extend -> render -> cues."""
import json
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from osky_remix.cli.main import app
from osky_remix.core.config import project_dir
from osky_remix.core.models import AssetKind, Provenance
from osky_remix.core.revisions import ProjectRepo
from osky_remix.core.assets import AssetStore
from tests.unit.test_dj import BAR, SR, dj_project

pytestmark = pytest.mark.audio
run = CliRunner().invoke


def J(res):
    assert res.exit_code == 0, res.output
    return json.loads(res.stdout.strip().splitlines()[-1])


def _bar_rms(x, tm, first_bar, bars, offset_ticks=0):
    out = []
    for b in range(bars):
        a = tm.tick_to_sample(offset_ticks + (first_bar + b) * BAR)
        z = tm.tick_to_sample(offset_ticks + (first_bar + b + 1) * BAR)
        out.append(10 * np.log10(np.mean(x[a:z] ** 2) + 1e-20))
    return np.array(out)


def test_extend_render_and_cues_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    t = np.arange(30 * SR) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    f = tmp_path / "voz.wav"
    sf.write(f, np.stack([tone, tone], 1), SR)
    root = project_dir("djx")
    a = AssetStore(root / "assets").ingest(f, AssetKind.stem, Provenance(tool="test"))
    repo = ProjectRepo.create(root, dj_project(asset=a))

    info = J(run(app, ["extend", "djx", "--from", "main", "--branch", "dj", "--intro-bars", "16",
                       "--outro-bars", "16", "--json"]))
    N = 16 * BAR
    assert info["intro_ticks"] == N and info["outro_start_tick"] == N + 32 * BAR and info["branch"] == "dj"
    assert run(app, ["extend", "djx", "--from", "main", "--branch", "dj"]).exit_code != 0      # rama existente
    base, ext = repo.load("main"), repo.load("dj")
    assert repo.load("main") == base and not base.meta.get("dj_extended")                     # la base no cambia

    rb = J(run(app, ["render", "djx", "--rev", "main", "--profile", "dynamic", "--out", str(tmp_path / "rb"), "--json"]))
    re_ = J(run(app, ["render", "djx", "--rev", "dj", "--profile", "dynamic", "--out", str(tmp_path / "re"), "--json"]))
    tb, te = base.tempo_map(), ext.tempo_map()
    grow = te.tick_to_sample((16 + 32 + 16) * BAR) - tb.tick_to_sample(32 * BAR)
    assert re_["length_samples"] - rb["length_samples"] == grow
    assert sf.info(re_["files"]["master"]).frames - sf.info(rb["files"]["master"]).frames == grow

    stem = lambda r, n: sf.read(str(tmp_path / r / "mix_stems" / f"{n}.wav"), dtype="float32", always_2d=True)[0][:, 0]
    s_n, s_half, s_out = te.tick_to_sample(N), te.tick_to_sample(N // 2), te.tick_to_sample(N + 32 * BAR)
    vox = stem("re", "vox")
    assert not np.any(vox[:s_n]) and not np.any(vox[s_out:])                 # sin voz en intro ni outro
    kick = stem("re", "kick")
    assert np.abs(kick[:s_half]).max() > 0.05 and np.abs(kick[s_out:]).max() > 0.05
    bass = stem("re", "bass")
    assert np.abs(bass[:s_half]).max() < 1e-6 < 0.01 < np.abs(bass[s_half:s_n]).max()   # el bajo entra a la mitad
    # la pista bloqueada suena igual, desplazada (RMS por compás, sin el primer compás por el sidechain)
    pad_b = _bar_rms(stem("rb", "pad"), tb, 1, 22)
    pad_e = _bar_rms(stem("re", "pad"), te, 1, 22, offset_ticks=N)
    assert np.max(np.abs(pad_b - pad_e)) < 0.1

    # cues: los tiempos del XML caen en las muestras donde empieza el audio
    out = tmp_path / "cues.xml"
    res = J(run(app, ["cues", "djx", "--rev", "dj", "--audio", re_["files"]["master"], "--out", str(out), "--json"]))
    assert res["cues"] == 5 and res["hot_cues"] == 5
    tr = ET.parse(out).getroot().find("COLLECTION/TRACK")
    (tempo,) = tr.findall("TEMPO")
    assert float(tempo.get("Inizio")) == 0 and float(tempo.get("Bpm")) == 124 and tempo.get("Battito") == "1"
    starts = {m.get("Name"): float(m.get("Start")) for m in tr.findall("POSITION_MARK") if m.get("Num") != "-1"}
    for m in ext.markers:
        assert abs(starts[m.label] * SR - te.tick_to_sample(m.tick)) < 1, m.label
    onset = int(np.argmax(np.abs(vox) > 1e-4))
    assert abs(onset - starts["voz"] * SR) <= 2                               # la voz empieza en el cue "voz"
    assert abs(int(np.argmax(np.abs(kick) > 1e-4)) - starts["intro_dj"] * SR) <= 2


def test_export_package_includes_dj_cues(tmp_path, repo):
    from osky_remix.audio.export import export_package, verify_package
    from osky_remix.core.models import Project
    from osky_remix.core.operations import Context
    from osky_remix.core.transactions import apply_ops, parse_ops
    kick = [{"tick": i * 960, "duration": 240, "pitch": 36, "velocity": 110} for i in range(16)]
    p = apply_ops(Project(name="pk", sample_rate=SR), parse_ops([
        {"op": "set_tempo", "points": [{"tick": 0, "bpm": 120}, {"tick": 2 * BAR, "bpm": 130}]},
        {"op": "add_track", "id": "kick", "name": "K", "kind": "instrument"},
        {"op": "add_plugin", "track": "kick", "plugin": {"id": "d", "backend": "builtin", "uri": "drums"}},
        {"op": "add_clip", "track": "kick", "clip": {"type": "midi", "id": "k", "start_tick": 0, "length_ticks": 4 * BAR, "notes": kick}},
        {"op": "add_marker", "marker": {"id": "a", "tick": 0, "label": "intro"}},
        {"op": "add_marker", "marker": {"id": "b", "tick": 3 * BAR, "label": "drop"}}]), Context(author="user"))
    out = tmp_path / "pkg"
    export_package(p, repo.assets, repo.cache, out, "streaming", "rev")
    assert verify_package(out) == []
    tr = ET.parse(out / "dj" / "cues.xml").getroot().find("COLLECTION/TRACK")
    assert tr.get("Location").endswith(str((out / "audio" / "master.wav").resolve()))
    assert [float(t.get("Bpm")) for t in tr.findall("TEMPO")] == [120, 130]
    drop = next(m for m in tr.findall("POSITION_MARK") if m.get("Name") == "drop")
    assert abs(float(drop.get("Start")) - (8 * 60 / 120 + 4 * 60 / 130)) * SR < 1
    assert (out / "dj" / "cues.csv").read_text().count("\ncue,") == 2
