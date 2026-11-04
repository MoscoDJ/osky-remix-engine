"""Transposición no destructiva (Rubber Band) y distancia mínima entre tonalidades."""
import json
import shutil

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from osky_remix.audio.pitch import parse_key, transpose_asset, transposition_to_key
from osky_remix.core.assets import sha256_file
from osky_remix.core.models import AssetKind, Provenance
from osky_remix.core.operations import RegisterAsset
from osky_remix.doctor.fixtures import f0_median

SR = 48000
needs_rb = pytest.mark.skipif(shutil.which("rubberband") is None, reason="rubberband no instalado")


@pytest.mark.parametrize("src,dst,want", [
    ("A minor", "C minor", 3), ("Am", "Cm", 3), ("C minor", "A minor", -3),
    ("F# minor", "C minor", -6), ("C minor", "F# minor", -6), ("Gb minor", "C minor", -6),
    ("D minor", "D minor", 0), ("D minor", "E minor", 2), ("B major", "C major", 1),
    ("C major", "B major", -1), ("G major", "C major", 5), ("C major", "G major", -5),
    ("A minor", "C major", 0), ("C major", "A minor", 0),          # relativas: mismas notas
    ("E minor", "C major", 5), ("Bb major", "A# major", 0),
])
def test_transposition_to_key(src, dst, want):
    assert transposition_to_key(src, dst) == want


def test_transposition_range_all_pairs():
    keys = [f"{n} {m}" for n in "C C# D D# E F F# G G# A A# B".split() for m in ("major", "minor")]
    for a in keys:
        for b in keys:
            d = transposition_to_key(a, b)
            assert -6 <= d <= 5
            pa, ma = parse_key(a)
            pb, mb = parse_key(b)
            if ma == mb:
                assert (pa + d) % 12 == pb


def test_parse_key_rejects_garbage():
    with pytest.raises(ValueError):
        parse_key("H minor")


def _tone_asset(repo, tmp_path, hz=440.0, seconds=2.0):
    t = np.arange(int(SR * seconds)) / SR
    x = (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    f = tmp_path / "voz.wav"
    sf.write(f, np.stack([x, x], 1), SR, subtype="FLOAT")
    a = repo.assets.ingest(f, AssetKind.stem, Provenance(tool="test"), name="voz.wav")
    repo.commit([RegisterAsset(asset=a)], "user", "voz")
    return a


@pytest.fixture
def repo48(tmp_path):
    from osky_remix.core.models import Project
    from osky_remix.core.revisions import ProjectRepo
    return ProjectRepo.create(tmp_path / "proj", Project(name="p", sample_rate=SR))


@needs_rb
@pytest.mark.parametrize("formant", [True, False])
def test_transpose_tone_plus_two(repo48, tmp_path, formant):
    a = _tone_asset(repo48, tmp_path)
    src_path = repo48.assets.find(a.sha256)
    before = sha256_file(src_path)
    out, cached = transpose_asset(repo48, a.sha256, 2, formant=formant)
    assert not cached and out.sha256 != a.sha256
    y, sr = sf.read(str(repo48.assets.find(out.sha256)), dtype="float32", always_2d=True)
    assert sr == SR and y.shape[1] == 2 and np.isfinite(y).all()
    want = 440 * 2 ** (2 / 12)                                      # 493.88 Hz
    assert abs(f0_median(y, sr) / want - 1) < 0.005
    assert abs(len(y) - a.frames) <= 512 and out.frames == a.frames
    # el original sigue intacto y la procedencia apunta a él
    assert sha256_file(src_path) == before == a.sha256
    assert out.provenance.tool == "rubberband" and out.provenance.parents == [a.sha256]
    assert out.provenance.params == {"semitones": 2.0, "formant": formant, "engine": "R3"}
    # segunda llamada: caché
    again, cached2 = transpose_asset(repo48, a.sha256, 2, formant=formant)
    assert cached2 and again.sha256 == out.sha256


@needs_rb
def test_cache_key_depends_on_options(repo48, tmp_path):
    a = _tone_asset(repo48, tmp_path)
    up, _ = transpose_asset(repo48, a.sha256, 2)
    down, cached = transpose_asset(repo48, a.sha256, -3)
    assert not cached and down.sha256 != up.sha256
    y, sr = sf.read(str(repo48.assets.find(down.sha256)), dtype="float32")
    assert abs(f0_median(y, sr) / (440 * 2 ** (-3 / 12)) - 1) < 0.005


def test_transpose_rejects_out_of_range(repo48, tmp_path):
    a = _tone_asset(repo48, tmp_path)
    with pytest.raises(ValueError):
        transpose_asset(repo48, a.sha256, 30)


# ------------------------------------------------------------------ CLI de punta a punta
run = CliRunner().invoke


def J(res):
    assert res.exit_code == 0, res.output
    return json.loads(res.stdout.strip().splitlines()[-1])


@needs_rb
def test_cli_transpose_to_project_key_replaces_clip(tmp_path, monkeypatch):
    from osky_remix.cli.main import app
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    t = np.arange(SR * 2) / SR
    vf = tmp_path / "voz.wav"
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)        # A
    sf.write(vf, np.stack([x, x], 1), SR, subtype="FLOAT")
    J(run(app, ["project", "new", "tv", "--sr", str(SR), "--key", "Cm", "--json"]))
    vsha = J(run(app, ["asset", "import", "tv", str(vf), "--kind", "stem", "--json"]))["sha256"]
    ops = [{"op": "add_track", "id": "vox", "name": "Vox", "kind": "audio"},
           {"op": "add_clip", "track": "vox", "clip": {"type": "audio", "id": "v1", "asset": vsha, "start_tick": 960,
                                                       "source_offset": 100, "source_length": SR, "gain_db": -2}}]
    J(run(app, ["op", "apply", "tv", "-", "-m", "voz", "--json"], input=json.dumps(ops)))
    res = J(run(app, ["transpose", "tv", vsha[:12], "--to-key", "proyecto", "--from-key", "A minor",
                      "--clip", "v1", "--json"]))
    assert res["semitones"] == 3 and res["cached"] is False and res["revision"]
    show = json.loads(run(app, ["project", "show", "tv", "--json"]).stdout)
    clip = show["tracks"][1]["clips"][0]
    assert clip["asset"] == res["asset"] != vsha
    assert (clip["start_tick"], clip["source_offset"], clip["source_length"], clip["gain_db"]) == (960, 100, SR, -2)
    assert vsha in show["assets"] and res["asset"] in show["assets"]          # el original sigue registrado
    from osky_remix.core.config import project_dir
    from osky_remix.core.revisions import ProjectRepo
    repo = ProjectRepo(project_dir("tv"))
    y, sr = sf.read(str(repo.assets.find(res["asset"])), dtype="float32")
    assert abs(f0_median(y, sr) / (440 * 2 ** (3 / 12)) - 1) < 0.005          # C = 523.25 Hz
    # repetir: caché, sin asset nuevo
    res2 = J(run(app, ["transpose", "tv", vsha[:12], "--semitones", "3", "--json"]))
    assert res2["cached"] is True and res2["asset"] == res["asset"] and res2["revision"] is None
    # ya en la tonalidad: no transpone
    res3 = J(run(app, ["transpose", "tv", vsha[:12], "--to-key", "C minor", "--from-key", "Eb major", "--json"]))
    assert res3["semitones"] == 0 and res3["asset"] == vsha
    # errores claros
    assert run(app, ["transpose", "tv", vsha[:12], "--json"]).exit_code != 0
    assert run(app, ["transpose", "tv", vsha[:12], "--to-key", "proyecto", "--json"]).exit_code != 0
