"""Biblioteca de samples: rasgos contra verdad sintética, categorías, búsqueda, caché y sampler."""
import json

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfilt
from typer.testing import CliRunner

from osky_remix.audio import synth
from osky_remix.audio.renderer import Renderer
from osky_remix.cli.main import app
from osky_remix.core.models import AssetKind, Project, Provenance
from osky_remix.core.revisions import ProjectRepo
from osky_remix.core.transactions import apply_ops, parse_ops
from osky_remix.core.operations import Context
from osky_remix.samples import features as F
from osky_remix.samples.features import extract
from osky_remix.samples.kit import kit_bytes
from osky_remix.samples.library import SampleLibrary

pytestmark = pytest.mark.audio
SR = 44100
run = CliRunner().invoke


def kick(f=50.0, tau=0.08, dur=0.8):
    t = np.arange(int(dur * SR)) / SR
    return 0.9 * np.sin(2 * np.pi * f * t) * np.exp(-t / tau)


def hat(tau=0.02, seed=1):
    n = int(0.15 * SR); t = np.arange(n) / SR
    x = sosfilt(butter(4, 8000, "high", fs=SR, output="sos"), np.random.default_rng(seed).standard_normal(n))
    return 0.5 * x * np.exp(-t / tau)


def snare():
    n = int(0.4 * SR); t = np.arange(n) / SR
    noise = sosfilt(butter(2, [1200, 5000], "band", fs=SR, output="sos"), np.random.default_rng(2).standard_normal(n))
    return 0.5 * (noise * np.exp(-t / 0.05) + 0.4 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.03))


def pluck():
    """La 4 (440 Hz) con 5 armónicos, ataque lineal de 10 ms y caída exponencial τ = 0.2 s."""
    t = np.arange(int(1.5 * SR)) / SR
    tone = sum(np.sin(2 * np.pi * 440 * k * t) / k for k in range(1, 6))
    return 0.4 * tone * np.minimum(1, t / 0.01) * np.exp(-t / 0.2)


def loop(bpm=124.0, bars=2):
    spb = 60 / bpm * SR
    n = int(round(bars * 4 * spb))
    y = np.zeros(n)
    k, h = kick(), hat()
    for b in range(bars * 4):
        for x, off in ((k, 0.0), (h, 0.5)):
            a = int(round((b + off) * spb))
            m = min(len(x), n - a)
            y[a:a + m] += x[:m]
    return y


GT = {"kick": kick, "kick2": lambda: kick(55.0, 0.07), "snare": snare, "hat": hat, "hat2": lambda: hat(0.025, 7),
      "pluck": pluck, "loop": loop}


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "pack"
    (d / "sub").mkdir(parents=True)
    for name, fn in GT.items():
        ext, sub = (".flac", "sub/") if name == "hat2" else (".wav", "")
        sf.write(d / f"{sub}{name}{ext}", fn().astype(np.float32), SR, subtype="FLOAT" if ext == ".wav" else "PCM_24")
    (d / "notas.txt").write_text("no es audio")
    return d


def test_features_match_ground_truth():
    fk, fs, fh, fp, fl = (extract(GT[k](), SR) for k in ("kick", "snare", "hat", "pluck", "loop"))
    assert fk["centroid_hz"] < fs["centroid_hz"] < fh["centroid_hz"]
    assert fk["brightness"] < fs["brightness"] < fh["brightness"]
    # bombo: 50 Hz, cola a −40 dB en τ·ln(100) = 0.368 s, ataque instantáneo
    assert abs(fk["f0_hz"] - 50) < 0.5 and fk["low_ratio"] > 0.95
    assert abs(fk["decay_s"] - 0.08 * np.log(100)) < 0.01 and fk["attack_s"] < 0.005
    # pluck: f0 440 → A4; ataque lineal 10 ms (10–90 % ≈ 7.9 ms); cola τ·ln(100) desde el pico
    assert abs(fp["f0_hz"] - 440) < 1.0 and fp["note"] == "A4" and fp["midi"] == 69
    assert abs(fp["attack_s"] - 0.0079) < 0.0015
    assert abs(fp["decay_s"] - 0.2 * np.log(100)) < 0.02
    assert fh["f0_hz"] is None and fh["note"] is None and abs(fh["decay_s"] - 0.02 * np.log(100)) < 0.02
    assert fl["is_loop"] and abs(fl["bpm"] - 124) <= 0.5 and fl["loop_bars"] == 2
    assert not any(f["is_loop"] for f in (fk, fs, fh, fp))
    assert abs(fk["duration_s"] - 0.8) < 1e-6 and abs(fk["peak_dbfs"] - 20 * np.log10(np.abs(kick()).max())) < 0.01


def test_categories():
    cats = {k: extract(fn(), SR)["category"] for k, fn in GT.items()}
    assert cats == {"kick": "kick", "kick2": "kick", "snare": "snare", "hat": "hat", "hat2": "hat",
                    "pluck": "tonal", "loop": "loop"}
    # la batería integrada del motor también se reconoce
    kit = {36: "kick", 38: "snare", 39: "clap", 42: "hat", 46: "hat"}
    for p, cat in kit.items():
        f = extract(synth.drum_hit(p, 110, SR, 0), SR)
        assert f["category"] == cat, (p, f["category_scores"])
        assert 0.5 <= f["category_conf"] <= 1


def test_import_search_and_cache(tmp_path, folder, monkeypatch):
    lib = SampleLibrary(tmp_path / "lib")
    rep = lib.import_dir(folder)
    assert len(rep.added) == len(GT) and rep.extracted == len(GT) and rep.cache_hits == 0 and not rep.failed
    by = {h.name.rsplit(".", 1)[0]: h for h in lib.all()}
    assert set(by) == set(GT)
    for h in by.values():                          # SHA-256 del archivo y almacén de solo lectura
        assert lib.path(h.sha).stat().st_mode & 0o222 == 0
    # re-importar: todo conocido, rasgos de caché, sin recalcular ni duplicar
    calls = []
    real = F.extract_file
    monkeypatch.setattr(F, "extract_file", lambda p: calls.append(p) or real(p))
    (folder / "copia_del_bombo.wav").write_bytes((folder / "kick.wav").read_bytes())
    rep2 = lib.import_dir(folder)
    assert not rep2.added and len(rep2.known) == len(GT) + 1 and rep2.cache_hits == len(GT) + 1
    assert rep2.extracted == 0 and calls == [] and len(lib.all()) == len(GT)
    names = lambda hits: sorted(h.name.rsplit(".", 1)[0] for h in hits)
    assert names(lib.search(cat="kick")) == ["kick", "kick2"]
    assert names(lib.search(cat="hat", max_decay=0.1)) == ["hat"]
    assert names(lib.search(bright="0.6-0.9", loop=False)) == ["snare"]
    assert names(lib.search(note="A4")) == ["pluck"]
    assert names(lib.search(note="G")) == ["kick"]                   # 50 Hz ≈ Sol 1
    assert names(lib.search(bpm="123.5-124.5")) == ["loop"]
    near = lib.search(like=by["kick"].sha[:10])
    assert near[0].name.startswith("kick2") and by["kick"].sha not in [h.sha for h in near]
    assert [h.distance for h in near] == sorted(h.distance for h in near)
    assert lib.search(like=by["hat"].sha)[0].name.startswith("hat2")
    assert names(lib.search(cat="hat", like=by["kick"].sha)) == ["hat", "hat2"]
    with pytest.raises(ValueError):
        lib.search(cat="guitarra")


def test_feature_cache_key_tracks_extractor_version(tmp_path, folder, monkeypatch):
    lib = SampleLibrary(tmp_path / "lib")
    lib.import_dir(folder)
    monkeypatch.setattr(F, "FEATURES_VERSION", "test-nueva")
    rep = lib.import_dir(folder)
    assert rep.extracted == len(GT) and rep.cache_hits == 0 and not rep.added


def _sampler_project(repo, sample_sha, kit_sha, uri="sampler", notes=None):
    notes = notes or [{"tick": t, "duration": 120, "pitch": 36, "velocity": v} for t, v in
                      ((0, 127), (960 + 7, 64), (3 * 960 + 333, 100), (7 * 960 + 1, 1))]
    ops = [{"op": "set_tempo", "points": [{"tick": 0, "bpm": 123.7}]},
           {"op": "add_track", "id": "smp", "name": "S", "kind": "instrument"},
           {"op": "add_plugin", "track": "smp", "plugin": {"id": "k", "backend": "builtin", "uri": uri, "preset": kit_sha}},
           {"op": "add_clip", "track": "smp", "clip": {"type": "midi", "id": "c", "start_tick": 960, "length_ticks": 8 * 960,
                                                       "notes": notes}}]
    p = repo.load()
    return apply_ops(p, parse_ops(ops), Context(author="user")), notes


def _kit_assets(repo, tmp_path, x, voices_pitch=36, gain_db=0.0):
    f = tmp_path / "s.wav"
    sf.write(f, np.stack([x, -0.5 * x], 1).astype(np.float32), SR, subtype="FLOAT")
    a = repo.assets.ingest(f, AssetKind.sample, Provenance(tool="test"))
    k = tmp_path / "kit.json"
    k.write_bytes(kit_bytes({voices_pitch: {"sample": a.sha256, "gain_db": gain_db}}))
    ka = repo.assets.ingest(k, AssetKind.preset, Provenance(tool="test"))
    repo.commit([{"op": "register_asset", "asset": a.model_dump(mode="json")},
                 {"op": "register_asset", "asset": ka.model_dump(mode="json")}], "user", "kit")
    return a, ka


def test_sampler_renders_sample_accurately(tmp_path):
    repo = ProjectRepo.create(tmp_path / "p", Project(name="s", sample_rate=SR))
    x = kick(tau=0.05, dur=0.3)
    a, ka = _kit_assets(repo, tmp_path, x, gain_db=-6.0)
    p, notes = _sampler_project(repo, a.sha256, ka.sha256)
    r = Renderer(p, repo.assets, repo.cache)
    r.validate_capabilities()
    r.output("smp")
    y = r.pre["smp"]
    tm = p.tempo_map()
    exp = np.zeros_like(y)
    src = np.stack([x, -0.5 * x], 1).astype(np.float32)
    starts = []
    for nt in notes:
        s = tm.tick_to_sample(960 + nt["tick"])
        starts.append(s)
        g = np.float32(nt["velocity"] / 127 * 10 ** (-6 / 20))
        exp[s:s + len(src)] += src * g
    assert np.allclose(y, exp, atol=1e-6)
    # el primer valor no nulo cae exactamente en la muestra de la nota
    assert np.flatnonzero(np.abs(y[:, 0]) > 0)[0] == starts[0] + np.flatnonzero(np.abs(x) > 0)[0]


def test_drum_kit_mapping_replaces_only_mapped_voice(tmp_path):
    repo = ProjectRepo.create(tmp_path / "p", Project(name="d", sample_rate=SR))
    x = kick(tau=0.05, dur=0.3)
    a, ka = _kit_assets(repo, tmp_path, x)
    notes = [{"tick": 0, "duration": 120, "pitch": 36, "velocity": 100},
             {"tick": 480, "duration": 120, "pitch": 42, "velocity": 90}]
    p, _ = _sampler_project(repo, a.sha256, ka.sha256, uri="drums", notes=notes)
    plain = apply_ops(p, parse_ops([{"op": "replace_plugin", "track": "smp", "plugin_id": "k",
                                     "plugin": {"id": "k", "backend": "builtin", "uri": "drums"}},
                                    {"op": "set_midi_notes", "clip": "c", "notes": notes[1:]}]),
                      Context(author="user"))
    r, r0 = Renderer(p, repo.assets, repo.cache), Renderer(plain, repo.assets, repo.cache)
    r.output("smp"); r0.output("smp")
    s = p.tempo_map().tick_to_sample(960)
    src = np.stack([x, -0.5 * x], 1).astype(np.float32) * np.float32(100 / 127)
    exp = r0.pre["smp"].copy()
    exp[s:s + len(src)] += src
    assert np.allclose(r.pre["smp"], exp, atol=1e-6)


def test_kit_with_unregistered_sample_is_rejected(tmp_path):
    repo = ProjectRepo.create(tmp_path / "p", Project(name="u", sample_rate=SR))
    f = tmp_path / "s.wav"
    sf.write(f, kick().astype(np.float32), SR)
    a = repo.assets.ingest(f, AssetKind.sample, Provenance(tool="test"))       # en el almacén, no registrado
    k = tmp_path / "kit.json"
    k.write_bytes(kit_bytes({36: {"sample": a.sha256}}))
    ka = repo.assets.ingest(k, AssetKind.preset, Provenance(tool="test"))
    repo.commit([{"op": "register_asset", "asset": ka.model_dump(mode="json")}], "user", "kit")
    p, _ = _sampler_project(repo, a.sha256, ka.sha256)
    with pytest.raises(ValueError, match="no está registrado"):
        Renderer(p, repo.assets, repo.cache).validate_capabilities()


def J(res):
    assert res.exit_code == 0, res.output
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_cli_import_search_use(tmp_path, folder, monkeypatch):
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    imp = J(run(app, ["samples", "import", str(folder), "--json"]))
    assert len(imp["added"]) == len(GT) and imp["total"] == len(GT)
    again = J(run(app, ["samples", "import", str(folder), "--json"]))
    assert again["added"] == [] and again["cache_hits"] == len(GT) and again["extracted"] == 0
    res = J(run(app, ["samples", "search", "--cat", "kick", "--max-decay", "0.4", "--json"]))["results"]
    assert sorted(r["name"] for r in res) == ["kick.wav", "kick2.wav"]
    ksha = next(r["sha"] for r in res if r["name"] == "kick.wav")
    near = J(run(app, ["samples", "search", "--like", ksha[:12], "--limit", "1", "--json"]))["results"]
    assert near[0]["name"] == "kick2.wav"
    J(run(app, ["project", "new", "kit", "--sr", str(SR), "--json"]))
    use = J(run(app, ["samples", "use", "kit", ksha[:12], "--track", "perc", "--create", "--json"]))
    assert use["pitch"] == 36 and use["sample"] == ksha                    # mismo contenido, mismo sha
    hsha = J(run(app, ["samples", "search", "--cat", "hat", "--json"]))["results"][0]["sha"]
    use2 = J(run(app, ["samples", "use", "kit", hsha, "--track", "perc", "--json"]))
    assert use2["pitch"] == 42
    show = json.loads(run(app, ["project", "show", "kit", "--json"]).stdout)
    assert ksha in show["assets"] and hsha in show["assets"] and use2["kit"] in show["assets"]
    slot = next(t for t in show["tracks"] if t["id"] == "perc")["plugins"][0]
    assert slot["uri"] == "sampler" and slot["preset"] == use2["kit"]
    repo = ProjectRepo(tmp_path / "ws" / "projects" / "kit")
    from osky_remix.samples.kit import load_kit
    kit = load_kit(repo.assets.find(use2["kit"]))
    assert {p: v["sample"] for p, v in kit.items()} == {36: ksha, 42: hsha}
    Renderer(repo.load(), repo.assets, repo.cache).validate_capabilities()
    bad = run(app, ["samples", "use", "kit", hsha, "--track", "nada", "--json"])
    assert bad.exit_code != 0
