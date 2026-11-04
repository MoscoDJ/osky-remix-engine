"""Etapa 4: grafo, automatización, envíos, estéreo, perfiles, A/B y paquete (sin Surge)."""
import json

import numpy as np
import pytest
import soundfile as sf

from osky_remix.audio.automation import curve
from osky_remix.audio.effects import apply_effect, stereo
from osky_remix.audio.export import export_package, verify_package
from osky_remix.audio.graph import build_graph
from osky_remix.audio.renderer import Renderer
from osky_remix.core.diff import diff_projects
from osky_remix.core.models import AssetKind, AutomationLane, Project, Provenance
from osky_remix.core.operations import Context, RegisterAsset
from osky_remix.core.transactions import TransactionError, apply_ops, parse_ops
from osky_remix.mastering.compare import compare
from osky_remix.mastering.dynamics import load_profile, master
from osky_remix.mastering.true_peak import true_peak_dbtp

SR = 44100
pytestmark = pytest.mark.audio
U = Context(author="user")


def _tone(tmp_path, repo, name="t.wav", secs=8.0, f=330.0):
    t = np.arange(int(secs * SR)) / SR
    x = (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)
    p = tmp_path / name
    sf.write(p, np.stack([x, x], 1), SR)
    return repo.assets.ingest(p, AssetKind.stem, Provenance(tool="test"))


def _base(tmp_path, repo, extra=()):
    a = _tone(tmp_path, repo)
    ops = [{"op": "set_tempo", "points": [{"tick": 0, "bpm": 120}]},
           {"op": "add_track", "id": "tone", "name": "T", "kind": "audio"},
           {"op": "add_clip", "track": "tone", "clip": {"type": "audio", "id": "c", "asset": a.sha256, "start_tick": 0,
                                                       "source_length": 8 * SR}}, *extra]
    return apply_ops(Project(name="m", sample_rate=SR), [RegisterAsset(asset=a)] + parse_ops(ops), U)


def test_routing_cycle_is_rejected_by_contract():
    p = apply_ops(Project(name="c"), parse_ops([
        {"op": "add_track", "id": "a", "name": "A", "kind": "bus"},
        {"op": "add_track", "id": "b", "name": "B", "kind": "bus", "output": "a"}]), U)
    with pytest.raises(TransactionError, match="circular"):
        apply_ops(p, parse_ops([{"op": "set_track", "track": "a", "output": "b"}]), U)
    order = build_graph(p).order
    assert order.index("b") < order.index("a") < order.index("master")


def test_automation_curve_linear_and_hold():
    p = Project(name="x", sample_rate=SR)
    tm = p.tempo_map()
    lane = AutomationLane(id="l", track="master", target="gain_db",
                          points=[{"tick": 960, "value": 0.0}, {"tick": 2 * 960, "value": 10.0, "curve": "hold"},
                                  {"tick": 3 * 960, "value": -5.0}])
    c = curve(lane, tm, tm.tick_to_sample(4 * 960))
    s1, s2, s3 = (tm.tick_to_sample(k * 960) for k in (1, 2, 3))
    assert c[0] == 0 and abs(c[(s1 + s2) // 2] - 5.0) < 0.01
    assert np.all(c[s2:s3] == 10.0) and c[-1] == -5.0


def test_gain_automation_and_pre_post_fader_sends(tmp_path, repo):
    p = _base(tmp_path, repo, [
        {"op": "add_track", "id": "pre", "name": "Pre", "kind": "return"},
        {"op": "add_track", "id": "post", "name": "Post", "kind": "return"},
        {"op": "add_send", "track": "tone", "send": {"target": "pre", "level_db": 0, "pre_fader": True}},
        {"op": "add_send", "track": "tone", "send": {"target": "post", "level_db": 0}},
        {"op": "set_automation", "lane": {"id": "g", "track": "tone", "target": "gain_db",
                                          "points": [{"tick": 0, "value": -40}, {"tick": 8 * 960, "value": 0}]}}])
    r = Renderer(p, repo.assets, repo.cache)
    r.output("master")
    tone, pre, post = r.output("tone").mean(1), r.output("pre").mean(1), r.output("post").mean(1)
    rms = lambda x, a, b: 20 * np.log10(np.sqrt(np.mean(x[int(a * SR):int(b * SR)] ** 2)) + 1e-12)
    assert rms(tone, 0, 0.5) < rms(tone, 3.5, 4.0) - 25          # sube de -40 a 0 dB en 4 s
    assert abs(rms(pre, 0, 0.5) - rms(pre, 3.5, 4.0)) < 0.5         # pre-fader: no sigue el fader
    assert np.allclose(post, tone, atol=1e-6)                       # post-fader: igual a la salida


def test_filter_automation_opens_spectrum(tmp_path, repo):
    t = np.arange(8 * SR) / SR
    noise = np.random.default_rng(0).standard_normal(len(t)).astype(np.float32) * 0.1
    f = tmp_path / "n.wav"
    sf.write(f, np.stack([noise, noise], 1), SR)
    a = repo.assets.ingest(f, AssetKind.stem, Provenance(tool="test"))
    ops = [{"op": "set_tempo", "points": [{"tick": 0, "bpm": 120}]},
           {"op": "add_track", "id": "n", "name": "N", "kind": "audio"},
           {"op": "add_plugin", "track": "n", "plugin": {"id": "lp", "backend": "builtin", "uri": "lowpass", "params": {"hz": 500}}},
           {"op": "add_clip", "track": "n", "clip": {"type": "audio", "id": "c", "asset": a.sha256, "start_tick": 0, "source_length": 8 * SR}},
           {"op": "set_automation", "lane": {"id": "a", "track": "n", "target": "lp.hz",
                                             "points": [{"tick": 0, "value": 300}, {"tick": 16 * 960, "value": 12000}]}}]
    p = apply_ops(Project(name="f", sample_rate=SR), [RegisterAsset(asset=a)] + parse_ops(ops), U)
    y = Renderer(p, repo.assets, repo.cache).output("n").mean(1)

    def hi_ratio(x):                                   # energía sobre 2 kHz / total
        s = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2; fr = np.fft.rfftfreq(len(x), 1 / SR)
        return s[fr > 2000].sum() / s.sum()
    assert hi_ratio(y[0:SR]) < 0.01                     # cerrado a ~300-1000 Hz (24 dB/oct)
    assert hi_ratio(y[6 * SR:7 * SR]) > 0.3             # abierto hacia el final


def test_unsupported_automation_target_warns_not_crashes(tmp_path, repo):
    p = _base(tmp_path, repo, [
        {"op": "add_plugin", "track": "tone", "plugin": {"id": "eq", "backend": "builtin", "uri": "eq", "params": {}}},
        {"op": "set_automation", "lane": {"id": "a", "track": "tone", "target": "eq.low_db",
                                          "points": [{"tick": 0, "value": 0}]}}])
    r = Renderer(p, repo.assets, repo.cache)
    r.output("master")
    assert any("no automatizables" in w for w in r.report.warnings)


def test_stereo_width_and_mono_bass():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((SR, 2)).astype(np.float32) * 0.1
    s2m = lambda y: 10 * np.log10(np.mean(((y[:, 0] - y[:, 1]) / 2) ** 2) / np.mean(((y[:, 0] + y[:, 1]) / 2) ** 2))
    assert abs(s2m(stereo(x, SR, 2.0)) - (s2m(x) + 6.02)) < 0.1
    assert np.allclose(stereo(x, SR, 0.0)[:, 0], stereo(x, SR, 0.0)[:, 1])
    y = stereo(x, SR, 1.0, mono_below_hz=150)
    side = (y[:, 0] - y[:, 1]) / 2
    w = np.hanning(len(side))
    sp = np.abs(np.fft.rfft(side * w)) ** 2; f = np.fft.rfftfreq(len(side), 1 / SR)
    low, ref = sp[(f > 5) & (f < 60)].mean(), sp[(f > 1000) & (f < 2000)].mean()
    assert 10 * np.log10(low / ref) < -40               # lado eliminado bajo 60 Hz


def test_mastering_profiles_from_files():
    prof = load_profile("melodic_techno_club")
    assert prof.target_lufs == -8.5 and prof.low_mono_hz == 120
    with pytest.raises(ValueError, match="desconocido"):
        load_profile("no-existe")
    rng = np.random.default_rng(2)
    t = np.arange(10 * SR) / SR
    kick = np.zeros_like(t)
    for k in range(0, len(t), SR // 2):
        n = min(int(0.3 * SR), len(t) - k)
        kick[k:k + n] = np.sin(2 * np.pi * 55 * t[:n]) * np.exp(-t[:n] * 8)
    mix = np.stack([kick + 0.05 * rng.standard_normal(len(t)), kick + 0.05 * rng.standard_normal(len(t))], 1)
    res = master(mix, SR, prof)
    assert true_peak_dbtp(res.audio, SR) <= prof.ceiling_dbtp + 1e-3
    assert abs(res.lufs - prof.target_lufs) <= 0.3             # el release corto ya no pone techo
    assert any("gain staging" in l for l in res.log) and any("mono" in l for l in res.log)


def test_ab_compare_and_diff(tmp_path, repo):
    p = _base(tmp_path, repo)
    q = apply_ops(p, parse_ops([{"op": "set_track", "track": "tone", "gain_db": -6}]), U)
    a = Renderer(p, repo.assets, repo.cache).output("master")
    b = Renderer(q, repo.assets, repo.cache).output("master")
    rep, bm = compare(a, b, SR, [("todo", 0, len(a))])
    assert abs(rep["b_gain_to_match_db"] - 6) < 0.05
    assert rep["null_residual_db"] < -60                        # a igual volumen son idénticas
    assert diff_projects(p, q) == ["tone.gain_db: 0 → -6"]


def test_export_package_is_complete_and_verifiable(tmp_path, repo):
    p = _base(tmp_path, repo, [
        {"op": "add_track", "id": "k", "name": "K", "kind": "instrument"},
        {"op": "add_plugin", "track": "k", "plugin": {"id": "d", "backend": "builtin", "uri": "drums"}},
        {"op": "add_clip", "track": "k", "clip": {"type": "midi", "id": "m", "start_tick": 0, "length_ticks": 3840,
                                                   "notes": [{"tick": i * 960, "duration": 240, "pitch": 36, "velocity": 110} for i in range(4)]}}])
    out = tmp_path / "pkg"
    res = export_package(p, repo.assets, repo.cache, out, "streaming", "rev-test")
    assert res["midi_notes"] == {"k": 4}
    import mido
    mf = mido.MidiFile(out / "midi" / "arrangement.mid")
    assert mf.ticks_per_beat == 960 and sum(1 for m in mf.tracks[1] if m.type == "note_on") == 4
    assert json.loads((out / "quality.json").read_text())["status"] in ("PASS", "WARN")
    assert verify_package(out) == []
    with open(out / "audio" / "stems" / "tone.wav", "r+b") as f:     # alterar un byte
        f.seek(2000); b = f.read(1); f.seek(2000); f.write(bytes([b[0] ^ 1]))
    assert any("alterado audio/stems/tone.wav" in x for x in verify_package(out))
    with pytest.raises(FileExistsError):
        export_package(p, repo.assets, repo.cache, out, "streaming")
