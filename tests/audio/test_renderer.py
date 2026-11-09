"""Renderizador con instrumentos integrados y verdad conocida (sin Surge)."""
import hashlib

import numpy as np
import pytest
import soundfile as sf

from osky_remix.audio.renderer import Renderer
from osky_remix.core.models import AssetKind, Project, Provenance
from osky_remix.core.operations import Context, RegisterAsset
from osky_remix.core.transactions import apply_ops, parse_ops
from osky_remix.mastering.true_peak import true_peak_dbtp

SR = 44100
pytestmark = pytest.mark.audio


def _project(tmp_path, repo):
    t = np.arange(SR) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    f = tmp_path / "tono.wav"
    sf.write(f, np.stack([tone, tone], 1), SR)
    a = repo.assets.ingest(f, AssetKind.stem, Provenance(tool="test"))
    ops = [
        {"op": "set_tempo", "points": [{"tick": 0, "bpm": 120}]},
        {"op": "add_track", "id": "kick", "name": "K", "kind": "instrument"},
        {"op": "add_plugin", "track": "kick", "plugin": {"id": "k", "backend": "builtin", "uri": "drums"}},
        {"op": "add_clip", "track": "kick", "clip": {"type": "midi", "id": "k1", "start_tick": 0, "length_ticks": 4 * 3840,
                                                      "notes": [{"tick": i * 960, "duration": 240, "pitch": 36, "velocity": 120} for i in range(16)]}},
        {"op": "add_track", "id": "bus", "name": "Bus", "kind": "bus"},
        {"op": "add_track", "id": "rev", "name": "Rev", "kind": "return"},
        {"op": "add_plugin", "track": "rev", "plugin": {"id": "r", "backend": "builtin", "uri": "reverb", "params": {"wet": 1.0}}},
        {"op": "add_track", "id": "pad", "name": "Pad", "kind": "instrument", "output": "bus"},
        {"op": "add_plugin", "track": "pad", "plugin": {"id": "s", "backend": "builtin", "uri": "synth"}},
        {"op": "add_clip", "track": "pad", "clip": {"type": "midi", "id": "p1", "start_tick": 0, "length_ticks": 4 * 3840,
                                                     "notes": [{"tick": 0, "duration": 4 * 3840 - 10, "pitch": 57, "velocity": 100}]}},
        {"op": "set_track", "track": "pad", "sidechain_from": "kick"},
        {"op": "add_plugin", "track": "pad", "plugin": {"id": "sc", "backend": "builtin", "uri": "sidechain", "params": {"depth": 0.8}}},
        {"op": "add_track", "id": "vox", "name": "Vox", "kind": "audio"},
        {"op": "add_clip", "track": "vox", "clip": {"type": "audio", "id": "v1", "asset": a.sha256, "start_tick": 3840,
                                                     "source_offset": 0, "source_length": SR // 2}},
    ]
    p = apply_ops(Project(name="r", sample_rate=SR), [RegisterAsset(asset=a)] + parse_ops(ops), Context(author="user"))
    return p


def test_render_positions_routing_sidechain_and_ceiling(tmp_path, repo):
    p = _project(tmp_path, repo)
    r = Renderer(p, repo.assets, repo.cache)
    rep = r.render(tmp_path / "out", "club", stems=True)
    kick = r.output("kick").mean(1)
    # ataques del bombo exactamente cada 0.5 s
    on = [i * SR // 2 for i in range(16)]
    assert np.abs(kick[:20]).max() > 0.1
    for o in on[1:]:
        assert np.abs(kick[o:o + 20]).max() > 10 * np.abs(kick[o - 60:o - 10]).max() + 1e-6
    vox = r.output("vox").mean(1)
    start = 2 * SR                                                  # compás 1 a 120 BPM = 2.0 s
    assert np.abs(vox[:start - 1]).max() == 0 and np.abs(vox[start:start + 100]).max() > 0
    assert np.abs(vox[start + SR // 2 + 1:]).max() == 0             # longitud exacta del clip
    pad = r.output("pad").mean(1)
    just_after = np.sqrt(np.mean(pad[SR // 2 + 200:SR // 2 + 2000] ** 2))
    before = np.sqrt(np.mean(pad[SR // 2 - 6000:SR // 2 - 200] ** 2))
    assert just_after < 0.5 * before                                 # el sidechain agacha tras el bombo
    assert np.allclose(r.output("bus"), r.output("pad"), atol=1e-6) # el bus recibe la pista
    m, _ = sf.read(rep.files["master"])
    assert true_peak_dbtp(m, SR) <= -1.0 + 0.05 and abs(rep.lufs + 9) <= 0.5
    assert not rep.warnings or all("silencio" in w for w in rep.warnings)


def test_render_is_reproducible(tmp_path, repo):
    p = _project(tmp_path, repo)
    h = []
    for i in (1, 2):
        rep = Renderer(p, repo.assets, repo.cache).render(tmp_path / f"o{i}", "club", stems=False)
        h.append(hashlib.sha256(open(rep.files["master"], "rb").read()).hexdigest())
    assert h[0] == h[1]


@pytest.mark.parametrize("solo", ["vox", "bus", "pad"])
def test_solo_routes_and_mute(tmp_path, repo, solo):
    p = _project(tmp_path, repo)
    p.track(solo).solo = True
    r = Renderer(p, repo.assets, repo.cache)
    r.output("master")
    assert not np.any(r.output("kick"))
    if solo == "vox":
        assert not np.any(r.output("pad"))
        assert np.any(r.output("vox"))
    else:
        assert np.any(r.output("pad"))
        assert np.allclose(r.output("bus"), r.output("pad"))
        assert not np.any(r.output("vox"))
    p.track(solo).mute = True
    muted = Renderer(p, repo.assets, repo.cache).output("master")
    assert not np.any(muted)


def test_multiple_solos_and_instrument_bypass(tmp_path, repo):
    p = _project(tmp_path, repo)
    p.track("vox").solo = p.track("kick").solo = True
    r = Renderer(p, repo.assets, repo.cache)
    r.output("master")
    assert np.any(r.output("vox")) and np.any(r.output("kick"))
    assert not np.any(r.output("pad"))
    p.track("kick").plugins[0].bypass = True
    assert not np.any(Renderer(p, repo.assets, repo.cache).output("kick"))


def test_solo_preserves_pre_and_post_sends(tmp_path, repo):
    p = _project(tmp_path, repo)
    p = apply_ops(p, parse_ops([
        {"op": "add_send", "track": "vox", "send": {"target": "rev", "pre_fader": True}},
        {"op": "set_track", "track": "vox", "solo": True}]), Context(author="user"))
    r = Renderer(p, repo.assets, repo.cache)
    r.output("master")
    assert np.any(r.output("rev"))
    assert not np.any(r.output("pad"))


@pytest.mark.parametrize("source_sr,target_sr", [(44100,48000), (48000,44100), (44100,44100)])
def test_source_sample_coordinates(tmp_path, repo, source_sr, target_sr):
    x = np.zeros((3 * source_sr, 2), np.float32)
    x[source_sr:2*source_sr] = 0.25
    f = tmp_path / "regions.wav"
    sf.write(f, x, source_sr, subtype="FLOAT")
    a = repo.assets.ingest(f, AssetKind.stem, Provenance(tool="test"))
    p = apply_ops(Project(name="rates", sample_rate=target_sr), [RegisterAsset(asset=a)] + parse_ops([
        {"op":"add_track", "id":"vox", "name":"V", "kind":"audio"},
        {"op":"add_clip", "track":"vox", "clip":{"type":"audio", "id":"v", "asset":a.sha256,
          "start_tick":960, "source_offset":source_sr, "source_length":source_sr}}]), Context(author="user"))
    r = Renderer(p, repo.assets, repo.cache)
    start = p.tempo_map().tick_to_sample(960)
    y = r.output("vox")
    assert np.max(np.abs(y[:start])) == 0
    assert np.allclose(y[start+100:start+target_sr-100], .25, atol=.001)
    assert not np.any(y[start+target_sr:])
    assert r.n == start + target_sr + 4 * target_sr



def test_silent_render_has_valid_wav_and_json(tmp_path, repo):
    import json
    r = Renderer(Project(name="empty", sample_rate=SR), repo.assets, repo.cache)
    report = r.render(tmp_path / "silent")
    x, sr = sf.read(report.files["master"])
    assert sr == SR and not np.any(x)
    assert report.mastering_status == "silent" and report.quality["status"] == "FAIL"
    json.dumps(report.__dict__, allow_nan=False)


def test_mastering_invalid_input_and_unmet_target():
    from osky_remix.mastering.dynamics import MasterProfile, master
    with pytest.raises(ValueError):
        MasterProfile("bad", float("nan"))
    with pytest.raises(ValueError):
        master(np.full((SR, 2), np.nan), SR, MasterProfile("test", -14))
    x = np.tile(.2 * np.sin(2*np.pi*440*np.arange(SR)/SR)[:, None], (1,2))
    result = master(x, SR, MasterProfile("impossible", 0, -20), max_iter=1)
    assert not result.converged and "loudness" in result.unmet
    assert np.isfinite(result.audio).all()


@pytest.mark.parametrize('source_sr,target_sr', [(44100,48000),(48000,44100)])
def test_impulse_offset_fades_and_tempo_change(tmp_path, repo, source_sr, target_sr):
    x = np.zeros((2*source_sr,2),np.float32)
    x[source_sr+source_sr//4] = .5
    f = tmp_path/'impulses.wav'
    sf.write(f,x,source_sr,subtype='FLOAT')
    a = repo.assets.ingest(f,AssetKind.stem,Provenance(tool='test'))
    p = apply_ops(Project(name='impulses',sample_rate=target_sr),[RegisterAsset(asset=a)]+parse_ops([
        {'op':'set_tempo','points':[{'tick':0,'bpm':120},{'tick':960,'bpm':60}]},
        {'op':'add_track','id':'v','name':'V','kind':'audio'},
        {'op':'add_clip','track':'v','clip':{'type':'audio','id':'c','asset':a.sha256,'start_tick':1920,
         'source_offset':source_sr,'source_length':source_sr,'fade_in':100,'fade_out':100}}
    ]),Context(author='user'))
    r = Renderer(p,repo.assets,repo.cache)
    y = r.output('v')[:,0]
    expected = round(1.75*target_sr)
    assert abs(int(np.argmax(np.abs(y)))-expected) <= 1
    assert r.n == round(1.5*target_sr)+target_sr+4*target_sr



def test_solo_keeps_sidechain_from_silent_source(tmp_path, repo):
    p = _project(tmp_path, repo)
    p.track("pad").solo = True                          # el bombo queda fuera del solo...
    r = Renderer(p, repo.assets, repo.cache)
    r.output("master")
    assert not np.any(r.output("kick"))
    pad = r.output("pad").mean(1)
    just_after = np.sqrt(np.mean(pad[SR // 2 + 200:SR // 2 + 2000] ** 2))
    before = np.sqrt(np.mean(pad[SR // 2 - 6000:SR // 2 - 200] ** 2))
    assert just_after < 0.5 * before                    # ...pero sigue disparando el sidechain


def test_solo_follows_nested_buses_and_returns(tmp_path, repo):
    p = _project(tmp_path, repo)
    p = apply_ops(p, parse_ops([
        {"op": "add_track", "id": "group", "name": "Group", "kind": "bus"},
        {"op": "set_track", "track": "bus", "output": "group"},
        {"op": "add_send", "track": "bus", "send": {"target": "rev", "level_db": -6}}]), Context(author="user"))
    p.track("pad").solo = True
    r = Renderer(p, repo.assets, repo.cache)
    master = r.output("master")
    assert np.any(master)
    for audible in ("pad", "bus", "group", "rev"):
        assert np.any(r.output(audible)), audible
    for silent in ("kick", "vox"):
        assert not np.any(r.output(silent)), silent
    p.track("pad").solo = False
    p.track("group").solo = True                        # solo de un bus: suenan sus fuentes
    r = Renderer(p, repo.assets, repo.cache)
    r.output("master")
    assert np.any(r.output("pad")) and not np.any(r.output("vox"))


def test_solo_on_return_plays_only_the_return(tmp_path, repo):
    p = _project(tmp_path, repo)
    p = apply_ops(p, parse_ops([
        {"op": "add_send", "track": "vox", "send": {"target": "rev"}}]), Context(author="user"))
    dry = Renderer(p, repo.assets, repo.cache)
    dry.output("master")
    p.track("rev").solo = True
    r = Renderer(p, repo.assets, repo.cache)
    master = r.output("master")
    assert np.any(r.output("vox"))                            # la voz se procesa para el envío...
    assert np.allclose(master, r.output("rev"), atol=1e-6)    # ...pero en el master solo está el retorno
    assert np.allclose(r.output("rev"), dry.output("rev"), atol=1e-6)
    assert not np.any(r.output("kick")) and not np.any(r.output("pad"))


def test_solo_on_return_fed_through_a_bus(tmp_path, repo):
    p = _project(tmp_path, repo)
    p = apply_ops(p, parse_ops([
        {"op": "add_send", "track": "bus", "send": {"target": "rev"}}]), Context(author="user"))
    dry = Renderer(p, repo.assets, repo.cache)
    dry.output("master")
    p.track("rev").solo = True
    r = Renderer(p, repo.assets, repo.cache)
    master = r.output("master")
    # pad -> bus -(envío)-> rev: el pad llega al bus, el bus alimenta el retorno, nada seco al master
    assert np.allclose(r.output("bus"), r.output("pad"), atol=1e-6)
    assert np.allclose(r.output("rev"), dry.output("rev"), atol=1e-6)
    assert np.allclose(master, r.output("rev"), atol=1e-6)


def test_solo_on_return_and_source_plays_both(tmp_path, repo):
    p = _project(tmp_path, repo)
    p = apply_ops(p, parse_ops([
        {"op": "add_send", "track": "vox", "send": {"target": "rev"}}]), Context(author="user"))
    p.track("rev").solo = p.track("vox").solo = True
    r = Renderer(p, repo.assets, repo.cache)
    master = r.output("master")
    assert np.allclose(master, r.output("vox") + r.output("rev"), atol=1e-6)


def test_solo_silences_sends_of_unsoloed_tracks_into_downstream_return(tmp_path, repo):
    p = _project(tmp_path, repo)
    p = apply_ops(p, parse_ops([
        {"op": "add_send", "track": "vox", "send": {"target": "rev"}},
        {"op": "add_send", "track": "kick", "send": {"target": "rev"}}]), Context(author="user"))
    p.track("vox").solo = True
    r = Renderer(p, repo.assets, repo.cache)
    r.output("master")
    assert not np.any(r.output("kick"))                       # el bombo no alimenta el reverb de la voz
    p.track("kick").sends = []
    ref = Renderer(p, repo.assets, repo.cache)
    ref.output("master")
    assert np.allclose(r.output("rev"), ref.output("rev"), atol=1e-6)


def test_feeder_reaches_only_the_soloed_return(tmp_path, repo):
    p = _project(tmp_path, repo)
    p = apply_ops(p, parse_ops([
        {"op": "add_track", "id": "dly", "name": "Delay", "kind": "return"},
        {"op": "set_track", "track": "rev", "output": "dly"},          # rev -> dly -> master
        {"op": "add_send", "track": "vox", "send": {"target": "rev"}},
        {"op": "add_send", "track": "vox", "send": {"target": "dly"}}]), Context(author="user"))
    p.track("rev").solo = True
    r = Renderer(p, repo.assets, repo.cache)
    r.output("master")
    # dly es audible (recibe al retorno en solo), pero el envío directo de la voz a dly calla
    assert np.allclose(r.output("dly"), r.output("rev"), atol=1e-6)

# ------------------------------------------------------------------------------- clipper suave
def test_soft_clip_linear_below_knee_and_bounded():
    from osky_remix.audio.effects import soft_clip
    sr = 44100
    t = np.arange(sr) / sr
    x = np.stack([0.2 * np.sin(2 * np.pi * 100 * t)] * 2, 1).astype(np.float32)
    assert np.array_equal(soft_clip(x, -6.0, 0.5), x)                          # 0.2 < 0.5·0.501: bit-exacto
    loud = np.stack([1.5 * np.sin(2 * np.pi * 100 * t)] * 2, 1)
    y = soft_clip(loud, -6.0, 0.5)
    ceil = 10 ** (-6 / 20)
    assert np.abs(y).max() <= ceil * 10 ** (0.3 / 20)                           # techo (+0.3 dB de filtro)
    assert np.abs(y).max() >= ceil * 0.97
    small = np.abs(loud[:, 0]) < 0.45 * ceil                                    # zona lineal intacta
    assert np.max(np.abs(y[small, 0] - loud[small, 0])) < 2e-3


def test_soft_clip_oversampling_reduces_aliasing():
    """Seno de 7 kHz recortado fuerte: sin sobremuestreo los armónicos 3.º y 5.º (21 y 35 kHz) se
    reflejan en 23.1 y 9.1 kHz; con 4x quedan muy por debajo."""
    from osky_remix.audio.effects import soft_clip
    sr = 44100
    t = np.arange(sr) / sr
    x = np.stack([2.0 * np.sin(2 * np.pi * 7000 * t)] * 2, 1)
    def alias_db(y):
        spec = np.abs(np.fft.rfft(y[:, 0] * np.hanning(len(y)))) ** 2
        f = np.fft.rfftfreq(len(y), 1 / sr)
        alias = spec[(np.abs(f - 9100) < 200)].sum()                            # 5.º armónico reflejado
        return 10 * np.log10(alias / spec[np.abs(f - 7000) < 200].sum())
    assert alias_db(soft_clip(x, -1.0, 0.3, oversample=1)) > -40
    assert alias_db(soft_clip(x, -1.0, 0.3, oversample=4)) < alias_db(soft_clip(x, -1.0, 0.3, oversample=1)) - 20
