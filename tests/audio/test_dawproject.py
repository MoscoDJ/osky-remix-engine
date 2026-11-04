"""Exportación DAWproject: se lee el XML de vuelta y se comparan valores exactos conocidos."""
import hashlib
import io
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from osky_remix.audio.dawproject import SURGE_XT_VST3_CID, export_dawproject, vst3_preset, vst3_preset_component
from osky_remix.audio.renderer import Renderer
from osky_remix.core.models import AssetKind, Project, Provenance
from osky_remix.core.operations import Context, RegisterAsset
from osky_remix.core.transactions import apply_ops, parse_ops

SR = 48000
SR_SRC = 44100                    # asset remuestreado: los segundos del clip se cuentan a 44.1 kHz
XSD = Path(__file__).parents[1] / "fixtures" / "dawproject"
pytestmark = pytest.mark.audio


def _asset(repo, tmp_path, name, sr, seconds, channels):
    t = np.arange(int(sr * seconds)) / sr
    x = (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    f = tmp_path / name
    sf.write(f, np.stack([x] * channels, 1) if channels > 1 else x, sr)
    return repo.assets.ingest(f, AssetKind.stem, Provenance(tool="test"))


def _project(tmp_path, repo):
    a = _asset(repo, tmp_path, "voz441.wav", SR_SRC, 3.0, 2)        # 132300 frames a 44.1 kHz
    b = _asset(repo, tmp_path, "perc48.wav", SR, 1.0, 1)            # mono, a la frecuencia del proyecto
    notes = [{"tick": 0, "duration": 960, "pitch": 60, "velocity": 127},
             {"tick": 480, "duration": 240, "pitch": 64, "velocity": 100}]
    ops = [
        {"op": "set_tempo", "points": [{"tick": 0, "bpm": 120}, {"tick": 16 * 960, "bpm": 90}]},
        {"op": "set_meter", "points": [{"bar": 0, "numerator": 4}, {"bar": 8, "numerator": 3, "denominator": 4}]},
        {"op": "add_track", "id": "bus", "name": "Bus", "kind": "bus"},
        {"op": "add_track", "id": "rev", "name": "Rev", "kind": "return"},
        {"op": "add_plugin", "track": "rev", "plugin": {"id": "r", "backend": "builtin", "uri": "reverb", "params": {"wet": 1.0}}},
        {"op": "add_track", "id": "vox", "name": "Voz", "kind": "audio"},
        {"op": "add_clip", "track": "vox", "clip": {"type": "audio", "id": "v1", "asset": a.sha256, "start_tick": 4 * 960,
                                                     "source_offset": SR_SRC // 2, "source_length": SR_SRC,
                                                     "fade_in": 4410, "fade_out": 2205, "gain_db": -3.0}},
        {"op": "add_clip", "track": "vox", "clip": {"type": "audio", "id": "v2", "asset": a.sha256, "start_tick": 14 * 960,
                                                     "source_offset": 0, "source_length": 3 * SR_SRC}},
        {"op": "set_track", "track": "vox", "gain_db": -6.0, "pan": -0.5},
        {"op": "add_send", "track": "vox", "send": {"target": "rev", "level_db": -6.0}},
        {"op": "add_track", "id": "perc", "name": "Perc", "kind": "audio", "output": "bus"},
        {"op": "add_clip", "track": "perc", "clip": {"type": "audio", "id": "p1", "asset": b.sha256, "start_tick": 0,
                                                      "source_offset": 4800, "source_length": 24000}},
        {"op": "add_track", "id": "lead", "name": "Lead", "kind": "instrument", "output": "bus"},
        {"op": "add_plugin", "track": "lead", "plugin": {"id": "s", "backend": "builtin", "uri": "synth"}},
        {"op": "add_plugin", "track": "lead", "plugin": {"id": "eq", "backend": "builtin", "uri": "eq", "params": {"low_db": -3}}},
        {"op": "add_clip", "track": "lead", "clip": {"type": "midi", "id": "m1", "start_tick": 1920, "length_ticks": 3840,
                                                      "notes": notes}},
        {"op": "add_send", "track": "lead", "send": {"target": "rev", "level_db": -12.0, "pre_fader": True}},
        {"op": "set_automation", "lane": {"id": "vg", "track": "vox", "target": "gain_db", "points": [
            {"tick": 0, "value": 0.0}, {"tick": 3840, "value": -6.0, "curve": "hold"}, {"tick": 7680, "value": -6.0}]}},
        {"op": "set_automation", "lane": {"id": "lp", "track": "lead", "target": "pan", "points": [
            {"tick": 0, "value": -1.0}, {"tick": 1920, "value": 1.0, "curve": "hold"}]}},
        {"op": "set_automation", "lane": {"id": "rw", "track": "rev", "target": "r.wet", "points": [
            {"tick": 0, "value": 0.5}]}},
        {"op": "add_marker", "marker": {"id": "intro", "tick": 0, "label": "Intro"}},
        {"op": "add_marker", "marker": {"id": "drop", "tick": 2 * 3840, "label": "Drop", "kind": "cue"}},
    ]
    p = apply_ops(Project(name="dp", sample_rate=SR), [RegisterAsset(asset=a), RegisterAsset(asset=b)] + parse_ops(ops),
                  Context(author="user"))
    return p, a, b


def _read(path):
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read("project.xml"))
    ids = {e.get("id"): e for e in root.iter() if e.get("id")}
    return z, root, ids


def _f(e, k):
    return float(e.get(k))


def _clip(root, name):
    return next(c for c in root.iter("Clip") if c.get("name") == name)


def _xsd_ok(xml_bytes, xsd, tmp_path):
    f = tmp_path / "check.xml"
    f.write_bytes(xml_bytes)
    r = subprocess.run(["xmllint", "--noout", "--schema", str(xsd), str(f)], capture_output=True, text=True)
    return r.returncode, r.stderr


@pytest.fixture
def exported(tmp_path, repo):
    p, a, b = _project(tmp_path, repo)
    out = tmp_path / "dp.dawproject"
    res = export_dawproject(p, repo.assets, repo.cache, out, bounce_instruments=False, revision="abc")
    return p, a, b, out, res


def test_transport_structure_and_routing(exported, tmp_path):
    p, a, b, out, res = exported
    z, root, ids = _read(out)
    assert root.tag == "Project" and root.get("version") == "1.0"
    tempo = root.find("Transport/Tempo")
    assert tempo.get("unit") == "bpm" and _f(tempo, "value") == 120.0
    ts = root.find("Transport/TimeSignature")
    assert (ts.get("numerator"), ts.get("denominator")) == ("4", "4")
    ta = root.find("Arrangement/TempoAutomation")
    assert ta.find("Target").get("parameter") == "tempo"
    assert [(_f(x, "time"), _f(x, "value"), x.get("interpolation")) for x in ta.findall("RealPoint")] == \
        [(0.0, 120.0, "hold"), (16.0, 90.0, "hold")]
    sa = root.find("Arrangement/TimeSignatureAutomation")
    # compás 8 en 4/4 = 32 negras
    assert [(_f(x, "time"), x.get("numerator"), x.get("denominator")) for x in sa.findall("TimeSignaturePoint")] == \
        [(0.0, "4", "4"), (32.0, "3", "4")]

    st = root.find("Structure")
    assert [t.get("id") for t in st.findall("Track")] == ["trk-bus", "trk-rev", "trk-vox", "trk-master"]
    bus = ids["trk-bus"]
    assert bus.get("contentType") == "tracks" and bus.find("Channel").get("role") == "submix"
    assert [t.get("id") for t in bus.findall("Track")] == ["trk-perc", "trk-lead"]       # pistas que salen al bus
    assert ids["ch-perc"].get("destination") == "ch-bus" and ids["ch-lead"].get("destination") == "ch-bus"
    assert ids["ch-bus"].get("destination") == "ch-master" and ids["ch-master"].get("destination") is None
    assert ids["ch-rev"].get("role") == "effect" and ids["ch-master"].get("role") == "master"
    assert ids["trk-lead"].get("contentType") == "notes" and ids["trk-vox"].get("contentType") == "audio"
    vox = ids["ch-vox"]
    assert _f(vox.find("Volume"), "value") == pytest.approx(10 ** (-6 / 20), abs=1e-12)
    assert vox.find("Volume").get("unit") == "linear"
    assert _f(vox.find("Pan"), "value") == 0.25 and vox.find("Pan").get("unit") == "normalized"
    assert vox.find("Mute").get("value") == "false"
    s = vox.find("Sends/Send")
    assert s.get("destination") == "ch-rev" and s.get("type") == "post"
    assert _f(s.find("Volume"), "value") == pytest.approx(10 ** (-6 / 20), abs=1e-12)
    s2 = ids["ch-lead"].find("Sends/Send")
    assert s2.get("type") == "pre" and _f(s2.find("Volume"), "value") == pytest.approx(10 ** (-12 / 20), abs=1e-12)
    # Todo IDREF apunta a un id existente
    for e in root.iter():
        for k in ("destination", "track", "parameter"):
            if e.get(k):
                assert e.get(k) in ids, (e.tag, k, e.get(k))


def test_notes_markers_and_automation(exported):
    p, a, b, out, res = exported
    z, root, ids = _read(out)
    m1 = _clip(root, "m1")
    assert (_f(m1, "time"), _f(m1, "duration")) == (2.0, 4.0)                       # tick 1920 = negra 2
    notes = [(_f(n, "time"), _f(n, "duration"), n.get("key"), _f(n, "vel")) for n in m1.find("Notes")]
    assert notes == [(0.0, 1.0, "60", 1.0), (0.5, 0.25, "64", 100 / 127)]
    mk = root.find("Arrangement/Markers")
    assert [(_f(m, "time"), m.get("name")) for m in mk] == [(0.0, "Intro"), (8.0, "Drop")]
    lane = ids["lanes-vox"]
    assert lane.get("track") == "trk-vox"
    vol = ids["auto-vol-vox"]
    assert vol.find("Target").get("parameter") == "vol-vox" and vol.get("unit") == "linear"
    pts = [(_f(x, "time"), _f(x, "value"), x.get("interpolation")) for x in vol.findall("RealPoint")]
    # 0 → -6 dB lineal en dB: 6 pasos de 1 dB entre las negras 0 y 4; luego hold y el último punto
    assert [t for t, _, _ in pts] == [0.0, 4 / 6, 8 / 6, 2.0, 16 / 6, 20 / 6, 4.0, 8.0]
    assert [v for _, v, _ in pts] == pytest.approx([10 ** (-k / 20) for k in range(7)] + [10 ** (-6 / 20)], abs=1e-12)
    assert [i for _, _, i in pts] == ["linear"] * 6 + ["hold", "linear"]
    pan = ids["auto-pan-lead"]
    assert pan.find("Target").get("parameter") == "pan-lead"
    assert [(_f(x, "time"), _f(x, "value"), x.get("interpolation")) for x in pan.findall("RealPoint")] == \
        [(0.0, 0.0, "linear"), (2.0, 1.0, "hold")]
    man = json.loads(z.read("osky/manifest.json"))
    assert man["tracks"]["rev"]["automation_not_mapped"] == ["r.wet"]
    assert [e["uri"] for e in man["tracks"]["rev"]["effects_not_mapped"]] == ["reverb"]
    assert [e["uri"] for e in man["tracks"]["lead"]["effects_not_mapped"]] == ["eq"]
    assert man["tracks"]["vox"]["clip_gain_db_not_mapped"] == {"v1": -3.0}
    assert man["tracks"]["lead"]["instrument"]["uri"] == "synth"
    assert man["master_profile_not_mapped"] == p.master_profile
    assert "eq" in ids["trk-lead"].get("comment") and "reverb" in ids["trk-rev"].get("comment")
    assert any("synth" in w for w in res["warnings"])                   # sin bounce no sonará en el DAW


def test_audio_clips_offsets_resampled_and_tempo_change(exported):
    p, a, b, out, res = exported
    z, root, ids = _read(out)
    # v1: asset a 44.1 kHz; empieza en la negra 4 (2.0 s a 120 BPM); offset 0.5 s; dura 1.0 s
    v1 = _clip(root, "v1")
    assert (_f(v1, "time"), _f(v1, "duration"), _f(v1, "playStart"), _f(v1, "playStop")) == (4.0, 2.0, 1.0, 3.0)
    assert v1.get("contentTimeUnit") == "beats" and v1.get("fadeTimeUnit") == "seconds"
    assert (_f(v1, "fadeInTime"), _f(v1, "fadeOutTime")) == (0.1, 0.05)
    assert "-3" in v1.get("comment")
    w = v1.find("Warps")
    assert w.get("contentTimeUnit") == "seconds" and w.get("timeUnit") == "beats"
    au = w.find("Audio")
    assert (au.get("sampleRate"), au.get("channels"), _f(au, "duration")) == ("44100", "2", 3.0)
    assert [(_f(x, "time"), _f(x, "contentTime")) for x in w.findall("Warp")] == [(0.0, 0.0), (6.0, 3.0)]
    # v2: cruza el cambio de tempo (negra 16 = 8.0 s): 1 s a 120 + 2 s a 90 = 2 + 3 negras
    v2 = _clip(root, "v2")
    assert (_f(v2, "time"), _f(v2, "duration"), _f(v2, "playStart"), _f(v2, "playStop")) == (14.0, 5.0, 0.0, 5.0)
    assert [(_f(x, "time"), _f(x, "contentTime")) for x in v2.find("Warps").findall("Warp")] == \
        [(0.0, 0.0), (2.0, 1.0), (5.0, 3.0)]
    # p1: asset a 48 kHz mono, offset 0.1 s en tick 0: el archivo empieza 0.2 negras antes del arreglo
    p1 = _clip(root, "p1")
    assert (_f(p1, "time"), _f(p1, "duration"), _f(p1, "playStart"), _f(p1, "playStop")) == \
        pytest.approx((0.0, 1.0, 0.2, 1.2), abs=1e-12)
    assert p1.find("Warps/Audio").get("channels") == "1"
    # los archivos referenciados están en el zip con el mismo SHA-256 que el asset
    paths = {c.get("name"): c.find("Warps/Audio/File").get("path") for c in (v1, v2, p1)}
    assert paths["v1"] == paths["v2"] and paths["v1"].startswith("audio/")
    for sha, rel in ((a.sha256, paths["v1"]), (b.sha256, paths["p1"])):
        assert hashlib.sha256(z.read(rel)).hexdigest() == sha
    man = json.loads(z.read("osky/manifest.json"))
    for name in z.namelist():
        if name != "osky/manifest.json":
            assert man["files"][name] == hashlib.sha256(z.read(name)).hexdigest()
    assert "metadata.xml" in z.namelist() and ET.fromstring(z.read("metadata.xml")).find("Title").text == "dp"


@pytest.mark.skipif(shutil.which("xmllint") is None, reason="xmllint no disponible")
def test_xml_validates_against_official_xsd(exported, tmp_path):
    p, a, b, out, res = exported
    z = zipfile.ZipFile(out)
    assert _xsd_ok(z.read("project.xml"), XSD / "Project.xsd", tmp_path) == (0, f"{tmp_path / 'check.xml'} validates\n")
    assert _xsd_ok(z.read("metadata.xml"), XSD / "MetaData.xsd", tmp_path)[0] == 0
    # control: el validador rechaza un documento inválido (atributo requerido ausente)
    bad = z.read("project.xml").replace(b'<Application name=', b'<Application nombre=', 1)
    assert _xsd_ok(bad, XSD / "Project.xsd", tmp_path)[0] != 0


def test_bounce_instrument_prefader(tmp_path, repo):
    p, a, b = _project(tmp_path, repo)
    out = tmp_path / "b.dawproject"
    res = export_dawproject(p, repo.assets, repo.cache, out, bounce_instruments=True)
    assert res["bounced"] == ["lead"]
    z, root, ids = _read(out)
    y, sr = sf.read(io.BytesIO(z.read("audio/bounce/lead.wav")), dtype="float32", always_2d=True)
    r = Renderer(p, repo.assets, repo.cache)
    r.output("lead")
    assert sr == SR and np.array_equal(y, r.pre["lead"])                  # pre-fader exacto
    on = SR                                                               # negra 2 a 120 BPM = 1.0 s
    assert np.abs(y[:on]).max() == 0 and np.abs(y[on:on + 2000]).max() > 0.01
    # la pista bounce está junto a la original (dentro del bus), con el mismo canal; la MIDI queda en mute
    bus = ids["trk-bus"]
    assert [t.get("id") for t in bus.findall("Track")] == ["trk-perc", "trk-lead", "bnc-lead"]
    assert ids["mute-lead"].get("value") == "true" and ids["mute-lead-bounce"].get("value") == "false"
    bc = ids["ch-lead-bounce"]
    assert bc.get("destination") == "ch-bus"
    assert bc.find("Sends/Send").get("type") == "pre" and bc.find("Sends/Send").get("destination") == "ch-rev"
    assert ids["auto-pan-lead-bounce"].find("Target").get("parameter") == "pan-lead-bounce"
    clip = _clip(root, "Lead (bounce)")
    assert _f(clip, "time") == 0.0 and _f(clip, "playStart") == 0.0
    secs = len(y) / SR
    assert _f(clip.find("Warps/Audio"), "duration") == pytest.approx(secs, abs=1e-12)
    # duración en negras con el mapa de tempo: 8 s a 120 BPM (16 negras) + el resto a 90
    assert _f(clip, "duration") == pytest.approx(16 + (secs - 8) * 1.5, abs=1e-9)
    if shutil.which("xmllint"):
        assert _xsd_ok(z.read("project.xml"), XSD / "Project.xsd", tmp_path)[0] == 0
    with pytest.raises(FileExistsError):
        export_dawproject(p, repo.assets, repo.cache, out)


def test_vst3_preset_roundtrip():
    chunk = bytes(range(256)) * 3
    cid, comp = vst3_preset_component(vst3_preset(SURGE_XT_VST3_CID, chunk))
    assert cid == SURGE_XT_VST3_CID.replace("-", "") and comp == chunk


@pytest.mark.compat
def test_surge_device_state_loads_and_sounds_like_fxp(tmp_path, repo):
    """El .vstpreset generado carga en un host VST3 real y suena igual que el .fxp del motor."""
    import tempfile as _tf
    from osky_remix.audio.plugin_host import (DETERMINISTIC_PRESET, PREROLL_S, SURGE_XT_VST3, fxp_chunk,
                                              fxp_to_state_file)
    if not SURGE_XT_VST3.exists():
        pytest.skip("Surge XT no instalado")
    import dawdreamer as daw
    fxp = repo.assets.ingest(DETERMINISTIC_PRESET, AssetKind.preset, Provenance(tool="surge-xt-factory"))
    ops = [{"op": "add_track", "id": "syn", "name": "Syn", "kind": "instrument"},
           {"op": "add_plugin", "track": "syn", "plugin": {"id": "s", "backend": "surge_xt", "uri": "Surge XT",
                                                           "preset": fxp.sha256}},
           {"op": "add_clip", "track": "syn", "clip": {"type": "midi", "id": "c", "start_tick": 0, "length_ticks": 960,
                                                        "notes": [{"tick": 0, "duration": 480, "pitch": 60, "velocity": 100}]}}]
    p = apply_ops(Project(name="sx", sample_rate=SR), [RegisterAsset(asset=fxp)] + parse_ops(ops), Context(author="user"))
    out = tmp_path / "s.dawproject"
    export_dawproject(p, repo.assets, repo.cache, out, bounce_instruments=False)
    z, root, ids = _read(out)
    dev = root.find(".//Channel/Devices/Vst3Plugin")
    assert dev.get("deviceID") == SURGE_XT_VST3_CID and dev.get("deviceRole") == "instrument"
    state = dev.find("State").get("path")
    assert hashlib.sha256(z.read("presets/syn-s.fxp")).hexdigest() == fxp.sha256
    assert vst3_preset_component(z.read(state))[1] == fxp_chunk(DETERMINISTIC_PRESET)

    def render(load):
        e = daw.RenderEngine(SR, 512)
        s = e.make_plugin_processor("i", str(SURGE_XT_VST3))
        with _tf.TemporaryDirectory() as td:
            load(s, Path(td))
        s.add_midi_note(60, 100, PREROLL_S, 0.5)
        e.load_graph([(s, [])])
        e.render(PREROLL_S + 1.5)
        return e.get_audio().T[int(PREROLL_S * SR):]

    def from_vstpreset(s, td):
        f = td / "p.vstpreset"
        f.write_bytes(z.read(state))
        assert s.load_vst3_preset(str(f))

    ref = render(lambda s, td: s.load_state(str(fxp_to_state_file(DETERMINISTIC_PRESET, td / "p.state"))))
    got = render(from_vstpreset)
    default = render(lambda s, td: None)
    assert np.abs(ref).max() > 0.05
    assert np.array_equal(got, ref) and np.abs(default - ref).max() > 0.05
