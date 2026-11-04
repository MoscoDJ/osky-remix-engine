"""Versión DJ: insert_time, extended mix (estructura y procedencia) y cues/beatgrid exactos."""
import csv
import json
from fractions import Fraction
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import soundfile as sf

from osky_remix.composition.arranger import _fingerprint, arrange, load_template, rearrange
from osky_remix.core.models import Asset, AssetKind, AudioClip, MidiClip, Project, Provenance
from osky_remix.core.operations import Context
from osky_remix.core.transactions import TransactionError, apply_ops, parse_ops
from osky_remix.dj.cues import COLORS, cue_points, write_cues
from osky_remix.dj.extend import extend
from tests.unit.test_composition import _ctx

SR = 44100
BAR = 3840
U, C = Context(author="user"), Context(author="claude")
FAKE = Asset(sha256="ab" * 32, kind=AssetKind.stem, name="voz.wav", ext=".wav", size_bytes=0, sample_rate=SR,
             channels=2, frames=SR * 30, provenance=Provenance(tool="test"))


def vocal_samples(bars: int, bpm: float = 124) -> int:
    return int(Fraction(bars * 4 * 60) / Fraction(str(bpm)) * SR) - 100


def dj_project(asset: Asset = FAKE, tempo_change: bool = False) -> Project:
    """Arreglo de 32 compases (intro 8, voz 16, drop 8) + voz manual, pad bloqueado, edición manual."""
    plan = arrange(load_template("melodic_techno"), _ctx(), 32, presets={},
                   compose_meta={"template": "melodic_techno", "bars": 32, "seed": 3})
    p = apply_ops(Project(name="dj", sample_rate=SR), parse_ops(plan.ops), C)
    ops = [{"op": "register_asset", "asset": asset.model_dump(mode="json")},
           {"op": "add_track", "id": "vox", "name": "Vox", "kind": "audio"},
           {"op": "add_clip", "track": "vox", "clip": {"type": "audio", "id": "v1", "asset": asset.sha256,
                                                       "start_tick": 8 * BAR, "source_length": vocal_samples(4)}},
           {"op": "add_clip", "track": "vox", "clip": {"type": "audio", "id": "v2", "asset": asset.sha256,
                                                       "start_tick": 24 * BAR, "source_offset": SR,
                                                       "source_length": vocal_samples(2)}},
           {"op": "add_clip", "track": "arp", "clip": {"type": "midi", "id": "mi-arp", "start_tick": 12 * BAR,
                                                       "length_ticks": BAR,
                                                       "notes": [{"tick": 0, "duration": 480, "pitch": 74, "velocity": 99}]}}]
    if tempo_change:
        ops.append({"op": "set_tempo", "points": [{"tick": 0, "bpm": 124}, {"tick": 24 * BAR, "bpm": 126}]})
    p = apply_ops(p, parse_ops(ops), U)
    bass_voz = p.find_clip("bass-voz")[1]
    edited = [{**n.model_dump(), "pitch": n.pitch + 12} for n in bass_voz.notes[:8]]
    return apply_ops(p, parse_ops([{"op": "set_midi_notes", "clip": "bass-voz", "notes": edited},
                                   {"op": "lock_track", "track": "pad"}]), U)


def extended(p: Project, intro=32, outro=32):
    plan = extend(p, intro, outro)
    return plan, apply_ops(p, parse_ops(plan.ops), C)


# ------------------------------------------------------------------ insert_time
def test_insert_time_shifts_everything_rigidly_and_keeps_automation_values():
    p = dj_project(tempo_change=True)
    p = apply_ops(p, parse_ops([{"op": "set_meter", "points": [{"bar": 0, "numerator": 4}, {"bar": 30, "numerator": 4}]},
                                {"op": "set_automation", "lane": {"id": "mi-gain", "track": "master", "target": "gain_db",
                                 "points": [{"tick": 0, "value": -6}, {"tick": 16 * BAR, "value": 0}]}}]), U)
    q = apply_ops(p, parse_ops([{"op": "insert_time", "at_tick": 8 * BAR, "ticks": 4 * BAR}]), C)
    assert q.find_clip("mi-arp")[1].start_tick == 16 * BAR                # después del punto: se mueve
    assert q.find_clip("v1")[1].start_tick == 12 * BAR                    # justo en el punto: se mueve
    assert q.find_clip("kick-intro")[1].start_tick == 0                   # antes: no se mueve
    assert [(t.tick, t.bpm) for t in q.tempo] == [(0, 124), (28 * BAR, 126)]
    assert [m.bar for m in q.meter] == [0, 34]
    assert [(m.label, m.tick) for m in q.markers] == [("intro", 0), ("voz", 12 * BAR), ("drop", 28 * BAR)]
    pts = [(x.tick, x.value, x.curve) for x in next(a for a in q.automation if a.id == "mi-gain").points]
    assert pts == [(0, -6, "linear"), (8 * BAR, -3, "hold"), (12 * BAR, -3, "linear"), (20 * BAR, 0, "linear")]
    pad0, pad1 = p.track("pad"), q.track("pad")                             # bloqueada: rígida
    assert [c.start_tick + (4 * BAR if c.start_tick >= 8 * BAR else 0) for c in pad0.clips] == [c.start_tick for c in pad1.clips]


def test_insert_time_rejects_splits_and_partial_bars():
    p = dj_project()
    for at, n in [(10 * BAR, BAR), (0, 100), (5, BAR)]:
        with pytest.raises(TransactionError):
            apply_ops(p, parse_ops([{"op": "insert_time", "at_tick": at, "ticks": n}]), C)


# ------------------------------------------------------------------ extended mix
@pytest.mark.parametrize("bars", [16, 32, 64])
def test_extended_mix_structure(bars):
    p = dj_project()
    plan, q = extended(p, bars, bars)
    N = bars * BAR
    assert plan.intro_ticks == N and plan.outro_start == N + 32 * BAR and plan.outro_ticks == N
    # la voz se desplaza exactamente la intro
    assert q.find_clip("v1")[1].start_tick == p.find_clip("v1")[1].start_tick + N == 8 * BAR + N
    # ni voz ni audio dentro de la intro ni del outro
    tm = q.tempo_map()
    for c in q.track("vox").clips:
        a, b = tm.tick_to_sample(c.start_tick), tm.tick_to_sample(c.start_tick) + q.audio_length(c)
        assert a >= tm.tick_to_sample(N) and b <= tm.tick_to_sample(plan.outro_start)
    # intro: bombo en cada tiempo, hats desde el inicio, el bajo entra a la mitad
    intro = lambda tr: [c for c in q.track(tr).clips if c.start_tick < N]
    assert [c.id for c in intro("kick")] == ["dj-kick-intro"]
    k = intro("kick")[0]
    assert (k.start_tick, k.length_ticks) == (0, N) and [n.tick for n in k.notes] == [i * 960 for i in range(bars * 4)]
    hats = [c.start_tick + n.tick for c in intro("drums") for n in c.notes if n.pitch == 42]
    assert min(hats) == 0 and max(hats) > N - BAR
    bass = intro("bass")
    assert len(bass) == 1 and bass[0].start_tick == N // 2 and bass[0].length_ticks == N // 2 and bass[0].notes
    assert {n.pitch % 12 for n in bass[0].notes} <= {2, 5, 7, 0}           # raíces del bajo existente (Dm Gm C F)
    for tr in ("pad", "arp", "lead", "fx", "vox"):
        assert not intro(tr), tr
    # outro: bajo solo en su primera mitad, bombo en todo el outro
    S = plan.outro_start
    out_bass = [c for c in q.track("bass").clips if c.start_tick >= S]
    assert [(c.start_tick, c.length_ticks) for c in out_bass] == [(S, N // 2)]
    assert [(c.start_tick, c.length_ticks) for c in q.track("kick").clips if c.start_tick >= S] == [(S, N)]
    # marcadores y secciones
    labels = [(m.label, m.tick) for m in q.markers]
    assert labels == [("intro_dj", 0), ("intro", N), ("voz", N + 8 * BAR), ("drop", N + 24 * BAR), ("outro_dj", S)]
    secs = q.meta["arrangement"]["sections"]
    assert [s["name"] for s in secs] == ["intro_dj", "intro", "voz", "drop", "outro_dj"]
    assert secs[0]["bars"] == secs[-1]["bars"] == bars and not secs[0]["vocal"] and not secs[-1]["vocal"]
    # tempo estable: sin puntos de tempo dentro de la intro ni del outro
    assert [t.tick for t in q.tempo] == [0]


def test_extended_mix_preserves_locked_manual_and_provenance():
    p = dj_project()
    plan, q = extended(p)
    N = plan.intro_ticks
    shift = lambda c: c.model_copy(update={"start_tick": c.start_tick + N})
    # pista bloqueada: idéntica salvo el desplazamiento (clips y automatización)
    assert [shift(c) for c in p.track("pad").clips] == q.track("pad").clips
    assert q.track("pad").model_dump(exclude={"clips"}) == p.track("pad").model_dump(exclude={"clips"})
    lp = [a for a in p.automation if a.track == "pad"]
    lq = [a for a in q.automation if a.track == "pad"]
    assert [[(x.tick + N, x.value, x.curve) for x in a.points] for a in lp] == [[(x.tick, x.value, x.curve) for x in a.points] for a in lq]
    # clip manual y clip editado: se desplazan intactos y siguen sin ser "generados"
    assert q.find_clip("mi-arp")[1] == shift(p.find_clip("mi-arp")[1])
    assert q.find_clip("bass-voz")[1] == shift(p.find_clip("bass-voz")[1])
    reg = q.meta["generated_content"]
    assert "mi-arp" not in reg["clips"]
    assert reg["clips"]["bass-voz"] != _fingerprint(q.find_clip("bass-voz")[1])
    # lo generado desplazado y lo nuevo quedan registrados con su huella actual
    for cid in ("kick-voz", "drums-drop", "dj-kick-intro", "dj-bass-intro", "dj-drums-outro-2"):
        assert reg["clips"][cid] == _fingerprint(q.find_clip(cid)[1]), cid
    for m in q.markers:
        assert reg["markers"][m.id] == _fingerprint(m)
    # el filtro generado de la batería: en la intro DJ abierto (valor del drop), luego el barrido original
    hp = next(a for a in q.automation if a.id == "auto-drums-drums-highpass-0-hz")
    assert [(x.tick, x.value) for x in hp.points][:2] == [(0, 180.0), (N, 3000.0)]
    assert q.meta["dj_extended"]["tracks"] == {"kick": "kick", "drums": "drums", "bass": "bass"}
    # determinista y no repetible sobre sí misma
    assert extend(p).ops == plan.ops
    with pytest.raises(ValueError, match="ya es una versión extendida"):
        extend(q)
    with pytest.raises(ValueError, match="versión extendida DJ"):
        rearrange(q, load_template("melodic_techno"), _ctx(), 32, "B", "b-")


def test_extended_mix_with_locked_rhythm_tracks_uses_new_tracks():
    p = apply_ops(dj_project(), parse_ops([{"op": "lock_track", "track": t} for t in ("kick", "drums", "bass")]), U)
    plan, q = extended(p, 16, 16)
    assert plan.tracks == {"kick": "dj-drums", "drums": "dj-drums", "bass": "dj-bass"}
    for t in ("kick", "drums", "bass"):
        assert [c.start_tick for c in q.track(t).clips] == [c.start_tick + 16 * BAR for c in p.track(t).clips]
    assert {c.id for c in q.track("dj-drums").clips} >= {"dj-kick-intro", "dj-drums-intro-1"}
    assert any("bloqueada 'bass'" in n for n in plan.notes)


def test_extended_mix_with_tempo_change_keeps_intro_at_first_tempo():
    p = dj_project(tempo_change=True)
    plan, q = extended(p, 16, 16)
    N = 16 * BAR
    assert [(t.tick, t.bpm) for t in q.tempo] == [(0, 124), (24 * BAR + N, 126)]
    assert q.tempo_map().seconds_at(N) == Fraction(16 * 4 * 60, 124)
    assert q.meta["dj_extended"]["bpm_outro"] == 126
    bad = apply_ops(p, parse_ops([{"op": "set_meta", "key": "arrangement", "value": {"sections": [
        {"name": "intro", "bars": 8, "energy": 1, "vocal": False}, {"name": "voz", "bars": 16, "energy": 2, "vocal": True}]}}]), U)
    with pytest.raises(ValueError, match="tempo estable"):
        extend(bad, 16, 16)


def test_extended_mix_rejects_bad_lengths():
    for n in (8, 24, 0):
        with pytest.raises(ValueError, match="frases enteras"):
            extend(dj_project(), n, 32)


# ------------------------------------------------------------------ cues y beatgrid
def _silence(path, seconds, sr=SR):
    sf.write(path, np.zeros((int(seconds * sr), 2), np.float32), sr)
    return path


def _sec(*segments):
    """Segundos exactos de tramos (beats, bpm), independiente del mapa de tempo."""
    return sum((Fraction(b * 60) / Fraction(str(bpm)) for b, bpm in segments), Fraction(0))


def test_rekordbox_xml_beatgrid_and_cues_with_tempo_change(tmp_path):
    p = dj_project(tempo_change=True)
    _, q = extended(p, 16, 16)
    # cambio de tempo en el tick 40*BAR (24 + 16 de intro); cue manual a mitad del tramo de 126
    q = apply_ops(q, parse_ops([{"op": "add_marker", "marker": {"id": "mi-cue", "tick": 44 * BAR + 2 * 960,
                                                                  "label": "golpe", "kind": "cue"}}]), U)
    audio = _silence(tmp_path / "master.wav", 140)
    res = write_cues(q, tmp_path / "cues.xml", "rekordbox", audio)
    assert res["tempo_points"] == 2 and res["cues"] == 6 and res["hot_cues"] == 6 and not res["warnings"]
    root = ET.parse(tmp_path / "cues.xml").getroot()
    assert root.tag == "DJ_PLAYLISTS"
    tr = root.find("COLLECTION/TRACK")
    assert tr.get("Location").startswith("file://localhost/") and tr.get("Location").endswith("master.wav")
    assert int(tr.get("TotalTime")) == 140 and int(tr.get("SampleRate")) == SR
    tempos = tr.findall("TEMPO")
    want_grid = [(Fraction(0), 124.0, 1), (_sec((40 * 4, 124)), 126.0, 1)]
    assert len(tempos) == 2
    for t, (sec, bpm, beat) in zip(tempos, want_grid):
        assert abs(float(t.get("Inizio")) * SR - round(sec * SR)) < 1
        assert float(t.get("Bpm")) == bpm and t.get("Metro") == "4/4" and int(t.get("Battito")) == beat
    marks = tr.findall("POSITION_MARK")
    memory = [m for m in marks if m.get("Num") == "-1"]
    hot = [m for m in marks if m.get("Num") != "-1"]
    expect = {  # nombre -> segundos exactos (tramos de 124 y 126 BPM)
        "intro_dj": Fraction(0), "intro": _sec((64, 124)), "voz": _sec((96, 124)),
        "drop": _sec((160, 124)), "golpe": _sec((160, 124), (18, 126)), "outro_dj": _sec((160, 124), (32, 126))}
    assert [m.get("Name") for m in memory] == list(expect)
    for m in memory + hot:
        assert abs(float(m.get("Start")) * SR - round(expect[m.get("Name")] * SR)) < 1, m.get("Name")
    assert [(m.get("Name"), int(m.get("Num"))) for m in hot] == [(n, i) for i, n in enumerate(expect)]
    colors = {m.get("Name"): (int(m.get("Red")), int(m.get("Green")), int(m.get("Blue"))) for m in hot}
    assert colors["intro_dj"] == COLORS["intro"] and colors["drop"] == COLORS["drop"]
    assert colors["voz"] == COLORS["voz"] and colors["outro_dj"] == COLORS["outro"] and colors["golpe"] == COLORS["cue"]
    # la conversión es la misma que usa el renderizador
    tm = q.tempo_map()
    assert [round(float(m.get("Start")) * SR) for m in memory] == [tm.tick_to_sample(m.tick) for m in q.markers]


def test_beat_number_and_csv_json(tmp_path):
    p = Project(name="g", sample_rate=48000)
    p = apply_ops(p, parse_ops([{"op": "set_tempo", "points": [{"tick": 0, "bpm": 120}, {"tick": 2 * BAR + 2 * 960, "bpm": 125.5}]},
                                {"op": "add_marker", "marker": {"id": "a", "tick": 0, "label": "intro"}},
                                {"op": "add_marker", "marker": {"id": "b", "tick": 4 * BAR, "label": "drop"}}]), U)
    res = write_cues(p, tmp_path / "c.csv", "csv")
    rows = list(csv.DictReader(open(tmp_path / "c.csv")))
    assert [r["type"] for r in rows] == ["tempo", "tempo", "cue", "cue"]
    assert rows[1]["beat"] == "3" and float(rows[1]["bpm"]) == 125.5
    s_drop = _sec((10, 120), (6, 125.5))
    assert int(rows[3]["sample"]) == round(s_drop * 48000) and rows[3]["hot_cue"] == "B" and rows[3]["bar"] == "5"
    write_cues(p, tmp_path / "c.json", "json")
    data = json.loads((tmp_path / "c.json").read_text())
    assert data["beatgrid"][1]["sample"] == round(_sec((10, 120)) * 48000) and data["cues"][0]["category"] == "intro"
    assert not res["warnings"]
    with pytest.raises(ValueError, match="necesita el audio"):
        write_cues(p, tmp_path / "x.xml", "rekordbox")
    with pytest.raises(ValueError, match="último cue"):
        write_cues(p, tmp_path / "x.xml", "rekordbox", _silence(tmp_path / "corto.wav", 1, 48000))


def test_more_sections_than_hot_cues_keeps_key_sections():
    names = ["intro", "verso", "pre", "break", "drop", "drop2", "puente", "coro2", "outro", "extra"]
    p = apply_ops(Project(name="h", sample_rate=SR), parse_ops(
        [{"op": "add_marker", "marker": {"id": f"m{i}", "tick": i * 16 * BAR, "label": n}} for i, n in enumerate(names)]), U)
    cues = cue_points(p)
    hot = [c for c in cues if c.hot is not None]
    assert len(cues) == 10 and len(hot) == 8 and [c.hot for c in hot] == list(range(8))
    assert {"intro", "drop", "verso", "outro", "break"} <= {c.name for c in hot}
    assert [c.tick for c in hot] == sorted(c.tick for c in hot)            # A-H en orden de tiempo
