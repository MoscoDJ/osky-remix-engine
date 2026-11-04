import pytest

from osky_remix.composition import bass as B
from osky_remix.composition import drums as D
from osky_remix.composition import harmony as H
from osky_remix.composition import melody as M
from osky_remix.composition.arranger import arrange, load_template
from osky_remix.composition.context import ComposeContext
from osky_remix.composition.theory import BAR, S16, chord_pcs, key_scale, parse_chord, voice_lead
from osky_remix.core.models import Project
from osky_remix.core.operations import Context
from osky_remix.core.transactions import apply_ops, parse_ops

PROG = ["Dm", "Dm", "Gm", "Gm", "C", "C", "F", "F"]


def test_chords_and_scales():
    assert parse_chord("A#m") == (10, "m") and parse_chord("N") is None
    assert chord_pcs("Dm") == [2, 5, 9] and chord_pcs("A") == [9, 1, 4]
    assert set(key_scale("D minor")) >= {2, 4, 5, 7, 9, 10, 0, 1}


def test_voice_leading_keeps_chord_tones_and_moves_little():
    prev = voice_lead(None, chord_pcs("Dm"))
    nxt = voice_lead(prev, chord_pcs("Gm"))
    assert {n % 12 for n in nxt} == set(chord_pcs("Gm"))
    assert sum(abs(a - b) for a, b in zip(sorted(prev), sorted(nxt))) <= 8


def test_drums_patterns():
    one = D.section_drums(4, 0, seed=1, with_fill=False, crash=False)
    assert [n.tick for n in one if n.pitch == D.KICK] == [i * 960 for i in range(16)]
    full = D.section_drums(4, 3, seed=1)
    assert any(n.pitch == D.SNARE for n in full)                 # redoble en el último compás
    assert any(n.pitch == D.CRASH and n.tick == 0 for n in full)
    assert D.section_drums(4, 3, seed=7) == D.section_drums(4, 3, seed=7)    # determinista


def test_bass_follows_roots():
    notes = B.rolling(PROG, seed=1, octave_jumps=False)
    assert {n.pitch % 12 for n in notes if n.tick < BAR} == {2}
    assert {n.pitch % 12 for n in notes if BAR <= n.tick < 2 * BAR} == {7}
    assert all(n.tick % 960 != 0 for n in notes)                 # nunca encima del bombo


def test_pads_and_arp_use_chord_tones():
    for n in H.pads(PROG):
        h = n.tick // 1920
        assert n.pitch % 12 in chord_pcs(PROG[h])
    for n in H.arp(PROG, seed=2):
        assert n.pitch % 12 in chord_pcs(PROG[n.tick // 1920])


def test_melody_from_transcription_quantizes_and_is_monophonic():
    notes = [(0.01, 0.5, 62), (0.48, 0.9, 65), (1.0, 1.2, 69)]
    out = M.from_transcription(notes, 120, 4 * BAR, register=74)
    assert all(n.tick % S16 == 0 for n in out)
    assert all(a.tick + a.duration <= b.tick for a, b in zip(out, out[1:]))
    assert abs(sum(n.pitch for n in out) / len(out) - 74) <= 6


def _ctx():
    beats = [i * 22050 for i in range(400)]          # 120 BPM a 44.1 kHz
    chords = []
    for i, lab in enumerate(PROG * 25):
        chords.append((i * 44100, (i + 1) * 44100, lab))
    return ComposeContext(bpm=124, key="D minor", source_bpm=120, beats=beats, b0=0, chords_src=chords, seed=3)


def test_context_halfbars_follow_source_beats():
    assert _ctx().halfbars(0, 4) == PROG


def test_arrangement_is_valid_atomic_batch():
    tpl = load_template("melodic_techno")
    plan = arrange(tpl, _ctx(), 32, presets={})
    ops = parse_ops(plan.ops)
    p = apply_ops(Project(name="t", sample_rate=44100), ops, Context(author="claude"))
    assert p.tempo[0].bpm == 124 and p.key == "Dm"
    assert [m.label for m in p.markers] == ["intro", "voz", "drop"]
    kick = p.track("kick")
    assert sum(len(c.notes) for c in kick.clips) == 32 * 4
    assert p.track("bass").sidechain_from == "kick"
    assert p.track("bass").plugins[0].uri == "synth"            # sin presets: sintetizador integrado
    with pytest.raises(Exception):
        apply_ops(p, parse_ops(plan.ops), Context(author="claude"))   # pistas ya existen: se rechaza todo


def test_unknown_template_and_length():
    with pytest.raises(ValueError, match="no existe"):
        load_template("polka")
    with pytest.raises(ValueError, match="compases"):
        arrange(load_template("melodic_techno"), _ctx(), 40, presets={})
