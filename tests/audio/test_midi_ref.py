"""MIDI de referencia: alineación con un audio de verdad conocida y uso en regiones."""
import mido
import numpy as np
import pytest

from osky_remix.analysis.midi_ref import build_reference, find_lyric, read_midi
from osky_remix.analysis.tempo import analyze_rhythm
from osky_remix.composition.context import ComposeContext
from osky_remix.doctor.songs import chord_notes, pop_song

pytestmark = pytest.mark.audio
SR = 44100


def write_reference_midi(path, gt, bpm=101.0, intro_beats=6):
    """MIDI 'de karaoke' del pop_song: otra introducción y tempo 1 % distinto (deriva)."""
    ppq = 480
    mf = mido.MidiFile(type=1, ticks_per_beat=ppq)
    meta, harm, drums = mido.MidiTrack(), mido.MidiTrack(), mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    harm.append(mido.Message("program_change", program=24, channel=0, time=0))
    ev_h, ev_d, ev_m = [], [], []
    starts = {s for s, _ in gt.sections}
    for i, (s, e, ch) in enumerate(gt.chords):
        t0 = (intro_beats + 4 * i) * ppq
        _, notes = chord_notes(ch, 3)
        for n in notes + [notes[0] - 12]:
            ev_h += [(t0, 1, n), (t0 + 4 * ppq - 10, 0, n)]
        if s in starts:
            label = next(l for st, l in gt.sections if st == s)
            ev_m.append((t0, f"/Linea {label} {i}"))
        if s >= gt.sections[1][0]:                     # batería solo donde el audio la tiene
            for b in range(4):
                ev_d += [(t0 + b * ppq, 1, 36), (t0 + b * ppq + 100, 0, 36)]
    for evs, tr, ch in ((ev_h, harm, 0), (ev_d, drums, 9)):
        now = 0
        for t, on, n in sorted(evs, key=lambda x: (x[0], x[1])):
            tr.append(mido.Message("note_on" if on else "note_off", note=n, velocity=100 if on else 0, channel=ch, time=t - now))
            now = t
    now = 0
    for t, txt in ev_m:
        meta.append(mido.MetaMessage("lyrics", text=txt, time=t - now))
        now = t
    mf.tracks += [meta, harm, drums]
    mf.save(str(path))


@pytest.fixture(scope="module")
def aligned(tmp_path_factory):
    a, gt = pop_song()
    r = analyze_rhythm(a, SR)
    p = tmp_path_factory.mktemp("m") / "ref.mid"
    write_reference_midi(p, gt)
    drums = np.zeros(len(a), np.float32)
    for k in gt.kicks:
        drums[k:k + 200] += np.hanning(200).astype(np.float32)
    ref = build_reference(read_midi(p), a, SR, r.beats, [(s, e, c.replace("Bb", "A#")) for s, e, c in gt.chords],
                          None, np.stack([drums, drums], 1))
    return a, gt, r, ref


def test_alignment_recovers_known_times(aligned):
    a, gt, r, ref = aligned
    assert ref.beat_anchors > 50
    assert ref.drum_error_ms is not None and ref.drum_error_ms < 25
    assert ref.chord_agreement >= 0.85
    for start, label in gt.sections[1:]:
        line = next(l for l in ref.lyrics if l.text.startswith(f"Linea {label}") and abs(l.start - start / SR) < 3)
        assert abs(line.start - start / SR) < 0.06, (label, line.start, start / SR)


def test_find_lyric_ignores_accents_and_counts_occurrences(aligned):
    *_, ref = aligned
    assert find_lyric(ref, "línea a", 2).text.startswith("Linea A")
    assert find_lyric(ref, "linea z") is None


def test_region_uses_reference_downbeat_and_measured_pickup(aligned):
    a, gt, r, ref = aligned
    bt = r.beats
    ctx = ComposeContext(bpm=124, key="D minor", source_bpm=100, beats=bt, b0=0, chords_src=[], downbeats=r.downbeats,
                         sr=SR, reference=ref)
    target = gt.sections[2][0]                                     # sección B
    period = 60 / 100
    env = np.full(int(len(a) / SR * 100) + 1, -60.0)
    onset = target / SR - 1.5 * period                             # canta 1.5 tiempos antes
    env[int(onset * 100):int((onset + 6) * 100)] = -12
    ctx.vocal_env = env
    reg = ctx.region({"lyric": "Linea B"})
    assert abs(bt[reg.beat] - target) < 0.03 * SR
    assert abs(reg.pickup_beats - 1.5) < 0.1


def test_arranger_regions_and_chops(aligned):
    from osky_remix.composition.arranger import generate_parts
    from osky_remix.composition.context import VocalSource
    a, gt, r, ref = aligned
    period = 60 / 100
    ctx = ComposeContext(bpm=124, key="D minor", source_bpm=100, beats=r.beats, b0=0, chords_src=[],
                         downbeats=r.downbeats, sr=SR, reference=ref)
    env = np.full(int(len(a) / SR * 100) + 1, -60.0)
    target = gt.sections[2][0] / SR
    env[int((target - 2 * period) * 100):int((target + 8) * 100)] = -12        # anacrusa de 2 tiempos
    ctx.vocal_env = env
    spb = 60 / 124 * SR
    ctx.vocal = VocalSource(asset="a" * 64, sr=SR, region_start=0, samples_per_beat=spb, frames=len(a))
    tpl = {"instruments": {"bass": {}, "pad": {}, "arp": {}, "lead": {}},
           "arrangements": {"8": [{"name": "intro", "bars": 4, "kick": True},
                                  {"name": "voz", "bars": 4, "vocal": "phrase", "vocal_bars": 3, "source": {"lyric": "Linea B"},
                                   "chops": [{"lyric": "Linea B", "at_bars": [1, 3], "max_beats": 2}]}]}}
    parts = generate_parts(tpl, ctx, 8)
    vox = next(c for t, c in parts.clips if t == "vox")
    assert abs(vox["start_tick"] - (4 * 3840 - 2 * 960)) <= 10             # entra 2 tiempos antes (±5 ms)
    assert abs(vox["source_length"] - (3 * 4 + 2) * spb) < 0.001 * SR     # 3 compases + anacrusa (±1 ms)
    chops = [c for t, c in parts.clips if t == "chops"]
    assert [round((c["start_tick"] + 1920) / 3840) for c in chops] == [5, 7]
    assert all(abs((c["start_tick"] + 1920) % 3840) <= 10 or abs((c["start_tick"] + 1920) % 3840 - 3840) <= 10 for c in chops)
    assert all(c["source_length"] <= 4 * spb + 0.001 * SR for c in chops)  # máx. 2 tiempos + anacrusa (±1 ms)
    assert parts.sections[1]["_region"]["label"].startswith("Linea B")
