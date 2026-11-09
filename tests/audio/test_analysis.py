"""Análisis contra canciones sintéticas con verdad conocida (tests/audio, CPU, ~1 min)."""
import numpy as np
import pytest

from osky_remix.analysis.features import analyze_features, detect_silences, detect_transients
from osky_remix.analysis.harmony import analyze_harmony
from osky_remix.analysis.structure import analyze_structure
from osky_remix.analysis.tempo import analyze_rhythm
from osky_remix.doctor.fixtures import harmonic_tone, midi_to_hz, sine
from osky_remix.doctor.songs import Section, chord_notes, make_song, pop_song

SR = 44100
pytestmark = pytest.mark.audio


def near(det, truth, tol_s):
    det = np.asarray(det)
    return sum(len(det) and np.min(np.abs(det - t)) <= tol_s * SR for t in truth)


def false_pos(det, truth, tol_s):
    truth = np.asarray(truth)
    return sum(np.min(np.abs(truth - d)) > tol_s * SR for d in det)


@pytest.fixture(scope="module")
def pop():
    a, gt = pop_song()
    return a, gt, analyze_rhythm(a, SR)


def test_tempo_beats_downbeats(pop):
    a, gt, r = pop
    assert abs(r.tempo.value - 100) < 0.3 and r.tempo.confidence > 0.7
    assert r.meter.value == 4
    assert near(r.beats, gt.beats, 0.03) == len(gt.beats)
    assert false_pos(r.beats, gt.beats, 0.03) <= 1
    assert near(r.downbeats, gt.downbeats, 0.05) == len(gt.downbeats)
    assert false_pos(r.downbeats, gt.downbeats, 0.05) <= 1
    assert not r.free_tempo and not r.phase_changes


def test_waltz_meter():
    a, gt = make_song([Section("A", 8, ["G", "C", "D", "G"], meter=3),
                       Section("B", 8, ["Em", "C", "D", "D"], meter=3, lead=True)], bpm=120)
    r = analyze_rhythm(a, SR)
    assert r.meter.value == 3
    assert near(r.downbeats, gt.downbeats, 0.05) == len(gt.downbeats)


def test_tempo_change_segments():
    a, gt = make_song([Section("A", 8, ["Dm", "Gm", "C", "F"]),
                       Section("B", 8, ["Bb", "C", "Dm", "A"], bpm=128, lead=True)], bpm=100)
    r = analyze_rhythm(a, SR)
    assert [round(s.bpm) for s in r.segments] == [100, 128]
    assert abs(r.segments[1].start - gt.tempo[1][0]) < 2.0 * SR
    assert near(r.downbeats, gt.downbeats, 0.05) == len(gt.downbeats)


def test_inserted_short_bar_is_reported():
    a, gt = pop_song(short_bar_after=13)
    r = analyze_rhythm(a, SR)
    assert len(r.phase_changes) == 1 and r.phase_changes[0].beats_shift == 2
    assert abs(r.phase_changes[0].sample - gt.inserted_short_bars[0]) < 2.5 * SR   # dentro de un compás
    assert near(r.downbeats, gt.downbeats, 0.05) == len(gt.downbeats)


def test_rubato_intro_is_free_tempo():
    rng = np.random.default_rng(3)
    t, parts = 0.0, []
    for ch in ["Dm", "Gm", "C", "A", "Dm", "F", "Gm", "A"]:       # acordes a tiempos irregulares
        d = float(rng.uniform(1.1, 2.9))
        _, notes = chord_notes(ch, 3)
        parts.append(sum(harmonic_tone(midi_to_hz(m), d, SR, n_harm=5, decay=0.8) for m in notes) * 0.2)
        t += d
    intro = np.concatenate(parts)
    song, gt = pop_song(lead_silence=0.0)
    a = np.concatenate([np.stack([intro, intro], 1).astype(np.float32), song])
    r = analyze_rhythm(a, SR)
    free = sum(b - a_ for a_, b in r.free_tempo)
    assert free >= 0.6 * len(intro), r.free_tempo
    assert abs(r.tempo.value - 100) < 0.3


def test_key_chords_and_modulation():
    a, gt = make_song([Section("A", 16, ["Dm", "Gm", "C", "F", "Dm", "Gm", "A", "Dm"]),
                       Section("B", 16, ["G", "D", "Bm", "Em", "A", "D", "G", "A"], lead=True)], bpm=96)
    r = analyze_rhythm(a, SR)
    h = analyze_harmony(a, SR, r.beats)
    assert [k.key for k in h.key_segments] == ["D minor", "D major"]
    assert abs(h.key_segments[1].start - gt.sections[1][0]) <= 2.5 * SR       # ± 1 compás
    hit = tot = 0
    for s, e, lab in gt.chords:
        for x in range(s, e, SR // 10):
            c = next((c for c in h.chords if c.start <= x < c.end), None)
            tot += 1
            hit += c is not None and c.label == lab.replace("Bb", "A#")
    assert hit / tot >= 0.9


def test_global_key(pop):
    a, gt, r = pop
    h = analyze_harmony(a, SR, r.beats)
    assert h.key.value == "D minor"


def test_structure_boundaries_and_labels():
    a, gt = make_song([Section("intro", 4, ["Dm"], drums=False), Section("A", 8, ["Dm", "Gm", "C", "F"]),
                       Section("B", 8, ["Bb", "C", "Dm", "A"], lead=True), Section("A", 8, ["Dm", "Gm", "C", "F"]),
                       Section("B", 8, ["Bb", "C", "Dm", "A"], lead=True), Section("outro", 4, ["Dm"], drums=False)])
    r = analyze_rhythm(a, SR)
    s = analyze_structure(a, SR, r.downbeats, r.beats)
    starts = [x.start for x in s.sections]
    assert len(s.sections) == 6
    assert all(abs(d - g) <= 0.1 * SR for d, (g, _) in zip(starts, gt.sections))
    lab = [x.label for x in s.sections]
    assert lab[1] == lab[3] and lab[2] == lab[4] and len({lab[1], lab[2], lab[0]}) == 3


def test_silences_transients_features(pop):
    a, gt, _ = pop
    sil = detect_silences(a, SR)
    assert any(abs(x - 0) < 0.05 * SR and abs(y - SR) < 0.05 * SR for x, y in sil)
    tr = [t.sample for t in detect_transients(a, SR)]
    assert near(tr, gt.kicks, 0.01) >= 0.9 * len(gt.kicks)
    f = analyze_features(np.stack([sine(1000, 20, amp=10 ** (-23 / 20))] * 2, 1), SR)
    assert abs(f.lufs + 23) < 0.15 and f.clipped_samples == 0 and abs(f.stereo_correlation - 1) < 1e-6


@pytest.mark.parametrize('source_sr,frames', [(48000,44100), (44100,22050)])
def test_analysis_rejects_misaligned_stems_before_processing(tmp_path, source_sr, frames):
    import soundfile as sf
    from osky_remix.analysis.pipeline import run_analysis
    mix, stem = tmp_path/'mix.wav', tmp_path/'stem.wav'
    sf.write(mix, np.zeros((44100,2)), 44100)
    sf.write(stem, np.zeros((frames,2)), source_sr)
    with pytest.raises(ValueError, match='alinear|remuestrear'):
        run_analysis(mix, {'vocals':stem})


def test_analysis_accepts_stems_a_few_samples_off(tmp_path, pop):
    import soundfile as sf
    from osky_remix.analysis.pipeline import run_analysis
    a = pop[0]
    x = a if a.ndim == 2 else np.stack([a, a], 1)
    mix, stem = tmp_path/'mix.wav', tmp_path/'stem.wav'
    sf.write(mix, x, SR)
    sf.write(stem, x[:-300], SR)                       # ~7 ms más corto, como algunos separadores
    rep = run_analysis(mix, {'vocals': stem})
    assert rep.sources["vocal_activity"] == "stem vocals"


def test_constant_tempo_with_long_breakdown_and_loud_offbeat_hats():
    """Techno a 125 BPM: 48 s de bombo + hats abiertos a contratiempo, 45 s de break sin bombo
    (pad y barrido de ruido) y 48 s de bombo otra vez. El rastreador deriva en el break; la
    rejilla debe ser una sola, a 125 BPM, y caer en los bombos (no en los hats) a ambos lados."""
    from scipy.signal import butter, sosfilt
    sr, bpm = 44100, 125.0
    spb = 60 / bpm
    n_beats = int(141 / spb)
    y = np.zeros(int((n_beats + 2) * spb * sr))
    rng = np.random.default_rng(4)
    t = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * np.cumsum(50 + 120 * np.exp(-t * 40)) / sr) * np.exp(-t * 14) * 0.8
    hat = sosfilt(butter(4, 7000, "high", fs=sr, output="sos"), rng.standard_normal(int(0.12 * sr))) \
        * np.exp(-np.arange(int(0.12 * sr)) / sr * 30) * 0.5
    brk = range(int(48 / spb), int(93 / spb))
    kicks = []
    for b in range(n_beats):
        a = int(round(b * spb * sr))
        if b not in brk:
            y[a:a + len(kick)] += kick
            kicks.append(a)
            h = int(round((b + 0.5) * spb * sr))
            y[h:h + len(hat)] += hat
    s0, s1 = int(brk[0] * spb * sr), int(brk[-1] * spb * sr)
    tt = np.arange(s1 - s0) / sr
    y[s0:s1] += 0.15 * sum(np.sin(2 * np.pi * f * tt) for f in (146.8, 174.6, 220.0)) \
        + 0.05 * sosfilt(butter(2, 2000, "high", fs=sr, output="sos"), rng.standard_normal(s1 - s0)) * (tt / tt[-1])
    r = analyze_rhythm(np.stack([y, y], 1).astype(np.float32), sr)
    assert len(r.segments) == 1 and r.segments[0].bpm == pytest.approx(bpm, abs=0.01)
    beats = np.array(r.beats)
    err = [np.min(np.abs(beats - k)) / sr * 1000 for k in kicks]
    assert np.median(err) < 5 and np.max(err) < 15              # en el bombo, no a contratiempo (240 ms)
