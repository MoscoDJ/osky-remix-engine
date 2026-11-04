from fractions import Fraction

import pytest

from osky_remix.core.timeline import PPQ, MeterMap, MeterPoint, TempoMap, TempoPoint


def test_constant_tempo_exact():
    tm = TempoMap([TempoPoint(0, 124.0)], 44100)
    bar = 4 * PPQ
    # 1 compás a 124 BPM = 240/124 s exactos
    assert tm.seconds_at(bar) == Fraction(240, 124)
    assert tm.tick_to_sample(64 * bar) == round(Fraction(64 * 240, 124) * 44100)


def test_decimal_bpm_has_no_binary_error():
    tm = TempoMap([TempoPoint(0, 86.1)], 48000)
    assert tm.seconds_at(PPQ) == Fraction(600, 861)


def test_tempo_change_accumulates():
    tm = TempoMap([TempoPoint(0, 120.0), TempoPoint(4 * PPQ, 60.0)], 48000)
    assert tm.seconds_at(4 * PPQ) == 2                  # 4 tiempos a 120
    assert tm.seconds_at(5 * PPQ) == 3                  # +1 tiempo a 60
    assert tm.tick_to_sample(5 * PPQ) == 3 * 48000


@pytest.mark.parametrize("bpms", [[124.0], [120.0, 87.5, 140.0]])
def test_roundtrip_tick_sample_within_one_tick(bpms):
    pts = [TempoPoint(i * 8 * PPQ, b) for i, b in enumerate(bpms)]
    tm = TempoMap(pts, 44100)
    for tick in range(0, 30 * PPQ, 337):
        assert abs(tm.sample_to_tick(tm.tick_to_sample(tick)) - tick) <= 1


def test_tempo_map_rejects_bad_points():
    with pytest.raises(ValueError):
        TempoMap([TempoPoint(10, 120)], 48000)
    with pytest.raises(ValueError):
        TempoMap([TempoPoint(0, 120), TempoPoint(0, 90)], 48000)


def test_meter_changes():
    mm = MeterMap([MeterPoint(0, 4, 4), MeterPoint(2, 3, 4), MeterPoint(4, 7, 8)])
    assert mm.bar_to_tick(2) == 2 * 4 * PPQ
    assert mm.bar_to_tick(3) == 2 * 4 * PPQ + 3 * PPQ
    assert mm.bar_to_tick(5) == 8 * PPQ + 6 * PPQ + 7 * PPQ // 2
    assert mm.bar_to_tick(1, beat=2) == 4 * PPQ + 2 * PPQ
