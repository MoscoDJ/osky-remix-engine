"""Visualización del master contra verdad conocida de señales sintéticas."""
import numpy as np
import pytest

from osky_remix.analysis.visualize import correlation, decode_spectrogram, short_term_lufs, spectrogram, visualize
from osky_remix.mastering.loudness import integrated_lufs

pytestmark = pytest.mark.audio
SR = 48000


def _tone(f, seconds, lufs, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    x = np.sin(2 * np.pi * f * t)
    st = np.stack([x, x], 1)
    return st * 10 ** ((lufs - integrated_lufs(st, sr)) / 20)


def test_short_term_lufs_step_at_section_change():
    x = np.vstack([_tone(1000, 6, -20), _tone(1000, 6, -10)])        # escalón en t = 6 s
    r = short_term_lufs(x, SR)
    t, v = np.array(r["t"]), np.array(r["lufs"], dtype=float)
    assert r["window_s"] == 3.0 and t[0] == pytest.approx(3.0) and t[-1] == pytest.approx(12.0)
    assert np.allclose(v[t <= 6.0], -20.0, atol=0.05)               # ventanas enteras en la 1.ª sección
    assert np.allclose(v[t >= 9.0], -10.0, atol=0.05)               # ventanas enteras en la 2.ª
    cross = t[np.argmax(v > -15.0)]
    assert 6.0 < cross < 9.0                                         # dentro de una ventana tras el cambio
    # punto medio en potencia: -20 + 10·log10(1 + 9f) = -15  =>  f ≈ 0.240 de la ventana
    assert cross == pytest.approx(6.0 + 3.0 * (10 ** 0.5 - 1) / 9, abs=0.11)


def test_short_term_lufs_matches_integrated_on_steady_tone():
    x = _tone(440, 5, -14)
    v = np.array(short_term_lufs(x, SR)["lufs"])
    assert np.allclose(v, -14.0, atol=0.05)


def test_correlation_mono_inverted_and_uncorrelated():
    x = _tone(220, 3, -18)
    assert set(correlation(x, SR)["correlation"]) == {1.0}
    inv = x.copy()
    inv[:, 1] *= -1
    c = correlation(inv, SR, [("todo", 0, len(inv))])
    assert set(c["correlation"]) == {-1.0} and c["sections"][0]["correlation"] == -1.0
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((3 * SR, 2)) * 0.1
    assert abs(np.nanmean(np.array(correlation(noise, SR)["correlation"], dtype=float))) < 0.05
    silent = np.zeros((SR, 2))
    assert set(correlation(silent, SR)["correlation"]) == {None}


def test_correlation_per_section():
    a = _tone(300, 2, -18)
    b = a.copy()
    b[:, 1] *= -1
    x = np.vstack([a, b])
    c = correlation(x, SR, [("mono", 0, len(a)), ("inv", len(a), len(x))])
    assert [s["correlation"] for s in c["sections"]] == [1.0, -1.0]
    t, v = np.array(c["t"]), np.array(c["correlation"], dtype=float)
    assert np.all(v[t <= 2.0] == 1.0) and np.all(v[t >= 2.0 + c["window_s"]] == -1.0)


@pytest.mark.parametrize("freq", [110.0, 1000.0, 6000.0])
def test_spectrogram_peak_at_tone_frequency(freq):
    spec = spectrogram(_tone(freq, 4, -14), SR)
    m = decode_spectrogram(spec)
    assert m.shape == (spec["rows"], spec["cols"]) and m.dtype == np.uint8
    row = int(m.mean(axis=1).argmax())
    lo, hi = spec["band_edges_hz"][row], spec["band_edges_hz"][row + 1]
    # la banda pico contiene el tono (o es adyacente si el tono cae justo en un borde)
    assert lo <= freq < hi or abs(np.log(freq / np.sqrt(lo * hi))) < np.log(hi / lo) * 1.5
    assert m[row].min() == 255 or m[row].mean() > 250                 # el máximo de la matriz
    far = [r for r in range(spec["rows"]) if spec["band_edges_hz"][r + 1] < freq / 4 or spec["band_edges_hz"][r] > freq * 4]
    assert m[far].mean() < 60                                         # lejos del tono: > 60 dB por debajo


def test_spectrogram_shows_tone_change_over_time():
    x = np.vstack([_tone(200, 3, -14), _tone(4000, 3, -14)])
    spec = spectrogram(x, SR)
    m = decode_spectrogram(spec)
    t = np.array(spec["times_s"])
    first, second = m[:, t < 2.5].mean(axis=1).argmax(), m[:, t > 3.5].mean(axis=1).argmax()
    assert spec["freqs_hz"][first] < 250 < 3500 < spec["freqs_hz"][second]


def test_visualize_bundle_compact_and_json_safe():
    import json
    x = np.vstack([_tone(1000, 6, -20), _tone(1000, 6, -10)])
    v = visualize(x, SR, [("intro", 0, 6 * SR), ("drop", 6 * SR, 12 * SR)], -9.0)
    json.dumps(v, allow_nan=False)
    assert [s["label"] for s in v["sections"]] == ["intro", "drop"] and v["sections"][1]["start_s"] == 6.0
    assert v["target_lufs"] == -9.0 and v["duration_s"] == 12.0
    assert v["spectrogram"]["cols"] <= 600 and len(v["lufs_s"]["t"]) <= 3000
