"""Control de calidad de stems con stems sintéticos (sin modelos)."""
import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from osky_remix.analysis.pipeline import vocal_activity
from osky_remix.doctor.fixtures import band_mixture
from osky_remix.separation.quality import prepare_stems

SR = 44100
pytestmark = pytest.mark.audio


@pytest.fixture(scope="module")
def parts():
    st = band_mixture(30.0)
    n = len(st["mixture"])
    t = np.arange(n) / SR
    voice = 0.25 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.2 * t) > -0.5)
    two = lambda x: np.stack([x, x], 1)
    return {k: two(v) for k, v in st.items()}, two(voice)


def test_measures_lag_and_gain(parts):
    st, voice = parts
    stems = {k: st[k] for k in ("drums", "bass", "other")}
    stems["vocals"] = voice
    mix = sum(stems.values())
    # el "separador" devuelve stems retrasados 300 muestras (salida más larga) y normalizados a 0.8
    shifted = {k: (np.r_[np.zeros((300, 2)), v] * 0.8, SR) for k, v in stems.items()}
    prep, q = prepare_stems(mix, SR, shifted)
    assert q.lag_samples == -300
    assert abs(q.gain_correction_db - 20 * np.log10(1 / 0.8)) < 0.05
    assert q.reconstruction_snr_db > 60
    assert not [r for r in q.regions if r.reason == "reconstruction_error"]


def test_reconstruction_error_region(parts):
    st, voice = parts
    stems = {k: st[k].copy() for k in ("drums", "bass", "other")}
    mix = sum(stems.values())
    stems["other"][10 * SR:14 * SR] = 0               # el separador "perdió" 4 s de un stem
    _, q = prepare_stems(mix, SR, {k: (v, SR) for k, v in stems.items()})
    regs = [r for r in q.regions if r.reason == "reconstruction_error"]
    assert regs and regs[0].start <= 10 * SR + SR and regs[-1].end >= 13 * SR


def test_bleed_vs_sibilants(parts):
    st, voice = parts
    rng = np.random.default_rng(0)
    sib = np.zeros(len(voice))
    for c in rng.uniform(1, 28, 40):                  # "eses": ruido agudo no sincronizado
        i, m = int(c * SR), int(0.08 * SR)
        sib[i:i + m] += rng.standard_normal(min(m, len(sib) - i)) * 0.15 * np.hanning(m)[: len(sib) - i]
    sib = sosfilt(butter(4, 5000, "high", fs=SR, output="sos"), sib)
    vox = voice + np.stack([sib, sib], 1)
    vox[20 * SR:26 * SR] += 0.5 * st["drums"][20 * SR:26 * SR]       # fuga real
    stems = {"vocals": vox, "drums": st["drums"], "bass": st["bass"], "other": st["other"]}
    _, q = prepare_stems(sum(stems.values()), SR, {k: (v, SR) for k, v in stems.items()})
    shared = [r for r in q.regions if r.stem == "vocals" and r.reason == "shared_with_drums"]
    assert shared, q.regions
    assert all(r.start >= 19 * SR and r.end <= 27 * SR for r in shared)      # sin falsos positivos
    covered = sum(r.end - r.start for r in shared)
    assert covered >= 5 * SR


def test_vocal_activity():
    n = 20 * SR
    v = np.zeros((n, 2))
    t = np.arange(4 * SR) / SR
    v[3 * SR:7 * SR] = np.stack([0.2 * np.sin(2 * np.pi * 300 * t)] * 2, 1)
    v[12 * SR:16 * SR] = np.stack([0.2 * np.sin(2 * np.pi * 300 * t)] * 2, 1)
    regs = vocal_activity(v, SR)
    assert len(regs) == 2
    assert abs(regs[0][0] - 3 * SR) < 0.1 * SR and abs(regs[1][1] - 16 * SR) < 0.1 * SR
