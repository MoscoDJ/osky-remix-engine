import numpy as np
import pytest

from osky_remix.analysis.transcribe import Note, is_harmonic_ghost
from osky_remix.audio.plugin_host import juce_b64_decode, juce_b64_encode, vst3_state_from_chunk
from osky_remix.doctor import fixtures as fx
from osky_remix.mastering.dynamics import PROFILES, limit, master
from osky_remix.mastering.loudness import integrated_lufs
from osky_remix.mastering.true_peak import sample_peak_dbfs, true_peak_dbtp

SR = 44100


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 64, 1001])
def test_juce_base64_roundtrip(n):
    data = np.random.default_rng(n).integers(0, 256, n, dtype=np.uint8).tobytes()
    assert juce_b64_decode(juce_b64_encode(data)) == data


def test_vst3_state_header_matches_payload():
    st = vst3_state_from_chunk(b"sub3" + b"\x00" * 20)
    assert st[:4] == b"VC2!"
    size = int.from_bytes(st[4:8], "little")
    assert len(st) == 8 + size + 1 and st.endswith(b"\x00")
    assert b"<IComponent>" in st


@pytest.mark.parametrize("level", [-23.0, -33.0])
def test_ebu3341_sine_loudness(level):
    x = np.stack([fx.sine(1000, 20, amp=10 ** (level / 20))] * 2, 1)
    assert abs(integrated_lufs(x, SR) - level) <= 0.1


def test_true_peak_detects_intersample_peak():
    s = fx.sine(SR / 4, 1.0, phase=np.pi / 4)
    assert abs(sample_peak_dbfs(s) + 3.01) < 0.05
    assert abs(true_peak_dbtp(s, SR)) < 0.2


def test_limiter_never_exceeds_ceiling():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((SR * 3, 2)) * 0.8 + np.stack([fx.sine(60, 3)] * 2, 1)
    y = limit(x, SR, ceiling_dbtp=-1.0)
    assert true_peak_dbtp(y, SR) <= -1.0 + 0.05
    assert np.all(np.abs(y) < 1.0)


def test_master_hits_profile():
    mix = fx.band_mixture(10.0)["mixture"]
    res = master(np.stack([mix, mix], 1) * 1.8, SR, PROFILES["club"])
    assert res.true_peak_dbtp <= -1.0 + 1e-3
    assert abs(res.lufs - PROFILES["club"].target_lufs) <= 0.5


def test_harmonic_ghost_filter():
    real = Note(0.0, 1.0, 60, 0.8)
    ghost = Note(0.01, 0.9, 84, 0.35)        # dos octavas arriba, más débil
    loud_octave = Note(0.0, 1.0, 72, 0.8)    # octava real con la misma fuerza: se conserva
    later = Note(2.0, 3.0, 84, 0.3)          # no simultánea: se conserva
    notes = [real, ghost, loud_octave, later]
    assert is_harmonic_ghost(ghost, notes)
    assert not is_harmonic_ghost(loud_octave, notes)
    assert not is_harmonic_ghost(later, notes)
    assert not is_harmonic_ghost(real, notes)
