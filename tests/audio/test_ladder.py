"""Filtro escalera lineal: misma respuesta que el LadderFilter de pedalboard a nivel bajo, sin su
saturación a nivel alto, y automatización sin saltos."""
import numpy as np
import pytest
from pedalboard import LadderFilter, Pedalboard

from osky_remix.audio.effects import apply_effect
from osky_remix.audio.ladder import ladder

pytestmark = pytest.mark.audio
SR = 44100
PB = {"LP24": LadderFilter.Mode.LPF24, "LP12": LadderFilter.Mode.LPF12,
      "HP24": LadderFilter.Mode.HPF24, "HP12": LadderFilter.Mode.HPF12}


def noise(level, n=3 * SR, seed=1):
    return np.random.default_rng(seed).standard_normal((n, 2)) * level


@pytest.mark.parametrize("mode,fc,res", [("LP24", 1000, 0.0), ("LP12", 4500, 0.0), ("HP24", 140, 0.0),
                                         ("HP12", 250, 0.0), ("LP24", 800, 0.6), ("HP24", 35, 0.3)])
def test_matches_pedalboard_ladder_at_low_level(mode, fc, res):
    x = noise(0.001)                                                   # −60 dBFS: tanh ≈ identidad
    ref = Pedalboard([LadderFilter(mode=PB[mode], cutoff_hz=fc, resonance=res, drive=1.0)])(
        x.T.astype(np.float32).copy(), SR).T[SR:]
    y = ladder(x, SR, mode, fc, res)[SR:]                              # tras el suavizado inicial de JUCE
    err = np.sqrt(np.mean((ref - y) ** 2)) / np.sqrt(np.mean(ref ** 2))
    assert 20 * np.log10(err) < -55


def test_linear_at_any_level_unlike_pedalboard():
    t = np.arange(SR) / SR
    x = np.stack([np.sin(2 * np.pi * 200 * t)] * 2, 1)
    def thd(y):
        y = y[SR // 4:, 0]
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
        f = np.fft.rfftfreq(len(y), 1 / SR)
        fund = spec[np.abs(f - 200) < 5].sum()
        return 10 * np.log10(sum(spec[np.abs(f - 200 * k) < 5].sum() for k in range(2, 10)) / fund + 1e-30)
    lo, hi = ladder(0.01 * x, SR, "HP24", 35), ladder(1.0 * x, SR, "HP24", 35)
    assert np.allclose(hi, 100 * lo, atol=1e-5)                       # escalar entrada = escalar salida
    assert thd(hi) < -90
    pb = Pedalboard([LadderFilter(mode=PB["HP24"], cutoff_hz=35, drive=1.0)])(x.T.astype(np.float32).copy(), SR).T
    assert thd(pb) > -30                                               # el original distorsiona a 0 dBFS


def test_attenuation_near_cutoff():
    """Respuesta medida (igual a la del LadderFilter, que por su cero en −0.3 se aplana lejos
    del corte): LP24 −24.7 dB a 2·fc y −45.5 a 4·fc; LP12 −10.8 / −21.5; HP24 −27.1 a fc/2."""
    x = noise(0.1, 4 * SR)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X = np.abs(np.fft.rfft(x[:, 0]))
    def at(y, c):
        m = (f > c / 1.05) & (f < c * 1.05)
        return 20 * np.log10(np.sqrt(np.mean((np.abs(np.fft.rfft(y[:, 0]))[m] / X[m]) ** 2)))
    lp, lp12, hp = ladder(x, SR, "LP24", 500), ladder(x, SR, "LP12", 500), ladder(x, SR, "HP24", 1000)
    assert at(lp, 1000) == pytest.approx(-24.7, abs=0.5) and at(lp, 2000) == pytest.approx(-45.5, abs=0.5)
    assert at(lp12, 1000) == pytest.approx(-10.8, abs=0.5) and at(lp12, 2000) == pytest.approx(-21.5, abs=0.5)
    assert at(hp, 500) == pytest.approx(-27.1, abs=0.5)


def test_automated_sweep_is_continuous_and_matches_static_ends():
    x = noise(0.1, 4 * SR, seed=3)
    hz = np.concatenate([np.geomspace(500, 6000, 3 * SR), np.full(SR, 6000.0)])   # barrido y luego fijo
    y = apply_effect("lowpass", {"hz": 500}, x.astype(np.float32), SR, 124, automation={"hz": hz})
    # sin clics: la derivada no tiene picos aislados respecto de la de la señal estática
    d = np.abs(np.diff(y[:, 0]))
    assert d.max() < 12 * np.percentile(d, 99.9)
    # en el tramo fijo final coincide con el filtro estático en 6 kHz (el estado converge)
    tail = slice(len(x) - SR // 2, len(x))
    st = ladder(x, SR, "LP24", 6000)
    assert np.sqrt(np.mean((y[tail] - st[tail]) ** 2)) / np.sqrt(np.mean(st[tail] ** 2)) < 0.02
