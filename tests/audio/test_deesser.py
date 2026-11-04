"""De-esser de banda dividida: fixture sintético con verdad conocida (tono armónico + ráfagas a 7 kHz)."""
import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from osky_remix.audio.deesser import LATENCY_SAMPLES
from osky_remix.audio.effects import AUTOMATABLE, EFFECTS, apply_effect
from osky_remix.core.models import AudioClip, AutomationLane, AutomationPoint, PluginSlot, Project, TempoPointM, Track

SR = 48000
BURSTS = [(0.50, 0.70), (1.50, 1.65), (2.30, 2.50)]       # segundos


def _bp(x, lo, hi):
    return sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=SR, output="sos"), x, axis=0)


def _db(x):
    return 10 * np.log10((x ** 2).mean() + 1e-30)


def vocal_like(seconds=3.0, sib=True):
    """Tono de 220 Hz con 8 armónicos (220–1760 Hz) + ráfagas de ruido 6.5–7.5 kHz en BURSTS."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    low = sum(0.15 / k * np.sin(2 * np.pi * 220 * k * t) for k in range(1, 9))
    x = low
    if sib:
        noise = _bp(np.random.default_rng(0).standard_normal(n), 6500, 7500)
        env = np.zeros(n)
        for a, b in BURSTS:
            env[int(a * SR):int(b * SR)] = 1
        s = noise * env
        x = low + s * 0.2 / np.sqrt((s[env > 0] ** 2).mean())
    return np.stack([x, x], 1).astype(np.float32)


def test_registered_and_automatable():
    assert "deesser" in EFFECTS
    assert set(AUTOMATABLE["deesser"]) == {"threshold_db", "range_db"}
    assert LATENCY_SAMPLES == 0


@pytest.mark.parametrize("range_db", [3.0, 6.0, 10.0])
def test_bursts_reduced_by_range_and_low_band_intact(range_db):
    x = vocal_like()
    y = apply_effect("deesser", {"threshold_db": -35, "range_db": range_db}, x, SR, 120)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()
    for a, b in BURSTS:
        i, j = int((a + 0.02) * SR), int((b - 0.02) * SR)          # interior (fuera del ataque)
        red = _db(_bp(x, 6000, 8000)[i:j]) - _db(_bp(y, 6000, 8000)[i:j])
        assert abs(red - range_db) < 0.3, (a, red)
        low = _db(_bp(x, 200, 2000)[i:j]) - _db(_bp(y, 200, 2000)[i:j])
        assert abs(low) < 0.2
    # entre ráfagas, pasada la liberación, la señal vuelve a ser la original
    i, j = int(1.0 * SR), int(1.4 * SR)
    assert np.abs(y[i:j] - x[i:j]).max() < 1e-4
    # 200–2000 Hz idénticos en toda la señal: ni ganancia ni desfase
    assert abs(_db(_bp(x, 200, 2000)) - _db(_bp(y, 200, 2000))) < 0.2
    d = _bp(x, 200, 2000)[:, 0] - _bp(y, 200, 2000)[:, 0]
    assert np.abs(d).max() < 1e-3


def test_timing_preserved():
    x = vocal_like()
    y = apply_effect("deesser", {"threshold_db": -35, "range_db": 6}, x, SR, 120)
    # el tono no se desplaza (correlación máxima en desfase 0)
    lo_x, lo_y = _bp(x, 200, 2000)[:, 0], _bp(y, 200, 2000)[:, 0]
    lags = range(-20, 21)
    best = max(lags, key=lambda k: float(np.dot(lo_x[1000:-1000], np.roll(lo_y, k)[1000:-1000])))
    assert best == 0
    # los bordes de las ráfagas siguen en su sitio: inicios y finales en la misma ventana de 4 ms
    hop = SR // 250
    env = lambda s: np.sqrt((_bp(s, 6000, 8000)[:, 0][: len(s) // hop * hop].reshape(-1, hop) ** 2).mean(1))
    edges = lambda e: np.flatnonzero(np.diff((e > e.max() * 0.1).astype(int)) != 0)
    want = sorted(int(round(t * 250)) - 1 for ab in BURSTS for t in ab)
    for e in (env(x), env(y)):
        assert len(edges(e)) == len(want) and np.abs(edges(e) - want).max() <= 1, edges(e)


def test_no_sibilance_untouched_and_deterministic():
    x = vocal_like(sib=False)
    y = apply_effect("deesser", {"threshold_db": -35, "range_db": 6}, x, SR, 120)
    assert np.array_equal(x, y)
    s = vocal_like()
    a = apply_effect("deesser", {"threshold_db": -35, "range_db": 6}, s, SR, 120)
    b = apply_effect("deesser", {"threshold_db": -35, "range_db": 6}, s, SR, 120)
    assert np.array_equal(a, b)
    z = apply_effect("deesser", {}, np.zeros((SR, 2), np.float32), SR, 120)
    assert np.isfinite(z).all() and not z.any()


def test_automation_range_per_sample():
    """range_db automatizado: 0 dB (sin efecto) en la primera ráfaga, 6 dB en las demás."""
    x = vocal_like()
    rng = np.where(np.arange(len(x)) < SR, 0.0, 6.0)
    y = apply_effect("deesser", {"threshold_db": -35}, x, SR, 120, automation={"range_db": rng})
    first = slice(int(0.52 * SR), int(0.68 * SR))
    assert np.abs(y[first] - x[first]).max() < 1e-5
    i, j = int(1.52 * SR), int(1.63 * SR)
    red = _db(_bp(x, 6000, 8000)[i:j]) - _db(_bp(y, 6000, 8000)[i:j])
    assert abs(red - 6) < 0.3


def test_renderer_accepts_deesser(tmp_path, repo):
    """El renderizador lo aplica, valida capacidades de entrega y acepta su automatización."""
    import soundfile as sf
    from osky_remix.audio.renderer import Renderer
    from osky_remix.core.models import AssetKind, Provenance
    x = vocal_like()
    f = tmp_path / "voz.wav"
    sf.write(f, x, SR, subtype="FLOAT")
    a = repo.assets.ingest(f, AssetKind.stem, Provenance(tool="test"))
    p = Project(name="d", sample_rate=SR, tempo=[TempoPointM(tick=0, bpm=120)], assets={a.sha256: a}, tracks=[
        Track(id="master", name="Master", kind="master", output=None),
        Track(id="vox", name="Vox", kind="audio", clips=[AudioClip(id="c", asset=a.sha256, start_tick=0,
                                                                   source_length=len(x))],
              plugins=[PluginSlot(id="ds", backend="builtin", uri="deesser", params={"threshold_db": -35, "range_db": 6})])],
        automation=[AutomationLane(id="l", track="vox", target="ds.range_db",
                                   points=[AutomationPoint(tick=0, value=6), AutomationPoint(tick=960 * 8, value=6)])])
    r = Renderer(p, repo.assets, repo.cache)
    r.validate_capabilities()
    pre = r.output("vox")
    assert not [w for w in r.report.warnings if "deesser" in w or "automatiz" in w]
    i, j = int(0.52 * SR), int(0.68 * SR)
    seg_x, seg_y = x[i:j], pre[i:j]
    gain = _db(_bp(seg_y, 200, 2000)) - _db(_bp(seg_x, 200, 2000))   # fader/paneo de la pista
    red = (_db(_bp(seg_x, 6000, 8000)) - _db(_bp(seg_y, 6000, 8000))) + gain
    assert abs(red - 6) < 0.3
