"""Doctor de mezcla: bandas, referencia comercial, enmascaramiento y sus acciones.

Todo contra fixtures sintéticos con valor conocido: ruido con espectro exacto (generado en
frecuencia), una banda de octava con +6 dB, canales con correlación conocida, tonos en bandas
conocidas.
"""
import json
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly
from typer.testing import CliRunner

from osky_remix.agents.actions import plan_actions
from osky_remix.agents.planner import assess, improve
from osky_remix.agents.policies import DEFAULT
from osky_remix.analysis import masking as mk
from osky_remix.analysis.bands import BANDS, band_spectrum
from osky_remix.analysis.reference_match import load_audio, match, punch
from osky_remix.core.models import AssetKind, Project, Provenance
from osky_remix.core.operations import Context
from osky_remix.core.revisions import ProjectRepo
from osky_remix.core.transactions import apply_ops, parse_ops

pytestmark = pytest.mark.audio
SR = 48000
U = Context(author="user")
B8K = next(i for i, b in enumerate(BANDS) if b["id"] == "8k")


def noise(psd, sr=SR, seconds=20, seed=1, gain=None, channels=2):
    """Ruido gaussiano con densidad espectral exacta ``psd(f)`` (y ganancia de potencia ``gain(f)``)."""
    n = sr * seconds
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1 / sr)
    S = np.zeros_like(f)
    k = f >= 20
    S[k] = psd(f[k])
    if gain:
        S *= gain(f)
    x = np.stack([np.fft.irfft((rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) * np.sqrt(S), n)
                  for _ in range(channels)], 1)
    return 0.1 * x / np.sqrt(np.mean(x ** 2))


def pink(**kw):
    return noise(lambda f: 1 / f, **kw)


def boost_band(i, db):
    lo, hi = BANDS[i]["lo"], BANDS[i]["hi"]
    return lambda f: np.where((f >= lo) & (f < hi), 10 ** (db / 10), 1.0)


# ------------------------------------------------------------------ bandas
@pytest.mark.parametrize("name,psd,energy", [
    ("rosa", lambda f: 1 / f, lambda lo, hi: np.log(hi / lo)),                   # ∫ 1/f
    ("marrón", lambda f: 1 / f ** 2, lambda lo, hi: 1 / lo - 1 / hi),            # ∫ 1/f²: −3 dB/oct
])
def test_band_energy_matches_known_tilt(name, psd, energy):
    bs = band_spectrum(noise(psd), SR)
    exp = [10 * np.log10(energy(max(b["lo"], 20.0), b["hi"])) for b in BANDS]
    ref = 5                                                                      # 1 kHz
    for b, e in zip(bs, exp):
        assert abs((b["db"] - bs[ref]["db"]) - (e - exp[ref])) < 0.5, (name, b["band"])


def test_width_mono_vs_decorrelated():
    n1, n2 = pink(seed=3, channels=1)[:, 0], pink(seed=4, channels=1)[:, 0]
    mono = band_spectrum(np.stack([n1, n1], 1), SR)
    indep = band_spectrum(np.stack([n1, n2], 1), SR)
    part = band_spectrum(np.stack([n1, 0.6 * n1 + 0.8 * n2], 1), SR)            # corr 0.6, lado/centro −6.02 dB
    for m, d, q in zip(mono[1:], indep[1:], part[1:]):
        assert m["corr"] > 0.999 and m["side_to_mid_db"] < -60
        assert abs(d["corr"]) < 0.06 and abs(d["side_to_mid_db"]) < 0.6
        assert abs(q["corr"] - 0.6) < 0.05 and abs(q["side_to_mid_db"] + 6.02) < 0.6


def test_punch_counts_kick_hits():
    t = np.arange(SR * 10) / SR
    kicks = np.zeros_like(t)
    for h in range(20):                                                          # 2 golpes/s
        a = h * SR // 2
        k = np.arange(int(0.3 * SR)) / SR
        kicks[a:a + len(k)] += np.sin(2 * np.pi * 55 * k) * np.exp(-k / 0.06)
    drone = 0.3 * np.sin(2 * np.pi * 55 * t)
    pk, pd = punch(kicks, SR), punch(drone, SR)
    assert abs(pk["transients_per_s"] - 2.0) < 0.15 and pd["transients_per_s"] < 0.2
    assert pk["punch_db"] > pd["punch_db"] + 6


# ------------------------------------------------------------------ referencia
def test_reference_match_finds_known_band_and_nothing_else(tmp_path):
    mix = pink(seed=1)
    ref_sr = 44100                                                               # otra frecuencia de muestreo
    ref = 0.5 * noise(lambda f: 1 / f, sr=ref_sr, seed=1, gain=boost_band(B8K, 6.0))   # además −6 dB de nivel
    f = tmp_path / "ref.wav"
    sf.write(f, ref.astype(np.float32), ref_sr, subtype="FLOAT")
    rep = match(mix, load_audio(f, SR), SR)
    assert abs(rep["tonal_delta_db"]["8k"] - 6.0) < 0.5
    assert all(abs(v) < 0.5 for k, v in rep["tonal_delta_db"].items() if k != "8k")
    assert [(x["code"], x.get("band")) for x in rep["findings"]] == [("reference_tonal", "8k")]
    assert "faltan ~6.0 dB entre 5.7–11.3 kHz" in rep["findings"][0]["detail"]
    assert rep["ref_gain_to_match_db"] > 4                                       # igualada en sonoridad
    same = match(mix, mix, SR)
    assert same["findings"] == [] and max(abs(v) for v in same["tonal_delta_db"].values()) < 1e-6


# ------------------------------------------------------------------ enmascaramiento (detector)
def vocal_band(seconds=8, seed=5):
    """Ruido limitado a 300 Hz–3 kHz (la "voz"), ~−20 dBFS RMS."""
    x = noise(lambda f: np.where((f >= 300) & (f < 3000), 1.0, 0.0), seconds=seconds, seed=seed)
    return x


def tone(hz, amp, seconds=8):
    y = amp * np.sin(2 * np.pi * hz * np.arange(SR * seconds) / SR)
    return np.stack([y, y], 1)


def test_masking_detector_flags_overlap_only():
    vox = vocal_band()
    post = {"vox": vox, "pad": tone(1000, 0.2), "lead": tone(8000, 0.5)}
    secs = [{"name": "todo", "start": 0, "end": len(vox), "vocal": True}]
    hits = mk.detect(post, SR, secs)
    flagged = mk.summarize(hits, DEFAULT.masking_threshold_db)
    assert [(g["target"], g["masker"], g["band"]) for g in flagged] == [("vox", "pad", "1k")]
    assert flagged[0]["masking_db"] > 3
    # la voz no tiene contenido en 8 kHz: el lead no se juzga allí
    assert not [h for h in hits if h.masker == "lead" and h.masking_db > DEFAULT.masking_threshold_db]
    # sección sin voz: no se mide
    assert mk.detect(post, SR, [{**secs[0], "vocal": False}]) == []


# ------------------------------------------------------------------ acciones con render real
def _project_with_audio(repo: ProjectRepo, tmp_path, tracks: dict[str, np.ndarray], profile="dynamic"):
    ops = [{"op": "set_master_profile", "profile": profile}]
    for tid, x in tracks.items():
        f = tmp_path / f"{tid}.wav"
        sf.write(f, x.astype(np.float32), SR, subtype="FLOAT")
        a = repo.assets.ingest(f, AssetKind.stem, Provenance(tool="test"))
        ops += [{"op": "register_asset", "asset": a.model_dump(mode="json")},
                {"op": "add_track", "id": tid, "name": tid, "kind": "audio"},
                {"op": "add_clip", "track": tid, "clip": {"type": "audio", "id": f"{tid}-c", "asset": a.sha256,
                                                          "start_tick": 0, "source_length": len(x)}}]
    repo.commit(ops, "user", "pistas")
    return repo.load()


@pytest.fixture
def mrepo(tmp_path):
    return ProjectRepo.create(tmp_path / "md", Project(name="md", sample_rate=SR))


def _masking_db(ev, target, masker, band):
    return next(m["masking_db"] for m in ev.metrics["masking"]
                if (m["target"], m["masker"], m["band"]) == (target, masker, band))


def test_unmask_eq_reduces_measured_masking(mrepo, tmp_path):
    p = _project_with_audio(mrepo, tmp_path, {"vox": vocal_band(), "pad": tone(1000, 0.2)})
    ev = assess(p, mrepo)
    before = _masking_db(ev, "vox", "pad", "1k")
    acts, blocked = plan_actions(p, ev)
    eq = [a for a in acts if a.name == "unmask_eq"]
    assert len(eq) == 1 and not blocked
    op = eq[0].ops[0]
    assert op["op"] == "add_plugin" and op["track"] == "pad" and op["plugin"]["uri"] == "eq"
    assert op["plugin"]["params"]["mid_hz"] == 1000 and op["plugin"]["params"]["mid_db"] == -DEFAULT.max_eq_step_db
    cand = apply_ops(p, parse_ops(eq[0].ops), Context(author="claude"))
    after = _masking_db(assess(cand, mrepo), "vox", "pad", "1k")
    assert abs((before - after) - DEFAULT.max_eq_step_db) < 0.5                 # el tono está en el centro
    # el bucle lo acepta porque el puntaje mejora, y el corte queda acotado
    res = improve(mrepo, "main", replace(DEFAULT, max_iterations=2), log=lambda *_: None)
    assert res.steps[0].accepted and res.final.score < res.initial.score
    slot = next(s for s in mrepo.load().track("pad").plugins if s.id == "auto-unmask-1k")
    assert -DEFAULT.max_eq_cut_db <= slot.params["mid_db"] < 0


def test_locked_masker_needs_user_and_no_op(mrepo, tmp_path):
    p = _project_with_audio(mrepo, tmp_path, {"vox": vocal_band(), "pad": tone(1000, 0.2)})
    p = apply_ops(p, parse_ops([{"op": "lock_track", "track": "pad"}]), U)
    acts, blocked = plan_actions(p, assess(p, mrepo))
    assert not [op for a in acts for op in a.ops if op.get("track") == "pad"]
    assert any("'pad' está bloqueada" in b and "enmascara" in b for b in blocked)


def test_bass_masking_kick_gets_ducked(mrepo, tmp_path):
    n = SR * 8
    kick = np.zeros(n)
    for h in range(16):
        k = np.arange(int(0.25 * SR)) / SR
        a = h * SR // 2
        kick[a:a + len(k)] += 0.8 * np.sin(2 * np.pi * 60 * k) * np.exp(-k / 0.05)
    bass = 0.3 * np.sin(2 * np.pi * 62 * np.arange(n) / SR)
    p = _project_with_audio(mrepo, tmp_path, {"kick": np.stack([kick, kick], 1), "bass": np.stack([bass, bass], 1)})
    ev = assess(p, mrepo)
    before = _masking_db(ev, "kick", "bass", "63")
    acts, _ = plan_actions(p, ev)
    duck = next(a for a in acts if a.name == "duck_bass")
    assert {"op": "set_track", "track": "bass", "sidechain_from": "kick"} in duck.ops
    cand = apply_ops(p, parse_ops(duck.ops), Context(author="claude"))
    assert _masking_db(assess(cand, mrepo), "kick", "bass", "63") < before - 2


def test_reference_in_evaluator_and_master_eq(mrepo, tmp_path):
    x = pink(seconds=12, seed=2)
    p = _project_with_audio(mrepo, tmp_path, {"music": x})
    ev0 = assess(p, mrepo)
    assert not [f for f in ev0.findings if f.code.startswith("reference")]       # sin referencia, nada
    rf = tmp_path / "ref.wav"
    sf.write(rf, resample_poly(pink(seconds=12, seed=9, gain=boost_band(B8K, 6.0)), 147, 160, axis=0).astype(np.float32),
             44100, subtype="FLOAT")
    a = mrepo.assets.ingest(rf, AssetKind.reference, Provenance(tool="test"))
    mrepo.commit([{"op": "register_asset", "asset": a.model_dump(mode="json")}, {"op": "set_meta", "key": "reference_track",
                                           "value": {"sha256": a.sha256, "name": "ref.wav"}}], "user", "ref")
    p = mrepo.load()
    ev = assess(p, mrepo)
    tonal = [f for f in ev.findings if f.code == "reference_tonal"]
    assert [f.data["band"] for f in tonal] == ["8k"] and abs(tonal[0].data["delta_db"] - 6) < 0.7
    acts, _ = plan_actions(p, ev)
    eq = next(a for a in acts if a.name == "reference_eq")
    assert eq.ops[0]["track"] == "master" and eq.ops[0]["plugin"]["params"]["mid_db"] == DEFAULT.max_eq_step_db
    cand = apply_ops(p, parse_ops(eq.ops), Context(author="claude"))
    d2 = assess(cand, mrepo).metrics["reference"]["tonal_delta_db"]["8k"]
    assert d2 < tonal[0].data["delta_db"] - 1.5
    locked = apply_ops(p, parse_ops([{"op": "lock_track", "track": "master"}]), U)
    acts, blocked = plan_actions(locked, ev)
    assert not [a for a in acts if a.name == "reference_eq"] and any("master está bloqueado" in b for b in blocked)


def test_match_cli_registers_reference(tmp_path, monkeypatch):
    from osky_remix.cli.main import app
    from osky_remix.core.config import project_dir
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    repo = ProjectRepo.create(project_dir("mt"), Project(name="mt", sample_rate=SR))
    _project_with_audio(repo, tmp_path, {"music": pink(seconds=8, seed=2)})
    rf = tmp_path / "ref.wav"
    sf.write(rf, pink(seconds=8, seed=4, gain=boost_band(B8K, -6.0)).astype(np.float32), SR, subtype="FLOAT")
    res = CliRunner().invoke(app, ["match", "mt", "--reference", str(rf), "--set", "--json"])
    assert res.exit_code == 0, res.output
    rep = json.loads(res.stdout.strip().splitlines()[-1])
    assert [f.get("band") for f in rep["findings"] if f["code"] == "reference_tonal"] == ["8k"]
    assert rep["tonal_delta_db"]["8k"] < -5
    p = repo.load()
    assert p.meta["reference_track"]["sha256"] == rep["reference_sha256"]
    assert p.assets[rep["reference_sha256"]].kind == AssetKind.reference
    # sin --reference usa la registrada (y el render en caché)
    again = CliRunner().invoke(app, ["match", "mt", "--json"])
    assert again.exit_code == 0, again.output
    assert json.loads(again.stdout.strip().splitlines()[-1])["tonal_delta_db"] == rep["tonal_delta_db"]
