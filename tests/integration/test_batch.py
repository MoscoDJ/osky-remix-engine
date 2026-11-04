"""Lote: pasos reales (proyecto, ingesta, análisis, composición, variantes, render, evaluación)
sobre canciones sintéticas, con la separación sustituida (demucs no está en las pruebas)."""
import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from osky_remix.batch import steps as S
from osky_remix.batch.manifest import Manifest
from osky_remix.batch.pipeline import BatchError, make_config, run_batch
from osky_remix.batch.report import write_reports
from osky_remix.cli.main import app
from osky_remix.core.config import project_dir
from osky_remix.core.models import AssetKind, Provenance
from osky_remix.core.operations import RegisterAsset
from osky_remix.core.revisions import ProjectRepo
from osky_remix.doctor.fixtures import harmonic_tone, midi_to_hz
from osky_remix.doctor.songs import pop_song
from osky_remix.mastering.loudness import integrated_lufs

pytestmark = pytest.mark.audio
SR = 44100
run = CliRunner().invoke


def stub_separate(voices: Path, ctx: S.SongContext) -> dict:
    """Separador de prueba: la voz sintética conocida + mezcla repartida en bass/other."""
    repo = ProjectRepo(project_dir(ctx.project))
    work = ctx.results["ingest"]["working"]
    mix, sr = sf.read(str(repo.assets.find(work)), dtype="float32", always_2d=True)
    voice, _ = sf.read(str(voices / f"{ctx.file.stem}.wav"), dtype="float32", always_2d=True)
    voice = np.pad(voice, ((0, max(0, len(mix) - len(voice))), (0, 0)))[: len(mix)]
    stems = {"vocals": voice, "bass": mix * 0.5, "other": mix * 0.5, "drums": mix * 0.0}
    assets, shas = [], {}
    d = ctx.file.parent.parent / "stub" / ctx.slug
    d.mkdir(parents=True, exist_ok=True)
    for k, x in stems.items():
        f = d / f"{k}.wav"
        sf.write(f, x, sr, subtype="FLOAT")
        a = repo.assets.ingest(f, AssetKind.stem, Provenance(tool="stub", parents=[work]), name=f"{k}.wav")
        assets.append(a)
        shas[k] = a.sha256
    rp = d / "separation_report.json"
    rp.write_text(json.dumps({"input": work, "backend": "stub", "model": "stub", "stems": shas}))
    ra = repo.assets.ingest(rp, AssetKind.analysis, Provenance(tool="stub", parents=[work]), name=rp.name)
    known = repo.load().assets
    uniq = {a.sha256: a for a in assets + [ra]}.values()          # bass y other son iguales
    ops = [RegisterAsset(asset=a) for a in uniq if a.sha256 not in known]
    if ops:
        repo.commit(ops, "claude", "separación de prueba")
    return {"report": ra.sha256, "stems": shas}


class Counter:
    def __init__(self):
        self.calls: dict[tuple[str, str], int] = {}
        self.crash: tuple[str, str] | None = None

    def wrap(self, step: S.Step) -> S.Step:
        def fn(ctx, _step=step):
            key = (ctx.slug, _step.name)
            self.calls[key] = self.calls.get(key, 0) + 1
            if self.crash == key:
                self.crash = None
                raise KeyboardInterrupt
            return _step.fn(ctx)
        return S.Step(step.name, fn, step.heavy)


def make_songs(tmp_path):
    """Una canción sintética (pop_song, 100 BPM, Re menor) con su voz conocida, y un archivo roto."""
    songs, voices = tmp_path / "canciones", tmp_path / "voces"
    songs.mkdir()
    voices.mkdir()
    song, gt = pop_song()
    sf.write(songs / "cancion-a.wav", song, SR, subtype="FLOAT")
    voice = np.zeros(len(song))
    a0 = gt.sections[1][0]
    for i, m in enumerate([62, 65, 69, 67] * 32):
        t = a0 + i * SR * 60 // 100
        x = harmonic_tone(midi_to_hz(m), 0.55, SR, n_harm=6, decay=1.5) * 0.4
        voice[t:t + len(x)] += x[: max(0, len(voice) - t)]
    sf.write(voices / "cancion-a.wav", np.stack([voice, voice], 1).astype(np.float32), SR, subtype="FLOAT")
    (songs / "roto.wav").write_text("esto no es audio")
    return songs, voices, a0 / SR


def J(res, code=0):
    assert res.exit_code == code, res.output
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_batch_resume_failure_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    # instrumentos integrados (deterministas) aunque Surge XT esté instalado
    monkeypatch.setattr("osky_remix.audio.plugin_host.SURGE_XT_VST3", tmp_path / "sin-surge.vst3")
    songs, voices, a0 = make_songs(tmp_path)
    real = [S.Step("separate", partial(stub_separate, voices), heavy=True) if s.name == "separate" else s
            for s in S.DEFAULT_STEPS]
    names = [s.name for s in real]
    c = Counter()
    steps = [c.wrap(s) for s in real]
    cfg = make_config(songs, "melodic_techno", ["A", "B"], profile=None, bars=32, separator="stub", sample_rate=SR)
    # 1) corte (Ctrl+C) al empezar el render: lo anterior queda hecho en el manifiesto
    c.crash = ("cancion-a", "render")
    with pytest.raises(KeyboardInterrupt):
        run_batch(cfg, steps)
    first = dict(c.calls)
    assert first == {("cancion-a", n): 1 for n in names[: names.index("render") + 1]}
    man = Manifest(next((tmp_path / "ws" / "batches").glob("canciones-*"))).read()
    st = man["songs"]["cancion-a"]["steps"]
    assert st["render"]["status"] == "interrupted" and st["vary"]["status"] == "done"
    assert set(st["vary"]["result"]) == {"A", "B"} and man["songs"]["roto"]["status"] == "pending"
    with pytest.raises(BatchError, match="--resume"):
        run_batch(cfg, steps)
    # 2) reanudar: solo lo pendiente; la canción rota falla sin frenar a la otra
    res = run_batch(cfg, steps, resume=True)
    assert res.songs == {"cancion-a": "done", "roto": "failed"}
    assert res.ran["cancion-a"] == ["render", "evaluate"]
    assert res.skipped["cancion-a"] == names[: names.index("render")]
    delta = {k: v - first.get(k, 0) for k, v in c.calls.items() if v - first.get(k, 0)}
    assert delta == {("cancion-a", "render"): 1, ("cancion-a", "evaluate"): 1,
                     ("roto", "project"): 1, ("roto", "ingest"): 1}
    man = Manifest(res.dir).read()
    assert man["config"] == cfg and set(man["songs"]) == {"cancion-a", "roto"}
    roto = man["songs"]["roto"]
    assert roto["status"] == "failed" and roto["steps"]["ingest"]["status"] == "failed"
    assert "Format not recognised" in roto["steps"]["ingest"]["error"] and "separate" not in roto["steps"]
    st = man["songs"]["cancion-a"]["steps"]
    assert st["render"]["attempts"] == 2 and st["analyze"]["attempts"] == 1
    assert all(x["status"] == "done" for x in st.values())
    # 3) reporte: valores iguales a la salida de evaluate del proyecto y al master renderizado
    rep = write_reports(res.dir)
    by = {r["song"]: r for r in rep["songs"]}
    r = by["cancion-a"]
    assert r["status"] == "done" and abs(r["tempo"] - 100) < 0.5 and r["key"] == "D minor"
    assert abs(r["vocal_segment"]["start_s"] - a0) < 0.15 and r["target_bpm"] == 124
    for v in ("A", "B"):
        x = r["variants"][v]
        assert x["branch"] == f"var-{v.lower()}"
        master, msr = sf.read(x["master"], dtype="float32", always_2d=True)
        assert abs(integrated_lufs(master, msr) - x["lufs"]) < 0.05
    ev = J(run(app, ["evaluate", r["project"], "--rev", "var-b", "--json"]))
    assert r["variants"]["B"]["score"] == ev["score"]
    assert r["variants"]["B"]["findings"] == [f["detail"] for f in ev["findings"]]
    assert r["best_variant"] == min(("A", "B"), key=lambda v: r["variants"][v]["score"])
    assert by["roto"]["failed_step"] == "ingest" and by["roto"]["tempo"] is None
    cmp = rep["comparison"]
    assert cmp["done"] == 1 and cmp["failed"] == ["roto"]
    assert cmp["mean_score"] == {v: round(r["variants"][v]["score"], 4) for v in ("A", "B")}
    md = (res.dir / "report.md").read_text()
    assert "cancion-a" in md and "roto" in md and (res.dir / "songs" / "roto.md").exists()
    assert json.loads((res.dir / "songs" / "cancion-a.json").read_text()) == r
    # 4) la CLI reanuda el mismo lote: todo hecho salvo reintentar la canción rota (sale con 1)
    out = J(run(app, ["batch", str(songs), "--variants", "A,B", "--separator", "stub", "--sr", str(SR),
                      "--resume", "--json"]), code=1)
    assert out["id"] == res.id and out["songs"] == res.songs and out["ran"]["cancion-a"] == []
    m2 = Manifest(res.dir).read()["songs"]["roto"]["steps"]
    assert m2["ingest"]["attempts"] == 2 and m2["project"]["attempts"] == 1


FAKE = {"x": (120.0, "A minor", [3.0, 1.5]), "y": (98.0, "C major", [0.5, 2.5]), "z": (128.0, "F minor", [1.0, 1.0])}


def fake_analyze(ctx):
    bpm, key, _ = FAKE[ctx.slug]
    return {"asset": "0" * 64, "tempo": bpm, "key": key, "meter": 4, "sections": 3}


def fake_segment(ctx):
    return {"at": "0:10.0", "start_s": 10.0, "end_s": 40.0, "reason": "prueba"}


def fake_compose(ctx):
    return {"revision": "r", "key": FAKE[ctx.slug][1], "bpm": 124, "vocals": "v" * 64}


def fake_vary(ctx):
    return {v: {"branch": f"var-{v.lower()}", "revision": v, "label": v} for v in ctx.variants}


def fake_render(ctx):
    return {v: {"master": f"/m/{ctx.slug}/{v}.wav", "lufs": -9.0 - i, "true_peak_dbtp": -1.2, "quality": "PASS",
                "warnings": [f"aviso {v}"] if v == "B" else []} for i, v in enumerate(ctx.variants)}


def fake_evaluate(ctx):
    sc = FAKE[ctx.slug][2]
    if ctx.slug == "z":
        raise RuntimeError("evaluador caído")
    return {v: {"score": sc[i], "findings": [f"hallazgo {v}"]} for i, v in enumerate(ctx.variants)}


def test_report_compares_songs(tmp_path, monkeypatch):
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    d = tmp_path / "in"
    d.mkdir()
    for n in FAKE:
        sf.write(d / f"{n}.wav", np.zeros(100, np.float32), SR)
    steps = [S.Step("analyze", fake_analyze), S.Step("vocal_segment", fake_segment), S.Step("compose", fake_compose),
             S.Step("vary", fake_vary), S.Step("render", fake_render), S.Step("evaluate", fake_evaluate)]
    res = run_batch(make_config(d, "melodic_techno", ["A", "B"], sample_rate=SR), steps)
    assert res.songs == {"x": "done", "y": "done", "z": "failed"}
    rep = write_reports(res.dir)
    by = {r["song"]: r for r in rep["songs"]}
    assert by["x"]["tempo"] == 120.0 and by["x"]["key"] == "A minor" and by["x"]["vocal_segment"]["at"] == "0:10.0"
    assert by["x"]["variants"]["A"]["score"] == 3.0 and by["x"]["variants"]["B"]["lufs"] == -10.0
    assert by["x"]["best_variant"] == "B" and by["y"]["best_variant"] == "A"
    assert by["x"]["warnings"] == ["B: aviso B"]
    assert by["z"]["failed_step"] == "evaluate" and "evaluador caído" in by["z"]["error"]
    assert by["z"]["variants"]["A"]["score"] is None and by["z"]["variants"]["A"]["lufs"] == -9.0
    cmp = rep["comparison"]
    assert cmp["mean_score"] == {"A": 1.75, "B": 2.0} and cmp["wins"] == {"A": 1, "B": 1}
    assert cmp["best_overall"] == "A" and cmp["failed"] == ["z"] and cmp["done"] == 2
    md = (res.dir / "report.md").read_text()
    assert "| x | done |" in md and "1.750" in md and "evaluador caído" in md


def stub_ok(ctx):
    return {"slug": ctx.slug, "file": ctx.file.name}


def stub_fail_b(ctx):
    if ctx.slug == "b":
        raise RuntimeError("fallo provocado")
    return {"ok": True}


def test_batch_parallel_processes_with_stub_steps(tmp_path, monkeypatch):
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    d = tmp_path / "in"
    d.mkdir()
    for n in ("a", "b", "c"):
        sf.write(d / f"{n}.wav", np.zeros(100, np.float32), SR)
    (d / "notas.txt").write_text("ignorado")
    cfg = make_config(d, "melodic_techno", ["A"], separator="stub", sample_rate=SR)
    steps = [S.Step("uno", stub_ok), S.Step("dos", stub_fail_b, heavy=True), S.Step("tres", stub_ok)]
    res = run_batch(cfg, steps, jobs=2)
    assert res.songs == {"a": "done", "b": "failed", "c": "done"}
    man = Manifest(res.dir).read()["songs"]
    assert man["a"]["steps"]["tres"]["result"] == {"slug": "a", "file": "a.wav"}
    assert man["b"]["steps"]["dos"]["status"] == "failed" and "fallo provocado" in man["b"]["error"]
    assert "tres" not in man["b"]["steps"]
    res2 = run_batch(cfg, steps, jobs=2, resume=True)
    assert res2.skipped["a"] == ["uno", "dos", "tres"] and res2.skipped["b"] == ["uno"]


def test_cli_call_maps_errors_and_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    from osky_remix.cli.project import project_new
    from osky_remix.cli.analyze import ingest
    out = S.call_cli(project_new, "uno", sample_rate=SR)
    assert out["project"] == "uno"
    with pytest.raises(S.StepError, match="no existe"):
        S.call_cli(ingest, "uno", tmp_path / "nada.wav")
