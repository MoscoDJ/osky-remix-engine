"""Comandos de la etapa 2: ingest, separate, analyze."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import soundfile as sf
import typer

from ..core.cache import cache_key
from ..core.models import AssetKind, Provenance
from ..core.operations import RegisterAsset
from ..core.revisions import ProjectRepo
from ..core.transactions import TransactionError
from .project import _author, _fail, _out, _repo, console


def _asset_path(repo: ProjectRepo, sha: str) -> tuple[str, Path]:
    p = repo.load()
    hits = [k for k in p.assets if k.startswith(sha)]
    if len(hits) != 1:
        _fail(f"asset '{sha}' {'no encontrado' if not hits else 'ambiguo'}")
    path = repo.assets.find(hits[0])
    if path is None:
        _fail(f"el archivo del asset {hits[0][:12]} no está en el almacén")
    return hits[0], path


def _register(repo: ProjectRepo, assets, author: str, message: str):
    known = repo.load().assets
    ops = [RegisterAsset(asset=a) for a in assets if a.sha256 not in known]
    if not ops:
        return None
    try:
        return repo.commit(ops, author, message)
    except TransactionError as e:
        _fail(str(e))


def _ffmpeg_version() -> str:
    return subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]


def ingest(name: str, file: Path, author: str = typer.Option("user", "--author", "-a"),
           as_json: bool = typer.Option(False, "--json")):
    """Preserva el original y crea la copia de trabajo PCM float32 a la frecuencia del proyecto."""
    if not file.exists():
        _fail(f"no existe {file}")
    repo = _repo(name)
    sr = repo.load().sample_rate
    orig = repo.assets.ingest(file, AssetKind.original, Provenance(tool="ingest", params={"source_path": str(file.resolve())}))
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / f"{file.stem}.wav"
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(file), "-map", "0:a:0", "-ar", str(sr),
                            "-c:a", "pcm_f32le", str(work)], capture_output=True, text=True)
        if p.returncode != 0:
            _fail("ffmpeg no pudo decodificar: " + p.stderr[-300:])
        working = repo.assets.ingest(work, AssetKind.working, Provenance(
            tool="ffmpeg", tool_version=_ffmpeg_version(), params={"sample_rate": sr, "format": "pcm_f32le"},
            parents=[orig.sha256]), name=f"{file.stem} (trabajo).wav")
    r = _register(repo, [orig, working], _author(author), f"Ingesta de {file.name}")
    _out(as_json, {"original": orig.sha256, "working": working.sha256, "revision": r.id if r else None},
         f"Original {orig.sha256[:12]} · trabajo {working.sha256[:12]} ({working.frames} muestras @ {sr} Hz)")


def separate(name: str, sha: str, backend: str = typer.Option("demucs", "--backend", "-b"),
             model: Optional[str] = typer.Option(None, "--model", "-m"),
             lead_backing: bool = typer.Option(False, "--lead-backing", help="Separar voz principal/coros (RoFormer karaoke)"),
             author: str = typer.Option("claude", "--author", "-a"), as_json: bool = typer.Option(False, "--json")):
    """Separa stems, mide y corrige alineación/ganancia, detecta artefactos y registra todo."""
    from ..separation.base import get_separator
    from ..separation.quality import prepare_stems
    from ..separation.roformer import KARAOKE_MODEL, RoformerSeparator
    repo = _repo(name)
    sha, src = _asset_path(repo, sha)
    try:
        sep = get_separator(backend)
    except ValueError as e:
        _fail(str(e))
    ok, why = sep.available()
    if not ok:
        _fail(why)
    model = model or sep.default_model
    key = cache_key("separate", "1", [sha], {"backend": backend, "model": model, "lead_backing": lead_backing})
    cached = repo.cache.get_json(key)
    if cached:
        _out(as_json, cached, f"(caché) reporte {cached['report'][:12]}; stems: "
             + ", ".join(f"{k} {v[:10]}" for k, v in cached["stems"].items()))
        return
    mix, sr = sf.read(str(src), dtype="float32", always_2d=True)
    assets, stem_sha, qualities = [], {}, {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # el separador recibe una copia con nombre simple (los nombres de salida dependen de él)
        inp = td / "input.wav"
        shutil.copyfile(src, inp)
        console.print(f"Separando con {backend}/{model}…")
        res = sep.separate(inp, td / "out", model)
        raw = {k: sf.read(str(v), dtype="float32", always_2d=True) for k, v in res.stems.items()}
        prepared, q = prepare_stems(mix, sr, {k: (x, s) for k, (x, s) in raw.items()})
        qualities[backend] = q
        jobs = [(res, prepared, sha)]
        if lead_backing:
            vk = "vocals"
            if vk not in prepared:
                _fail("--lead-backing necesita un stem 'vocals'")
            vpath = td / "vocals_for_karaoke.wav"
            sf.write(vpath, prepared[vk], sr, subtype="FLOAT")
            kres = RoformerSeparator().separate(vpath, td / "karaoke", KARAOKE_MODEL)
            kraw = {k: sf.read(str(v), dtype="float32", always_2d=True) for k, v in kres.stems.items()}
            kprep, kq = prepare_stems(prepared[vk], sr, {k: (x, s) for k, (x, s) in kraw.items()},
                                       drums_ref=prepared.get("drums"))
            qualities["lead_backing"] = kq
            jobs.append((kres, kprep, None))       # el padre se resuelve tras registrar 'vocals'
        for r, prep, parent in jobs:
            for stem, x in prep.items():
                f = td / f"{stem}.wav"
                sf.write(f, x, sr, subtype="FLOAT")
                par = [parent] if parent else [sha, stem_sha.get("vocals", sha)]
                q = qualities["lead_backing" if parent is None else backend]
                a = repo.assets.ingest(f, AssetKind.stem, Provenance(
                    tool=r.backend, tool_version=r.tool_version, model=r.model,
                    params={**r.params, "device": r.device, "lag_samples": q.lag_samples,
                            "gain_correction_db": q.gain_correction_db, "seconds": r.seconds},
                    parents=list(dict.fromkeys(par))), name=f"{stem}.wav")
                assets.append(a)
                stem_sha[stem] = a.sha256
        report = {"input": sha, "backend": backend, "model": model, "stems": stem_sha,
                  "quality": {k: v.model_dump() for k, v in qualities.items()}}
        rp = td / "separation_report.json"
        rp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        ra = repo.assets.ingest(rp, AssetKind.analysis, Provenance(tool="osky-separation-qc", tool_version="1",
                                                                   parents=[sha, *stem_sha.values()]),
                                name="separation_report.json")
        assets.append(ra)
    rev = _register(repo, assets, _author(author), f"Separación {backend}/{model}")
    summary = {"report": ra.sha256, "stems": stem_sha, "revision": rev.id if rev else None}
    repo.cache.put_json(key, summary)
    if as_json:
        print(json.dumps({**summary, "quality": report["quality"]}, ensure_ascii=False))
        return
    for k, q in qualities.items():
        console.print(f"[bold]{k}[/]: reconstrucción {q.reconstruction_snr_db} dB, desfase {q.lag_samples} muestras, "
                      f"ganancia {q.gain_correction_db:+.2f} dB, {len(q.regions)} regiones a revisar")
        for g in q.regions[:12]:
            console.print(f"   {g.stem:14s} {g.start / sr:7.1f}–{g.end / sr:7.1f}s  {g.reason:20s} severidad {g.severity:.2f}")
    console.print("Stems: " + ", ".join(f"{k} {v[:10]}" for k, v in stem_sha.items()))
    console.print(f"Reporte {ra.sha256[:12]}")


def analyze(name: str, sha: str, stems: Optional[str] = typer.Option(None, "--stems", help="sha del reporte de separación"),
            md: Optional[Path] = typer.Option(None, "--md", help="Guardar el resumen en Markdown"),
            author: str = typer.Option("claude", "--author", "-a"), as_json: bool = typer.Option(False, "--json")):
    """Tempo, compás, tiempos fuertes, tonalidad, acordes, secciones, silencios, transitorios y espectro."""
    from ..analysis.pipeline import ANALYSIS_VERSION, AnalysisReport, report_markdown, run_analysis
    repo = _repo(name)
    sha, src = _asset_path(repo, sha)
    stem_paths, inputs = {}, [sha]
    if stems:
        rsha, rpath = _asset_path(repo, stems)
        rep = json.loads(rpath.read_text())
        for k, v in rep["stems"].items():
            stem_paths[k] = repo.assets.find(v)
            inputs.append(v)
    key = cache_key("analysis", ANALYSIS_VERSION, inputs)
    cached = repo.cache.get_json(key)
    if cached:
        report = AnalysisReport.model_validate(cached["report"])
        asha = cached["asset"]
    else:
        console.print("Analizando…")
        report = run_analysis(src, stem_paths)
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "analysis.json"
            f.write_text(report.model_dump_json())
            a = repo.assets.ingest(f, AssetKind.analysis, Provenance(tool="osky-analysis", tool_version=ANALYSIS_VERSION,
                                                                     parents=inputs), name="analysis.json")
        _register(repo, [a], _author(author), f"Análisis de {sha[:12]}")
        asha = a.sha256
        repo.cache.put_json(key, {"report": json.loads(report.model_dump_json()), "asset": asha})
    text = report_markdown(report, f"Análisis {sha[:12]}")
    if md:
        md.write_text(text)
    if as_json:
        print(json.dumps({"asset": asha, "tempo": report.rhythm.tempo.value, "key": report.harmony.key.value,
                          "meter": report.rhythm.meter.value, "sections": len(report.structure.sections)}))
    else:
        console.print(text)
        console.print(f"[dim]Reporte {asha[:12]}[/]")


def reference(name: str, midi: Path, analysis: str = typer.Option(..., "--analysis"),
              stems: str = typer.Option(..., "--stems", help="sha del reporte de separación"),
              author: str = typer.Option("claude", "--author", "-a"), as_json: bool = typer.Option(False, "--json")):
    """Importa un MIDI de referencia (p. ej. karaoke), lo alinea con la grabación y lo registra."""
    from ..analysis.midi_ref import build_reference, read_midi
    from ..analysis.pipeline import AnalysisReport
    if not midi.exists():
        _fail(f"no existe {midi}")
    repo = _repo(name)
    asha, apath = _asset_path(repo, analysis)
    rep = AnalysisReport.model_validate_json(apath.read_text())
    ssha, spath = _asset_path(repo, stems)
    sep = json.loads(spath.read_text())
    load = lambda k: sf.read(str(repo.assets.find(sep["stems"][k])), dtype="float32", always_2d=True)[0]
    if not {"bass", "other", "drums"} <= set(sep["stems"]):
        _fail("el reporte de separación necesita stems bass, other y drums")
    msha_asset = repo.assets.ingest(midi, AssetKind.midi, Provenance(tool="import", params={"source_path": str(midi.resolve())}))
    vkey = next((k for k in ("lead_vocals", "vocals") if k in sep["stems"]), None)
    vocal_notes = None
    if vkey:
        from ..analysis.transcribe import transcribe
        vn_key = cache_key("vocal_notes", "1", [sep["stems"][vkey]], {})
        hit = repo.cache.get_json(vn_key)
        if hit is None:
            console.print("Transcribiendo la voz original (juez de la alineación)…")
            clean, _ = transcribe(repo.assets.find(sep["stems"][vkey]))
            hit = {"notes": [(n.start, n.end, n.pitch) for n in clean]}
            repo.cache.put_json(vn_key, hit)
        vocal_notes = [tuple(x) for x in hit["notes"]]
    console.print("Alineando el MIDI con la grabación…")
    ref = build_reference(read_midi(midi), load("bass") + load("other"), rep.sample_rate, rep.rhythm.beats,
                          [(c.start, c.end, c.label) for c in rep.harmony.chords], rep.vocal_activity, load("drums"),
                          vocal_notes)
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "midi_reference.json"
        f.write_text(ref.model_dump_json())
        ra = repo.assets.ingest(f, AssetKind.analysis, Provenance(tool="osky-midi-reference", tool_version=ref.version,
                                                                  parents=[msha_asset.sha256, asha, ssha]),
                                name="midi_reference.json")
    _register(repo, [msha_asset, ra], _author(author), f"Referencia MIDI {midi.name}")
    data = {"reference": ra.sha256, "midi": msha_asset.sha256, "confidence": ref.confidence,
            "alignment_similarity": ref.alignment_similarity, "drum_error_ms": ref.drum_error_ms,
            "lyric_vocal_overlap": ref.lyric_vocal_overlap, "chord_agreement": ref.chord_agreement,
            "beat_anchors": ref.beat_anchors, "lines": len(ref.lyrics), "vocal_fit": ref.vocal_fit,
            "channels": {c.channel: f"{c.role} ({len(c.notes)})" for c in ref.channels}}
    if as_json:
        print(json.dumps(data, ensure_ascii=False))
        return
    console.print(f"Referencia {ra.sha256[:12]} · confianza [bold]{ref.confidence:.2f}[/]\n"
                  f"  similitud armónica {ref.alignment_similarity}, batería vs. stem {ref.drum_error_ms} ms, "
                  f"letra sobre voz {ref.lyric_vocal_overlap:.0%}, acordes iguales al audio {ref.chord_agreement:.0%}, "
                  f"{ref.beat_anchors} negras ancladas")
    if ref.vocal_fit:
        best = max(ref.vocal_fit, key=ref.vocal_fit.get)
        console.print("  juez (voz original en la armonía del MIDI, por desfase): "
                      + "  ".join(f"{k:+d}:{v:.0%}" for k, v in sorted(ref.vocal_fit.items()))
                      + ("  ✓" if best in (0, -1) else "  [red]✗ la alineación parece corrida[/]"))
    console.print("  canales: " + ", ".join(f"{k}={v}" for k, v in data["channels"].items()))
    for l in ref.lyrics:
        console.print(f"  {int(l.start // 60)}:{l.start % 60:05.2f}  {l.text}")
