"""Comandos de la etapa 3: compose y render."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import soundfile as sf
import typer

from ..core.cache import cache_key
from ..core.models import AssetKind, Provenance
from ..core.operations import RegisterAsset
from ..core.transactions import TransactionError
from ..core.revisions import RepoError
from .analyze import _asset_path
from .project import _author, _fail, _out, _repo, console


def _parse_time(s: str) -> float:
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


def compose(name: str,
            analysis: str = typer.Option(..., "--analysis", help="sha del reporte de análisis"),
            vocals: Optional[str] = typer.Option(None, "--vocals", help="sha del stem de voz (sin estirar)"),
            at: str = typer.Option(..., "--at", help="inicio de la frase vocal en el original (m:ss.s)"),
            bars: int = typer.Option(32, "--bars"),
            template: str = typer.Option("melodic_techno", "--template", "-t"),
            bpm: Optional[float] = typer.Option(None, "--bpm"),
            seed: int = typer.Option(1, "--seed"),
            branch: Optional[str] = typer.Option(None, "--branch"),
            author: str = typer.Option("claude", "--author", "-a"),
            message: Optional[str] = typer.Option(None, "-m"),
            as_json: bool = typer.Option(False, "--json")):
    """Genera un arreglo (operaciones atómicas) a partir del análisis y la voz extraída."""
    from ..analysis.pipeline import AnalysisReport
    from ..analysis.transcribe import transcribe
    from ..audio.plugin_host import SURGE_FACTORY, SURGE_XT_VST3
    from ..audio.stretch import rubberband_version, stretch
    from ..composition.arranger import arrange, load_template
    from ..composition.context import VocalSource, build_context
    repo = _repo(name)
    try:
        tpl = load_template(template)
    except ValueError as e:
        _fail(str(e))
    bpm = bpm or tpl["bpm"]
    _, apath = _asset_path(repo, analysis)
    report = AnalysisReport.model_validate_json(apath.read_text())
    ctx = build_context(report, _parse_time(at), bpm, seed)
    sr = report.sample_rate
    new_assets = []
    if vocals:
        vsha, vpath = _asset_path(repo, vocals)
        ratio = ctx.ratio
        key = cache_key("stretch", "1", [vsha], {"ratio": round(ratio, 10)})
        hit = repo.cache.get_json(key)
        if hit and repo.assets.exists(hit["sha"]):
            st_asset = repo.load().assets.get(hit["sha"])
        else:
            st_asset = None
        if st_asset is None:
            console.print(f"Estirando la voz ×{ratio:.4f} ({ctx.source_bpm:.3f} → {bpm} BPM)…")
            with tempfile.TemporaryDirectory() as td:
                out = stretch(vpath, Path(td) / "vocals_stretched.wav", ratio)
                st_asset = repo.assets.ingest(out, AssetKind.stem, Provenance(
                    tool="rubberband", tool_version=rubberband_version(),
                    params={"tempo_ratio": ratio, "from_bpm": ctx.source_bpm, "to_bpm": bpm, "formant": True},
                    parents=[vsha]), name=f"voz ×{ratio:.3f}.wav")
            new_assets.append(st_asset)
            repo.cache.put_json(key, {"sha": st_asset.sha256})
        spb = 60 / bpm * sr
        region = int(round(ctx.beats[ctx.b0] / ratio))
        ctx.vocal = VocalSource(st_asset.sha256, sr, region, spb, st_asset.frames)
        # melodía de la voz para el lead (Basic Pitch, en caché)
        mkey = cache_key("vocal_melody", "1", [st_asset.sha256], {"region": region, "bars": bars})
        mel = repo.cache.get_json(mkey)
        if mel is None:
            x, _ = sf.read(str(repo.assets.find(st_asset.sha256)), dtype="float32", always_2d=True)
            seg = x[region: region + int(bars * 4 * spb)]
            with tempfile.TemporaryDirectory() as td:
                f = Path(td) / "seg.wav"
                sf.write(f, seg, sr)
                clean, _raw = transcribe(f)
            mel = {"notes": [(n.start, n.end, n.pitch) for n in clean]}
            repo.cache.put_json(mkey, mel)
        ctx.melody = [tuple(n) for n in mel["notes"]]
    presets = {}
    if SURGE_XT_VST3.exists():
        for role, spec in tpl["instruments"].items():
            fxp = SURGE_FACTORY / f"{spec['surge_preset']}.fxp"
            if fxp.exists():
                a = repo.assets.ingest(fxp, AssetKind.preset, Provenance(tool="surge-xt-factory",
                                                                         params={"preset": spec["surge_preset"]}))
                presets[role] = a.sha256
                new_assets.append(a)
    else:
        console.print("[yellow]Surge XT no disponible: se usa el sintetizador integrado[/]")
    plan = arrange(tpl, ctx, bars, presets)
    known = repo.load(branch).assets if branch else repo.load().assets
    reg = [{"op": "register_asset", "asset": a.model_dump(mode="json")} for a in
           {a.sha256: a for a in new_assets}.values() if a.sha256 not in known]
    try:
        rev = repo.commit(reg + plan.ops, _author(author),
                          message or f"Arreglo {template} de {bars} compases (seed {seed})", branch)
    except TransactionError as e:
        _fail(f"{e} — no se aplicó ningún cambio (¿pistas ya existentes? usa una rama nueva)", 2)
    except RepoError as e:
        _fail(str(e))
    info = {"revision": rev.id, "ops": len(rev.ops), "tracks": plan.tracks, "key": ctx.key,
            "source_bpm": ctx.source_bpm, "bpm": bpm,
            "harmony": ctx.halfbars(0, min(bars, 8))[::2], "melody_notes": len(ctx.melody)}
    _out(as_json, info,
         f"[green]OK[/] revisión {rev.id[:12]} · {len(rev.ops)} operaciones · pistas {', '.join(plan.tracks)}\n"
         f"Tonalidad {ctx.key}; voz {ctx.source_bpm:.3f} → {bpm} BPM; armonía (primeros compases): "
         + " ".join(info["harmony"]) + f"; melodía del lead: {len(ctx.melody)} notas transcritas")


def render(name: str, rev: Optional[str] = typer.Option(None, "--rev"),
           out: Optional[Path] = typer.Option(None, "--out", help="carpeta de salida"),
           profile: str = typer.Option("club", "--profile"),
           stems: bool = typer.Option(True, "--stems/--no-stems"),
           as_json: bool = typer.Option(False, "--json")):
    """Renderiza la revisión: premaster float, master WAV 24 bits, stems y reporte."""
    from ..audio.renderer import Renderer
    from ..core.config import workspace
    repo = _repo(name)
    try:
        p = repo.load(rev)
        rid = repo.resolve(rev) if rev else repo.head()
    except RepoError as e:
        _fail(str(e))
    out = out or workspace() / "renders" / name / rid[:12]
    console.print(f"Renderizando {name}@{rid[:12]}…")
    r = Renderer(p, repo.assets, repo.cache).render(out, profile, stems)
    data = {"revision": rid, **r.__dict__}
    if as_json:
        print(json.dumps(data, ensure_ascii=False))
        return
    console.print(f"[green]Master[/] {r.files['master']}\n  {r.lufs} LUFS, {r.true_peak_dbtp} dBTP "
                  f"(premaster {r.premaster_lufs} LUFS), {r.length_samples / p.sample_rate:.1f} s")
    for t, m in r.tracks.items():
        console.print(f"  {t:8s} pico {m['peak_dbfs']:7.2f} dBFS  rms {m['rms_dbfs']:7.2f} dBFS")
    for w in r.warnings:
        console.print(f"[yellow]aviso:[/] {w}")
