"""Pasos del lote. Cada paso es ``fn(ctx) -> dict`` (el dict queda en el manifiesto).

Los pasos reales llaman a las mismas funciones de la CLI (en proceso, con ``--json``), así que un
lote hace exactamente lo que haría una persona con ``osky-remix ingest/separate/...``; todos son
idempotentes (assets por SHA, resultados en caché, ramas existentes reutilizadas), de modo que
reanudar tras un corte a mitad de un paso es seguro.
"""
from __future__ import annotations

import inspect
import io
import json
import re
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import click
import typer


class StepError(RuntimeError):
    pass


@dataclass
class Step:
    name: str
    fn: Callable[["SongContext"], dict]
    heavy: bool = False          # separación: un solo proceso a la vez aunque haya --jobs N


@dataclass
class SongContext:
    slug: str
    file: Path
    project: str
    config: dict
    results: dict = field(default_factory=dict)        # resultados de pasos anteriores
    log: Callable[[str], None] = lambda *_: None

    @property
    def variants(self) -> list[str]:
        return list(self.config["variants"])


def _strip_markup(s: str) -> str:
    return re.sub(r"\[/?[a-z ]+\]", "", s).strip()


def call_cli(fn, *args, **kwargs):
    """Ejecuta un comando de la CLI en proceso y devuelve su salida ``--json`` (última línea).

    Los parámetros no indicados toman el valor por defecto de su ``typer.Option``; un error de la
    CLI (``typer.Exit`` con código ≠ 0) se convierte en ``StepError`` con el mensaje de stderr.
    """
    from typer.models import ParameterInfo
    sig = inspect.signature(fn)
    bound = sig.bind_partial(*args, **kwargs)
    for name, p in sig.parameters.items():
        if name in bound.arguments:
            continue
        d = p.default
        if isinstance(d, ParameterInfo):
            if d.default is ...:
                raise TypeError(f"{fn.__name__}: falta el parámetro obligatorio '{name}'")
            bound.arguments[name] = d.default
        elif d is inspect.Parameter.empty:
            raise TypeError(f"{fn.__name__}: falta el parámetro '{name}'")
    if "as_json" in sig.parameters:
        bound.arguments["as_json"] = True
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            fn(**bound.arguments)
    except (click.exceptions.Exit, typer.Exit) as e:
        if e.exit_code:
            raise StepError(_strip_markup(err.getvalue()).removeprefix("error:").strip() or f"código {e.exit_code}")
    lines = [ln for ln in out.getvalue().strip().splitlines() if ln.strip()]
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except ValueError:
        raise StepError(f"{fn.__name__}: salida no JSON: {lines[-1][:200]}")


def _repo(ctx: SongContext):
    from ..core.config import project_dir
    from ..core.revisions import ProjectRepo
    return ProjectRepo(project_dir(ctx.project))


def _fmt_time(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


# ------------------------------------------------------------------ pasos reales
def step_project(ctx: SongContext) -> dict:
    from ..cli.project import project_new
    from ..core.config import project_dir
    if (project_dir(ctx.project) / "project.db").exists():
        return {"project": ctx.project, "created": False}
    call_cli(project_new, ctx.project, sample_rate=int(ctx.config["sample_rate"]), author=ctx.config["author"])
    return {"project": ctx.project, "created": True}


def step_ingest(ctx: SongContext) -> dict:
    from ..cli.analyze import ingest
    r = call_cli(ingest, ctx.project, ctx.file, author=ctx.config["author"])
    return {"original": r["original"], "working": r["working"]}


def step_separate(ctx: SongContext) -> dict:
    from ..cli.analyze import separate
    r = call_cli(separate, ctx.project, ctx.results["ingest"]["working"], backend=ctx.config["separator"],
                 author=ctx.config["author"])
    return {"report": r["report"], "stems": r["stems"]}


def step_analyze(ctx: SongContext) -> dict:
    from ..cli.analyze import analyze
    r = call_cli(analyze, ctx.project, ctx.results["ingest"]["working"], stems=ctx.results["separate"]["report"],
                 author=ctx.config["author"])
    return {"asset": r["asset"], "tempo": r["tempo"], "key": r["key"], "meter": r["meter"], "sections": r["sections"]}


def choose_vocal_segment(report, bars: int = 8) -> dict:
    """Inicio de la frase vocal: la primera región con voz que dure al menos ``bars`` compases
    del tempo original (si ninguna, la más larga); sin voz detectada, el inicio de la 2.ª sección."""
    sr = report.sample_rate
    need = bars * report.rhythm.meter.value * 60 / max(report.rhythm.tempo.value, 1e-6) * sr
    regs = report.vocal_activity or []
    if regs:
        long_enough = [r for r in regs if r[1] - r[0] >= need]
        a, b = long_enough[0] if long_enough else max(regs, key=lambda r: r[1] - r[0])
        reason = f"primera región con voz ≥ {bars} compases" if long_enough else "región con voz más larga"
    else:
        secs = report.structure.sections
        a = secs[1].start if len(secs) > 1 else (secs[0].start if secs else 0)
        b = secs[1].end if len(secs) > 1 else report.duration_samples
        reason = "sin actividad vocal: inicio de la segunda sección"
    return {"at": _fmt_time(a / sr), "start_s": round(a / sr, 3), "end_s": round(b / sr, 3), "reason": reason}


def step_vocal_segment(ctx: SongContext) -> dict:
    from ..analysis.pipeline import AnalysisReport
    repo = _repo(ctx)
    path = repo.assets.find(ctx.results["analyze"]["asset"])
    if path is None:
        raise StepError("el análisis no está en el almacén")
    return choose_vocal_segment(AnalysisReport.model_validate_json(path.read_text()))


def _vocals(ctx: SongContext) -> str | None:
    stems = ctx.results.get("separate", {}).get("stems", {})
    return stems.get("lead_vocals") or stems.get("vocals")


def step_compose(ctx: SongContext) -> dict:
    from ..cli.produce import compose
    repo = _repo(ctx)
    seg = ctx.results["vocal_segment"]
    meta = repo.load().meta.get("compose")
    want = {"analysis": ctx.results["analyze"]["asset"], "at": seg["at"], "bars": ctx.config["bars"],
            "template": ctx.config["template"]}
    if meta and all(meta.get(k) == v for k, v in want.items()):
        # el arreglo ya se confirmó (corte entre el commit y el manifiesto): no se repite
        return {"revision": repo.head(), "recovered": True, "key": repo.load().key, "bpm": meta.get("bpm"),
                "vocals": meta.get("vocals")}
    vocals = _vocals(ctx)
    r = call_cli(compose, ctx.project, analysis=want["analysis"], vocals=vocals, at=seg["at"], bars=ctx.config["bars"],
                 template=ctx.config["template"], seed=ctx.config["seed"], author=ctx.config["author"])
    return {"revision": r["revision"], "key": r["key"], "bpm": r["bpm"], "source_bpm": r["source_bpm"],
            "tracks": r["tracks"], "melody_notes": r["melody_notes"], "vocals": vocals}


def _branch(v: str) -> str:
    return f"var-{v.lower()}"


def step_vary(ctx: SongContext) -> dict:
    from ..cli.agent import vary_cmd
    from ..composition.arranger import VARIANTS
    repo = _repo(ctx)
    base = repo.head("main")
    out = {}
    for v in ctx.variants:
        br = _branch(v)
        branches = repo.branches()
        if br in branches and branches[br] != base:
            out[v] = {"branch": br, "revision": branches[br], "label": VARIANTS.get(v, {}).get("label"), "reused": True}
            continue
        if br in branches:
            raise StepError(f"la rama '{br}' existe sin la variante (borrarla o renombrarla a mano)")
        rows = call_cli(vary_cmd, ctx.project, base="main", variants=v, prefix="var", evaluate_=False)
        row = rows[0]
        out[v] = {"branch": row["branch"], "revision": row["revision"], "label": row["label"]}
    return out


def step_render(ctx: SongContext) -> dict:
    from ..audio.render_service import render_cached
    repo = _repo(ctx)
    out = {}
    for v, info in ctx.results["vary"].items():
        rid, rep, d = render_cached(repo, ctx.project, info["branch"], ctx.config.get("profile"))
        out[v] = {"revision": rid, "master": rep["files"]["master"], "lufs": rep["lufs"],
                  "true_peak_dbtp": rep["true_peak_dbtp"], "quality": rep["quality"].get("status"),
                  "profile": rep["profile"].get("name"), "warnings": rep["warnings"]}
    return out


def step_evaluate(ctx: SongContext) -> dict:
    from ..cli.agent import evaluate_cmd
    out = {}
    for v, info in ctx.results["vary"].items():
        r = call_cli(evaluate_cmd, ctx.project, rev=info["branch"], profile=ctx.config.get("profile"))
        out[v] = {"score": r["score"], "findings": [f["detail"] for f in r["findings"]],
                  "lufs": r["metrics"].get("lufs"), "true_peak_dbtp": r["metrics"].get("true_peak_dbtp"),
                  "sections": r["metrics"].get("sections")}
    return out


DEFAULT_STEPS = [
    Step("project", step_project),
    Step("ingest", step_ingest),
    Step("separate", step_separate, heavy=True),
    Step("analyze", step_analyze),
    Step("vocal_segment", step_vocal_segment),
    Step("compose", step_compose),
    Step("vary", step_vary),
    Step("render", step_render),
    Step("evaluate", step_evaluate),
]
