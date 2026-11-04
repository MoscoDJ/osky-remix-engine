"""Planificador: renderiza, evalúa, propone, prueba en memoria y solo guarda lo que mejora.

Cada iteración: acciones -> proyecto candidato (apply_ops, misma validación que el contrato) ->
render en memoria -> evaluación. Si el puntaje mejora al menos ``min_improvement``, se confirma
como revisión en la rama (autor claude, mensaje con el motivo y las métricas). Si no, se descarta
sin dejar rastro en el historial.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..audio.renderer import Renderer
from ..core.operations import Context
from ..core.revisions import ProjectRepo
from ..core.transactions import TransactionError, apply_ops, parse_ops
from .actions import plan_actions
from .evaluator import Evaluation, evaluate
from .policies import DEFAULT, Policy


@dataclass
class Step:
    iteration: int
    actions: list[str]
    score_before: float
    score_after: float | None
    accepted: bool
    revision: str | None
    note: str = ""


@dataclass
class ImproveResult:
    initial: Evaluation
    final: Evaluation
    steps: list[Step] = field(default_factory=list)
    needs_user: list[str] = field(default_factory=list)


def assess(p, repo: ProjectRepo, policy: Policy = DEFAULT, profile: str | None = None) -> Evaluation:
    r = Renderer(p, repo.assets, repo.cache)
    _, res, _ = r.run(profile)
    return evaluate(p, r, res.audio, r.report.quality, policy)


def improve(repo: ProjectRepo, branch: str, policy: Policy = DEFAULT, profile: str | None = None,
            log=print) -> ImproveResult:
    p = repo.load(branch)
    ev = assess(p, repo, policy, profile)
    log(f"inicial: {ev.summary()}")
    out = ImproveResult(initial=ev, final=ev)
    for it in range(1, policy.max_iterations + 1):
        acts, blocked = plan_actions(p, ev, policy)
        out.needs_user = blocked
        if not acts:
            log("sin acciones posibles" + (f" ({len(blocked)} hallazgos requieren al usuario)" if blocked else ""))
            break
        ops = [op for a in acts for op in a.ops]
        try:
            cand = apply_ops(p, parse_ops(ops), Context(author=policy.author))
        except TransactionError as e:
            out.steps.append(Step(it, [a.rationale for a in acts], ev.score, None, False, None, f"rechazado por el contrato: {e}"))
            log(f"it{it}: rechazado por el contrato: {e}")
            break
        ev2 = assess(cand, repo, policy, profile)
        better = ev2.score <= ev.score - policy.min_improvement or (ev2.score == 0 and ev.score > 0)
        if better:
            msg = "Agente: " + "; ".join(a.rationale for a in acts) + f" (puntaje {ev.score:.2f}→{ev2.score:.2f})"
            rev = repo.commit(parse_ops(ops), policy.author, msg[:500], branch)
            out.steps.append(Step(it, [a.rationale for a in acts], ev.score, ev2.score, True, rev.id))
            log(f"it{it}: ACEPTADO {ev.score:.2f}→{ev2.score:.2f} · " + "; ".join(a.rationale for a in acts))
            p, ev = cand, ev2
            out.final = ev
            if ev.score == 0:
                break
        else:
            out.steps.append(Step(it, [a.rationale for a in acts], ev.score, ev2.score, False, None, "no mejora"))
            log(f"it{it}: descartado ({ev.score:.2f}→{ev2.score:.2f}) · " + "; ".join(a.rationale for a in acts))
            break
    log(f"final: {out.final.summary()}")
    return out
