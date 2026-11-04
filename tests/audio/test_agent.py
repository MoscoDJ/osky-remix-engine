"""Etapa 5: evaluador, acciones, planificador y variaciones (instrumentos integrados)."""
from dataclasses import replace

import pytest

from osky_remix.agents.actions import ENERGY_LANE, plan_actions
from osky_remix.agents.planner import assess, improve
from osky_remix.agents.policies import DEFAULT
from osky_remix.composition.arranger import arrange, load_template, rearrange
from osky_remix.core.models import Project
from osky_remix.core.operations import Context
from osky_remix.core.revisions import ProjectRepo
from osky_remix.core.transactions import apply_ops, parse_ops

pytestmark = pytest.mark.audio
U = Context(author="user")
BAR = 3840


def flat_project() -> Project:
    """3 secciones de 4 compases a igual nivel (energía plana), integradas, con un pad bloqueado."""
    sec = [("intro", 1), ("voz", 2), ("drop", 3)]
    kick = [{"tick": i * 960, "duration": 240, "pitch": 36, "velocity": 115} for i in range(48)]
    pad = [{"tick": b * BAR, "duration": BAR - 10, "pitch": 57, "velocity": 90} for b in range(12)]
    ops = [{"op": "set_tempo", "points": [{"tick": 0, "bpm": 124}]},
           {"op": "set_master_profile", "profile": "dynamic"},
           {"op": "add_track", "id": "kick", "name": "K", "kind": "instrument"},
           {"op": "add_plugin", "track": "kick", "plugin": {"id": "k", "backend": "builtin", "uri": "drums"}},
           {"op": "add_clip", "track": "kick", "clip": {"type": "midi", "id": "k1", "start_tick": 0, "length_ticks": 12 * BAR, "notes": kick}},
           {"op": "add_track", "id": "pad", "name": "P", "kind": "instrument"},
           {"op": "add_plugin", "track": "pad", "plugin": {"id": "s", "backend": "builtin", "uri": "synth"}},
           {"op": "add_clip", "track": "pad", "clip": {"type": "midi", "id": "p1", "start_tick": 0, "length_ticks": 12 * BAR, "notes": pad}},
           {"op": "set_meta", "key": "arrangement", "value": {"sections": [
               {"name": n, "bars": 4, "energy": e, "vocal": False} for n, e in sec]}}]
    ops += [{"op": "add_marker", "marker": {"id": f"m-{n}", "tick": i * 4 * BAR, "label": n}} for i, (n, _) in enumerate(sec)]
    ops += [{"op": "lock_track", "track": "pad"}]
    return apply_ops(Project(name="agente", sample_rate=44100), parse_ops(ops), U)


@pytest.fixture
def agent_repo(tmp_path):
    return ProjectRepo.create(tmp_path / "ag", flat_project())


def test_evaluator_detects_flat_energy(agent_repo):
    ev = assess(agent_repo.load(), agent_repo)
    codes = [f.code for f in ev.findings]
    assert codes.count("energy_contrast") == 2 and ev.score > 1.0
    secs = {s["name"]: s["lufs"] for s in ev.metrics["sections"]}
    assert abs(secs["intro"] - secs["drop"]) < 1.0


def test_actions_respect_locks(agent_repo):
    p = agent_repo.load()
    ev = assess(p, agent_repo)
    acts, blocked = plan_actions(p, ev)
    assert acts and all(op["lane"]["track"] == "master" for a in acts for op in a.ops)
    locked_master = apply_ops(p, parse_ops([{"op": "lock_track", "track": "master"}]), U)
    acts2, _ = plan_actions(locked_master, ev)
    assert not [a for a in acts2 if a.name == "energy_contrast"]


def test_improve_accepts_only_improvements_and_keeps_locked(agent_repo):
    before = agent_repo.load()
    head0 = agent_repo.head()
    res = improve(agent_repo, "main", replace(DEFAULT, allow_loudness_tradeoff=False), log=lambda *_: None)
    assert res.final.score < res.initial.score
    accepted = [s for s in res.steps if s.accepted]
    assert accepted and len(agent_repo.log()) == 1 + len(accepted)          # solo lo aceptado queda
    after = agent_repo.load()
    assert after.track("pad").model_dump() == before.track("pad").model_dump()
    lane = next(a for a in after.automation if a.id == ENERGY_LANE)
    vals = {pt.tick: pt.value for pt in lane.points}
    assert vals[0] < vals[4 * BAR] <= vals[8 * BAR] == 0.0               # intro < voz < drop
    # una política que exige una mejora imposible no deja rastro en el historial
    head1 = agent_repo.head()
    improve(agent_repo, "main", replace(DEFAULT, min_improvement=1e9), log=lambda *_: None)
    assert agent_repo.head() == head1 != head0


def test_variants_keep_locked_tracks_and_change_the_rest():
    from tests.unit.test_composition import _ctx
    tpl = load_template("melodic_techno")
    p = apply_ops(Project(name="v", sample_rate=44100), parse_ops(arrange(tpl, _ctx(), 32, presets={}).ops),
                  Context(author="claude"))
    p = apply_ops(p, parse_ops([{"op": "set_track", "track": "bass", "gain_db": -2.5},
                                {"op": "lock_track", "track": "bass"}]), U)
    outs = {}
    for v in ("A", "B", "C"):
        plan = rearrange(p, tpl, _ctx(), 32, v, prefix=f"{v.lower()}-")
        q = apply_ops(p, parse_ops(plan.ops), Context(author="claude"))
        assert q.track("bass").model_dump() == p.track("bass").model_dump()
        assert any("bloqueada 'bass'" in s for s in plan.skipped)
        assert q.meta["arrangement"]["variant"] == v
        outs[v] = q
    notes = lambda q, t: sorted((c.start_tick + n.tick, n.pitch) for c in q.track(t).clips for n in c.notes)
    assert notes(outs["A"], "arp") != notes(outs["B"], "arp") != notes(outs["C"], "arp")
    assert notes(outs["A"], "drums") != notes(outs["C"], "drums")
