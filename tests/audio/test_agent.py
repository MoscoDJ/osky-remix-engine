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



def test_regeneration_keeps_unlocked_manual_content():
    from tests.unit.test_composition import _ctx
    tpl = load_template("melodic_techno")
    p = apply_ops(Project(name="manual", sample_rate=44100), parse_ops(arrange(tpl, _ctx(), 32, {}).ops), U)
    clip = p.track("arp").clips[0]
    p = apply_ops(p, parse_ops([
        {"op":"move_clip", "clip":clip.id, "start_tick":clip.start_tick+120},
        {"op":"add_marker", "marker":{"id":"my-marker", "tick":100, "label":"manual"}},
        {"op":"set_automation", "lane":{"id":"my-auto", "track":"pad", "target":"gain_db",
        "points":[{"tick":0, "value":-3}]}}]), U)
    plan = rearrange(p, tpl, _ctx(), 32, "B", "b-")
    q = apply_ops(p, parse_ops(plan.ops), Context(author="claude"))
    assert q.track("arp") == p.track("arp")
    assert q.track("pad") == p.track("pad")
    assert next(m for m in q.markers if m.id == "my-marker").label == "manual"
    assert next(a for a in q.automation if a.id == "my-auto").points[0].value == -3


def test_agent_does_not_commit_after_manual_edit(agent_repo, monkeypatch):
    import osky_remix.agents.planner as planner
    original = planner.assess
    calls = 0
    def assess_then_edit(*args, **kwargs):
        nonlocal calls
        value = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            agent_repo.commit([{'op':'set_track','track':'kick','gain_db':-1}], 'user', 'manual')
        return value
    monkeypatch.setattr(planner, 'assess', assess_then_edit)
    result = planner.improve(agent_repo, 'main', log=lambda *_:None)
    assert result.needs_user and not any(s.accepted for s in result.steps)
    assert agent_repo.load().track('kick').gain_db == -1
    assert len(agent_repo.log()) == 2



def test_regeneration_does_not_resurrect_manually_deleted_clips():
    from tests.unit.test_composition import _ctx
    tpl = load_template('melodic_techno')
    p = apply_ops(Project(name='deleted',sample_rate=44100),parse_ops(arrange(tpl,_ctx(),32,{}).ops),U)
    cid = p.track('arp').clips[0].id
    p = apply_ops(p,parse_ops([{'op':'delete_clip','clip':cid}]),U)
    q = apply_ops(p,parse_ops(rearrange(p,tpl,_ctx(),32,'B','b-').ops),Context(author='claude'))
    assert q.track('arp') == p.track('arp')



def test_legacy_project_is_preserved_unless_adopted():
    from tests.unit.test_composition import _ctx
    tpl = load_template('melodic_techno')
    p = apply_ops(Project(name='legacy', sample_rate=44100), parse_ops(arrange(tpl, _ctx(), 32, {}).ops), U)
    del p.meta['generated_content']                      # proyecto creado antes del registro
    plan = rearrange(p, tpl, _ctx(), 32, 'B', 'b-')
    q = apply_ops(p, parse_ops(plan.ops), Context(author='claude'))
    assert [t.model_dump() for t in q.tracks] == [t.model_dump() for t in p.tracks]
    assert any('--adopt-legacy' in s for s in plan.skipped)
    adopted = rearrange(p, tpl, _ctx(), 32, 'B', 'b-', adopt_legacy=True)
    r = apply_ops(p, parse_ops(adopted.ops), Context(author='claude'))
    assert adopted.tracks and r.track('arp') != p.track('arp')
    assert 'generated_content' in r.meta and not any('--adopt-legacy' in s for s in adopted.skipped)
    # tras adoptar, el registro protege las ediciones manuales como en un proyecto nuevo
    clip = r.track('arp').clips[0]
    r = apply_ops(r, parse_ops([{'op': 'move_clip', 'clip': clip.id, 'start_tick': clip.start_tick + 120}]), U)
    again = apply_ops(r, parse_ops(rearrange(r, tpl, _ctx(), 32, 'C', 'c-').ops), Context(author='claude'))
    assert again.track('arp') == r.track('arp')