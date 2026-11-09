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

def test_arrange_assigns_kits_and_surge_params():
    """Kits de samples en kick/drums/fx y parámetros estáticos de Surge desde el juego de sonidos."""
    from tests.unit.test_composition import _ctx
    tpl = load_template("melodic_techno")
    k, f, b = "1" * 64, "2" * 64, "3" * 64
    ops = arrange(tpl, _ctx(), 32, {"kick": k, "fx": f, "bass": b}, preset_params={"bass": {"Volume": 0.4}}).ops
    plug = {o["track"]: o["plugin"] for o in ops if o["op"] == "add_plugin" and o["plugin"]["id"].endswith("-inst")}
    assert plug["kick"] == {"id": "kick-inst", "backend": "builtin", "uri": "drums", "preset": k}
    assert plug["fx"]["uri"] == "fx" and plug["fx"]["preset"] == f
    assert "preset" not in plug["drums"]
    assert plug["bass"]["backend"] == "surge_xt" and plug["bass"]["params"] == {"Volume": 0.4}


def test_drum_clip_comes_before_loudness_tradeoff():
    """Con recortes por sección agotados y el limitador trabajando: primero el clipper del bus de
    batería (2 dB por paso, máximo 6 dB); solo entonces se cede volumen."""
    from osky_remix.agents.actions import DRUM_CLIP_ID
    from osky_remix.agents.evaluator import Evaluation, Finding
    from tests.unit.test_composition import _ctx
    tpl = load_template("melodic_techno")
    p = apply_ops(Project(name="c", sample_rate=44100), parse_ops(arrange(tpl, _ctx(), 32, {}).ops), U)
    secs = [m for m in p.meta.get("sections", [])] or [{"name": "intro", "tick": 0}]
    lane = {"op": "set_automation", "lane": {"id": ENERGY_LANE, "track": "master", "target": "gain_db",
                                               "points": [{"tick": 0, "value": -7.0, "curve": "hold"}]}}
    p = apply_ops(p, parse_ops([lane]), Context(author="claude"))
    ev = Evaluation(3.0, [Finding("energy_contrast", 1.0, "falta contraste", {"section": "intro", "deficit": 1.0})],
                    {"limiting_db": 9.0, "bus_input_peak_dbfs": {"drumbus": 1.5}})
    shaves = []
    for _ in range(4):
        acts, _ = plan_actions(p, ev)
        if acts[0].name != "drum_clip":
            break
        p = apply_ops(p, parse_ops(acts[0].ops), Context(author="claude"))
        slot = next(s for s in p.track("drumbus").plugins if s.id == DRUM_CLIP_ID)
        assert p.track("drumbus").plugins[0].id == DRUM_CLIP_ID                # antes del compresor
        shaves.append((slot.params["shave_db"], slot.params["ceiling_db"]))
    assert shaves == [(2.0, -0.5), (4.0, -2.5), (6.0, -4.5)]
    assert acts[0].name == "loudness_tradeoff"
    # bus bloqueado: no se recorta, se pasa directo al intercambio de volumen
    q = apply_ops(apply_ops(Project(name="c", sample_rate=44100), parse_ops(arrange(tpl, _ctx(), 32, {}).ops), U),
                  parse_ops([lane]), Context(author="claude"))
    q = apply_ops(q, parse_ops([{"op": "lock_track", "track": "drumbus"}]), U)
    assert plan_actions(q, ev)[0][0].name == "loudness_tradeoff"


def _lane(points):
    return {"op": "set_automation", "lane": {"id": ENERGY_LANE, "track": "master", "target": "gain_db",
                                              "points": [{"tick": t, "value": v, "curve": "hold"} for t, v in points]}}


def test_energy_family_order_uses_cuts_of_deficient_sections():
    """El intro ya lleva -7 dB, pero la sección con déficit (voz) solo -2: se recorta antes de ceder."""
    from osky_remix.agents.evaluator import Evaluation, Finding
    pol = replace(DEFAULT, profile_ladder=("dynamic", "streaming"))
    p = apply_ops(flat_project(), parse_ops([_lane([(0, -7.0), (4 * BAR, -2.0), (8 * BAR, 0.0)])]), U)
    ev = Evaluation(2.0, [Finding("energy_contrast", 1.0, "voz plana", {"section": "voz", "deficit": 1.0})],
                    {"limiting_db": 9.0, "bus_input_peak_dbfs": {}})
    assert plan_actions(p, ev, pol)[0][0].name == "energy_contrast"
    p2 = apply_ops(flat_project(), parse_ops([_lane([(0, -7.0), (4 * BAR, -6.5), (8 * BAR, 0.0)])]), U)
    assert plan_actions(p2, ev, pol)[0][0].name == "loudness_tradeoff"      # sin bus de batería: no hay clipper
    # exclusión: la siguiente alternativa es seguir recortando la sección
    assert plan_actions(p2, ev, pol, exclude=frozenset({"loudness_tradeoff"}))[0][0].name == "energy_contrast"


def test_planner_tries_next_alternative_when_first_does_not_improve(agent_repo, monkeypatch):
    """Evaluación simulada: ceder volumen empeora y recortar la sección mejora. El planificador,
    tras descartar lo primero, prueba lo segundo en la misma iteración y confirma solo eso."""
    from osky_remix.agents import planner
    from osky_remix.agents.evaluator import Evaluation, Finding
    agent_repo.commit([_lane([(0, -7.0), (4 * BAR, -6.5), (8 * BAR, 0.0)])], "claude", "cortes previos")
    finding = [Finding("energy_contrast", 2.0, "voz plana", {"section": "voz", "deficit": 1.0})]

    def fake_assess(p, repo, policy=DEFAULT, profile=None):
        if p.master_profile != "dynamic":
            return Evaluation(3.0, finding, {"limiting_db": 9.0, "bus_input_peak_dbfs": {}})
        lane = next(a for a in p.automation if a.id == ENERGY_LANE)
        cut = -next(pt.value for pt in lane.points if pt.tick == 4 * BAR)
        score = max(0.0, 2.0 - (cut - 6.5))
        return Evaluation(score, finding if score else [], {"limiting_db": 9.0, "bus_input_peak_dbfs": {}})
    monkeypatch.setattr(planner, "assess", fake_assess)
    pol = replace(DEFAULT, profile_ladder=("dynamic", "streaming"))
    n0 = len(agent_repo.log("main", 50))
    res = improve(agent_repo, "main", pol, log=lambda *_: None)
    kinds = [("tradeoff" if "perfil" in s.actions[0] else "corte", s.accepted) for s in res.steps]
    assert kinds[:2] == [("tradeoff", False), ("corte", True)]            # misma iteración: alternativa
    assert all(s.iteration == 1 for s in res.steps[:2])
    final = agent_repo.load()
    assert final.master_profile == "dynamic"                               # el descartado no dejó rastro
    assert len(agent_repo.log("main", 50)) - n0 == sum(s.accepted for s in res.steps)


def test_loudness_tradeoff_needs_clear_improvement(agent_repo, monkeypatch):
    """Ceder volumen para borrar un residuo pequeño (0.30) no compensa: se descarta."""
    from osky_remix.agents import planner
    from osky_remix.agents.evaluator import Evaluation, Finding
    agent_repo.commit([_lane([(0, -9.0), (4 * BAR, -9.0), (8 * BAR, 0.0)])], "claude", "cortes al máximo")
    finding = [Finding("energy_contrast", 0.3, "voz casi bien", {"section": "voz", "deficit": 0.2})]
    monkeypatch.setattr(planner, "assess", lambda p, *a, **k: Evaluation(
        0.0 if p.master_profile != "dynamic" else 0.3, [] if p.master_profile != "dynamic" else finding,
        {"limiting_db": 9.0, "bus_input_peak_dbfs": {}}))
    res = improve(agent_repo, "main", replace(DEFAULT, profile_ladder=("dynamic", "streaming")), log=lambda *_: None)
    assert not any(s.accepted for s in res.steps) and agent_repo.load().master_profile == "dynamic"
