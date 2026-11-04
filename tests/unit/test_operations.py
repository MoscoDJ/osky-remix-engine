import pytest

from osky_remix.core.models import AudioClip, MidiClip, MidiNote, PluginSlot, Project
from osky_remix.core.operations import (AddClip, AddPlugin, AddTrack, Context, LockTrack, MoveClip,
                                        RemovePlugin, RemoveTrack, ReplacePlugin, SetAutomation,
                                        SetTempo, TransposeMidi, UnlockTrack)
from osky_remix.core.transactions import TransactionError, apply_ops, parse_ops

CLAUDE = Context(author="claude")
USER = Context(author="user")


def base() -> Project:
    p = Project(name="t")
    return apply_ops(p, [
        AddTrack(id="bass", name="Bajo", kind="instrument"),
        AddClip(track="bass", clip=MidiClip(id="b1", start_tick=0, length_ticks=3840,
                                            notes=[MidiNote(tick=0, duration=240, pitch=38, velocity=100),
                                                   MidiNote(tick=480, duration=240, pitch=45, velocity=90)])),
        AddPlugin(track="bass", plugin=PluginSlot(id="synth", backend="surge_xt", uri="Surge XT.vst3")),
    ], USER)


def test_ops_return_new_project_and_never_mutate_input():
    p = base()
    snapshot = p.canonical_json()
    q = apply_ops(p, [TransposeMidi(clip="b1", semitones=5)], CLAUDE)
    assert p.canonical_json() == snapshot
    assert [n.pitch for n in q.find_clip("b1")[1].notes] == [43, 50]


def test_batch_is_atomic():
    p = base()
    snapshot = p.canonical_json()
    with pytest.raises(TransactionError) as e:
        apply_ops(p, [TransposeMidi(clip="b1", semitones=2), MoveClip(clip="no-existe", start_tick=0)], CLAUDE)
    assert e.value.index == 1
    assert p.canonical_json() == snapshot


def test_transpose_precondition_range():
    with pytest.raises(TransactionError, match="rango MIDI"):
        apply_ops(base(), [TransposeMidi(clip="b1", semitones=-40)], CLAUDE)


def test_locked_track_blocks_claude_and_unlock_needs_user():
    p = apply_ops(base(), [LockTrack(track="bass")], USER)
    for op in (TransposeMidi(clip="b1", semitones=1), MoveClip(clip="b1", start_tick=960),
               ReplacePlugin(track="bass", plugin_id="synth", plugin=PluginSlot(id="x", backend="builtin", uri="saw")),
               RemoveTrack(track="bass")):
        with pytest.raises(TransactionError, match="bloquead"):
            apply_ops(p, [op], CLAUDE)
    with pytest.raises(TransactionError, match="solo el usuario"):
        apply_ops(p, [UnlockTrack(track="bass")], CLAUDE)
    q = apply_ops(p, [UnlockTrack(track="bass")], USER)
    assert not q.track("bass").locked


def test_tempo_change_refused_when_locked_tracks_have_clips():
    p = apply_ops(base(), [LockTrack(track="bass")], USER)
    with pytest.raises(TransactionError, match="tempo"):
        apply_ops(p, [SetTempo(points=[{"tick": 0, "bpm": 128}])], CLAUDE)


def test_replace_plugin_keeps_position():
    p = apply_ops(base(), [AddPlugin(track="bass", plugin=PluginSlot(id="eq", backend="builtin", uri="eq"))], USER)
    q = apply_ops(p, [ReplacePlugin(track="bass", plugin_id="synth",
                                    plugin=PluginSlot(id="synth2", backend="faust", uri="process=_;"))], CLAUDE)
    assert [s.id for s in q.track("bass").plugins] == ["synth2", "eq"]


def test_remove_plugin_refused_if_automated():
    p = apply_ops(base(), [SetAutomation(lane={"id": "cut", "track": "bass", "target": "synth.cutoff",
                                                "points": [{"tick": 0, "value": 0.1}, {"tick": 3840, "value": 0.9}]})], CLAUDE)
    with pytest.raises(TransactionError, match="automatizaciones"):
        apply_ops(p, [RemovePlugin(track="bass", plugin_id="synth")], CLAUDE)


def test_automation_target_must_exist():
    with pytest.raises(TransactionError, match="plugin"):
        apply_ops(base(), [SetAutomation(lane={"id": "x", "track": "bass", "target": "nope.cutoff",
                                                "points": [{"tick": 0, "value": 0}]})], CLAUDE)


def test_audio_clip_requires_registered_asset_and_audio_track():
    p = apply_ops(base(), [AddTrack(id="vox", name="Voz", kind="audio")], USER)
    clip = AudioClip(id="v1", asset="a" * 64, start_tick=0, source_length=100)
    with pytest.raises(TransactionError, match="asset"):
        apply_ops(p, [AddClip(track="vox", clip=clip)], CLAUDE)
    with pytest.raises(TransactionError, match="pista audio"):
        apply_ops(p, [AddClip(track="bass", clip=clip)], CLAUDE)


def test_cannot_remove_track_that_is_routed_to():
    p = apply_ops(base(), [AddTrack(id="bus", name="Bus", kind="bus"),
                           AddTrack(id="pad", name="Pad", kind="instrument", output="bus")], USER)
    with pytest.raises(TransactionError, match="destino"):
        apply_ops(p, [RemoveTrack(track="bus")], CLAUDE)


def test_parse_ops_reports_index_and_rejects_unknown():
    with pytest.raises(TransactionError) as e:
        parse_ops([{"op": "add_track", "id": "a", "name": "A", "kind": "audio"}, {"op": "borrar_todo"}])
    assert e.value.index == 1
    with pytest.raises(TransactionError):
        parse_ops([{"op": "add_track", "id": "a", "name": "A", "kind": "audio", "extra": 1}])
