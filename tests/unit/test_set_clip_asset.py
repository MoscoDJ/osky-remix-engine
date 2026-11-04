"""Operación set_clip_asset: cambia el asset de un clip de audio y nada más."""
import pytest

from osky_remix.core.models import Asset, AudioClip, MidiClip, Project, Provenance
from osky_remix.core.operations import (AddClip, AddTrack, Context, LockTrack, RegisterAsset, SetClipAsset)
from osky_remix.core.transactions import TransactionError, apply_ops, parse_ops

CLAUDE = Context(author="claude")
USER = Context(author="user")


def _asset(c: str, frames=48000, sr=48000) -> Asset:
    return Asset(sha256=c * 64, kind="stem", name=f"{c}.wav", ext=".wav", size_bytes=1, sample_rate=sr, channels=2,
                 frames=frames, provenance=Provenance(tool="test"))


def base() -> Project:
    return apply_ops(Project(name="t"), [
        RegisterAsset(asset=_asset("a")), RegisterAsset(asset=_asset("b")),
        RegisterAsset(asset=_asset("c", frames=1000)), RegisterAsset(asset=_asset("d", sr=44100)),
        AddTrack(id="vox", name="Voz", kind="audio"), AddTrack(id="lead", name="Lead", kind="instrument"),
        AddClip(track="vox", clip=AudioClip(id="v1", asset="a" * 64, start_tick=960, source_offset=100,
                                            source_length=20000, gain_db=-3, fade_in=10, fade_out=20)),
        AddClip(track="lead", clip=MidiClip(id="m1", start_tick=0, length_ticks=960)),
    ], USER)


def test_replaces_only_the_asset_and_keeps_the_old_one():
    p = base()
    q = apply_ops(p, parse_ops([{"op": "set_clip_asset", "clip": "v1", "asset": "b" * 64}]), CLAUDE)
    old, new = p.find_clip("v1")[1], q.find_clip("v1")[1]
    assert new.asset == "b" * 64 and old.asset == "a" * 64                  # entrada sin mutar
    assert new.model_dump(exclude={"asset"}) == old.model_dump(exclude={"asset"})
    assert "a" * 64 in q.assets


@pytest.mark.parametrize("op,msg", [
    (SetClipAsset(clip="v1", asset="e" * 64), "no registrado"),
    (SetClipAsset(clip="m1", asset="b" * 64), "solo aplica a clips de audio"),
    (SetClipAsset(clip="zz", asset="b" * 64), "no existe el clip"),
    (SetClipAsset(clip="v1", asset="c" * 64), "más corto"),
    (SetClipAsset(clip="v1", asset="d" * 64), "44100 Hz"),
])
def test_preconditions(op, msg):
    p = base()
    with pytest.raises(TransactionError, match=msg):
        apply_ops(p, [op], CLAUDE)


def test_locked_track_and_atomic_with_register():
    p = apply_ops(base(), [LockTrack(track="vox")], USER)
    with pytest.raises(TransactionError, match="bloqueada"):
        apply_ops(p, [SetClipAsset(clip="v1", asset="b" * 64)], CLAUDE)
    # registrar y usar en el mismo lote
    q = apply_ops(base(), [RegisterAsset(asset=_asset("f")), SetClipAsset(clip="v1", asset="f" * 64)], CLAUDE)
    assert q.find_clip("v1")[1].asset == "f" * 64


def test_schema_rejects_bad_sha():
    with pytest.raises(TransactionError, match="esquema"):
        parse_ops([{"op": "set_clip_asset", "clip": "v1", "asset": "xyz"}])
