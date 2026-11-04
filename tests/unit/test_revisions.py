import json
import os
import stat

import pytest

from osky_remix.core.cache import CacheStore, cache_key
from osky_remix.core.revisions import MergeConflict, ProjectRepo, RepoError
from osky_remix.core.transactions import TransactionError

TRACK = lambda i: {"op": "add_track", "id": i, "name": i, "kind": "instrument"}
CLIP = lambda c, t, start=0: {"op": "add_clip", "track": t, "clip": {"type": "midi", "id": c, "start_tick": start,
                                                                     "length_ticks": 3840, "notes": []}}


def test_commit_creates_immutable_chain(repo):
    r1 = repo.commit([TRACK("bass")], "claude", "bajo")
    r2 = repo.commit([CLIP("b1", "bass")], "claude", "clip")
    log = repo.log()
    assert [r.id for r in log[:2]] == [r2.id, r1.id]
    assert log[0].parent == r1.id
    assert repo.load(r1.id).track("bass").clips == []
    assert len(repo.load().track("bass").clips) == 1


def test_failed_commit_leaves_head_untouched(repo):
    h = repo.head()
    with pytest.raises(TransactionError):
        repo.commit([TRACK("bass"), CLIP("b1", "no-existe")], "claude", "malo")
    assert repo.head() == h
    assert all(t.id != "bass" for t in repo.load().tracks)


def test_revert_creates_new_revision(repo):
    r1 = repo.commit([TRACK("bass")], "claude", "bajo")
    repo.commit([TRACK("pad")], "claude", "pad")
    r3 = repo.revert(r1.id, "claude")
    assert r3.kind == "revert" and r3.parent != r1.id
    assert {t.id for t in repo.load().tracks} == {"master", "bass"}
    assert len(repo.log()) == 4                      # historial preservado


def test_revert_that_alters_locked_track_needs_user(repo):
    r1 = repo.commit([TRACK("bass")], "claude", "bajo")
    repo.commit([CLIP("b1", "bass"), {"op": "lock_track", "track": "bass"}], "user", "edición manual")
    with pytest.raises(RepoError, match="bloqueadas"):
        repo.revert(r1.id, "claude")
    repo.revert(r1.id, "user")


def test_fast_forward_and_divergent_merge(repo):
    repo.commit([TRACK("bass")], "user", "bajo")
    repo.create_branch("idea")
    repo.checkout("idea")
    repo.commit([CLIP("b1", "bass")], "claude", "clip en rama")
    repo.checkout("main")
    r = repo.merge("idea", "claude")                 # main no avanzó: avance rápido
    assert repo.head() == repo.head("idea") and r.kind == "ops"
    # divergencia sin conflicto
    repo.create_branch("v2")
    repo.commit([TRACK("pad")], "user", "pad en main")
    repo.checkout("v2")
    repo.commit([TRACK("lead"), CLIP("l1", "lead")], "claude", "lead en v2")
    repo.checkout("main")
    m = repo.merge("v2", "claude")
    assert m.kind == "merge" and m.merge_parent == repo.head("v2")
    assert {t.id for t in repo.load().tracks} == {"master", "bass", "pad", "lead"}
    assert repo.merge("v2", "claude") is None        # ya fusionada


def test_merge_conflict_is_atomic(repo):
    repo.commit([TRACK("bass"), CLIP("b1", "bass")], "user", "base")
    repo.create_branch("alt")
    repo.commit([{"op": "lock_track", "track": "bass"}], "user", "bloqueo manual")
    repo.checkout("alt")
    repo.commit([{"op": "transpose_midi", "clip": "b1", "semitones": 3}], "claude", "transponer")
    repo.checkout("main")
    h = repo.head()
    with pytest.raises(MergeConflict, match="bloquead"):
        repo.merge("alt", "claude")
    assert repo.head() == h                          # nada escrito


def test_corrupted_snapshot_is_detected(repo):
    import sqlite3
    r = repo.commit([TRACK("bass")], "claude", "bajo")
    repo.engine.dispose()
    con = sqlite3.connect(repo.root / "project.db")
    snap = json.loads(con.execute("select snapshot from revisions where id=?", (r.id,)).fetchone()[0])
    snap["tracks"][1]["gain_db"] = 6.0
    con.execute("update revisions set snapshot=? where id=?", (json.dumps(snap), r.id))
    con.commit(); con.close()
    with pytest.raises(RepoError, match="corrupta"):
        ProjectRepo(repo.root).load()


def test_asset_store_dedupes_and_is_read_only(repo_with_asset, wav):
    repo, a = repo_with_asset
    from osky_remix.core.models import AssetKind, Provenance
    again = repo.assets.ingest(wav, AssetKind.original, Provenance(tool="test"))
    assert again.sha256 == a.sha256 and a.frames == 88200 and a.sample_rate == 44100
    p = repo.assets.find(a.sha256)
    assert not os.stat(p).st_mode & stat.S_IWUSR
    assert repo.assets.verify(a)


def test_register_asset_requires_file_in_store(repo, wav):
    from osky_remix.core.models import AssetKind, Provenance
    from osky_remix.core.assets import AssetStore
    other = AssetStore(repo.root.parent / "otro")
    a = other.ingest(wav, AssetKind.original, Provenance(tool="test"))
    with pytest.raises(TransactionError, match="almacén"):
        repo.commit([{"op": "register_asset", "asset": a.model_dump(mode="json")}], "claude", "x")


def test_cache_key_and_atomic_put(tmp_path):
    c = CacheStore(tmp_path / "cache")
    k = cache_key("tempo", "1", ["b" * 64, "a" * 64], {"hop": 512})
    assert k == cache_key("tempo", "1", ["a" * 64, "b" * 64], {"hop": 512})
    assert k != cache_key("tempo", "2", ["a" * 64, "b" * 64], {"hop": 512})
    assert c.get_json(k) is None
    c.put_json(k, {"bpm": 124})
    assert c.get_json(k) == {"bpm": 124}
