import os
import time

from osky_remix.storage.gc import collect


def _age(path, days):
    t = time.time() - days * 86400
    os.utime(path, (t, t))


def test_gc_removes_only_stale_derived_results(tmp_path):
    key = "c" * 64
    renders = tmp_path / "renders" / "demo"
    old, fresh, orphan = renders / ("a" * 64), renders / key, renders / f"{key}-x1"
    for d in (old, fresh, orphan):
        d.mkdir(parents=True)
        (d / "mix_master.wav").write_bytes(b"0" * 1000)
    stale_lock = renders / ("d" * 64 + ".lock")
    stale_lock.write_text("")
    project = tmp_path / "projects" / "demo" / "assets"
    project.mkdir(parents=True)
    (project / "x.wav").write_bytes(b"1")
    for p in (old, orphan, stale_lock, project):
        _age(p, 60)
    _age(project.parent, 60)

    dry = collect(30, root=tmp_path)
    assert old.exists() and orphan.exists()             # por defecto solo informa
    assert {"renders/demo/" + p.name for p in (old, orphan, stale_lock)} == set(dry.removed)

    rep = collect(30, root=tmp_path, dry_run=False)
    assert rep.freed_bytes >= 2000
    assert not old.exists() and not orphan.exists() and not stale_lock.exists()
    assert fresh.exists() and (project / "x.wav").exists()
