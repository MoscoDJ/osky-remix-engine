from dataclasses import replace
import json
import pytest
from osky_remix.audio.render_service import render_cached, render_identity
from osky_remix.mastering.dynamics import load_profile
from tests.audio.test_renderer import _project

pytestmark = pytest.mark.audio


def test_render_profile_identity_and_repair(tmp_path, repo, monkeypatch):
    monkeypatch.setenv('OSKY_WORKSPACE', str(tmp_path / 'ws'))
    p = _project(tmp_path, repo)
    from osky_remix.core.revisions import ProjectRepo
    r = ProjectRepo.create(tmp_path / 'renderrepo', p)
    r.assets = repo.assets
    profile = load_profile('club')
    rid, report, out = render_cached(r, 'demo', 'main', profile)
    before = (out / 'mix_master.wav').stat().st_mtime_ns
    assert render_cached(r, 'demo', rid, profile)[2] == out
    assert (out / 'mix_master.wav').stat().st_mtime_ns == before
    changed = replace(profile, target_lufs=-14)
    _, second, out2 = render_cached(r, 'demo', rid, changed)
    assert out2 != out and abs(second['lufs'] + 14) < .3
    assert render_identity(rid, profile, {'version':1}) != render_identity(rid, profile, {'version':2})
    (out / 'mix_master.wav').write_bytes(b'broken')
    render_cached(r, 'demo', rid, profile)
    assert (out / 'mix_master.wav').read_bytes()[:4] == b'RIFF'
    assert json.loads((out / 'mix_report.json').read_text())['revision'] == rid


def test_same_named_profile_files_change_identity(tmp_path, repo, monkeypatch):
    from dataclasses import asdict
    monkeypatch.setenv('OSKY_WORKSPACE',str(tmp_path/'ws'))
    profile = asdict(load_profile('streaming'))
    paths = [tmp_path/'a'/'same.json',tmp_path/'b'/'same.json']
    for path in paths:
        path.parent.mkdir()
        path.write_text(json.dumps(profile))
    key = render_identity(repo.head(),load_profile(str(paths[0])))
    profile['target_lufs'] = -16
    paths[0].write_text(json.dumps(profile))
    assert render_identity(repo.head(),load_profile(str(paths[0]))) != key
    assert render_identity(repo.head(),load_profile(str(paths[1]))) == key
