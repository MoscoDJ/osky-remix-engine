import hashlib
import json
import pytest
from osky_remix.core.environment import separator_environment


def test_model_recipe_drift_is_rejected_across_hf_revisions(tmp_path, monkeypatch):
    import osky_remix.core.environment as env
    import osky_remix.separation.base as base
    monkeypatch.setattr(base, 'tool_python', lambda _:None)
    monkeypatch.setattr(env, 'REPO_ROOT', tmp_path)
    monkeypatch.setenv('HF_HUB_CACHE', str(tmp_path/'hub'))
    monkeypatch.setenv('TORCH_HOME', str(tmp_path/'torch'))
    root = tmp_path/'hub/models--adefossez--HTDemucs-ft'
    (root/'refs').mkdir(parents=True)
    old, new = root/'snapshots/old', root/'snapshots/new'
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old/'htdemucs_ft.yaml').write_text('original recipe')
    (new/'htdemucs_ft.yaml').write_text('changed recipe')
    (tmp_path/'requirements').mkdir()
    pin = {'demucs/models--adefossez--HTDemucs-ft/htdemucs_ft.yaml':hashlib.sha256(b'original recipe').hexdigest()}
    (tmp_path/'requirements/model-hashes.json').write_text(json.dumps(pin))
    (root/'refs/main').write_text('old')
    assert separator_environment('demucs')['models'] == pin
    (root/'refs/main').write_text('new')
    with pytest.raises(ValueError, match='modelo alterado'):
        separator_environment('demucs')
