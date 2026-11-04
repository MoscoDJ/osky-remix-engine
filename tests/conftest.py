from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from osky_remix.core.models import AssetKind, Project, Provenance
from osky_remix.core.revisions import ProjectRepo


@pytest.fixture
def wav(tmp_path) -> Path:
    p = tmp_path / "tono.wav"
    sf.write(p, (0.3 * np.sin(2 * np.pi * 220 * np.arange(44100 * 2) / 44100)).astype(np.float32), 44100)
    return p


@pytest.fixture
def repo(tmp_path) -> ProjectRepo:
    return ProjectRepo.create(tmp_path / "proj", Project(name="prueba", sample_rate=44100))


@pytest.fixture
def repo_with_asset(repo, wav):
    from osky_remix.core.operations import RegisterAsset
    a = repo.assets.ingest(wav, AssetKind.original, Provenance(tool="test"))
    repo.commit([RegisterAsset(asset=a)], "user", "asset")
    return repo, a
