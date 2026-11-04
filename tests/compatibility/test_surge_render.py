"""Surge XT a través del renderizador: calibración de transposición y congelado."""
import numpy as np
import pytest

from osky_remix.audio.plugin_host import SURGE_FACTORY, SURGE_XT_VST3
from osky_remix.audio.renderer import Renderer
from osky_remix.core.models import AssetKind, Project, Provenance
from osky_remix.core.operations import Context, RegisterAsset
from osky_remix.core.transactions import apply_ops, parse_ops
from osky_remix.doctor.fixtures import f0_median

pytestmark = [pytest.mark.compat, pytest.mark.skipif(not SURGE_XT_VST3.exists(), reason="sin Surge XT")]
SR = 44100


def _proj(repo, preset: str, pitch: int):
    a = repo.assets.ingest(SURGE_FACTORY / preset, AssetKind.preset, Provenance(tool="test"))
    ops = [{"op": "set_tempo", "points": [{"tick": 0, "bpm": 120}]},
           {"op": "add_track", "id": "s", "name": "S", "kind": "instrument"},
           {"op": "add_plugin", "track": "s", "plugin": {"id": "i", "backend": "surge_xt", "uri": "Surge XT.vst3", "preset": a.sha256}},
           {"op": "add_clip", "track": "s", "clip": {"type": "midi", "id": "c", "start_tick": 0, "length_ticks": 3840,
                                                    "notes": [{"tick": 0, "duration": 1920, "pitch": pitch, "velocity": 100}]}}]
    return apply_ops(Project(name="s", sample_rate=SR), [RegisterAsset(asset=a)] + parse_ops(ops), Context(author="user"))


def test_preset_transposition_is_compensated(repo):
    # "Rubber Bass" suena una octava abajo por diseño: el MIDI A2 (110 Hz) debe sonar a 110 Hz
    r = Renderer(_proj(repo, "Basses/Rubber Bass.fxp", 45), repo.assets, repo.cache)
    y = r.output("s")
    f0 = f0_median(y[: SR], SR, 30, 1000)
    assert abs(12 * np.log2(f0 / 110.0)) < 0.5, f0


def test_stochastic_preset_is_frozen(repo):
    p = _proj(repo, "Pads/Pad 1.fxp", 57)
    a = Renderer(p, repo.assets, repo.cache).output("s")
    b = Renderer(p, repo.assets, repo.cache).output("s")
    assert np.array_equal(a, b)            # Pad 1 es estocástico: la 2a vez sale del congelado
