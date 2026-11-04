"""Renderizador con instrumentos integrados y verdad conocida (sin Surge)."""
import hashlib

import numpy as np
import pytest
import soundfile as sf

from osky_remix.audio.renderer import Renderer
from osky_remix.core.models import AssetKind, Project, Provenance
from osky_remix.core.operations import Context, RegisterAsset
from osky_remix.core.transactions import apply_ops, parse_ops
from osky_remix.mastering.true_peak import true_peak_dbtp

SR = 44100
pytestmark = pytest.mark.audio


def _project(tmp_path, repo):
    t = np.arange(SR) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    f = tmp_path / "tono.wav"
    sf.write(f, np.stack([tone, tone], 1), SR)
    a = repo.assets.ingest(f, AssetKind.stem, Provenance(tool="test"))
    ops = [
        {"op": "set_tempo", "points": [{"tick": 0, "bpm": 120}]},
        {"op": "add_track", "id": "kick", "name": "K", "kind": "instrument"},
        {"op": "add_plugin", "track": "kick", "plugin": {"id": "k", "backend": "builtin", "uri": "drums"}},
        {"op": "add_clip", "track": "kick", "clip": {"type": "midi", "id": "k1", "start_tick": 0, "length_ticks": 4 * 3840,
                                                      "notes": [{"tick": i * 960, "duration": 240, "pitch": 36, "velocity": 120} for i in range(16)]}},
        {"op": "add_track", "id": "bus", "name": "Bus", "kind": "bus"},
        {"op": "add_track", "id": "rev", "name": "Rev", "kind": "return"},
        {"op": "add_plugin", "track": "rev", "plugin": {"id": "r", "backend": "builtin", "uri": "reverb", "params": {"wet": 1.0}}},
        {"op": "add_track", "id": "pad", "name": "Pad", "kind": "instrument", "output": "bus"},
        {"op": "add_plugin", "track": "pad", "plugin": {"id": "s", "backend": "builtin", "uri": "synth"}},
        {"op": "add_clip", "track": "pad", "clip": {"type": "midi", "id": "p1", "start_tick": 0, "length_ticks": 4 * 3840,
                                                     "notes": [{"tick": 0, "duration": 4 * 3840 - 10, "pitch": 57, "velocity": 100}]}},
        {"op": "set_track", "track": "pad", "sidechain_from": "kick"},
        {"op": "add_plugin", "track": "pad", "plugin": {"id": "sc", "backend": "builtin", "uri": "sidechain", "params": {"depth": 0.8}}},
        {"op": "add_track", "id": "vox", "name": "Vox", "kind": "audio"},
        {"op": "add_clip", "track": "vox", "clip": {"type": "audio", "id": "v1", "asset": a.sha256, "start_tick": 3840,
                                                     "source_offset": 0, "source_length": SR // 2}},
    ]
    p = apply_ops(Project(name="r", sample_rate=SR), [RegisterAsset(asset=a)] + parse_ops(ops), Context(author="user"))
    return p


def test_render_positions_routing_sidechain_and_ceiling(tmp_path, repo):
    p = _project(tmp_path, repo)
    r = Renderer(p, repo.assets, repo.cache)
    rep = r.render(tmp_path / "out", "club", stems=True)
    kick = r.output("kick").mean(1)
    # ataques del bombo exactamente cada 0.5 s
    on = [i * SR // 2 for i in range(16)]
    assert np.abs(kick[:20]).max() > 0.1
    for o in on[1:]:
        assert np.abs(kick[o:o + 20]).max() > 10 * np.abs(kick[o - 60:o - 10]).max() + 1e-6
    vox = r.output("vox").mean(1)
    start = 2 * SR                                                  # compás 1 a 120 BPM = 2.0 s
    assert np.abs(vox[:start - 1]).max() == 0 and np.abs(vox[start:start + 100]).max() > 0
    assert np.abs(vox[start + SR // 2 + 1:]).max() == 0             # longitud exacta del clip
    pad = r.output("pad").mean(1)
    just_after = np.sqrt(np.mean(pad[SR // 2 + 200:SR // 2 + 2000] ** 2))
    before = np.sqrt(np.mean(pad[SR // 2 - 6000:SR // 2 - 200] ** 2))
    assert just_after < 0.5 * before                                 # el sidechain agacha tras el bombo
    assert np.allclose(r.output("bus"), r.output("pad"), atol=1e-6) # el bus recibe la pista
    m, _ = sf.read(rep.files["master"])
    assert true_peak_dbtp(m, SR) <= -1.0 + 0.05 and abs(rep.lufs + 9) <= 0.5
    assert not rep.warnings or all("silencio" in w for w in rep.warnings)


def test_render_is_reproducible(tmp_path, repo):
    p = _project(tmp_path, repo)
    h = []
    for i in (1, 2):
        rep = Renderer(p, repo.assets, repo.cache).render(tmp_path / f"o{i}", "club", stems=False)
        h.append(hashlib.sha256(open(rep.files["master"], "rb").read()).hexdigest())
    assert h[0] == h[1]
