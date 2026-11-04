"""Etapa 2 por la CLI: ingesta (original + copia de trabajo) y análisis registrado como asset."""
import json

import pytest
import soundfile as sf
from typer.testing import CliRunner

from osky_remix.cli.main import app
from osky_remix.doctor.songs import pop_song

run = CliRunner().invoke
pytestmark = pytest.mark.audio


def J(res):
    assert res.exit_code == 0, res.output
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_ingest_and_analyze(tmp_path, monkeypatch):
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    a, gt = pop_song()
    src = tmp_path / "cancion.flac"
    sf.write(src, a, 44100, subtype="PCM_24")
    J(run(app, ["project", "new", "p", "--sr", "48000", "--json"]))
    ing = J(run(app, ["ingest", "p", str(src), "--json"]))
    show = json.loads(run(app, ["project", "show", "p", "--json"]).stdout)
    work = show["assets"][ing["working"]]
    assert work["kind"] == "working" and work["sample_rate"] == 48000
    assert work["provenance"]["parents"] == [ing["original"]]
    assert show["assets"][ing["original"]]["ext"] == ".flac"           # original intacto
    md = tmp_path / "a.md"
    r = J(run(app, ["analyze", "p", ing["working"][:12], "--md", str(md), "--json"]))
    assert abs(r["tempo"] - 100) < 0.3 and r["key"] == "D minor" and r["meter"] == 4
    assert "## Secciones" in md.read_text()
    show = json.loads(run(app, ["project", "show", "p", "--json"]).stdout)
    an = show["assets"][r["asset"]]
    assert an["kind"] == "analysis" and an["provenance"]["parents"] == [ing["working"]]
    again = J(run(app, ["analyze", "p", ing["working"][:12], "--json"]))       # caché
    assert again["asset"] == r["asset"]
