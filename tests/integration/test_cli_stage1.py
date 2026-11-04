"""Demostración de la etapa 1 de punta a punta a través de la CLI (como la usaría Claude)."""
import json

from typer.testing import CliRunner

from osky_remix.cli.main import app

run = CliRunner().invoke


def J(res):
    assert res.exit_code == 0, res.output
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_stage1_demo(tmp_path, wav, monkeypatch):
    monkeypatch.setenv("OSKY_WORKSPACE", str(tmp_path / "ws"))
    J(run(app, ["project", "new", "demo", "--sr", "44100", "--json"]))
    sha = J(run(app, ["asset", "import", "demo", str(wav), "--kind", "stem", "--json"]))["sha256"]
    ops = [
        {"op": "add_track", "id": "vox", "name": "Voz", "kind": "audio"},
        {"op": "add_clip", "track": "vox", "clip": {"type": "audio", "id": "v1", "asset": sha,
                                                    "start_tick": 15360, "source_length": 88200}},
        {"op": "add_track", "id": "bass", "name": "Bajo", "kind": "instrument"},
        {"op": "add_clip", "track": "bass", "clip": {"type": "midi", "id": "b1", "start_tick": 0, "length_ticks": 3840,
                                                     "notes": [{"tick": 240, "duration": 200, "pitch": 38, "velocity": 100}]}},
        {"op": "set_key", "key": "Dm"},
    ]
    f = tmp_path / "ops.json"
    f.write_text(json.dumps(ops))
    r = J(run(app, ["op", "apply", "demo", str(f), "-a", "claude", "-m", "base", "--json"]))
    assert r["ok"] and r["ops"] == 5

    # lote inválido: código 2, motivo con índice, sin cambios
    bad = run(app, ["op", "apply", "demo", "-", "-a", "claude", "-m", "x", "--json"],
              input=json.dumps([{"op": "transpose_midi", "clip": "b1", "semitones": 2},
                                {"op": "delete_clip", "clip": "no-existe"}]))
    assert bad.exit_code == 2
    assert json.loads(bad.stdout)["index"] == 1
    show = json.loads(run(app, ["project", "show", "demo", "--json"]).stdout)
    assert show["tracks"][2]["clips"][0]["notes"][0]["pitch"] == 38

    # tres arreglos en ramas sin perder la edición manual (bloqueada) de la voz
    J(run(app, ["op", "apply", "demo", "-", "-a", "user", "-m", "voz final", "--json"],
          input=json.dumps([{"op": "lock_track", "track": "vox"}])))
    for i, semis in enumerate((0, 3, -2), start=1):
        assert run(app, ["project", "branch", "demo", f"arreglo{i}", "--from", "main"]).exit_code == 0
        J(run(app, ["op", "apply", "demo", "-", "-a", "claude", "-m", f"arreglo {i}", "--branch", f"arreglo{i}", "--json"],
              input=json.dumps([{"op": "transpose_midi", "clip": "b1", "semitones": semis},
                                {"op": "move_clip", "clip": "b1", "start_tick": 3840 * i}])))
    for i in (1, 2, 3):
        assert run(app, ["project", "checkout", "demo", f"arreglo{i}"]).exit_code == 0
        p = json.loads(run(app, ["project", "show", "demo", "--json"]).stdout)
        vox = next(t for t in p["tracks"] if t["id"] == "vox")
        assert vox["locked"] and vox["clips"][0]["start_tick"] == 15360

    assert run(app, ["asset", "verify", "demo"]).exit_code == 0
    log = json.loads(run(app, ["project", "log", "demo", "--json"]).stdout)
    assert [x["message"] for x in log[:2]] == ["arreglo 3", "voz final"]
