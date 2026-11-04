"""Registro de escucha ciega: orden X/Y sorteado con semilla, traducción a A/B y acuerdo con el evaluador."""
import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from osky_remix.storage.listening import ListeningError, ListeningStore, blind_order, summarize

from .test_api import client, wait_job  # noqa: F401  (fixture)

pytestmark = pytest.mark.audio


def test_blind_order_seeded_and_balanced():
    orders = [blind_order(s) for s in range(400)]
    assert orders == [blind_order(s) for s in range(400)]             # reproducible
    assert set(orders) == {"ab", "ba"} and 160 < orders.count("ab") < 240


def _store(tmp_path):
    return ListeningStore(tmp_path / "listening.db")


def _session(store, seed, sa=1.0, sb=2.0, profile="club"):
    return store.create_session(rev_a="ra", rev_b="rb", label_a="main", label_b="alt", profile=profile,
                                render_key_a="ka", render_key_b="kb", score_a=sa, score_b=sb, b_gain_db=0.5, seed=seed)


def test_verdict_translates_xy_to_ab(tmp_path):
    st = _store(tmp_path)
    s_ab = next(s for s in range(100) if blind_order(s) == "ab")
    s_ba = next(s for s in range(100) if blind_order(s) == "ba")
    a = _session(st, s_ab)
    b = _session(st, s_ba)
    assert a["order"] == "ab" and b["order"] == "ba"
    va = st.record(a["session"], "x", 5, 2, "X brilla")                 # X = A
    vb = st.record(b["session"], "x", 5, 2)                             # X = B
    assert (va["preferred"], va["preferred_rev"], va["rating_a"], va["rating_b"], va["x_was"]) == ("a", "ra", 5, 2, "a")
    assert (vb["preferred"], vb["preferred_rev"], vb["rating_a"], vb["rating_b"], vb["x_was"]) == ("b", "rb", 2, 5, "b")
    assert va["note"] == "X brilla" and va["seed"] == s_ab and va["score_a"] == 1.0
    with pytest.raises(ListeningError):
        st.record(a["session"], "y", 3, 3)                              # un solo veredicto por sesión
    with pytest.raises(ListeningError):
        st.record(_session(st, 1)["session"], "x", 6, 1)
    assert [v["session"] for v in st.verdicts()] == [b["session"], a["session"]]   # más reciente primero


def test_agreement_statistic_on_known_data():
    def v(pref, sa, sb, profile="club", ra=3, rb=3):
        return {"preferred": pref, "score_a": sa, "score_b": sb, "profile": profile, "rating_a": ra, "rating_b": rb}
    data = [v("a", 1.0, 2.0, ra=5, rb=2),            # acuerdo (A preferida, A menor puntaje)
            v("b", 1.0, 2.0, ra=2, rb=4),            # desacuerdo
            v("b", 3.0, 0.5, ra=1, rb=5),            # acuerdo
            v("a", 0.2, 0.9, "streaming", 4, 3),     # acuerdo
            v("none", 1.0, 2.0),                     # sin preferencia: fuera del %
            v("a", 1.0, 1.0),                        # empate de puntaje: fuera del %
            v("a", None, 2.0)]                       # sin puntaje: fuera del %
    s = summarize(data)
    assert (s["verdicts"], s["no_preference"], s["pairs"], s["agree"], s["score_ties"]) == (7, 1, 4, 3, 1)
    assert s["agreement_pct"] == 75.0
    assert s["by_profile"] == {"club": {"pairs": 3, "agree": 2, "agreement_pct": 66.7},
                               "streaming": {"pairs": 1, "agree": 1, "agreement_pct": 100.0}}
    assert s["mean_rating_preferred"] == round((5 + 4 + 5 + 4 + 3 + 3) / 6, 2)
    assert s["mean_rating_other"] == round((2 + 2 + 1 + 3 + 3 + 3) / 6, 2)
    assert summarize([])["agreement_pct"] is None


def test_api_blind_round_trip_and_cli(client, tmp_path):  # noqa: F811
    client.post("/api/projects/demo/branches", json={"name": "alt", "from_rev": "main"})
    client.post("/api/projects/demo/ops", json={"ops": [{"op": "set_track", "track": "kick", "gain_db": -6}],
                                                "message": "alt", "branch": "alt"})
    heads = client.get("/api/projects/demo").json()["branches"]
    seed = next(s for s in range(100) if blind_order(s) == "ba")       # X = B (alt)
    j = wait_job(client, client.post("/api/projects/demo/jobs", json={"kind": "blind", "a": "main", "b": "alt",
                                                                      "profile": "club", "seed": seed}).json()["job"])
    assert j["status"] == "done", j["error"]
    res = j["result"]
    assert set(res) == {"session", "x_url", "y_url", "profile"}          # nada revela la identidad
    assert res["x_url"].endswith("/X.wav") and res["y_url"].endswith("/Y.wav")
    root = tmp_path
    x, sr = sf.read(root / res["x_url"][len("/files/"):], always_2d=True)
    y, _ = sf.read(root / res["y_url"][len("/files/"):], always_2d=True)
    from osky_remix.mastering.loudness import integrated_lufs
    assert abs(integrated_lufs(x, sr) - integrated_lufs(y, sr)) < 0.2  # igualadas en sonoridad
    assert not np.allclose(x, y)
    # X es B (alt): su bombo está 6 dB abajo, así que X y el master de A no coinciden
    ra = wait_job(client, client.post("/api/projects/demo/jobs", json={"kind": "render", "rev": "main", "profile": "club"}).json()["job"])
    ma, _ = sf.read(root / ra["result"]["master_url"][len("/files/"):], always_2d=True)
    assert np.allclose(y, ma, atol=1e-6) and not np.allclose(x[: len(ma)], ma, atol=1e-3)
    assert client.get(res["x_url"]).status_code == 200

    bad = client.post("/api/projects/demo/listening", json={"session": res["session"], "preferred": "x",
                                                            "rating_x": 0, "rating_y": 3})
    assert bad.status_code == 400
    r = client.post("/api/projects/demo/listening", json={"session": res["session"], "preferred": "y",
                                                          "rating_x": 2, "rating_y": 4, "note": "más pegada"})
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["x_was"] == "b" and v["preferred"] == "a" and v["preferred_rev"] == heads["main"]
    assert (v["rev_a"], v["rev_b"], v["label_a"], v["label_b"]) == (heads["main"], heads["alt"], "main", "alt")
    assert (v["rating_a"], v["rating_b"], v["note"], v["profile"], v["seed"]) == (4, 2, "más pegada", "club", seed)
    assert v["render_key_a"] == ra["result"]["cache_key"] and len(v["render_key_b"]) == 64
    ev = wait_job(client, client.post("/api/projects/demo/jobs", json={"kind": "evaluate", "rev": "main", "profile": "club"}).json()["job"])
    assert v["score_a"] == ev["result"]["score"] and isinstance(v["score_b"], float)
    assert client.post("/api/projects/demo/listening", json={"session": res["session"], "preferred": "x",
                                                             "rating_x": 3, "rating_y": 3}).status_code == 400
    got = client.get("/api/projects/demo/listening").json()
    assert [g["id"] for g in got["verdicts"]] == [v["id"]] and got["verdicts"][0] == v
    expected = 100.0 if v["score_a"] < v["score_b"] else (0.0 if v["score_a"] > v["score_b"] else None)
    assert got["summary"]["verdicts"] == 1 and got["summary"]["agreement_pct"] == expected

    from osky_remix.cli.main import app
    out = CliRunner().invoke(app, ["listening", "demo", "--json"])
    assert out.exit_code == 0, out.output
    import json
    data = json.loads(out.output)
    assert data["verdicts"][0]["preferred_rev"] == heads["main"] and data["summary"] == got["summary"]
    txt = CliRunner().invoke(app, ["listening", "demo"])
    flat = "".join(txt.output.split())                                 # la tabla de rich parte las celdas
    assert txt.exit_code == 0 and "Acuerdocon" in flat and "pegada" in flat
    if expected is not None:
        assert f"{expected:.1f}%(" in flat


def test_api_visualize_job_values_and_cache(client, tmp_path):  # noqa: F811
    j = wait_job(client, client.post("/api/projects/demo/jobs", json={"kind": "visualize", "profile": "club"}).json()["job"])
    assert j["status"] == "done", j["error"]
    v = j["result"]
    assert [s["label"] for s in v["sections"]] == ["intro", "voz", "drop"]
    # 124 BPM, 4 compases por sección: cada una dura 16 · 60/124 s
    assert v["sections"][1]["start_s"] == pytest.approx(16 * 60 / 124, abs=1e-3)
    assert v["target_lufs"] == -9.0
    r = wait_job(client, client.post("/api/projects/demo/jobs", json={"kind": "render", "profile": "club"}).json()["job"])
    assert v["render_key"] == r["result"]["cache_key"]
    vals = [x for x in v["lufs_s"]["lufs"] if x is not None]
    assert vals and max(vals) <= 0 and abs(np.median(vals) - r["result"]["lufs"]) < 3   # alrededor del integrado
    from osky_remix.analysis.visualize import decode_spectrogram
    assert decode_spectrogram(v["spectrogram"]).max() == 255
    from osky_remix.core.cache import cache_key
    from osky_remix.core.revisions import ProjectRepo
    repo = ProjectRepo(tmp_path / "projects" / "demo")
    assert repo.cache.get_json(cache_key("visualize", "1", [v["render_key"]])) is not None
