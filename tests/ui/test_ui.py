"""Pruebas end-to-end de la interfaz (Playwright): el navegador hace lo que haría el usuario y
se verifica el efecto en el historial del proyecto (autor, operación, valores)."""
import socket
import threading
import time

import pytest

from osky_remix.api.app import FRONTEND_DIST

pytestmark = [pytest.mark.ui, pytest.mark.skipif(not FRONTEND_DIST.exists(), reason="frontend sin compilar")]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import os

    import uvicorn
    ws = tmp_path_factory.mktemp("ws")
    os.environ["OSKY_WORKSPACE"] = str(ws)
    from osky_remix.core.revisions import ProjectRepo
    from tests.integration.test_api import make_ws_project
    repo, _ = make_ws_project(ws)
    repo.create_branch("alt")
    from osky_remix.api.app import create_app
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.1)
    yield {"url": f"http://127.0.0.1:{port}", "repo": ProjectRepo(ws / "projects" / "demo")}
    srv.should_exit = True
    th.join(timeout=5)
    os.environ.pop("OSKY_WORKSPACE", None)


def last(repo, branch="main"):
    return repo.log(branch, 1)[0]


def open_app(page, server):
    page.goto(server["url"])
    page.wait_for_selector("[data-clip='k1']")


def test_timeline_shows_tracks_and_clips(page, server):
    open_app(page, server)
    for t in ("kick", "pad", "vox"):
        assert page.locator(f"[data-track='{t}']").count() == 1
    assert page.locator("[data-clip='v1'] path.wave").count() == 1          # forma de onda real
    assert page.locator("text=intro").first.is_visible()


def test_lock_toggle_creates_user_revision(page, server):
    open_app(page, server)
    page.click("button[aria-label='bloqueo kick']")
    page.wait_for_selector("[data-track='kick'].locked")
    r = last(server["repo"])
    assert r.author == "user" and r.ops[0]["op"] == "lock_track"
    page.click("button[aria-label='bloqueo kick']")
    page.wait_for_selector("[data-track='kick']:not(.locked)")
    assert last(server["repo"]).ops[0]["op"] == "unlock_track"


def test_fader_commits_on_release(page, server):
    open_app(page, server)
    page.click("text=Mezclador")
    fader = page.locator("input[aria-label='ganancia kick']")
    fader.focus()
    for _ in range(6):
        page.keyboard.press("ArrowDown")                                    # -0.5 dB por paso
    page.wait_for_function("document.querySelector(\"[data-strip='kick'] .db\").textContent.includes('-3.0')")
    deadline = time.time() + 10
    while time.time() < deadline and server["repo"].load().track("kick").gain_db != -3.0:
        time.sleep(0.2)
    assert server["repo"].load().track("kick").gain_db == -3.0
    r = last(server["repo"])
    assert r.author == "user" and r.ops == [{"op": "set_track", "track": "kick", "gain_db": -3.0, "name": None,
                                             "pan": None, "mute": None, "solo": None, "output": None,
                                             "sidechain_from": None}]          # 6 teclas = 1 revisión


def test_drag_clip_moves_and_locked_clip_does_not(page, server):
    open_app(page, server)
    clip = page.locator("[data-clip='v1'] rect").first
    box = clip.bounding_box()
    page.mouse.move(box["x"] + 10, box["y"] + 10)
    page.mouse.down()
    page.mouse.move(box["x"] + 10 + 28 * 2, box["y"] + 10, steps=6)          # 2 compases a 28 px/compás
    page.mouse.up()
    deadline = time.time() + 10
    while time.time() < deadline and server["repo"].load().find_clip("v1")[1].start_tick == 3840:
        time.sleep(0.2)
    assert server["repo"].load().find_clip("v1")[1].start_tick == 3 * 3840
    head = server["repo"].head()
    pclip = page.locator("[data-clip='p1'] rect").first                      # 'pad' está bloqueada
    b = pclip.bounding_box()
    page.mouse.move(b["x"] + 20, b["y"] + 10)
    page.mouse.down()
    page.mouse.move(b["x"] + 120, b["y"] + 10, steps=6)
    page.mouse.up()
    time.sleep(1)
    assert server["repo"].head() == head


def test_midi_editor_add_note_and_save(page, server):
    open_app(page, server)
    server["repo"].commit([{"op": "unlock_track", "track": "pad"}], "user", "editar pad")
    page.reload()
    page.wait_for_selector("[data-clip='p1']")
    page.click("[data-clip='p1'] rect")
    page.wait_for_selector("text=Guardar")
    n0 = len(server["repo"].load().find_clip("p1")[1].notes)
    grid = page.locator("[data-roll='grid']")
    g = grid.bounding_box()
    page.mouse.click(g["x"] + 5, g["y"] + 5)                                 # nota nueva arriba a la izquierda
    page.click("text=Guardar")
    deadline = time.time() + 10
    while time.time() < deadline and len(server["repo"].load().find_clip("p1")[1].notes) == n0:
        time.sleep(0.2)
    assert len(server["repo"].load().find_clip("p1")[1].notes) == n0 + 1
    assert last(server["repo"]).ops[0]["op"] == "set_midi_notes"


def test_render_and_ab(page, server):
    server["repo"].commit([{"op": "set_track", "track": "kick", "gain_db": -6}], "user", "alt", "alt")
    open_app(page, server)
    page.click("text=Renderizar")
    page.wait_for_selector("[data-testid=master-wave][data-ready='1']", timeout=120000)
    assert page.locator("text=calidad").count() >= 1
    page.click("nav.tabs >> text=A/B")
    page.select_option("select[aria-label=A]", "main")
    page.select_option("select[aria-label=B]", "alt")
    page.click("text=Comparar")
    page.wait_for_selector("[data-testid=ab-report]", timeout=120000)
    assert page.locator("text=kick.gain_db").count() >= 1
    page.click("[data-testid=ab-toggle]")
    assert "Escuchando B" in page.inner_text("[data-testid=ab-toggle]")
