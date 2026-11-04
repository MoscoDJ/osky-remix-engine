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


def test_selected_branch_revert_preserves_main(page, server):
    repo = server['repo']
    name = 'ui-revert'
    repo.create_branch(name, 'main')
    initial = repo.head(name)
    repo.commit([{'op':'set_track','track':'kick','gain_db':-11}], 'user', 'branch edit', name)
    main = repo.head('main')
    open_app(page, server)
    page.select_option('select[aria-label=rama]', name)
    page.wait_for_function("document.querySelector('aside').textContent.includes('branch edit')")
    page.locator(f'[data-rev="{initial}"] button').click()
    deadline = time.time() + 10
    while time.time() < deadline and repo.log(name,1)[0].kind != 'revert':
        time.sleep(.1)
    assert repo.head('main') == main
    assert repo.load(name).content_hash() == repo.load(initial).content_hash()


def test_solo_button_changes_rendered_signal(page, server):
    import numpy as np
    from osky_remix.audio.renderer import Renderer
    open_app(page, server)
    page.click('text=Mezclador')
    page.locator('[data-strip="kick"] button', has_text='S').click()
    deadline = time.time() + 10
    repo = server['repo']
    while time.time() < deadline and not repo.load('main').track('kick').solo:
        time.sleep(.1)
    r = Renderer(repo.load('main'), repo.assets, repo.cache)
    r.output('master')
    assert np.any(r.output('kick'))
    assert not np.any(r.output('vox'))
    assert not np.any(r.output('pad'))


def test_render_shows_visualization_panels(page, server):
    open_app(page, server)
    page.click("text=Renderizar")
    page.wait_for_selector("[data-testid=visuals]", timeout=180000)
    for testid in ("vis-markers-spec", "vis-markers-lufs", "vis-markers-corr"):
        assert page.inner_text(f"[data-testid={testid}]").split() == ["intro", "voz", "drop"]
    # marcadores en su tiempo: 'voz' empieza a 1/3 de los 12 compases (más la cola del render)
    left = page.eval_on_selector("[data-testid=vis-markers-lufs] .vis-marker:nth-child(2)", "e => parseFloat(e.style.left)")
    assert 20 < left < 34
    assert "objetivo -16 LUFS" in page.inner_text(".visuals")          # perfil del proyecto: dynamic
    spec = page.locator("[data-testid=vis-spectrogram]")
    assert spec.get_attribute("data-rows") == "128" and int(spec.get_attribute("data-cols")) > 50
    assert int(page.get_attribute("[data-testid=vis-lufs]", "data-points")) > 50
    assert int(page.get_attribute("[data-testid=vis-corr]", "data-points")) > 50
    # los canvas tienen dibujo real (no solo el elemento)
    lit = page.evaluate("""() => ['vis-spectrogram', 'vis-lufs', 'vis-corr'].map(id => {
        const c = document.querySelector(`[data-testid=${id}]`); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let n = 0; for (let i = 0; i < d.length; i += 4) if (d[i] + d[i + 1] + d[i + 2] > 150) n++; return n; })""")
    assert all(n > 500 for n in lit), lit
    rows = page.inner_text("[data-testid=vis-corr-sections]")
    for lab in ("intro", "voz", "drop"):
        assert lab in rows


def test_blind_listening_records_verdict(page, server):
    from osky_remix.storage.listening import ListeningStore
    repo = server["repo"]
    open_app(page, server)
    page.click("nav.tabs >> text=Escucha ciega")
    page.select_option("select[aria-label='escucha A']", "main")
    page.select_option("select[aria-label='escucha B']", "alt")
    page.click("text=Preparar escucha ciega")
    page.wait_for_selector("[data-testid=wave-x][data-ready='1']", timeout=180000)
    page.wait_for_selector("[data-testid=wave-y][data-ready='1']")
    assert "main" not in page.inner_text("[data-testid=blind-session]")    # nada revela la identidad
    page.click("[data-testid=blind-toggle]")
    assert "Escuchando Y" in page.inner_text("[data-testid=blind-toggle]")
    page.check("input[aria-label='prefiero x']")
    page.click("button[aria-label='X 4']")
    page.click("button[aria-label='Y 2']")
    page.fill("textarea[aria-label=nota]", "X tiene más pegada")
    page.click("text=Guardar y revelar")
    page.wait_for_selector("[data-testid=reveal]")
    v = ListeningStore.for_project(repo.root).verdicts()[0]
    x_is_a = v["x_was"] == "a"
    assert v["preferred"] == ("a" if x_is_a else "b")
    assert (v["rating_a"], v["rating_b"]) == ((4, 2) if x_is_a else (2, 4))
    assert v["rev_a"] == repo.head("main") and v["rev_b"] == repo.head("alt") and v["note"] == "X tiene más pegada"
    reveal = page.inner_text("[data-testid=reveal]")
    assert f"X era {v['x_was'].upper()}" in reveal and f"Preferiste {v['preferred'].upper()}" in reveal
    row = page.locator(f"[data-testid=listening-log] tr[data-verdict='{v['id']}']")
    row.wait_for()
    cells = row.locator("td").all_inner_texts()
    assert cells[1].startswith("main") and cells[2].startswith("alt") and cells[4] == v["preferred"].upper()
    assert cells[5] == f"{v['rating_a']} / {v['rating_b']}" and cells[7] == "X tiene más pegada"
    assert "veredictos: 1" in page.inner_text("[data-testid=listening-summary]")
