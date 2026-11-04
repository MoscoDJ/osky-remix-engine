import os
import time
import pytest
from osky_remix.storage.jobs import JobQueue


def ok(project, params):
    return {"project": project, "value": params["value"]}


def crash(project, params):
    os._exit(23)


def hang(project, params):
    time.sleep(30)


def wait(queue, jid, statuses=("done", "failed", "cancelled")):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = queue.get(jid)
        if job['status'] in statuses:
            return job
        time.sleep(.05)
    raise AssertionError(queue.get(jid))


def test_crash_is_contained_and_next_job_succeeds(tmp_path):
    q = JobQueue(tmp_path / 'jobs.db', {'crash':crash, 'ok':ok})
    try:
        assert wait(q, q.submit('demo', 'crash', {}))['status'] == 'failed'
        assert wait(q, q.submit('demo', 'ok', {'value':42}))['result'] == {'project':'demo', 'value':42}
        with pytest.raises(RuntimeError, match='ya existe'):
            JobQueue(tmp_path / 'jobs.db', {'ok':ok})
    finally:
        q.stop()


def test_timeout_and_cancel(tmp_path):
    q = JobQueue(tmp_path / 'jobs.db', {'hang':hang, 'ok':ok}, timeout_s=1.5)
    try:
        job = wait(q, q.submit('demo','hang',{}))
        assert job['status'] == 'failed' and 'tiempo máximo' in job['error']
        jid = q.submit('demo', 'hang', {})
        wait(q, jid, ('running',))
        assert q.cancel(jid)
        assert wait(q, jid)['status'] == 'cancelled'
        assert wait(q, q.submit('demo','ok',{'value':7}))['result']['value'] == 7
    finally:
        q.stop()


def test_restart_marks_interrupted_jobs_and_resumes_queued(tmp_path):
    from sqlalchemy.orm import Session
    from osky_remix.storage.jobs import JobRow, _now
    db = tmp_path / 'restart.db'
    q = JobQueue(db, {'ok':ok})
    q.stop()
    with Session(q.engine) as s, s.begin():
        import json
        for jid,status in [('interrupted','running'), ('waiting','queued')]:
            s.add(JobRow(id=jid,project='demo',kind='ok',params=json.dumps({'value':9}),status=status,created=_now(),updated=_now()))
    resumed = JobQueue(db, {'ok':ok})
    try:
        assert resumed.get('interrupted')['status'] == 'failed'
        assert wait(resumed,'waiting')['result']['value'] == 9
    finally:
        resumed.stop()



def test_cancel_publishes_cancelled_not_failed(tmp_path):
    events = []
    q = JobQueue(tmp_path / 'jobs.db', {'hang': hang}, notify=events.append)
    try:
        jid = q.submit('demo', 'hang', {})
        wait(q, jid, ('running',))
        started = time.monotonic()
        assert q.cancel(jid)
        assert wait(q, jid)['status'] == 'cancelled'
        deadline = time.monotonic() + 5
        while q._running is not None and time.monotonic() < deadline:
            time.sleep(.05)
        assert time.monotonic() - started < 3          # el evento corta sin esperar al proceso
        statuses = [e['status'] for e in events if e.get('id') == jid]
        assert statuses[-1] == 'cancelled' and 'failed' not in statuses
    finally:
        q.stop()


def test_cancelled_render_leaves_no_orphans(tmp_path):
    from osky_remix.core.cache import CacheStore
    store = CacheStore(tmp_path / 'cache')
    key = 'ab' * 32
    orphan = tmp_path / 'cache' / key[:2] / f'{key}-dead'
    orphan.mkdir(parents=True)                          # temporal de un proceso terminado a la fuerza
    store.put_json(key, {'v': 1})
    assert store.get_json(key) == {'v': 1} and not orphan.exists()