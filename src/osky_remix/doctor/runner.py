"""Ejecuta las pruebas aisladas en subprocesos y arma el reporte."""
from __future__ import annotations

import datetime as dt
import json
import multiprocessing as mp
import platform
import tempfile
import time
import traceback
from pathlib import Path

from .checks import CHECKS, Check


def _child(name: str, work: str, q) -> None:
    from .checks import CHECKS as _C
    chk = next(c for c in _C if c.name == name)
    try:
        q.put(chk.fn(Path(work)))
    except Exception as e:  # noqa: BLE001
        q.put({"status": "FAIL", "detail": f"{type(e).__name__}: {e}",
               "metrics": {"traceback": traceback.format_exc()[-1500:]}})


def run_check(chk: Check, base: Path) -> dict:
    work = base / chk.name
    work.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(chk.name, str(work), q))
    t = time.time()
    p.start()
    p.join(chk.timeout_s)
    if p.is_alive():
        p.kill(); p.join()
        res = {"status": "FAIL", "detail": f"tiempo agotado ({chk.timeout_s}s); proceso aislado y terminado", "metrics": {}}
    elif not q.empty():
        res = q.get()
    else:
        res = {"status": "FAIL", "detail": f"el proceso terminó sin resultado (código {p.exitcode}); "
                                           "posible crash de plugin, aislado", "metrics": {}}
    res.update(name=chk.name, title=chk.title, critical=chk.critical, seconds=round(time.time() - t, 1))
    return res


def run_all(only: list[str] | None, report_dir: Path, keep: bool, on_result=None) -> dict:
    checks = [c for c in CHECKS if not only or c.name in only]
    base = Path(tempfile.mkdtemp(prefix="osky-doctor-"))
    results = []
    for c in checks:
        r = run_check(c, base)
        results.append(r)
        if on_result:
            on_result(r)
    crit_fail = [r["name"] for r in results if r["critical"] and r["status"] != "PASS"]
    report = {
        "tool": "osky-remix doctor",
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "host": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()},
        "results": results,
        "critical_failures": crit_fail,
        "stage0_gate": "PASS" if not crit_fail else "FAIL",
        "artifacts_dir": str(base) if keep else None,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"doctor-{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    report["report_path"] = str(path)
    if not keep:
        import shutil
        shutil.rmtree(base, ignore_errors=True)
    return report
