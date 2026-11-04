"""Ejecuta cada prueba del doctor como test (lento; usa herramientas reales).

    uv run pytest -m compat
"""
import pytest

from osky_remix.doctor.checks import CHECKS
from osky_remix.doctor.runner import run_check

pytestmark = pytest.mark.compat


@pytest.mark.parametrize("chk", [c for c in CHECKS if c.critical], ids=lambda c: c.name)
def test_critical_capability(chk, tmp_path):
    r = run_check(chk, tmp_path)
    assert r["status"] == "PASS", r["detail"]
