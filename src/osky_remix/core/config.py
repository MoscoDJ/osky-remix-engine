"""Rutas de trabajo. ``OSKY_WORKSPACE`` permite cambiar el workspace (pruebas, otros discos)."""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def workspace() -> Path:
    return Path(os.environ.get("OSKY_WORKSPACE", REPO_ROOT / "workspace")).expanduser()


def project_dir(name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_\-]{0,63}", name):
        raise ValueError("nombre de proyecto inválido: minúsculas, dígitos, '-' y '_' (máx. 64)")
    return workspace() / "projects" / name
