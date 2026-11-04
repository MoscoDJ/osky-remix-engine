"""Interfaz común de separadores. Los modelos son intercambiables; cada backend declara
qué sabe hacer y se verifica con el doctor."""
from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

STEMS_4 = ("vocals", "drums", "bass", "other")


@dataclass
class SeparationResult:
    stems: dict[str, Path]
    backend: str
    model: str
    tool_version: str | None
    device: str
    seconds: float
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Capabilities:
    stems: tuple[str, ...]
    lead_backing: bool = False


class Separator(ABC):
    name: str = "base"
    default_model: str = ""

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """(disponible, motivo)."""

    @abstractmethod
    def capabilities(self, model: str | None = None) -> Capabilities: ...

    @abstractmethod
    def separate(self, src: Path, out_dir: Path, model: str | None = None) -> SeparationResult: ...


def tool_python(exe_name: str) -> str | None:
    """Intérprete del entorno aislado de una herramienta instalada con uv tool / pipx."""
    exe = shutil.which(exe_name)
    if not exe:
        return None
    first = Path(exe).read_text(errors="ignore").splitlines()[0]
    return first[2:].strip() if first.startswith("#!") else None


def cuda_available(exe_name: str) -> bool:
    py = tool_python(exe_name)
    if not py:
        return False
    r = subprocess.run([py, "-c", "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"],
                       capture_output=True, timeout=120)
    return r.returncode == 0


def package_version(exe_name: str, dist: str) -> str | None:
    py = tool_python(exe_name)
    if not py:
        return None
    r = subprocess.run([py, "-c", f"from importlib.metadata import version;print(version('{dist}'))"],
                       capture_output=True, text=True, timeout=60)
    return r.stdout.strip() or None


def get_separator(name: str) -> Separator:
    from .demucs import DemucsSeparator
    from .roformer import RoformerSeparator
    table = {"demucs": DemucsSeparator, "roformer": RoformerSeparator}
    if name not in table:
        raise ValueError(f"backend desconocido '{name}' (disponibles: {', '.join(table)})")
    return table[name]()
