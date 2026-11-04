from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .base import STEMS_4, Capabilities, SeparationResult, Separator, cuda_available, package_version


class DemucsSeparator(Separator):
    """Demucs v4 (herramienta independiente: ``uv tool install demucs``)."""
    name = "demucs"
    default_model = "htdemucs_ft"

    def available(self):
        return (True, "ok") if shutil.which("demucs") else (False, "demucs no está instalado")

    def capabilities(self, model=None):
        m = model or self.default_model
        stems = STEMS_4 + (("guitar", "piano") if m == "htdemucs_6s" else ())
        return Capabilities(stems=stems, lead_backing=False)

    def separate(self, src: Path, out_dir: Path, model: str | None = None) -> SeparationResult:
        model = model or self.default_model
        dev = "cuda" if cuda_available("demucs") else "cpu"
        out_dir.mkdir(parents=True, exist_ok=True)
        t = time.time()
        p = subprocess.run(["demucs", "-n", model, "-d", dev, "--float32", "-o", str(out_dir), str(src)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError("demucs falló: " + p.stderr.strip()[-400:])
        d = out_dir / model / Path(src).stem
        stems = {s: d / f"{s}.wav" for s in self.capabilities(model).stems}
        missing = [s for s, f in stems.items() if not f.exists()]
        if missing:
            raise RuntimeError(f"demucs no produjo: {', '.join(missing)}")
        return SeparationResult(stems, self.name, model, package_version("demucs", "demucs"), dev,
                                round(time.time() - t, 2), {"float32": True})
