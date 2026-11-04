"""Adaptador RoFormer vía ``audio-separator`` (herramienta independiente).

Modelos verificados:
- ``model_bs_roformer_ep_317_sdr_12.9755.ckpt`` (BS-Roformer Viperx 1297): vocals / instrumental.
- ``mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt``: aplicado a un stem de voz,
  "Vocals" = voz principal y "Instrumental" = coros (lead/backing).

Notas: audio-separator normaliza el pico a 0.9 y escribe 16 bits; ``separation.quality``
corrige ganancia y alineación midiendo contra la mezcla.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from .base import Capabilities, SeparationResult, Separator, cuda_available, package_version

VOCALS_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
KARAOKE_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
MODEL_DIR = Path(os.environ.get("OSKY_SEPARATOR_MODELS", Path.home() / ".cache" / "audio-separator-models"))

# stem del archivo de salida -> nombre canónico, por modelo
STEM_NAMES = {
    VOCALS_MODEL: {"Vocals": "vocals", "Instrumental": "instrumental"},
    KARAOKE_MODEL: {"Vocals": "lead_vocals", "Instrumental": "backing_vocals"},
}


class RoformerSeparator(Separator):
    name = "roformer"
    default_model = VOCALS_MODEL

    def available(self):
        return (True, "ok") if shutil.which("audio-separator") else (False, "audio-separator no está instalado")

    def capabilities(self, model=None):
        m = model or self.default_model
        names = tuple(STEM_NAMES.get(m, {"Vocals": "vocals", "Instrumental": "instrumental"}).values())
        return Capabilities(stems=names, lead_backing=m == KARAOKE_MODEL)

    def separate(self, src: Path, out_dir: Path, model: str | None = None) -> SeparationResult:
        model = model or self.default_model
        out_dir.mkdir(parents=True, exist_ok=True)
        dev = "cuda" if cuda_available("audio-separator") else "cpu"
        t = time.time()
        p = subprocess.run(["audio-separator", str(src), "-m", model, "--output_dir", str(out_dir),
                            "--output_format", "WAV", "--model_file_dir", str(MODEL_DIR)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError("audio-separator falló: " + (p.stderr or p.stdout).strip()[-400:])
        mapping = STEM_NAMES.get(model, {"Vocals": "vocals", "Instrumental": "instrumental"})
        stems = {}
        for f in out_dir.glob(f"{Path(src).stem}_(*)_*.wav"):
            tag = f.name[len(Path(src).stem) + 2:].split(")", 1)[0]
            if tag in mapping:
                stems[mapping[tag]] = f
        missing = set(mapping.values()) - set(stems)
        if missing:
            raise RuntimeError(f"audio-separator no produjo: {', '.join(sorted(missing))}")
        return SeparationResult(stems, self.name, model, package_version("audio-separator", "audio-separator"), dev,
                                round(time.time() - t, 2), {"normalization": 0.9, "bit_depth": 16})
