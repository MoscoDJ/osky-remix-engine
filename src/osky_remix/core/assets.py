"""Almacén de assets direccionado por contenido (SHA-256). Nunca modifica ni borra originales."""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import soundfile as sf

from .models import Asset, AssetKind, Provenance

AUDIO_EXT = {".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3"}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


class AssetStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, sha: str, ext: str) -> Path:
        return self.root / sha[:2] / f"{sha}{ext}"

    def find(self, sha: str) -> Path | None:
        d = self.root / sha[:2]
        if not d.exists():
            return None
        hits = list(d.glob(f"{sha}.*"))
        return hits[0] if hits else None

    def exists(self, sha: str) -> bool:
        return self.find(sha) is not None

    def ingest(self, src: Path, kind: AssetKind, provenance: Provenance, name: str | None = None) -> Asset:
        """Copia ``src`` al almacén (si no estaba) y devuelve su descriptor. Idempotente."""
        src = Path(src)
        ext = src.suffix.lower() or ".bin"
        sha = sha256_file(src)
        dst = self.path_for(sha, ext)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(ext + ".part")
            shutil.copyfile(src, tmp)
            if sha256_file(tmp) != sha:
                tmp.unlink()
                raise IOError(f"la copia de {src} no coincide con su hash")
            os.replace(tmp, dst)          # atómico en el mismo sistema de archivos
            os.chmod(dst, 0o444)          # solo lectura: el original es inmutable
        sr = ch = frames = None
        if ext in AUDIO_EXT:
            info = sf.info(str(dst))
            sr, ch, frames = info.samplerate, info.channels, info.frames
        return Asset(sha256=sha, kind=kind, name=name or src.name, ext=ext, size_bytes=dst.stat().st_size,
                     sample_rate=sr, channels=ch, frames=frames, provenance=provenance)

    def verify(self, asset: Asset) -> bool:
        p = self.find(asset.sha256)
        return p is not None and sha256_file(p) == asset.sha256
