"""Biblioteca de samples del workspace (``workspace/samples/``).

- ``assets/``: archivos en un ``AssetStore`` (SHA-256, solo lectura); importar dos veces el mismo
  contenido no duplica nada, aunque venga con otro nombre.
- ``cache/``: rasgos por ``cache_key("sample_features", FEATURES_VERSION, [sha])``: si cambia el
  extractor cambia la clave y se recalcula; si no, re-importar es un acierto de caché.
- ``library.db``: índice SQLite (sha, nombre, origen, rasgos, categoría) para buscar.
"""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..core.assets import AssetStore, sha256_file
from ..core.cache import CacheStore, cache_key
from ..core.config import workspace
from ..core.models import Asset, AssetKind, Provenance
from . import features as F

SAMPLE_EXT = {".wav", ".flac", ".aif", ".aiff"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    sha TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    ext TEXT NOT NULL,
    source_path TEXT,
    imported_at TEXT NOT NULL,
    features_version TEXT NOT NULL,
    features TEXT NOT NULL,
    category TEXT NOT NULL,
    category_conf REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS samples_category ON samples(category);
"""


@dataclass
class ImportReport:
    added: list[str] = field(default_factory=list)          # shas nuevos en el índice
    known: list[str] = field(default_factory=list)          # ya estaban (mismo contenido)
    cache_hits: int = 0
    extracted: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Hit:
    sha: str
    name: str
    category: str
    features: dict
    distance: float | None = None


def _parse_range(spec: str | None) -> tuple[float, float] | None:
    if not spec:
        return None
    lo, _, hi = spec.partition("-")
    try:
        a = float(lo) if lo else -np.inf
        b = float(hi) if hi else (a if not _ else np.inf)
    except ValueError:
        raise ValueError(f"rango inválido '{spec}' (usa a-b, a- o -b)")
    if a > b:
        raise ValueError(f"rango inválido '{spec}': mínimo mayor que máximo")
    return a, b


FLATS = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}


def _note_matches(f: dict, note: str) -> bool:
    """``C`` = cualquier Do; ``A4`` = exactamente La 4. Solo coinciden samples tonales."""
    m = re.fullmatch(r"([A-G][#B]?)(-?\d+)?", note.strip().upper())
    if not m:
        raise ValueError(f"nota inválida '{note}' (p. ej. C, F#, Bb, A4)")
    if f.get("midi") is None:
        return False
    pc = FLATS.get(m.group(1), m.group(1))
    if pc not in F.NOTE_NAMES:
        raise ValueError(f"nota inválida '{note}'")
    if m.group(2) is not None:
        return f["midi"] == F.NOTE_NAMES.index(pc) + 12 * (int(m.group(2)) + 1)
    return F.NOTE_NAMES[f["midi"] % 12] == pc


class SampleLibrary:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else workspace() / "samples"
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets = AssetStore(self.root / "assets")
        self.cache = CacheStore(self.root / "cache")
        self.db_path = self.root / "library.db"
        with self._db() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        try:
            with db:                                   # transacción: todo o nada
                yield db
        finally:
            db.close()

    # ------------------------------------------------------------------ rasgos
    def features(self, sha: str) -> tuple[dict, bool]:
        """(rasgos, acierto_de_caché) del sample ``sha`` ya almacenado."""
        key = cache_key("sample_features", F.FEATURES_VERSION, [sha])
        hit = self.cache.get_json(key)
        if hit is not None:
            return hit, True
        path = self.assets.find(sha)
        if path is None:
            raise FileNotFoundError(f"sample {sha[:12]} no está en el almacén")
        f = F.extract_file(path)
        self.cache.put_json(key, f)
        return f, False

    # ------------------------------------------------------------------ importación
    def import_dir(self, folder: Path, log=lambda *_: None) -> ImportReport:
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"no es una carpeta: {folder}")
        rep = ImportReport()
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SAMPLE_EXT)
        for src in files:
            try:
                rep_sha = self._import_one(src, rep)
                log(f"{rep_sha[:12]}  {src.name}")
            except Exception as e:                     # un archivo roto no frena la importación
                rep.failed.append((str(src), str(e)))
                log(f"error {src.name}: {e}")
        return rep

    def _import_one(self, src: Path, rep: ImportReport) -> str:
        sha = sha256_file(src)
        asset = self.assets.ingest(src, AssetKind.sample, Provenance(
            tool="samples-import", params={"source_path": str(src.resolve())}), name=src.name)
        assert asset.sha256 == sha
        f, cached = self.features(sha)
        rep.cache_hits += cached
        rep.extracted += not cached
        with self._db() as db:
            row = db.execute("SELECT features_version FROM samples WHERE sha = ?", (sha,)).fetchone()
            if row is None:
                db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)",
                           (sha, src.name, asset.ext, str(src.resolve()), datetime.now(timezone.utc).isoformat(),
                            F.FEATURES_VERSION, json.dumps(f, sort_keys=True), f["category"], f["category_conf"]))
                rep.added.append(sha)
            else:
                if row[0] != F.FEATURES_VERSION:        # extractor nuevo: se actualizan los rasgos
                    db.execute("UPDATE samples SET features_version=?, features=?, category=?, category_conf=? "
                               "WHERE sha=?", (F.FEATURES_VERSION, json.dumps(f, sort_keys=True), f["category"],
                                               f["category_conf"], sha))
                rep.known.append(sha)
        return sha

    # ------------------------------------------------------------------ consulta
    def all(self) -> list[Hit]:
        with self._db() as db:
            rows = db.execute("SELECT sha, name, category, features FROM samples ORDER BY name, sha").fetchall()
        return [Hit(r[0], r[1], r[2], json.loads(r[3])) for r in rows]

    def resolve(self, prefix: str) -> Hit:
        hits = [h for h in self.all() if h.sha.startswith(prefix.lower())]
        if len(hits) != 1:
            raise ValueError(f"sample '{prefix}' {'no encontrado' if not hits else 'ambiguo'} en la biblioteca")
        return hits[0]

    def path(self, sha: str) -> Path:
        p = self.assets.find(sha)
        if p is None:
            raise FileNotFoundError(f"el archivo del sample {sha[:12]} no está en el almacén")
        return p

    def asset(self, sha: str) -> Asset:
        """Descriptor para registrar el sample en un proyecto (mismo sha, procedencia la biblioteca)."""
        h = self.resolve(sha)
        return self.assets.ingest(self.path(h.sha), AssetKind.sample, Provenance(
            tool="sample-library", tool_version=F.FEATURES_VERSION,
            params={"library": str(self.root), "category": h.category}), name=h.name)

    def search(self, cat: str | None = None, bright: str | None = None, max_decay: float | None = None,
               min_decay: float | None = None, note: str | None = None, loop: bool | None = None,
               bpm: str | None = None, like: str | None = None, limit: int = 20) -> list[Hit]:
        """Filtros sobre los rasgos y, con ``like``, orden por vecinos más cercanos.

        Distancia: euclídea sobre los vectores de ``features.vector`` normalizados por z-score con la
        media y desviación de toda la biblioteca (cada rasgo pesa igual).
        """
        items = self.all()
        br, bp = _parse_range(bright), _parse_range(bpm)
        if cat and cat not in F.CATEGORIES:
            raise ValueError(f"categoría desconocida '{cat}' (hay: {', '.join(F.CATEGORIES)})")
        ref = self.resolve(like) if like else None
        out = []
        for h in items:
            f = h.features
            if ref and h.sha == ref.sha:
                continue
            if cat and h.category != cat:
                continue
            if br and not br[0] <= f["brightness"] <= br[1]:
                continue
            if max_decay is not None and f["decay_s"] > max_decay:
                continue
            if min_decay is not None and f["decay_s"] < min_decay:
                continue
            if note and not _note_matches(f, note):
                continue
            if loop is not None and f["is_loop"] != loop:
                continue
            if bp and not (f["bpm"] is not None and bp[0] <= f["bpm"] <= bp[1]):
                continue
            out.append(h)
        if ref:
            mat = np.array([F.vector(h.features) for h in items])
            mu, sd = mat.mean(axis=0), mat.std(axis=0)
            sd[sd < 1e-9] = 1.0
            z = lambda f: (F.vector(f) - mu) / sd
            zr = z(ref.features)
            for h in out:
                h.distance = round(float(np.linalg.norm(z(h.features) - zr)), 4)
            out.sort(key=lambda h: (h.distance, h.sha))
        return out[:limit]
