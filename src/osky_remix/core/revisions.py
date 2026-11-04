"""Repositorio de un proyecto: revisiones inmutables, ramas, reversión y fusión.

Estructura en disco::

    <workspace>/projects/<nombre>/
        project.db      revisiones, ramas, metadatos (SQLite)
        assets/         almacén por SHA-256 (solo lectura)
        cache/          resultados derivados
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..storage.database import DB_VERSION, BranchRow, MetaRow, RevisionRow, make_engine
from .assets import AssetStore
from .cache import CacheStore
from .models import Project
from .operations import Author, Context, Op
from .transactions import TransactionError, apply_ops, parse_ops


class RepoError(Exception):
    pass


class MergeConflict(RepoError):
    pass


@dataclass(frozen=True)
class Revision:
    id: str
    parent: str | None
    merge_parent: str | None
    branch: str
    author: str
    message: str
    kind: str
    ops: list[dict]
    content_hash: str
    created_at: str

    @classmethod
    def from_row(cls, r: RevisionRow) -> "Revision":
        return cls(r.id, r.parent, r.merge_parent, r.branch, r.author, r.message, r.kind,
                   json.loads(r.ops), r.content_hash, r.created_at)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _ops_json(ops: list[Op]) -> list[dict]:
    return [o.model_dump(mode="json") for o in ops]


class ProjectRepo:
    def __init__(self, root: Path):
        self.root = Path(root)
        db = self.root / "project.db"
        if not db.exists():
            raise RepoError(f"no hay proyecto en {self.root}")
        self.engine = make_engine(db)
        self.assets = AssetStore(self.root / "assets")
        self.cache = CacheStore(self.root / "cache")

    # ------------------------------------------------------------ creación
    @classmethod
    def create(cls, root: Path, project: Project, author: Author = "user") -> "ProjectRepo":
        root = Path(root)
        if (root / "project.db").exists():
            raise RepoError(f"ya existe un proyecto en {root}")
        root.mkdir(parents=True, exist_ok=True)
        eng = make_engine(root / "project.db")
        with Session(eng) as s, s.begin():
            rev = cls._new_row(None, None, "main", author, "Proyecto creado", "create", [], project)
            s.add(rev)
            s.flush()
            s.add(BranchRow(name="main", head=rev.id))
            s.add(MetaRow(key="current_branch", value="main"))
            s.add(MetaRow(key="db_version", value=str(DB_VERSION)))
            s.add(MetaRow(key="name", value=project.name))
        eng.dispose()
        return cls(root)

    @staticmethod
    def _new_row(parent, merge_parent, branch, author, message, kind, ops, project: Project) -> RevisionRow:
        snap = project.canonical_json()
        ch = project.content_hash()
        created = _now()
        rid = hashlib.sha256(json.dumps([parent, merge_parent, ch, ops, author, message, kind, created],
                                        sort_keys=True).encode()).hexdigest()
        return RevisionRow(id=rid, parent=parent, merge_parent=merge_parent, branch=branch, author=author,
                           message=message, kind=kind, ops=json.dumps(ops, ensure_ascii=False),
                           snapshot=snap, content_hash=ch, created_at=created)

    # ------------------------------------------------------------ lectura
    def _meta(self, s: Session, key: str) -> str:
        return s.get(MetaRow, key).value

    @property
    def current_branch(self) -> str:
        with Session(self.engine) as s:
            return self._meta(s, "current_branch")

    def branches(self) -> dict[str, str]:
        with Session(self.engine) as s:
            return {b.name: b.head for b in s.scalars(select(BranchRow))}

    def head(self, branch: str | None = None) -> str:
        with Session(self.engine) as s:
            b = s.get(BranchRow, branch or self._meta(s, "current_branch"))
            if b is None:
                raise RepoError(f"no existe la rama '{branch}'")
            return b.head

    def _row(self, s: Session, rev: str) -> RevisionRow:
        """Acepta un nombre de rama, un id completo o un prefijo único de id."""
        b = s.get(BranchRow, rev)
        if b is not None:
            rev = b.head
        rows = list(s.scalars(select(RevisionRow).where(RevisionRow.id.startswith(rev))))
        if not rows:
            raise RepoError(f"no existe la revisión '{rev}'")
        if len(rows) > 1:
            raise RepoError(f"el prefijo '{rev}' es ambiguo")
        return rows[0]

    def resolve(self, rev: str) -> str:
        with Session(self.engine) as s:
            return self._row(s, rev).id

    def revision(self, rev: str) -> Revision:
        with Session(self.engine) as s:
            return Revision.from_row(self._row(s, rev))

    def load(self, rev: str | None = None) -> Project:
        with Session(self.engine) as s:
            if rev is None:
                rev = s.get(BranchRow, self._meta(s, "current_branch")).head
            row = self._row(s, rev)
            p = Project.model_validate_json(row.snapshot)
            if p.content_hash() != row.content_hash:
                raise RepoError(f"la revisión {row.id[:12]} está corrupta (hash no coincide)")
            return p

    def log(self, branch: str | None = None, limit: int = 50) -> list[Revision]:
        out = []
        with Session(self.engine) as s:
            rid = s.get(BranchRow, branch or self._meta(s, "current_branch")).head
            while rid and len(out) < limit:
                r = s.scalar(select(RevisionRow).where(RevisionRow.id == rid))
                out.append(Revision.from_row(r))
                rid = r.parent
        return out

    def _ancestors(self, s: Session, rid: str) -> list[str]:
        """Ancestros (incluido ``rid``) en orden BFS, siguiendo ambos padres."""
        seen, order, queue = set(), [], [rid]
        while queue:
            x = queue.pop(0)
            if x is None or x in seen:
                continue
            seen.add(x)
            order.append(x)
            r = s.scalar(select(RevisionRow).where(RevisionRow.id == x))
            queue += [r.parent, r.merge_parent]
        return order

    # ------------------------------------------------------------ escritura
    def _advance(self, s: Session, branch: str, expected_head: str, row: RevisionRow) -> None:
        b = s.get(BranchRow, branch)
        if b.head != expected_head:
            raise RepoError("la rama cambió durante la operación; reintenta")
        s.add(row)
        s.flush()
        b.head = row.id

    def commit(self, ops: list[Op] | list[dict], author: Author, message: str, branch: str | None = None) -> Revision:
        """Aplica un lote de operaciones de forma atómica y crea una revisión."""
        if ops and isinstance(ops[0], dict):
            ops = parse_ops(ops)
        if not ops:
            raise RepoError("el lote de operaciones está vacío")
        with Session(self.engine) as s, s.begin():
            br = branch or self._meta(s, "current_branch")
            head = s.get(BranchRow, br)
            if head is None:
                raise RepoError(f"no existe la rama '{br}'")
            before = Project.model_validate_json(s.scalar(select(RevisionRow.snapshot).where(RevisionRow.id == head.head)))
            after = apply_ops(before, ops, Context(author=author, asset_exists=self.assets.exists))
            row = self._new_row(head.head, None, br, author, message, "ops", _ops_json(ops), after)
            self._advance(s, br, head.head, row)
            return Revision.from_row(row)

    def create_branch(self, name: str, from_rev: str | None = None) -> str:
        with Session(self.engine) as s, s.begin():
            if s.get(BranchRow, name):
                raise RepoError(f"ya existe la rama '{name}'")
            src = self._row(s, from_rev).id if from_rev else s.get(BranchRow, self._meta(s, "current_branch")).head
            s.add(BranchRow(name=name, head=src))
            return src

    def checkout(self, branch: str) -> None:
        with Session(self.engine) as s, s.begin():
            if not s.get(BranchRow, branch):
                raise RepoError(f"no existe la rama '{branch}'")
            s.get(MetaRow, "current_branch").value = branch

    def revert(self, to_rev: str, author: Author, message: str | None = None) -> Revision:
        """Nueva revisión cuyo contenido es el de ``to_rev``. El historial no se reescribe.

        Si la reversión alteraría pistas bloqueadas, solo el usuario puede hacerla.
        """
        with Session(self.engine) as s, s.begin():
            br = self._meta(s, "current_branch")
            head_id = s.get(BranchRow, br).head
            target = self._row(s, to_rev)
            cur = Project.model_validate_json(s.scalar(select(RevisionRow.snapshot).where(RevisionRow.id == head_id)))
            old = Project.model_validate_json(target.snapshot)
            if author != "user":
                touched = []
                for t in cur.tracks:
                    if t.locked:
                        prev = next((x for x in old.tracks if x.id == t.id), None)
                        if prev is None or prev.model_dump() != t.model_dump():
                            touched.append(t.id)
                if touched:
                    raise RepoError(f"revertir alteraría pistas bloqueadas ({', '.join(touched)}); requiere al usuario")
            row = self._new_row(head_id, None, br, author, message or f"Revertir a {target.id[:12]}",
                                "revert", [{"op": "__revert__", "to": target.id}], old)
            self._advance(s, br, head_id, row)
            return Revision.from_row(row)

    def merge(self, source: str, author: Author, message: str | None = None) -> Revision | None:
        """Fusiona la rama ``source`` en la actual reaplicando sus operaciones.

        - Si ya está contenida: no hace nada (None).
        - Si la actual es ancestro de ``source``: avance rápido (mueve la cabeza).
        - Si divergieron: reaplica las operaciones de ``source`` desde el ancestro común sobre
          la cabeza actual, de forma atómica. Cualquier precondición fallida es un conflicto
          y no se escribe nada.
        """
        with Session(self.engine) as s, s.begin():
            br = self._meta(s, "current_branch")
            if source == br:
                raise RepoError("no se puede fusionar una rama consigo misma")
            src_row = s.get(BranchRow, source)
            if src_row is None:
                raise RepoError(f"no existe la rama '{source}'")
            t_head, s_head = s.get(BranchRow, br).head, src_row.head
            t_anc = self._ancestors(s, t_head)
            if s_head in t_anc:
                return None
            s_anc = self._ancestors(s, s_head)
            if t_head in s_anc:
                s.get(BranchRow, br).head = s_head
                return self.revision_in(s, s_head)
            t_set = set(t_anc)
            lca = next(x for x in s_anc if x in t_set)
            # operaciones de la rama fuente desde el ancestro común (primer padre)
            chain, rid = [], s_head
            while rid != lca:
                r = s.scalar(select(RevisionRow).where(RevisionRow.id == rid))
                if r.kind == "revert":
                    raise MergeConflict(f"la rama '{source}' contiene una reversión ({r.id[:12]}); "
                                        "no se puede reaplicar como operaciones")
                chain.append(r)
                rid = r.parent
                if rid is None:
                    raise MergeConflict("no se encontró el ancestro común por la línea principal")
            raw = [op for r in reversed(chain) for op in json.loads(r.ops)]
            base = Project.model_validate_json(s.scalar(select(RevisionRow.snapshot).where(RevisionRow.id == t_head)))
            try:
                merged = apply_ops(base, parse_ops(raw), Context(author=author, asset_exists=self.assets.exists))
            except TransactionError as e:
                raise MergeConflict(f"conflicto al reaplicar '{source}': {e}") from None
            row = self._new_row(t_head, s_head, br, author, message or f"Fusionar '{source}' en '{br}'",
                                "merge", raw, merged)
            self._advance(s, br, t_head, row)
            return Revision.from_row(row)

    def revision_in(self, s: Session, rid: str) -> Revision:
        return Revision.from_row(s.scalar(select(RevisionRow).where(RevisionRow.id == rid)))
