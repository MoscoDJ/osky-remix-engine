"""Esquema SQLite (SQLAlchemy 2). Un archivo ``project.db`` por proyecto."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DB_VERSION = 1


class Base(DeclarativeBase):
    pass


class RevisionRow(Base):
    """Revisión inmutable: nunca se actualiza ni se borra una fila."""
    __tablename__ = "revisions"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    parent: Mapped[str | None] = mapped_column(String(64), ForeignKey("revisions.id"), nullable=True)
    merge_parent: Mapped[str | None] = mapped_column(String(64), ForeignKey("revisions.id"), nullable=True)
    branch: Mapped[str] = mapped_column(String(64))
    author: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(16))          # create | ops | revert | merge
    ops: Mapped[str] = mapped_column(Text)                 # JSON de las operaciones aplicadas
    snapshot: Mapped[str] = mapped_column(Text)            # JSON canónico del proyecto resultante
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[str] = mapped_column(String(40))


class BranchRow(Base):
    __tablename__ = "branches"
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    head: Mapped[str] = mapped_column(String(64), ForeignKey("revisions.id"))


class MetaRow(Base):
    __tablename__ = "meta"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


def make_engine(db_path: Path) -> Engine:
    eng = create_engine(f"sqlite:///{db_path}", future=True)

    @event.listens_for(eng, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=FULL")
        cur.close()

    # SQLite + pysqlite: transacciones explícitas con BEGIN IMMEDIATE (un solo escritor)
    @event.listens_for(eng, "connect")
    def _no_autobegin(dbapi_conn, _):
        dbapi_conn.isolation_level = None

    @event.listens_for(eng, "begin")
    def _begin(conn):
        conn.exec_driver_sql("BEGIN IMMEDIATE")

    Base.metadata.create_all(eng)
    return eng
