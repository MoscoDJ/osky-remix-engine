"""Registro de escucha ciega: sesiones X/Y y veredictos del usuario (``<proyecto>/listening.db``).

Una sesión fija qué versión suena como X y cuál como Y (orden aleatorio, con semilla guardada),
las revisiones, el perfil, las claves de render y los puntajes del evaluador en ese momento. El
veredicto se envía en términos de X/Y y aquí se traduce a A/B; una sesión admite un solo veredicto.
El puntaje del evaluador es una suma de severidades: **menor es mejor**.
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Float, ForeignKey, Integer, String, Text, create_engine, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class _Base(DeclarativeBase):
    pass


class SessionRow(_Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created: Mapped[str] = mapped_column(String(40))
    rev_a: Mapped[str] = mapped_column(String(64))
    rev_b: Mapped[str] = mapped_column(String(64))
    label_a: Mapped[str] = mapped_column(String(128))
    label_b: Mapped[str] = mapped_column(String(128))
    profile: Mapped[str] = mapped_column(String(64))
    render_key_a: Mapped[str | None] = mapped_column(String(64), nullable=True)
    render_key_b: Mapped[str | None] = mapped_column(String(64), nullable=True)
    score_a: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_b: Mapped[float | None] = mapped_column(Float, nullable=True)
    b_gain_db: Mapped[float | None] = mapped_column(Float, nullable=True)
    seed: Mapped[int] = mapped_column(Integer)
    order: Mapped[str] = mapped_column(String(2))          # "ab": X = A · "ba": X = B


class VerdictRow(_Base):
    __tablename__ = "verdicts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session: Mapped[str] = mapped_column(String(32), ForeignKey("sessions.id"), unique=True)
    created: Mapped[str] = mapped_column(String(40))
    preferred: Mapped[str] = mapped_column(String(4))      # a | b | none
    rating_a: Mapped[int] = mapped_column(Integer)
    rating_b: Mapped[int] = mapped_column(Integer)
    note: Mapped[str] = mapped_column(Text, default="")


class ListeningError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def blind_order(seed: int) -> str:
    """Orden ciego reproducible: ``"ab"`` (X = A) o ``"ba"`` (X = B), equiprobable."""
    return "ab" if random.Random(seed).random() < 0.5 else "ba"


class ListeningStore:
    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")

        @event.listens_for(self.engine, "connect")
        def _pragmas(conn, _):
            cur = conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        _Base.metadata.create_all(self.engine)

    @classmethod
    def for_project(cls, project_root: Path) -> "ListeningStore":
        return cls(Path(project_root) / "listening.db")

    def create_session(self, *, rev_a: str, rev_b: str, label_a: str, label_b: str, profile: str,
                       render_key_a: str | None, render_key_b: str | None, score_a: float | None,
                       score_b: float | None, b_gain_db: float | None, seed: int) -> dict:
        sid = uuid.uuid4().hex[:16]
        row = SessionRow(id=sid, created=_now(), rev_a=rev_a, rev_b=rev_b, label_a=label_a, label_b=label_b,
                         profile=profile, render_key_a=render_key_a, render_key_b=render_key_b, score_a=score_a,
                         score_b=score_b, b_gain_db=b_gain_db, seed=seed, order=blind_order(seed))
        order = row.order
        with Session(self.engine) as s, s.begin():
            s.add(row)
        return {"session": sid, "order": order}

    def session_order(self, sid: str) -> str:
        with Session(self.engine) as s:
            r = s.get(SessionRow, sid)
            if r is None:
                raise ListeningError("sesión de escucha desconocida")
            return r.order

    def record(self, sid: str, preferred_xy: str, rating_x: int, rating_y: int, note: str = "") -> dict:
        """Guarda el veredicto (en términos X/Y) y devuelve el registro revelado (A/B)."""
        if preferred_xy not in ("x", "y", "none"):
            raise ListeningError("preferencia: 'x', 'y' o 'none'")
        for v in (rating_x, rating_y):
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 5:
                raise ListeningError("las calificaciones van de 1 a 5")
        with Session(self.engine) as s, s.begin():
            sess = s.get(SessionRow, sid)
            if sess is None:
                raise ListeningError("sesión de escucha desconocida")
            if s.scalars(select(VerdictRow).where(VerdictRow.session == sid)).first() is not None:
                raise ListeningError("esta sesión ya tiene veredicto")
            x_is_a = sess.order == "ab"
            pref = "none" if preferred_xy == "none" else ("a" if (preferred_xy == "x") == x_is_a else "b")
            ra, rb = (rating_x, rating_y) if x_is_a else (rating_y, rating_x)
            s.add(VerdictRow(session=sid, created=_now(), preferred=pref, rating_a=ra, rating_b=rb, note=note or ""))
        return next(v for v in self.verdicts() if v["session"] == sid)

    def verdicts(self) -> list[dict]:
        with Session(self.engine) as s:
            rows = s.execute(select(VerdictRow, SessionRow).join(SessionRow, VerdictRow.session == SessionRow.id)
                             .order_by(VerdictRow.id.desc())).all()
            return [_verdict_dict(v, se) for v, se in rows]


def _verdict_dict(v: VerdictRow, se: SessionRow) -> dict:
    return {"id": v.id, "session": se.id, "created": v.created, "session_created": se.created,
            "rev_a": se.rev_a, "rev_b": se.rev_b, "label_a": se.label_a, "label_b": se.label_b,
            "profile": se.profile, "render_key_a": se.render_key_a, "render_key_b": se.render_key_b,
            "score_a": se.score_a, "score_b": se.score_b, "b_gain_db": se.b_gain_db, "seed": se.seed,
            "x_was": "a" if se.order == "ab" else "b", "preferred": v.preferred,
            "preferred_rev": {"a": se.rev_a, "b": se.rev_b}.get(v.preferred),
            "rating_a": v.rating_a, "rating_b": v.rating_b, "note": v.note}


def summarize(verdicts: list[dict]) -> dict:
    """Acuerdo entre la preferencia humana y el evaluador (menor puntaje = mejor).

    ``agreement_pct``: de los pares con preferencia y puntajes distintos, % en que la versión
    preferida también tiene el menor puntaje. Los empates de puntaje y "sin preferencia" se cuentan
    aparte y no entran en el porcentaje.
    """
    decisive = [v for v in verdicts if v["preferred"] in ("a", "b")]
    scored = [v for v in decisive if v["score_a"] is not None and v["score_b"] is not None]
    comparable = [v for v in scored if v["score_a"] != v["score_b"]]

    def agrees(v):
        better = "a" if v["score_a"] < v["score_b"] else "b"
        return v["preferred"] == better

    agree = sum(agrees(v) for v in comparable)
    pref_r = [v["rating_a"] if v["preferred"] == "a" else v["rating_b"] for v in decisive]
    other_r = [v["rating_b"] if v["preferred"] == "a" else v["rating_a"] for v in decisive]
    by_profile: dict[str, dict] = {}
    for v in comparable:
        d = by_profile.setdefault(v["profile"], {"pairs": 0, "agree": 0})
        d["pairs"] += 1
        d["agree"] += agrees(v)
    for d in by_profile.values():
        d["agreement_pct"] = round(100 * d["agree"] / d["pairs"], 1)
    mean = (lambda xs: round(sum(xs) / len(xs), 2) if xs else None)
    return {"verdicts": len(verdicts), "no_preference": len(verdicts) - len(decisive),
            "pairs": len(comparable), "score_ties": len(scored) - len(comparable), "agree": agree,
            "agreement_pct": round(100 * agree / len(comparable), 1) if comparable else None,
            "mean_rating_preferred": mean(pref_r), "mean_rating_other": mean(other_r), "by_profile": by_profile}
