"""Database engine, session factory, and ``init_db()`` (Task 1.2).

Provides a default SQLite engine pointing at ``data/iar.db`` (resolved relative
to the project root), a session factory, and ``init_db()`` to create all tables.
Callers (and tests) can pass a different path/engine for an isolated database.

Usage
-----
    from iar.db.session import init_db, get_session

    init_db()                       # create data/iar.db with all tables
    with get_session() as s:
        s.add(obj)
        s.commit()
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from iar.db.models import Base

# project root = .../IAR MVP  (this file is iar/db/session.py -> parents[2])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "iar.db"


def _enable_sqlite_fk(engine: Engine) -> None:
    """Turn on foreign-key enforcement (off by default in SQLite)."""

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, _connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def make_engine(db_path: str | Path = DEFAULT_DB_PATH, echo: bool = False) -> Engine:
    """Create a SQLite engine for ``db_path`` (parent dir created if needed).

    Pass ``db_path=":memory:"`` for an in-memory database (handy in tests).
    """
    if db_path == ":memory:":
        url = "sqlite:///:memory:"
    else:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
    engine = create_engine(url, echo=echo, future=True)
    _enable_sqlite_fk(engine)
    return engine


# Default engine + session factory used by the app.
engine: Engine = make_engine()
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, expire_on_commit=False, future=True
)


def init_db(eng: Engine | None = None) -> Engine:
    """Create all tables on ``eng`` (defaults to the module engine)."""
    eng = eng or engine
    Base.metadata.create_all(eng)
    return eng


@contextmanager
def get_session() -> Iterator[Session]:
    """Session context manager that rolls back on error and always closes."""
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
