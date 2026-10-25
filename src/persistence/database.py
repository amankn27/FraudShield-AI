"""SQLAlchemy database manager with optional logging when DATABASE_URL is absent."""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import logging
import os
from typing import Iterator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.config import PROJECT_ROOT  # also loads the local .env file
from src.persistence.models import Base


logger = logging.getLogger(__name__)


class DatabaseNotConfigured(RuntimeError):
    """Raised when a persistence operation is attempted without a database URL."""


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def normalize_database_url(url: str) -> str:
    """Normalize common PostgreSQL URL prefixes to the psycopg 3 driver."""
    clean = url.strip()
    if clean.startswith("postgres://"):
        return "postgresql+psycopg://" + clean[len("postgres://"):]
    if clean.startswith("postgresql://"):
        return "postgresql+psycopg://" + clean[len("postgresql://"):]
    return clean


class DatabaseManager:
    """Own an Engine and session factory; can also use SQLite for tests."""

    def __init__(self, url: str | None = None, required: bool | None = None):
        # url=None means read from environment; url="" explicitly disables DB.
        raw_url = os.getenv("DATABASE_URL", "") if url is None else url
        self.url = normalize_database_url(raw_url) if raw_url and raw_url.strip() else ""
        default_required = bool(self.url)
        self.required = _bool_env("DATABASE_REQUIRED", default_required) if required is None else required
        self.engine: Engine | None = None
        self.session_factory: sessionmaker[Session] | None = None
        self.initialization_error: Exception | None = None

        if self.url:
            kwargs = {"pool_pre_ping": True, "future": True}
            if self.url.startswith("sqlite"):
                kwargs["connect_args"] = {"check_same_thread": False}
                if ":memory:" in self.url:
                    kwargs["poolclass"] = StaticPool
            try:
                self.engine = create_engine(self.url, **kwargs)
                self.session_factory = sessionmaker(
                    bind=self.engine, autoflush=False, expire_on_commit=False, class_=Session
                )
            except Exception as exc:
                # Keep the HTTP process alive so health/readiness can report degraded state.
                self.initialization_error = exc
                logger.exception("Could not initialize configured database engine")

    @property
    def enabled(self) -> bool:
        return self.engine is not None and self.session_factory is not None

    def ping(self) -> bool:
        """Return whether SELECT 1 succeeds; false if DB is disabled or unavailable."""
        if not self.enabled or self.engine is None:
            return False
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def create_schema_for_local_dev(self) -> None:
        """Create tables for local prototyping/tests. Production should use Alembic."""
        if not self.enabled or self.engine is None:
            raise DatabaseNotConfigured("Set DATABASE_URL before initializing the database.")
        # Ensure all model declarations are imported before create_all.
        from src.persistence import models as _models  # noqa: F401
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        if self.session_factory is None:
            raise DatabaseNotConfigured("Database persistence is not configured.")
        session = self.session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        if self.engine is not None:
            self.engine.dispose()


@lru_cache(maxsize=1)
def get_database_manager() -> DatabaseManager:
    """Cache database configuration/engine once per API process."""
    return DatabaseManager()


def clear_database_manager_cache() -> None:
    """Clear cached engine configuration after changing environment or in tests."""
    if get_database_manager.cache_info().currsize:
        manager = get_database_manager()
        manager.dispose()
    get_database_manager.cache_clear()
