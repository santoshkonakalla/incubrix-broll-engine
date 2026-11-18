"""Database layer (SQLAlchemy) supporting PostgreSQL (Docker) and SQLite (local/tests)."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class JobRecord(Base):
    __tablename__ = "jobs"

    job_id = Column(String(64), primary_key=True, index=True)
    status = Column(String(24), nullable=False, default="queued", index=True)
    progress = Column(Integer, nullable=False, default=0)
    stage = Column(String(64), nullable=False, default="queued")
    request_payload = Column(JSON, nullable=False)
    result_keys = Column(JSON, nullable=True)
    warnings = Column(JSON, nullable=False, default=list)
    error_payload = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    correlation_id = Column(String(128), nullable=False, default="-")
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_keys"

    idempotency_key = Column(String(256), primary_key=True)
    endpoint = Column(String(128), nullable=False)
    request_hash = Column(String(64), nullable=False)
    job_id = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


_engine = None
_SessionLocal = None


def init_db(database_url: Optional[str] = None) -> None:
    global _engine, _SessionLocal
    url = database_url or get_settings().database_url
    connect_args: Dict[str, Any] = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    _engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True, future=True)
    _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(bind=_engine)


def get_session() -> Session:
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def check_db_ready() -> bool:
    try:
        with get_session() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    stage: Optional[str] = None,
    result_keys: Optional[Dict[str, Any]] = None,
    warnings: Optional[list] = None,
    error_payload: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Optional[JobRecord]:
    with get_session() as db:
        rec = db.get(JobRecord, job_id)
        if not rec:
            return None
        if status is not None:
            rec.status = status
        if progress is not None:
            rec.progress = max(0, min(100, progress))
        if stage is not None:
            rec.stage = stage
        if result_keys is not None:
            rec.result_keys = result_keys
        if warnings is not None:
            rec.warnings = warnings
        if error_payload is not None:
            rec.error_payload = error_payload
        if metrics is not None:
            rec.metrics = metrics
        rec.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(rec)
        return rec
