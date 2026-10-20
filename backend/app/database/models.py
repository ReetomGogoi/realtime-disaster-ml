"""
SQLAlchemy ORM models for the realtime-disaster-ml project.

All timestamp columns use timezone=True so they are stored as UTC and remain
compatible when migrating from SQLite to PostgreSQL.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

class Event(Base):
    """A raw disaster event fetched from an external source (e.g. GDACS)."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    disaster_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    country: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    event_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    raw_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("source", "source_event_id", name="uq_event_source_id"),
    )

    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="event", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class Prediction(Base):
    """ML model output associated with a specific Event."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("events.id", ondelete="CASCADE"), nullable=False, index=True
    )
    impact_priority: Mapped[str | None] = mapped_column(String(32), nullable=True)
    affected_people_range: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    event: Mapped["Event"] = relationship("Event", back_populates="predictions")


# ---------------------------------------------------------------------------
# ModelMetadata
# ---------------------------------------------------------------------------

class ModelMetadata(Base):
    """Registry of trained ML model artifacts."""

    __tablename__ = "model_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    model_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    training_source: Mapped[str | None] = mapped_column(String(256), nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    feature_schema_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )


# ---------------------------------------------------------------------------
# SyncLog
# ---------------------------------------------------------------------------

class SyncLog(Base):
    """Audit log for each data-ingestion run."""

    __tablename__ = "sync_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    fetched_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    inserted_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    skipped_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
