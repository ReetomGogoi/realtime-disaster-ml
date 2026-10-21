"""
Event-related API routes.

  POST /sync-live-events   — fetch GDACS feed and persist new events
  GET  /events             — paginated, filterable list (no raw_payload)
  GET  /events/{event_id}  — single event detail (no raw_payload)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import Event, SyncLog
from app.database.session import get_db
from app.ingestion.gdacs import fetch_gdacs_events
from app.schemas.events import EventListOut, EventOut, EventMLPredictionOut
from app.ml_inference import predict_event

logger = logging.getLogger(__name__)

router = APIRouter()


def _event_response(row: Event, request: Request, include_predictions: bool) -> EventOut:
    response = EventOut.model_validate(row)
    if include_predictions:
        response.ml_prediction = EventMLPredictionOut.model_validate(
            predict_event(row, getattr(request.app.state, "ml_bundle", None))
        )
    return response


# ---------------------------------------------------------------------------
# POST /sync-live-events
# ---------------------------------------------------------------------------

# Mutable fields refreshed on every sync for existing events.
# ID and created_at are intentionally excluded to preserve row identity.
_UPSERT_FIELDS = (
    "title", "disaster_type", "country",
    "latitude", "longitude", "event_time",
    "source_url", "raw_payload", "fetched_at",
)


def _persist_failed_sync(db: Session, started_at: datetime, exc: Exception) -> None:
    """Best-effort failure audit after rolling back the active transaction."""
    db.rollback()
    try:
        db.add(
            SyncLog(
                source="GDACS",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status="error",
                fetched_count=0,
                inserted_count=0,
                skipped_count=0,
                error_message=str(exc),
            )
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to write error SyncLog")


@router.post(
    "/sync-live-events",
    summary="Fetch live GDACS events and persist new ones",
    response_description="Counts of fetched, inserted, updated, and skipped events",
)
async def sync_live_events(db: Session = Depends(get_db)) -> dict:
    """
    1. Fetch the GDACS 7-day RSS feed.
    2. Deduplicate against existing rows using (source, source_event_id).
    3. Insert genuinely new events.
    4. Update mutable fields on already-known events (title, coordinates,
       event_time, source_url, raw_payload, fetched_at); ID and created_at
       are never changed.
    5. Write a SyncLog record regardless of success or failure.
    6. Return a summary dict.
    """
    started_at = datetime.now(timezone.utc)

    # Fetch before the first Session operation so a slow upstream request does
    # not hold a database connection or transaction open.
    try:
        raw_events = await fetch_gdacs_events()
    except httpx.TimeoutException as exc:
        _persist_failed_sync(db, started_at, exc)
        logger.exception("GDACS request timed out")
        raise HTTPException(
            status_code=504,
            detail="GDACS request timed out.",
        ) from exc
    except (httpx.HTTPError, ET.ParseError) as exc:
        _persist_failed_sync(db, started_at, exc)
        logger.exception("GDACS upstream request or feed parsing failed")
        raise HTTPException(
            status_code=502,
            detail="GDACS upstream request or feed parsing failed.",
        ) from exc
    except Exception as exc:
        _persist_failed_sync(db, started_at, exc)
        logger.exception("Unexpected GDACS ingestion failure")
        raise HTTPException(
            status_code=502,
            detail="GDACS ingestion failed.",
        ) from exc

    fetched_count = len(raw_events)
    sync_log = SyncLog(source="GDACS", started_at=started_at)

    try:
        db.add(sync_log)
        # ── 2. Load existing Event rows keyed by (source, source_event_id) ─
        existing: dict[tuple[str, str], Event] = {
            (row.source, row.source_event_id): row
            for row in db.execute(select(Event)).scalars().all()
        }

        # ── 3. Insert new / update existing ────────────────────────────
        inserted_count = 0
        updated_count  = 0
        skipped_count  = 0
        seen: set[tuple[str, str]] = set()  # guard against intra-batch dupes

        for ev in raw_events:
            key = (ev["source"], ev["source_event_id"])

            if key in seen:
                # Exact duplicate within this feed batch — skip silently
                skipped_count += 1
                continue
            seen.add(key)

            existing_row = existing.get(key)
            if existing_row is None:
                # Brand-new event — insert it
                db.add(Event(**ev))
                inserted_count += 1
            else:
                # Already known — refresh mutable fields from latest feed data
                for field in _UPSERT_FIELDS:
                    setattr(existing_row, field, ev[field])
                updated_count += 1
                skipped_count += 1  # counts toward dedup metric as requested

        # ── 4. Finalise SyncLog ─────────────────────────────────────────
        sync_log.completed_at   = datetime.now(timezone.utc)
        sync_log.status         = "success"
        sync_log.fetched_count  = fetched_count
        sync_log.inserted_count = inserted_count
        sync_log.skipped_count  = skipped_count

        db.commit()
        logger.info(
            "GDACS sync complete — fetched=%d inserted=%d updated=%d skipped=%d",
            fetched_count, inserted_count, updated_count, skipped_count,
        )

    except Exception as exc:
        _persist_failed_sync(db, started_at, exc)
        logger.exception("GDACS event storage failed")
        raise HTTPException(
            status_code=503,
            detail="GDACS events could not be stored.",
        ) from exc

    return {
        "status":         "success",
        "fetched_count":  fetched_count,
        "inserted_count": inserted_count,
        "updated_count":  updated_count,
        "skipped_count":  skipped_count,
    }


# ---------------------------------------------------------------------------
# GET /events
# ---------------------------------------------------------------------------

@router.get(
    "/events",
    response_model=EventListOut,
    response_model_exclude_unset=True,
    summary="List disaster events (paginated)",
)
def list_events(
    request:       Request,
    db:            Session = Depends(get_db),
    limit:         Annotated[int, Query(ge=1, le=500, description="Max records to return (1-500)")] = 100,
    offset:        Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    disaster_type: Annotated[str | None, Query(description="Filter by disaster type, e.g. Earthquake")] = None,
    country:       Annotated[str | None, Query(description="Filter by country name")] = None,
    include_predictions: Annotated[bool, Query(description="Include guarded ML status and, only when eligible, model class probabilities")] = False,
) -> EventListOut:
    """
    Return events ordered by event_time descending (newest first).
    Supports pagination via limit/offset and optional filters.
    raw_payload is excluded from the response.
    """
    stmt = select(Event)

    if disaster_type:
        stmt = stmt.where(Event.disaster_type == disaster_type)
    if country:
        stmt = stmt.where(Event.country == country)

    # Total count (before pagination)
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total: int = db.execute(count_stmt).scalar_one()

    # Ordered, paginated results
    stmt = (
        stmt
        .order_by(Event.event_time.desc().nulls_last(), Event.id.desc())
        .offset(offset)
        .limit(limit)
    )
    rows = db.execute(stmt).scalars().all()

    return EventListOut(
        total=total,
        limit=limit,
        offset=offset,
        events=[_event_response(row, request, include_predictions) for row in rows],
    )


# ---------------------------------------------------------------------------
# GET /events/{event_id}
# ---------------------------------------------------------------------------

@router.get(
    "/events/{event_id}",
    response_model=EventOut,
    response_model_exclude_unset=True,
    summary="Get a single disaster event by ID",
)
def get_event(
    event_id: int,
    request: Request,
    db: Session = Depends(get_db),
    include_predictions: Annotated[bool, Query(description="Include guarded ML status and, only when eligible, model class probabilities")] = False,
) -> EventOut:
    """
    Retrieve one event by its internal integer ID.
    Returns HTTP 404 when no matching event exists.
    raw_payload is excluded from the response.
    """
    stmt = select(Event).where(Event.id == event_id)
    row = db.execute(stmt).scalar_one_or_none()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

    return _event_response(row, request, include_predictions)
