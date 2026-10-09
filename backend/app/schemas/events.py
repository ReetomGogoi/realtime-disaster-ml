"""
Pydantic response schemas for Event endpoints.

raw_payload is intentionally excluded from all list and detail responses
to keep payloads compact.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EventOut(BaseModel):
    """Serialised representation of a single Event row."""

    model_config = ConfigDict(from_attributes=True)

    id:              int
    source:          str
    source_event_id: str
    title:           str
    disaster_type:   str | None
    country:         str | None
    latitude:        float | None
    longitude:       float | None
    event_time:      datetime | None
    source_url:      str | None
    fetched_at:      datetime
    created_at:      datetime


class EventListOut(BaseModel):
    """Paginated list of events."""

    total:   int
    limit:   int
    offset:  int
    events:  list[EventOut]
