"""
Pydantic response schemas for Event endpoints.

raw_payload is intentionally excluded from all list and detail responses
to keep payloads compact.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class TargetPredictionOut(BaseModel):
    status: Literal[
        "mapping_incomplete", "unsupported_hazard", "unsupported_for_target",
        "model_unavailable", "predicted", "inference_error",
    ]
    reasons: list[str]
    predicted_label: str | None = None
    # Model class probabilities; not verified operational confidence.
    probabilities: dict[str, float] | None = None


class EventMLPredictionOut(BaseModel):
    affected_range: TargetPredictionOut
    impact_priority: TargetPredictionOut


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
    alert_level:     Literal["Green", "Orange", "Red"] | None = None
    fetched_at:      datetime
    created_at:      datetime
    ml_prediction: EventMLPredictionOut | None = None


class EventListOut(BaseModel):
    """Paginated list of events."""

    total:   int
    limit:   int
    offset:  int
    events:  list[EventOut]
