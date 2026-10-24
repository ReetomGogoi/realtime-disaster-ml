"""Opt-in localhost research demo; no database writes or GDACS inference."""
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.ml_inference import FEATURES, TARGETS

router = APIRouter(prefix='/local-demo')
ROOT = Path(__file__).resolve().parents[4]
FEATURE_PATH = ROOT / 'datasets/processed/gdacs_cyclone_extraction/features.jsonl'
SCHEMA = 'gdacs-cyclone-observations/1.0'


def local_only(request: Request):
    if (os.getenv('ENABLE_LOCAL_ML_DEMO') != '1' or os.getenv('VERCEL')
            or request.client is None or request.client.host not in {'127.0.0.1', '::1', 'localhost', 'testclient'}):
        raise HTTPException(404, 'Local demo disabled')


class ObservationField(BaseModel):
    source_path: str
    attribute: str | None = None
    occurrences: list[dict] = Field(default_factory=list)
    value: str | float | None = None
    unit: str | None = None
    status: Literal['available', 'missing', 'malformed', 'ambiguous']
    failure: str | None = None

    @model_validator(mode='after')
    def consistent_status(self):
        if (self.status == 'available') != (self.value is not None):
            raise ValueError('Observation value/status disagree')
        return self

    @field_validator('value')
    @classmethod
    def finite(cls, value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Nonfinite observation')
        return value


class Observation(BaseModel):
    schema_version: Literal['gdacs-cyclone-observations/1.0']
    snapshot_id: str
    source_url: Literal['https://www.gdacs.org/xml/rss_7d.xml']
    retrieved_at_utc: datetime
    item_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    response_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    availability_evidence: str
    fields: dict[str, ObservationField]
    country_occurrences: list[str | None] = Field(default_factory=list)
    population_semantics: str

    @field_validator('retrieved_at_utc')
    @classmethod
    def timezone_required(cls, value):
        if value.utcoffset() is None:
            raise ValueError('Timezone required')
        return value


@router.get('/cyclones', dependencies=[Depends(local_only)])
def cyclones():
    if not FEATURE_PATH.is_file():
        return {'status': 'unavailable', 'records': [], 'reason': 'Local extraction output missing'}
    try:
        records = [Observation.model_validate(json.loads(line)).model_dump(mode='json')
                   for line in FEATURE_PATH.read_text(encoding='utf-8').splitlines() if line.strip()]
        return {'status': 'available', 'records': records,
                'notice': 'Archived GDACS observations, not ML predictions or observed impact'}
    except (OSError, ValueError, ValidationError):
        raise HTTPException(422, 'Invalid local extraction schema; rerun verified extractor') from None


class HistoricalInput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    disaster_type: str = Field(min_length=1)
    disaster_subtype: str = Field(min_length=1)
    iso: str = Field(min_length=1)
    region: str = Field(min_length=1)
    subregion: str = Field(min_length=1)
    season: Literal['DJF', 'MAM', 'JJA', 'SON', 'Unknown']
    start_month: int | None = Field(ge=1, le=12)
    magnitude: float | None
    magnitude_available: Literal[0, 1]


@router.post('/historical-predict', dependencies=[Depends(local_only)])
def historical_predict(data: HistoricalInput, request: Request):
    if (data.magnitude is not None) != bool(data.magnitude_available):
        raise HTTPException(422, 'Magnitude and availability indicator disagree')
    expected_season = ('DJF' if data.start_month in {12, 1, 2} else
                       'MAM' if data.start_month in {3, 4, 5} else
                       'JJA' if data.start_month in {6, 7, 8} else
                       'SON' if data.start_month in {9, 10, 11} else 'Unknown')
    if data.season != expected_season:
        raise HTTPException(422, 'Month and season disagree')
    import pandas as pd
    values = data.model_dump()
    frame = pd.DataFrame([values], columns=list(FEATURES))
    for column in ['start_month', 'magnitude']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    output = {}
    for target in TARGETS:
        state = getattr(request.app.state, 'ml_bundle', {}).get(target)
        result = {'status': 'model_unavailable', 'predicted_label': None, 'probabilities': None}
        if state is not None and state.pipeline is not None:
            if data.disaster_type not in state.supported_hazards:
                result['status'] = 'unsupported_for_target'
            else:
                try:
                    label = str(state.pipeline.predict(frame)[0])
                    scores = state.pipeline.predict_proba(frame)[0]
                    classes = state.pipeline.named_steps['classifier'].classes_
                    probabilities = {str(k): float(v) for k, v in zip(classes, scores, strict=True)}
                    if (label not in probabilities or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values())
                            or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6)):
                        raise ValueError('Invalid probabilities')
                    result.update(status='predicted', predicted_label=label, probabilities=probabilities)
                except Exception:
                    result['status'] = 'inference_error'
        output[target] = result
    return {'mode': 'retrospective_manual_features', 'results': output,
            'notice': 'Historical model demonstration only; no GDACS mapping. Probabilities are not operational confidence.'}
