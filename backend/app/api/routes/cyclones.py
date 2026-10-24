"""Latest successful refresh only. Local archives are never accessed here."""
from datetime import timezone
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.models import Event, SyncLog
from app.database.session import get_db
from app.api.routes.local_demo import ObservationField
from pydantic import ValidationError

router = APIRouter()


def utc(value):
    return (value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)).isoformat()


@router.get('/cyclones/latest')
def latest_cyclones(db: Session = Depends(get_db)):
    sync = db.execute(select(SyncLog).where(SyncLog.source == 'GDACS', SyncLog.status == 'success',
        SyncLog.completed_at.is_not(None)).order_by(SyncLog.completed_at.desc(), SyncLog.id.desc()).limit(1)).scalar_one_or_none()
    if sync is None:
        return {'status': 'unavailable', 'records': [], 'last_successful_refresh_utc': None,
                'reason': 'No successful GDACS refresh recorded'}
    started = utc(sync.started_at)
    rows = db.execute(select(Event).where(Event.source == 'GDACS', Event.disaster_type == 'Tropical Cyclone').order_by(Event.id)).scalars().all()
    records = []
    legacy = False
    for row in rows:
        raw = row.raw_payload or {}
        if raw.get('gdacs_sync_started_at_utc') != started:
            # Unmarked legacy records in the refresh interval need a new sync;
            # do not guess batch membership or display stale cyclones as latest.
            if not raw.get('gdacs_sync_started_at_utc') and utc(sync.started_at) <= utc(row.fetched_at) <= utc(sync.completed_at):
                legacy = True
            continue
        fields = raw.get('gdacs_cyclone_observations_v1')
        validated = {}
        allowed = {'event_id','episode_id','version','alert_level','publication_time',
                   'modification_time','event_start_time','event_end_time','severity',
                   'population_exposure','vulnerability','alert_score','latitude','longitude'}
        for name, value in (fields.items() if isinstance(fields, dict) else []):
            if name not in allowed:
                continue
            try:
                validated[name] = ObservationField.model_validate(value).model_dump(mode='json')
            except (ValidationError, TypeError, ValueError):
                continue
        records.append({'source_event_id': row.source_event_id, 'snapshot_id': f'sync-{sync.id}',
            'retrieved_at_utc': utc(row.fetched_at), 'source_url': row.source_url,
            'country_occurrences': [v for v in raw.get('gdacs_country_occurrences', []) if isinstance(v, str)] if isinstance(raw.get('gdacs_country_occurrences'), list) else [],
            'fields': validated,
            'provenance_kind': 'saved_refresh', 'refresh_started_at_utc': started})
    return {'status': 'refresh_required' if legacy else 'available', 'records': records,
            'last_successful_refresh_utc': utc(sync.completed_at),
            'reason': 'Legacy records lack refresh provenance; refresh GDACS again' if legacy else None,
            'notice': 'Near-real-time monitoring after manual refresh, not streaming or ML predictions'}
