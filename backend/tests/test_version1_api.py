"""Offline API regressions for the deployed Version 1 monitoring path."""

import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.models import Event, SyncLog


# Ensure importing the application cannot select the normal local SQLite file.
with patch.dict(
    os.environ,
    {"DATABASE_URL": "sqlite+pysqlite:///:memory:", "VERCEL": "0"},
):
    from app.database.session import get_db
    from app.main import app


_LONG_GDACS_COUNTRY = (
    "Austria, Bosnia & Herzegovina, Belgium, Belarus, Switzerland, Czech Republic, "
    "Germany, Denmark, Spain, France, Croatia, Hungary, Ireland, Italy, Liechtenstein, "
    "Luxembourg, Moldova, Montenegro, Netherlands, Poland, Romania, Serbia, Sweden, "
    "Slovenia, Slovakia, Ukraine,"
)


class Version1ApiRegressionTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        def isolated_database():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        self.original_overrides = app.dependency_overrides.copy()
        app.dependency_overrides[get_db] = isolated_database
        # Deliberately do not enter TestClient as a context manager: that would
        # run the production lifespan, which owns a different configured engine.
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        app.dependency_overrides.clear()
        app.dependency_overrides.update(self.original_overrides)
        self.engine.dispose()

    @staticmethod
    def event(
        source_event_id: str,
        *,
        country: str = "India",
        title: str = "Test earthquake",
        event_time: datetime | None = None,
    ) -> dict:
        observed_at = event_time or datetime(2026, 10, 21, 10, tzinfo=timezone.utc)
        return {
            "source": "GDACS",
            "source_event_id": source_event_id,
            "title": title,
            "disaster_type": "Earthquake",
            "country": country,
            "latitude": 28.6139,
            "longitude": 77.2090,
            "event_time": observed_at,
            "source_url": f"https://www.gdacs.org/report.aspx?eventid={source_event_id}",
            "raw_payload": {
                "country": country,
                "gdacs_country_occurrences": [country],
            },
            "fetched_at": observed_at + timedelta(minutes=5),
        }

    def test_health_contract(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "service": "realtime-disaster-ml-api",
                "version": "0.1.0",
            },
        )

    def test_events_list_filters_paginates_and_excludes_raw_payload(self):
        older = self.event(
            "EQ:older",
            country="Nepal",
            event_time=datetime(2026, 10, 20, 8, tzinfo=timezone.utc),
        )
        newer = self.event(
            "EQ:newer",
            country=_LONG_GDACS_COUNTRY,
            event_time=datetime(2026, 10, 21, 8, tzinfo=timezone.utc),
        )
        with Session(self.engine) as session:
            session.add_all([Event(**older), Event(**newer)])
            session.commit()
            newer_id = session.execute(
                select(Event.id).where(Event.source_event_id == "EQ:newer")
            ).scalar_one()

        page = self.client.get("/events", params={"limit": 1, "offset": 0})
        self.assertEqual(page.status_code, 200)
        payload = page.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["limit"], 1)
        self.assertEqual(payload["offset"], 0)
        self.assertEqual(len(payload["events"]), 1)
        self.assertEqual(payload["events"][0]["source_event_id"], "EQ:newer")
        self.assertEqual(payload["events"][0]["country"], _LONG_GDACS_COUNTRY)
        self.assertNotIn("raw_payload", payload["events"][0])
        self.assertNotIn("ml_prediction", payload["events"][0])

        filtered = self.client.get("/events", params={"country": "Nepal"})
        self.assertEqual(filtered.status_code, 200)
        self.assertEqual(filtered.json()["total"], 1)
        self.assertEqual(filtered.json()["events"][0]["source_event_id"], "EQ:older")

        detail = self.client.get(f"/events/{newer_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["country"], _LONG_GDACS_COUNTRY)
        self.assertNotIn("raw_payload", detail.json())

        missing = self.client.get("/events/999999")
        self.assertEqual(missing.status_code, 404)

    def test_mocked_sync_preserves_long_country_and_deduplicates(self):
        first = self.event(
            "DR:1018332",
            country=_LONG_GDACS_COUNTRY,
            title=f"Drought is on going in {_LONG_GDACS_COUNTRY}",
        )

        with patch(
            "app.api.routes.events.fetch_gdacs_events",
            new=AsyncMock(return_value=[first, dict(first)]),
        ):
            response = self.client.post("/sync-live-events")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "success",
                "fetched_count": 2,
                "inserted_count": 1,
                "updated_count": 0,
                "skipped_count": 1,
            },
        )

        with Session(self.engine) as session:
            stored = session.execute(select(Event)).scalar_one()
            original_id = stored.id
            original_created_at = stored.created_at
            self.assertEqual(stored.country, _LONG_GDACS_COUNTRY)
            self.assertEqual(
                stored.raw_payload["gdacs_country_occurrences"],
                [_LONG_GDACS_COUNTRY],
            )
            self.assertEqual(
                session.execute(select(func.count()).select_from(SyncLog)).scalar_one(),
                1,
            )

        refreshed = dict(first)
        refreshed["title"] = "Updated GDACS drought title"
        refreshed["fetched_at"] = first["fetched_at"] + timedelta(hours=1)
        with patch(
            "app.api.routes.events.fetch_gdacs_events",
            new=AsyncMock(return_value=[refreshed]),
        ):
            second_response = self.client.post("/sync-live-events")

        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(second_response.json()["inserted_count"], 0)
        self.assertEqual(second_response.json()["updated_count"], 1)
        self.assertEqual(second_response.json()["skipped_count"], 1)

        with Session(self.engine) as session:
            stored = session.execute(select(Event)).scalar_one()
            self.assertEqual(stored.id, original_id)
            self.assertEqual(stored.created_at, original_created_at)
            self.assertEqual(stored.title, "Updated GDACS drought title")
            self.assertEqual(stored.country, _LONG_GDACS_COUNTRY)
            self.assertEqual(
                session.execute(select(func.count()).select_from(SyncLog)).scalar_one(),
                2,
            )


if __name__ == "__main__":
    unittest.main()
