"""Offline API regressions for the deployed Version 1 monitoring path."""

import os
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event as sqlalchemy_event, func, select
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
        newer["raw_payload"]["alertlevel"] = "Orange"
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
        self.assertEqual(payload["events"][0]["alert_level"], "Orange")
        self.assertNotIn("raw_payload", payload["events"][0])
        self.assertNotIn("ml_prediction", payload["events"][0])

        filtered = self.client.get("/events", params={"country": "Nepal"})
        self.assertEqual(filtered.status_code, 200)
        self.assertEqual(filtered.json()["total"], 1)
        self.assertEqual(filtered.json()["events"][0]["source_event_id"], "EQ:older")
        self.assertIsNone(filtered.json()["events"][0]["alert_level"])

        detail = self.client.get(f"/events/{newer_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["country"], _LONG_GDACS_COUNTRY)
        self.assertNotIn("raw_payload", detail.json())

        with Session(self.engine) as session:
            row = session.execute(
                select(Event).where(Event.source_event_id == "EQ:newer")
            ).scalar_one()
            row.raw_payload = {**row.raw_payload, "alertlevel": "unsupported"}
            session.commit()
        self.assertIsNone(self.client.get(f"/events/{newer_id}").json()["alert_level"])

        with Session(self.engine) as session:
            row = session.execute(
                select(Event).where(Event.source_event_id == "EQ:newer")
            ).scalar_one()
            row.raw_payload = {
                "alertlevel": "Red",
                "gdacs_cyclone_observations_v1": {
                    "alert_level": {"status": "ambiguous", "value": None}
                },
            }
            session.commit()
        self.assertIsNone(self.client.get(f"/events/{newer_id}").json()["alert_level"])

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

    def test_successful_fetch_finishes_before_database_checkout(self):
        checkouts = 0

        def record_checkout(*_args):
            nonlocal checkouts
            checkouts += 1

        async def fetch_without_database_connection():
            self.assertEqual(checkouts, 0)
            return []

        sqlalchemy_event.listen(self.engine, "checkout", record_checkout)
        try:
            with patch(
                "app.api.routes.events.fetch_gdacs_events",
                new=fetch_without_database_connection,
            ):
                response = self.client.post("/sync-live-events")
        finally:
            sqlalchemy_event.remove(self.engine, "checkout", record_checkout)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "success",
                "fetched_count": 0,
                "inserted_count": 0,
                "updated_count": 0,
                "skipped_count": 0,
            },
        )
        self.assertGreater(checkouts, 0)

    def test_upstream_timeout_returns_504_without_exposing_details(self):
        request = httpx.Request("GET", "https://www.gdacs.org/xml/rss_7d.xml")
        failure = httpx.ReadTimeout(
            "private timeout diagnostics",
            request=request,
        )

        with patch(
            "app.api.routes.events.fetch_gdacs_events",
            new=AsyncMock(side_effect=failure),
        ):
            response = self.client.post("/sync-live-events")

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json(), {"detail": "GDACS request timed out."})
        self.assertNotIn("private timeout diagnostics", response.text)
        self.assert_failed_sync_log("private timeout diagnostics")

    def test_upstream_http_error_returns_502_without_exposing_details(self):
        request = httpx.Request("GET", "https://www.gdacs.org/xml/rss_7d.xml")
        upstream_response = httpx.Response(503, request=request)
        failure = httpx.HTTPStatusError(
            "private upstream response details",
            request=request,
            response=upstream_response,
        )

        with patch(
            "app.api.routes.events.fetch_gdacs_events",
            new=AsyncMock(side_effect=failure),
        ):
            response = self.client.post("/sync-live-events")

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json(),
            {"detail": "GDACS upstream request or feed parsing failed."},
        )
        self.assertNotIn("private upstream response details", response.text)
        self.assert_failed_sync_log("private upstream response details")

    def test_malformed_upstream_xml_returns_502_without_exposing_details(self):
        failure = ET.ParseError("private malformed XML details")

        with patch(
            "app.api.routes.events.fetch_gdacs_events",
            new=AsyncMock(side_effect=failure),
        ):
            response = self.client.post("/sync-live-events")

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json(),
            {"detail": "GDACS upstream request or feed parsing failed."},
        )
        self.assertNotIn("private malformed XML details", response.text)
        self.assert_failed_sync_log("private malformed XML details")

    def test_database_failure_rolls_back_events_and_returns_sanitized_503(self):
        valid = self.event("EQ:valid")
        invalid = self.event("EQ:invalid")
        invalid["title"] = None

        with patch(
            "app.api.routes.events.fetch_gdacs_events",
            new=AsyncMock(return_value=[valid, invalid]),
        ):
            response = self.client.post("/sync-live-events")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"detail": "GDACS events could not be stored."},
        )
        self.assertNotIn("INSERT INTO", response.text)
        self.assertNotIn("EQ:invalid", response.text)

        with Session(self.engine) as session:
            self.assertEqual(
                session.execute(select(func.count()).select_from(Event)).scalar_one(),
                0,
            )
            logs = session.execute(select(SyncLog)).scalars().all()
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0].status, "error")
            self.assertIn("NOT NULL", logs[0].error_message.upper())

    def assert_failed_sync_log(self, internal_detail: str):
        with Session(self.engine) as session:
            self.assertEqual(
                session.execute(select(func.count()).select_from(Event)).scalar_one(),
                0,
            )
            logs = session.execute(select(SyncLog)).scalars().all()
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0].status, "error")
            self.assertEqual(logs[0].fetched_count, 0)
            self.assertIn(internal_detail, logs[0].error_message)


if __name__ == "__main__":
    unittest.main()
