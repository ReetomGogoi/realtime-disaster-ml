"""Offline storage tests for lossless GDACS event text; no external DB access."""

import unittest
import xml.etree.ElementTree as ET

from sqlalchemy import Text, create_engine, inspect, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from app.database.base import Base
from app.database.models import Event
from app.ingestion.gdacs import _item_to_dict


_LONG_GDACS_COUNTRY = (
    "Austria, Bosnia & Herzegovina, Belgium, Belarus, Switzerland, Czech Republic, "
    "Germany, Denmark, Spain, France, Croatia, Hungary, Ireland, Italy, Liechtenstein, "
    "Luxembourg, Moldova, Montenegro, Netherlands, Poland, Romania, Serbia, Sweden, "
    "Slovenia, Slovakia, Ukraine,"
)
_LONG_GDACS_TITLE = f"Drought is on going in {_LONG_GDACS_COUNTRY}"
_GDACS_NAMESPACE = "http://www.gdacs.org"


class EventStorageTests(unittest.TestCase):
    def test_country_column_is_unbounded_text_for_supported_databases(self):
        self.assertIsInstance(Event.__table__.c.country.type, Text)
        postgresql_ddl = str(
            CreateTable(Event.__table__).compile(dialect=postgresql.dialect())
        )
        self.assertIn("country TEXT", postgresql_ddl)

        engine = create_engine("sqlite:///:memory:")
        try:
            Base.metadata.create_all(engine)
            columns = {column["name"]: column for column in inspect(engine).get_columns("events")}
            self.assertIsInstance(columns["country"]["type"], Text)
        finally:
            engine.dispose()

    def test_verified_long_gdacs_country_round_trips_without_truncation(self):
        item = ET.Element("item")
        ET.SubElement(item, "title").text = _LONG_GDACS_TITLE
        ET.SubElement(item, f"{{{_GDACS_NAMESPACE}}}eventtype").text = "DR"
        ET.SubElement(item, f"{{{_GDACS_NAMESPACE}}}eventid").text = "1018332"
        ET.SubElement(item, f"{{{_GDACS_NAMESPACE}}}country").text = _LONG_GDACS_COUNTRY
        parsed = _item_to_dict(item)
        self.assertEqual(parsed["source_event_id"], "DR:1018332")
        self.assertEqual(parsed["disaster_type"], "Drought")
        self.assertEqual(parsed["title"], _LONG_GDACS_TITLE)
        self.assertEqual(len(parsed["country"]), 267)
        self.assertEqual(parsed["country"], _LONG_GDACS_COUNTRY)

        engine = create_engine("sqlite:///:memory:")
        try:
            Base.metadata.create_all(engine)
            with Session(engine) as session:
                session.add(Event(**parsed))
                session.commit()
                stored_country = session.execute(select(Event.country)).scalar_one()
            self.assertEqual(stored_country, _LONG_GDACS_COUNTRY)
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
