"""Offline tests for lossless GDACS source-field occurrence preservation."""

import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from app.ingestion.gdacs import _item_to_dict


class GdacsCountryOccurrenceTests(unittest.TestCase):
    def parse_event(self, countries):
        country_xml = "".join(
            f"<gdacs:country>{country}</gdacs:country>" for country in countries
        )
        item = ET.fromstring(
            "<item xmlns:gdacs='http://www.gdacs.org'>"
            "<title>Test event</title>"
            "<gdacs:eventtype>EQ</gdacs:eventtype>"
            "<gdacs:eventid>123</gdacs:eventid>"
            f"{country_xml}"
            "</item>"
        )
        return _item_to_dict(item)

    def test_one_country(self):
        event = self.parse_event(["  India  "])

        self.assertEqual(event["country"], "India")
        self.assertEqual(event["raw_payload"]["gdacs_country_occurrences"], ["India"])

    def test_multiple_different_countries_preserve_source_order(self):
        event = self.parse_event([" India ", " Nepal ", " Bhutan "])

        self.assertEqual(event["country"], "India")
        self.assertEqual(
            event["raw_payload"]["gdacs_country_occurrences"],
            ["India", "Nepal", "Bhutan"],
        )

    def test_repeated_identical_countries_are_not_deduplicated(self):
        event = self.parse_event(["India", " India ", "India"])

        self.assertEqual(event["country"], "India")
        self.assertEqual(
            event["raw_payload"]["gdacs_country_occurrences"],
            ["India", "India", "India"],
        )

    def test_missing_country_produces_empty_occurrence_list(self):
        event = self.parse_event([])

        self.assertIsNone(event["country"])
        self.assertEqual(event["raw_payload"]["gdacs_country_occurrences"], [])


class GdacsFromdateOccurrenceTests(unittest.TestCase):
    def parse_event(self, fromdates):
        fromdate_xml = "".join(
            f"<gdacs:fromdate>{fromdate}</gdacs:fromdate>" for fromdate in fromdates
        )
        item = ET.fromstring(
            "<item xmlns:gdacs='http://www.gdacs.org'>"
            "<title>Test event</title>"
            "<pubDate>Wed, 31 Dec 2025 23:00:00 GMT</pubDate>"
            f"{fromdate_xml}"
            "</item>"
        )
        return _item_to_dict(item)

    def test_one_fromdate(self):
        event = self.parse_event(["  Thu, 01 Jan 2026 01:00:00 GMT  "])

        self.assertEqual(
            event["raw_payload"]["gdacs_fromdate_occurrences"],
            ["Thu, 01 Jan 2026 01:00:00 GMT"],
        )
        self.assertEqual(event["raw_payload"]["fromdate"], "  Thu, 01 Jan 2026 01:00:00 GMT  ")
        self.assertEqual(event["event_time"], datetime(2026, 1, 1, 1, tzinfo=timezone.utc))

    def test_repeated_identical_fromdates_are_not_deduplicated(self):
        value = "Thu, 01 Jan 2026 01:00:00 GMT"
        event = self.parse_event([value, f" {value} ", value])

        self.assertEqual(
            event["raw_payload"]["gdacs_fromdate_occurrences"],
            [value, value, value],
        )

    def test_multiple_different_fromdates_preserve_source_order(self):
        first = "Thu, 01 Jan 2026 01:00:00 GMT"
        second = "Fri, 02 Jan 2026 02:00:00 GMT"
        event = self.parse_event([f" {first} ", f" {second} "])

        self.assertEqual(
            event["raw_payload"]["gdacs_fromdate_occurrences"],
            [first, second],
        )
        self.assertEqual(event["raw_payload"]["fromdate"], f" {second} ")
        self.assertEqual(event["event_time"], datetime(2026, 1, 1, 1, tzinfo=timezone.utc))

    def test_missing_fromdate_produces_empty_list_and_keeps_pubdate_fallback(self):
        event = self.parse_event([])

        self.assertEqual(event["raw_payload"]["gdacs_fromdate_occurrences"], [])
        self.assertNotIn("fromdate", event["raw_payload"])
        self.assertEqual(event["event_time"], datetime(2025, 12, 31, 23, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
