"""Offline tests for lossless GDACS country-occurrence preservation."""

import unittest
import xml.etree.ElementTree as ET

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


if __name__ == "__main__":
    unittest.main()
