"""Offline tests for guarded GDACS start-date feature mapping."""

import math
import unittest
from datetime import datetime, timezone

from app import ml_inference as ml


_DEFAULT = object()
_MISSING = object()
_VALID_GMT = "Thu, 01 Jan 2026 01:00:00 GMT"


class StartDateMappingTests(unittest.TestCase):
    def event(self, disaster_type="Earthquake", occurrences=_DEFAULT):
        raw_payload = {
            "gdacs_country_occurrences": ["India"],
            "fromdate": "Tue, 01 Jan 2030 00:00:00 GMT",
            "pubDate": "Wed, 01 Jan 2031 00:00:00 GMT",
        }
        if occurrences is _DEFAULT:
            raw_payload["gdacs_fromdate_occurrences"] = [_VALID_GMT]
        elif occurrences is not _MISSING:
            raw_payload["gdacs_fromdate_occurrences"] = occurrences
        return {
            "source": "GDACS",
            "disaster_type": disaster_type,
            "country": "India",
            "raw_payload": raw_payload,
            "event_time": datetime(2032, 1, 1, tzinfo=timezone.utc),
            "fetched_at": datetime(2033, 1, 1, tzinfo=timezone.utc),
        }

    def assert_date_unresolved(self, event, reason):
        mapped = ml.map_event_features(event)
        self.assertTrue(math.isnan(mapped.features["start_month"]))
        self.assertEqual(mapped.features["season"], "Unknown")
        self.assertTrue(any(reason in text for text in mapped.reasons), mapped.reasons)

    def test_one_valid_utc_or_gmt_fromdate(self):
        for zone in ("UTC", "GMT"):
            with self.subTest(zone=zone):
                value = f"Thu, 01 Jan 2026 01:00:00 {zone}"
                mapped = ml.map_event_features(self.event(occurrences=[value]))
                self.assertEqual(mapped.features["start_month"], 1)
                self.assertEqual(mapped.features["season"], "DJF")

    def test_missing_occurrence_list(self):
        self.assert_date_unresolved(self.event(occurrences=_MISSING), "provenance unavailable")

    def test_empty_occurrence_list(self):
        self.assert_date_unresolved(self.event(occurrences=[]), "No GDACS fromdate occurrences")

    def test_duplicate_identical_occurrence(self):
        self.assert_date_unresolved(
            self.event(occurrences=[_VALID_GMT, _VALID_GMT]),
            "Multiple GDACS fromdate occurrences",
        )

    def test_conflicting_multiple_occurrences(self):
        self.assert_date_unresolved(
            self.event(occurrences=[_VALID_GMT, "Fri, 02 Jan 2026 01:00:00 GMT"]),
            "Multiple GDACS fromdate occurrences",
        )

    def test_malformed_timestamp(self):
        self.assert_date_unresolved(self.event(occurrences=["not a timestamp"]), "malformed")

    def test_timezone_free_timestamp(self):
        self.assert_date_unresolved(
            self.event(occurrences=["Thu, 01 Jan 2026 01:00:00"]),
            "has no timezone",
        )

    def test_non_utc_offset(self):
        self.assert_date_unresolved(
            self.event(occurrences=["Thu, 01 Jan 2026 01:00:00 +0530"]),
            "not GMT/UTC",
        )

    def test_supported_earthquake(self):
        mapped = ml.map_event_features(
            self.event("Earthquake", ["Thu, 31 Dec 2026 23:00:00 GMT"])
        )
        self.assertEqual(mapped.features["start_month"], 12)
        self.assertEqual(mapped.features["season"], "DJF")

    def test_supported_landslide(self):
        mapped = ml.map_event_features(
            self.event("Landslide", ["Wed, 15 Apr 2026 12:00:00 UTC"])
        )
        self.assertEqual(mapped.features["start_month"], 4)
        self.assertEqual(mapped.features["season"], "MAM")

    def test_blocked_flood(self):
        self.assert_date_unresolved(self.event("Flood"), "unsupported for this hazard")

    def test_blocked_storm(self):
        self.assert_date_unresolved(
            self.event("Tropical Cyclone"), "unsupported for this hazard"
        )

    def test_blocked_volcanic_activity(self):
        self.assert_date_unresolved(
            self.event("Volcanic Event"), "unsupported for this hazard"
        )

    def test_blocked_wildfire(self):
        self.assert_date_unresolved(self.event("Wildfire"), "unsupported for this hazard")


if __name__ == "__main__":
    unittest.main()
