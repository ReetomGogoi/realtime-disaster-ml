"""Offline geography tests; no model artifacts, datasets, or GDACS requests."""

import copy
import csv
import tempfile
import unicodedata
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app import ml_inference as ml


class CountryEnrichmentTests(unittest.TestCase):
    def setUp(self):
        ml._load_country_geography.cache_clear()

    def tearDown(self):
        ml._load_country_geography.cache_clear()

    def event(self, country, occurrences=...):
        raw_payload = {"fromdate": "01 Jan 2026 00:00:00 GMT"}
        if occurrences is ...:
            raw_payload["gdacs_country_occurrences"] = [country]
        elif occurrences is not None:
            raw_payload["gdacs_country_occurrences"] = occurrences
        return {"source": "GDACS", "disaster_type": "Earthquake", "country": country,
                "raw_payload": raw_payload}

    def assert_unresolved(self, country, reason):
        mapped = ml.map_event_features(self.event(country))
        for feature in ("iso", "region", "subregion"):
            self.assertIsNone(mapped.features[feature])
        self.assertTrue(any(reason in text for text in mapped.reasons), mapped.reasons)

    def test_all_authoritative_rows_map_to_exact_tuples(self):
        with ml._COUNTRY_CROSSWALK.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(country=row["Country"]):
                geo, reason = ml.resolve_country_geography(row["Country"])
                self.assertIsNone(reason)
                self.assertEqual(geo, {"iso": row["ISO"], "region": row["region"],
                                       "subregion": row["subregion"]})

    def test_conservative_case_whitespace_and_unicode_normalization(self):
        for country in ["  INDIA  ", "\tIndia\n", "india"]:
            geo, _ = ml.resolve_country_geography(country)
            self.assertEqual(geo, {"iso": "IND", "region": "Asia", "subregion": "Southern Asia"})
        composed, _ = ml.resolve_country_geography("Côte d’Ivoire")
        decomposed, _ = ml.resolve_country_geography(unicodedata.normalize("NFD", "Côte d’Ivoire"))
        self.assertEqual(composed, decomposed)
        self.assertIsNotNone(composed)

    def test_original_event_preserved_and_nine_features_unchanged(self):
        event = self.event("  india  ")
        original = copy.deepcopy(event)
        mapped = ml.map_event_features(event)
        self.assertEqual(event, original)
        self.assertEqual(set(mapped.features), set(ml.FEATURES))
        self.assertEqual(mapped.features["iso"], "IND")
        self.assertEqual(mapped.status, "mapping_incomplete")
        self.assertFalse(any("country impact scope" in reason for reason in mapped.reasons))
        self.assertFalse(any("No verified region" in reason for reason in mapped.reasons))

    def test_one_country_occurrence_validates_scope_for_geography(self):
        mapped = ml.map_event_features(self.event("India", ["India"]))
        self.assertEqual(
            {name: mapped.features[name] for name in ("iso", "region", "subregion")},
            {"iso": "IND", "region": "Asia", "subregion": "Southern Asia"},
        )
        self.assertFalse(any("scope" in reason.lower() for reason in mapped.reasons))

    def test_multiple_different_country_occurrences_block_geography(self):
        mapped = ml.map_event_features(self.event("India", ["India", "Nepal"]))
        self.assertTrue(all(mapped.features[name] is None for name in ("iso", "region", "subregion")))
        self.assertTrue(any("Multiple GDACS country occurrences" in reason for reason in mapped.reasons))

    def test_repeated_identical_country_occurrences_block_geography(self):
        mapped = ml.map_event_features(self.event("India", ["India", "India"]))
        self.assertTrue(all(mapped.features[name] is None for name in ("iso", "region", "subregion")))
        self.assertTrue(any("Multiple GDACS country occurrences" in reason for reason in mapped.reasons))

    def test_empty_country_occurrence_list_does_not_fall_back_to_display_country(self):
        mapped = ml.map_event_features(self.event("India", []))
        self.assertTrue(all(mapped.features[name] is None for name in ("iso", "region", "subregion")))
        self.assertTrue(any("No GDACS country occurrences" in reason for reason in mapped.reasons))

    def test_legacy_event_without_country_occurrences_keeps_scope_blocker(self):
        mapped = ml.map_event_features(self.event("India", None))
        self.assertTrue(all(mapped.features[name] is None for name in ("iso", "region", "subregion")))
        self.assertTrue(any("scope remains unvalidated" in reason for reason in mapped.reasons))

    def test_territories_and_historical_codes_are_not_replaced(self):
        for country, code in [("Puerto Rico", "PRI"), ("American Samoa", "ASM"),
                              ("Azores Islands", "AZO"), ("Canary Islands", "SPI")]:
            with self.subTest(country=country):
                geo, _ = ml.resolve_country_geography(country)
                self.assertEqual(geo["iso"], code)

    def test_punctuation_in_verified_single_country_names(self):
        for country in ["China, Hong Kong Special Administrative Region", "Antigua and Barbuda",
                        "United Kingdom of Great Britain and Northern Ireland"]:
            self.assertIsNotNone(ml.resolve_country_geography(country)[0])

    def test_missing_and_non_text_countries(self):
        for value in [None, "", "  ", ["India", "Nepal"], {"country": "India"}]:
            with self.subTest(country=value):
                self.assert_unresolved(value, "not a single country-text value")

    def test_no_unreviewed_aliases_or_fuzzy_matching(self):
        for value in ["Indai", "Korea", "United States", "Turkey", "Cote d'Ivoire"]:
            with self.subTest(country=value):
                self.assert_unresolved(value, "no exact verified historical match")

    def test_multi_country_never_chooses_first(self):
        for value in ["India, Nepal", "India;Nepal", "India / Nepal", "India and Nepal",
                      "India & Nepal", "India|Nepal", "India\nNepal"]:
            with self.subTest(country=value):
                self.assert_unresolved(value, "multi-country")

    def test_coordinates_cannot_supply_missing_or_unknown_country(self):
        event = self.event("Unmatched name")
        event.update(latitude=28.6, longitude=77.2, iso="IND", region="Asia", subregion="Southern Asia")
        mapped = ml.map_event_features(event)
        self.assertTrue(all(mapped.features[k] is None for k in ("iso", "region", "subregion")))

    def test_ambiguous_normalized_name_blocks_atomic_mapping(self):
        self.check_fixture(
            "Country,ISO,region,subregion\nIndia,IND,Asia,Southern Asia\n"
            " INDIA ,NPL,Asia,Southern Asia\n", "India", "Ambiguous")

    def test_conflicting_geography_for_same_iso_blocks_mapping(self):
        self.check_fixture(
            "Country,ISO,region,subregion\nIndia,IND,Asia,Southern Asia\n"
            "France,IND,Europe,Western Europe\n", "India", "conflicting")

    def test_incomplete_or_invalid_crosswalk_is_not_partially_used(self):
        for content in ["Country,ISO,region,subregion\nIndia,IND,Asia,\n",
                        "country,ISO,region,subregion\nIndia,IND,Asia,Southern Asia\n",
                        "Country,ISO,region,subregion\n"]:
            self.check_fixture(content, "India", "unavailable or invalid")

    def check_fixture(self, content, country, reason):
        # Deliberately invalid test tables are never installed as production data.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.csv"
            path.write_text(content, encoding="utf-8")
            with patch.object(ml, "_COUNTRY_CROSSWALK", path):
                ml._load_country_geography.cache_clear()
                self.assert_unresolved(country, reason)
        ml._load_country_geography.cache_clear()

    def test_missing_crosswalk_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(ml, "_COUNTRY_CROSSWALK", Path(folder) / "absent.csv"):
                self.assert_unresolved("India", "unavailable or invalid")

    def test_lookup_cached_without_changing_source(self):
        ml.resolve_country_geography("India")
        ml.resolve_country_geography("Nepal")
        self.assertEqual(ml._load_country_geography.cache_info().misses, 1)
        self.assertEqual(ml._load_country_geography.cache_info().hits, 1)

    def test_geography_success_does_not_enable_predictions(self):
        pipeline = Mock()
        bundle = {target: ml.ModelState(pipeline=pipeline, supported_hazards=frozenset({"Earthquake"}))
                  for target in ml.TARGETS}
        for result in ml.predict_event(self.event("India"), bundle).values():
            self.assertEqual(result["status"], "mapping_incomplete")
            self.assertIsNone(result["predicted_label"])
            self.assertIsNone(result["probabilities"])
        pipeline.predict.assert_not_called()
        pipeline.predict_proba.assert_not_called()


if __name__ == "__main__":
    unittest.main()
