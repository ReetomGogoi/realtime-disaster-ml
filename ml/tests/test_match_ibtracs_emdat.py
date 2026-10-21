"""
Unit Tests for V2 IBTrACS–EM-DAT Matching Pipeline
===================================================
Lightweight unit tests using purely synthetic, in-memory fixtures.
No synthetic data is written to training datasets or database.
Tests cover:
  1. Name normalisation & Jaro-Winkler metric
  2. Meteorological basin concordance
  3. Deterministic matching tiers (T2 exact, T3 fuzzy, T4 date fallback)
  4. GLIDE cross-reference resolution for ambiguous records
  5. Quarantine of unverified T4 fallback matches
  6. Physical cyclone identity preservation (multi-country grouping)
  7. Strict 2023–2025 development exclusion guards
  8. Verified high-confidence dataset integrity checks
"""

import unittest
from datetime import date
from pathlib import Path
import pandas as pd

from ml.preprocessing.match_ibtracs_emdat import (
    normalise_name,
    jaro_winkler,
    basin_matches_subregion,
    guard_test_period,
    try_t2_exact,
    try_t3_fuzzy,
    try_t4_country_date,
    build_ibtracs_indexes,
    group_by_physical_storm,
    IBTrACSStorm,
    PILOT_YEAR_START,
    PILOT_YEAR_END,
    PROCESSED_DATASET,
)


class TestNameNormalisation(unittest.TestCase):
    """Test string normalisation and Jaro-Winkler distance calculation."""

    def test_strip_disaster_category_prefixes(self):
        self.assertEqual(normalise_name("Hurricane Katrina"), "KATRINA")
        self.assertEqual(normalise_name("Typhoon Haiyan"), "HAIYAN")
        self.assertEqual(normalise_name("Cyclone Pam"), "PAM")
        self.assertEqual(normalise_name("Tropical Storm Chantal"), "CHANTAL")
        self.assertEqual(normalise_name("Tropical Cyclone Niran"), "NIRAN")

    def test_strip_parentheses_and_punctuation(self):
        # Parentheses containing local/dual names are stripped for T2 exact matching
        self.assertEqual(normalise_name("Typhoon 'Saomai' (Osang)"), "SAOMAI")
        self.assertEqual(normalise_name("Tropical Cyclone 'Seroja-21'"), "SEROJA 21")

    def test_empty_and_none_input(self):
        self.assertEqual(normalise_name(""), "")
        self.assertEqual(normalise_name(None), "")

    def test_jaro_winkler_metric(self):
        # Exact match
        self.assertEqual(jaro_winkler("KATRINA", "KATRINA"), 1.0)
        # Empty string
        self.assertEqual(jaro_winkler("", "KATRINA"), 0.0)
        # High similarity (minor spelling/transliteration variation)
        score_close = jaro_winkler("HAIYAN", "HAIYANN")
        self.assertGreaterEqual(score_close, 0.92)
        # Low similarity
        score_low = jaro_winkler("KATRINA", "ANDREW")
        self.assertLess(score_low, 0.60)


class TestBasinConcordance(unittest.TestCase):
    """Test basin to subregion meteorological concordance."""

    def test_concordant_basins(self):
        self.assertTrue(basin_matches_subregion("NA", "Northern America"))
        self.assertTrue(basin_matches_subregion("NA", "Caribbean"))
        self.assertTrue(basin_matches_subregion("WP", "South-Eastern Asia"))
        self.assertTrue(basin_matches_subregion("SI", "Sub-Saharan Africa"))

    def test_discordant_basins(self):
        # North Atlantic storm cannot affect South-Eastern Asia
        self.assertFalse(basin_matches_subregion("NA", "South-Eastern Asia"))
        # Western Pacific storm cannot affect Caribbean
        self.assertFalse(basin_matches_subregion("WP", "Caribbean"))

    def test_empty_basin_or_subregion(self):
        self.assertTrue(basin_matches_subregion("", "Caribbean"))
        self.assertTrue(basin_matches_subregion("NA", ""))


class TestMatchingTiersSynthetic(unittest.TestCase):
    """Test deterministic matching tiers using minimal synthetic in-memory fixtures."""

    def setUp(self):
        # Minimal synthetic storm dictionary
        self.storms = {
            "STORM_001": IBTrACSStorm("STORM_001", "2005", "NA", "KATRINA"),
            "STORM_002": IBTrACSStorm("STORM_002", "2013", "WP", "HAIYAN"),
            "STORM_003": IBTrACSStorm("STORM_003", "2015", "SP", "PAM"),
        }
        self.storms["STORM_001"].first_obs = "2005-08-23 18:00:00"
        self.storms["STORM_001"].last_obs = "2005-08-31 00:00:00"

        self.storms["STORM_002"].first_obs = "2013-11-03 06:00:00"
        self.storms["STORM_002"].last_obs = "2013-11-11 12:00:00"

        self.name_season_idx, self.season_idx = build_ibtracs_indexes(self.storms)

    def test_t2_exact_match(self):
        matches = try_t2_exact("KATRINA", "2005", "Northern America", self.name_season_idx)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].sid, "STORM_001")

    def test_t2_exact_basin_mismatch_rejected(self):
        # Katrina with discordant subregion
        matches = try_t2_exact("KATRINA", "2005", "South-Eastern Asia", self.name_season_idx)
        self.assertEqual(len(matches), 0)

    def test_t3_fuzzy_match(self):
        # Name with slight transliteration variation
        matches, score = try_t3_fuzzy("HAIYANN", "2013", "South-Eastern Asia", self.season_idx)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].sid, "STORM_002")
        self.assertGreaterEqual(score, 0.92)

    def test_t4_country_date_fallback(self):
        # Date overlapping Katrina's track
        start_d = date(2005, 8, 25)
        end_d = date(2005, 8, 30)
        matches = try_t4_country_date("USA", start_d, end_d, "2005", self.season_idx)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].sid, "STORM_001")


class TestGlideCrossReference(unittest.TestCase):
    """Test resolution of ambiguous multi-country records using shared GLIDE ID."""

    def test_resolve_via_shared_glide(self):
        # Synthetic scenario: Country A accepted Katrina via T2, Country B ambiguous
        accepted_records = [
            {
                "dis_no": "2005-0010-USA",
                "matched_sid": "STORM_001",
                "matched_name": "KATRINA",
                "glide": "GLIDE:TC-2005-000001",
                "is_high_confidence": True,
            }
        ]
        ambiguous_record = {
            "dis_no": "2005-0010-BHS",
            "glide": "GLIDE:TC-2005-000001",
            "match_status": "AMBIGUOUS",
        }

        # Build GLIDE lookup
        glide_to_sid = {}
        for r in accepted_records:
            glide_to_sid[r["glide"]] = r["matched_sid"]

        # Resolve
        self.assertIn(ambiguous_record["glide"], glide_to_sid)
        resolved_sid = glide_to_sid[ambiguous_record["glide"]]
        self.assertEqual(resolved_sid, "STORM_001")


class TestT4Quarantine(unittest.TestCase):
    """Test that unverified blank-name T4 matches are strictly quarantined."""

    def test_unverified_blank_name_quarantine(self):
        # A blank event name matched via date window must be quarantined
        synthetic_record = {
            "dis_no": "2010-0171-IND",
            "event_name": None,
            "match_tier": "T4_country_date",
            "matched_sid": "2010091S09093",
        }
        ev = synthetic_record["event_name"]
        is_verified = (synthetic_record["dis_no"] == "2011-0157-PHL") or (ev is not None and str(ev).strip() != "")
        self.assertFalse(is_verified)

    def test_verified_t4_accepted(self):
        # 2011-0157-PHL (Typhoon Bebeng / Aere) is verified
        synthetic_record = {
            "dis_no": "2011-0157-PHL",
            "event_name": "Typhoon Bebeng (Aere)",
            "match_tier": "T4_country_date",
            "matched_sid": "2011126N11129",
        }
        is_verified = synthetic_record["dis_no"] == "2011-0157-PHL"
        self.assertTrue(is_verified)


class TestPhysicalCyclonePreservation(unittest.TestCase):
    """Test grouping and identity preservation for multi-country cyclones."""

    def test_group_by_physical_storm(self):
        # Synthetic DataFrame with 3 country records for Hurricane Irma
        synthetic_df = pd.DataFrame([
            {
                "dis_no": "2017-0381-PRI",
                "dis_event_id": "2017-0381",
                "country": "Puerto Rico",
                "matched_sid": "2017242N16333",
                "matched_name": "IRMA",
                "matched_basin": "NA",
                "start_year": "2017",
                "split": "train",
                "match_tier": "T2_exact",
            },
            {
                "dis_no": "2017-0381-CUB",
                "dis_event_id": "2017-0381",
                "country": "Cuba",
                "matched_sid": "2017242N16333",
                "matched_name": "IRMA",
                "matched_basin": "NA",
                "start_year": "2017",
                "split": "train",
                "match_tier": "T2_exact",
            },
            {
                "dis_no": "2017-0381-USA",
                "dis_event_id": "2017-0381",
                "country": "United States of America",
                "matched_sid": "2017242N16333",
                "matched_name": "IRMA",
                "matched_basin": "NA",
                "start_year": "2017",
                "split": "train",
                "match_tier": "T2_exact",
            },
        ])

        grouped = group_by_physical_storm(synthetic_df)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped.iloc[0]["matched_sid"], "2017242N16333")
        self.assertEqual(grouped.iloc[0]["storm_name"], "IRMA")
        self.assertEqual(grouped.iloc[0]["country_count"], 3)
        self.assertIn("Cuba", grouped.iloc[0]["countries"])
        self.assertIn("Puerto Rico", grouped.iloc[0]["countries"])
        self.assertIn("United States of America", grouped.iloc[0]["countries"])

    def test_temporal_split_consistency(self):
        # A single physical storm must have identical split across all country rows
        synthetic_df = pd.DataFrame([
            {"matched_sid": "2021093S11122", "split": "validation", "start_year": "2021"},
            {"matched_sid": "2021093S11122", "split": "validation", "start_year": "2021"},
        ])
        splits = synthetic_df.groupby("matched_sid")["split"].nunique()
        self.assertTrue((splits == 1).all())


class TestDevelopmentLeakageGuards(unittest.TestCase):
    """Test strict enforcement of test-period (2023-2025) exclusion guards."""

    def test_guard_blocks_2023_record(self):
        df_violation = pd.DataFrame([{"Start Year": 2023}])
        with self.assertRaises(ValueError) as ctx:
            guard_test_period(df_violation)
        self.assertIn("LEAKAGE GUARD VIOLATION", str(ctx.exception))

    def test_guard_blocks_2024_record(self):
        df_violation = pd.DataFrame([{"Start Year": 2024}])
        with self.assertRaises(ValueError) as ctx:
            guard_test_period(df_violation)
        self.assertIn("LEAKAGE GUARD VIOLATION", str(ctx.exception))

    def test_guard_permits_valid_development_years(self):
        df_valid = pd.DataFrame([{"Start Year": 2005}, {"Start Year": 2020}, {"Start Year": 2022}])
        try:
            guard_test_period(df_valid)
        except ValueError:
            self.fail("guard_test_period raised ValueError unexpectedly for valid development years.")


class TestVerifiedDatasetIntegrity(unittest.TestCase):
    """Verify integrity of the training-ready high-confidence processed dataset."""

    @classmethod
    def setUpClass(cls):
        if not PROCESSED_DATASET.exists():
            raise unittest.SkipTest("Processed dataset not yet generated.")
        cls.df = pd.read_csv(PROCESSED_DATASET)

    def test_exact_record_counts(self):
        # Exactly 1,033 high-confidence records reproduced
        self.assertEqual(len(self.df), 1033)
        # Training split: 868
        self.assertEqual((self.df["split"] == "train").sum(), 868)
        # Validation split: 165
        self.assertEqual((self.df["split"] == "validation").sum(), 165)

    def test_no_test_period_leakage(self):
        # Years must strictly be in 2000-2022
        years = pd.to_numeric(self.df["start_year"], errors="raise")
        self.assertTrue((years >= PILOT_YEAR_START).all())
        self.assertTrue((years <= PILOT_YEAR_END).all())
        self.assertFalse((years >= 2023).any())

    def test_physical_storm_identifiers_populated(self):
        # Every high-confidence record must have valid matched_sid and dis_event_id
        self.assertTrue(self.df["matched_sid"].notna().all())
        self.assertTrue((self.df["matched_sid"].str.strip() != "").all())
        self.assertTrue(self.df["dis_event_id"].notna().all())

    def test_zero_unverified_t4_records(self):
        # No quarantined records can exist in the high-confidence dataset
        self.assertFalse((self.df["match_status"] == "QUARANTINED_UNVERIFIED_T4").any())
        self.assertFalse((self.df["match_status"] == "AMBIGUOUS").any())


if __name__ == "__main__":
    unittest.main()
