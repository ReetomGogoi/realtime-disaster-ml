"""Focused tests for the conservative IBTrACS/EM-DAT matching pipeline."""

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from ml.preprocessing.match_ibtracs_emdat import (
    EMDAT_MATCH_COLUMNS,
    IBTrACSStorm,
    basin_matches_subregion,
    build_ibtracs_indexes,
    group_by_physical_storm,
    guard_test_period,
    is_supported_sid,
    jaro_winkler,
    load_emdat_tc,
    load_ibtracs_index,
    match_emdat_record,
    normalise_geography,
    normalise_name,
    run_matching_pipeline,
    storm_overlaps_event,
    try_t2_exact,
    try_t3_fuzzy,
    try_t4_country_date,
    validate_event_country_identifiers,
)


def make_storm(
    sid: str,
    season: str,
    basin: str,
    name: str,
    first: str,
    last: str,
) -> IBTrACSStorm:
    storm = IBTrACSStorm(sid=sid, season=season)
    storm.add_observation(basin, name, first)
    storm.add_observation(basin, name, last)
    return storm


def prepared_event(**overrides: object) -> pd.Series:
    values: dict[str, object] = {
        "DisNo.": "2005-0001-USA",
        "Event Name": "Hurricane Katrina",
        "Country": "United States of America",
        "ISO": "USA",
        "Subregion": "Northern America",
        "External IDs": "",
        "_norm_name": "KATRINA",
        "_start_year": 2005,
        "_start_month": 8,
        "_start_day": 25,
        "_end_year": 2005,
        "_end_month": 8,
        "_end_day": 30,
        "_iso": "USA",
        "_subregion": "northern america",
        "_split": "train",
    }
    values.update(overrides)
    return pd.Series(values)


def accepted_record(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "matched_sid": "2005235N12345",
        "matched_name": "KATRINA",
        "matched_basin": "NA",
        "matched_season": "2005",
        "start_year": "2005",
        "split": "train",
        "country": "United States of America",
        "iso": "USA",
        "event_country_id": "2005-0001-USA",
        "dis_no": "2005-0001-USA",
        "dis_event_id": "2005-0001",
        "match_tier": "T2_exact",
    }
    values.update(overrides)
    return values


def write_ibtracs(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SID", "SEASON", "BASIN", "NAME", "ISO_TIME"])
        writer.writerow(["", "year", "", "", "UTC"])
        writer.writerows(rows)


def emdat_row(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "DisNo.": "2005-0001-USA",
        "Disaster Group": "Natural",
        "Disaster Type": "Storm",
        "Disaster Subtype": "Tropical cyclone",
        "Event Name": "Hurricane Katrina",
        "ISO": "USA",
        "Country": "United States of America",
        "Subregion": "Northern America",
        "External IDs": "",
        "Start Year": 2005,
        "Start Month": 8,
        "Start Day": 25,
        "End Year": 2005,
        "End Month": 8,
        "End Day": 30,
    }
    values.update(overrides)
    return values


class TestNormalization(unittest.TestCase):
    def test_name_prefixes_and_punctuation(self) -> None:
        self.assertEqual(normalise_name("Hurricane 'Katrina'"), "KATRINA")
        self.assertEqual(normalise_name("Tropical Cyclone Pam"), "PAM")

    def test_name_missing(self) -> None:
        self.assertEqual(normalise_name(None), "")
        self.assertEqual(normalise_name(float("nan")), "")

    def test_jaro_winkler(self) -> None:
        self.assertEqual(jaro_winkler("HAIYAN", "HAIYAN"), 1.0)
        self.assertGreaterEqual(jaro_winkler("HAIYANN", "HAIYAN"), 0.92)
        self.assertLess(jaro_winkler("HAIYAN", "KATRINA"), 0.7)

    def test_geography_normalization_is_conservative(self) -> None:
        self.assertEqual(normalise_geography("  South-Eastern   Asia "), "south-eastern asia")


class TestGuardsAndCompatibility(unittest.TestCase):
    def test_guard_blocks_test_period(self) -> None:
        with self.assertRaisesRegex(ValueError, "LEAKAGE GUARD"):
            guard_test_period(pd.DataFrame({"Start Year": [2022, 2023]}))

    def test_guard_requires_year_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required year"):
            guard_test_period(pd.DataFrame({"year": [2020]}))

    def test_supported_sid_format(self) -> None:
        self.assertTrue(is_supported_sid("2005235N12345"))
        self.assertFalse(is_supported_sid("STORM_001"))
        self.assertFalse(is_supported_sid(""))

    def test_unknown_or_incompatible_geography_rejected(self) -> None:
        self.assertFalse(basin_matches_subregion("", "Caribbean"))
        self.assertFalse(basin_matches_subregion("XX", "Caribbean"))
        self.assertFalse(basin_matches_subregion("NA", ""))
        self.assertFalse(basin_matches_subregion("NA", "South-Eastern Asia"))
        self.assertTrue(basin_matches_subregion("NA", "Northern America"))

    def test_track_overlap_is_direct(self) -> None:
        storm = make_storm("2005235N12345", "2005", "NA", "KATRINA", "2005-08-23 00:00:00", "2005-08-31 00:00:00")
        self.assertTrue(storm_overlaps_event(storm, date(2005, 8, 25), date(2005, 8, 30)))
        self.assertFalse(storm_overlaps_event(storm, date(2005, 9, 2), date(2005, 9, 4)))

    def test_identifier_validation_requires_nonempty_unique_dis_no(self) -> None:
        valid = pd.DataFrame({"DisNo.": [" 2005-0001-USA ", "2005-0001-BHS"]})
        self.assertEqual(
            validate_event_country_identifiers(valid).tolist(),
            ["2005-0001-USA", "2005-0001-BHS"],
        )
        with self.assertRaisesRegex(ValueError, "missing DisNo"):
            validate_event_country_identifiers(pd.DataFrame({"DisNo.": [" "]}))
        with self.assertRaisesRegex(ValueError, "duplicate DisNo"):
            validate_event_country_identifiers(
                pd.DataFrame({"DisNo.": ["2005-0001-USA", "2005-0001-USA"]})
            )


class TestProductionMatchingFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.katrina = make_storm("2005235N12345", "2005", "NA", "KATRINA", "2005-08-23 00:00:00", "2005-08-31 00:00:00")
        self.name_index, self.season_index = build_ibtracs_indexes({self.katrina.sid: self.katrina})

    def test_t2_requires_date_overlap(self) -> None:
        matches = try_t2_exact("KATRINA", "2005", "Northern America", date(2005, 8, 25), date(2005, 8, 30), self.name_index)
        self.assertEqual([storm.sid for storm in matches], [self.katrina.sid])
        no_overlap = try_t2_exact("KATRINA", "2005", "Northern America", date(2005, 9, 5), date(2005, 9, 6), self.name_index)
        self.assertEqual(no_overlap, [])

    def test_t2_rejects_missing_geography(self) -> None:
        matches = try_t2_exact("KATRINA", "2005", "", date(2005, 8, 25), date(2005, 8, 30), self.name_index)
        self.assertEqual(matches, [])

    def test_t2_rejects_incompatible_geography(self) -> None:
        matches = try_t2_exact("KATRINA", "2005", "South-Eastern Asia", date(2005, 8, 25), date(2005, 8, 30), self.name_index)
        self.assertEqual(matches, [])

    def test_t3_requires_date_and_geography(self) -> None:
        matches, score = try_t3_fuzzy("KATRINAA", "2005", "Northern America", date(2005, 8, 25), date(2005, 8, 30), self.season_index)
        self.assertEqual([storm.sid for storm in matches], [self.katrina.sid])
        self.assertGreaterEqual(score, 0.92)

    def test_reused_name_is_disambiguated_by_track_dates(self) -> None:
        older = make_storm("2004235N12345", "2004", "NA", "KATRINA", "2004-08-20 00:00:00", "2004-08-25 00:00:00")
        names, _ = build_ibtracs_indexes({older.sid: older, self.katrina.sid: self.katrina})
        matches = try_t2_exact("KATRINA", "2005", "Northern America", date(2005, 8, 25), date(2005, 8, 30), names)
        self.assertEqual([storm.sid for storm in matches], [self.katrina.sid])

    def test_ambiguous_same_name_and_dates_are_rejected(self) -> None:
        duplicate = make_storm("2005236N12346", "2005", "NA", "KATRINA", "2005-08-24 00:00:00", "2005-08-30 00:00:00")
        names, seasons = build_ibtracs_indexes({self.katrina.sid: self.katrina, duplicate.sid: duplicate})
        result = match_emdat_record(prepared_event(), names, seasons)
        self.assertEqual(result["match_status"], "AMBIGUOUS")
        self.assertFalse(result["is_high_confidence"])
        self.assertEqual(result["candidate_count"], 2)

    def test_unsupported_sid_cannot_match(self) -> None:
        unsupported = make_storm("STORM_001", "2005", "NA", "KATRINA", "2005-08-23", "2005-08-31")
        names, seasons = build_ibtracs_indexes({unsupported.sid: unsupported})
        result = match_emdat_record(prepared_event(), names, seasons)
        self.assertEqual(result["match_status"], "NO_MATCH")

    def test_country_date_only_never_matches_wrong_country(self) -> None:
        matches = try_t4_country_date("JPN", date(2005, 8, 25), date(2005, 8, 30), "2005", self.season_index)
        self.assertEqual(matches, [])

    def test_actual_record_match_rejects_missing_geography(self) -> None:
        result = match_emdat_record(prepared_event(_iso="", _subregion=""), self.name_index, self.season_index)
        self.assertEqual(result["reject_reason"], "missing_geography")
        self.assertFalse(result["is_high_confidence"])

    def test_match_preserves_identifier_and_canonical_start_date(self) -> None:
        result = match_emdat_record(prepared_event(), self.name_index, self.season_index)
        self.assertEqual(result["event_country_id"], "2005-0001-USA")
        self.assertEqual(result["start_year"], 2005)
        self.assertEqual(result["start_month"], 8)
        self.assertEqual(result["start_day"], 25)
        self.assertEqual(result["start_date"], "2005-08-25")

    def test_match_rejects_missing_or_inconsistent_identifier(self) -> None:
        missing = match_emdat_record(
            prepared_event(**{"DisNo.": "", "_event_country_id": ""}),
            self.name_index,
            self.season_index,
        )
        self.assertEqual(missing["reject_reason"], "missing_event_country_id")
        inconsistent = match_emdat_record(
            prepared_event(_event_country_id="2005-9999-USA"),
            self.name_index,
            self.season_index,
        )
        self.assertEqual(inconsistent["reject_reason"], "inconsistent_event_country_id")

    def test_invalid_or_fractional_start_date_is_rejected(self) -> None:
        invalid = match_emdat_record(
            prepared_event(_start_month=2, _start_day=30),
            self.name_index,
            self.season_index,
        )
        self.assertEqual(invalid["reject_reason"], "missing_or_invalid_start_date")
        self.assertEqual(invalid["start_date"], "")
        fractional = match_emdat_record(
            prepared_event(_start_month=8.5),
            self.name_index,
            self.season_index,
        )
        self.assertEqual(fractional["reject_reason"], "missing_or_invalid_start_date")


class TestPhysicalStormGrouping(unittest.TestCase):
    def test_multi_country_records_become_one_physical_storm(self) -> None:
        frame = pd.DataFrame([
            accepted_record(),
            accepted_record(country="Bahamas", iso="BHS", event_country_id="2005-0001-BHS", dis_no="2005-0001-BHS"),
        ])
        grouped = group_by_physical_storm(frame)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped.iloc[0]["country_record_count"], 2)
        self.assertEqual(grouped.iloc[0]["isos"], "BHS; USA")
        self.assertEqual(
            grouped.iloc[0]["event_country_ids"],
            "2005-0001-BHS; 2005-0001-USA",
        )

    def test_cross_split_sid_is_rejected(self) -> None:
        frame = pd.DataFrame([
            accepted_record(),
            accepted_record(split="validation", start_year="2020", event_country_id="2020-0001-BHS", dis_no="2020-0001-BHS"),
        ])
        with self.assertRaisesRegex(ValueError, "spans temporal splits"):
            group_by_physical_storm(frame)

    def test_unsupported_sid_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            group_by_physical_storm(pd.DataFrame([accepted_record(matched_sid="STORM_001")]))

    def test_inconsistent_or_duplicate_event_country_ids_are_rejected(self) -> None:
        inconsistent = pd.DataFrame([
            accepted_record(event_country_id="2005-9999-USA"),
        ])
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            group_by_physical_storm(inconsistent)
        duplicated = pd.DataFrame([accepted_record(), accepted_record()])
        with self.assertRaisesRegex(ValueError, "duplicate event-country"):
            group_by_physical_storm(duplicated)

    def test_empty_input_has_grouped_schema(self) -> None:
        empty = pd.DataFrame(columns=accepted_record().keys())
        grouped = group_by_physical_storm(empty)
        self.assertTrue(grouped.empty)
        self.assertIn("matched_sid", grouped.columns)


class TestLoadersAndPipeline(unittest.TestCase):
    def test_ibtracs_loader_requires_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("SID,SEASON\n,year\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_ibtracs_index(path)

    def test_emdat_loader_filters_test_year_and_does_not_load_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emdat.xlsx"
            rows = [emdat_row(), emdat_row(**{"DisNo.": "2023-0001-USA", "Start Year": 2023, "End Year": 2023})]
            frame = pd.DataFrame(rows)
            frame["Total Affected"] = [1000, 2000]
            frame.to_excel(path, sheet_name="EM-DAT Data", index=False)
            loaded = load_emdat_tc(path)
            self.assertEqual(len(loaded), 1)
            self.assertNotIn("Total Affected", loaded.columns)
            self.assertEqual(set(EMDAT_MATCH_COLUMNS).difference(frame.columns), set())

    def test_emdat_loader_releases_workbook_handle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emdat.xlsx"
            pd.DataFrame([emdat_row()]).to_excel(
                path,
                sheet_name="EM-DAT Data",
                index=False,
            )
            loaded = load_emdat_tc(path)
            self.assertEqual(len(loaded), 1)

            # Windows raises PermissionError [WinError 32] here when the
            # openpyxl workbook remains open. Deleting it explicitly makes
            # handle release part of the test instead of an incidental error
            # during TemporaryDirectory cleanup.
            path.unlink()
            self.assertFalse(path.exists())

    def test_emdat_loader_rejects_missing_and_duplicate_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_path = root / "missing.xlsx"
            duplicate_path = root / "duplicate.xlsx"
            pd.DataFrame([emdat_row(**{"DisNo.": ""})]).to_excel(
                missing_path, sheet_name="EM-DAT Data", index=False
            )
            with self.assertRaisesRegex(ValueError, "missing DisNo"):
                load_emdat_tc(missing_path)
            pd.DataFrame([emdat_row(), emdat_row()]).to_excel(
                duplicate_path, sheet_name="EM-DAT Data", index=False
            )
            with self.assertRaisesRegex(ValueError, "duplicate DisNo"):
                load_emdat_tc(duplicate_path)

    def test_pipeline_matches_and_mandatorily_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            emdat_path = root / "emdat.xlsx"
            ibtracs_path = root / "ibtracs.csv"
            pd.DataFrame([
                emdat_row(),
                emdat_row(**{"DisNo.": "2005-0001-BHS", "ISO": "BHS", "Country": "Bahamas"}),
            ]).to_excel(emdat_path, sheet_name="EM-DAT Data", index=False)
            write_ibtracs(ibtracs_path, [
                ["2005235N12345", "2005", "NA", "KATRINA", "2005-08-23 00:00:00"],
                ["2005235N12345", "2005", "NA", "KATRINA", "2005-08-31 00:00:00"],
            ])
            audit, grouped = run_matching_pipeline(emdat_path, ibtracs_path)
            self.assertEqual(int(audit["is_high_confidence"].sum()), 2)
            self.assertEqual(
                set(audit["event_country_id"]),
                {"2005-0001-USA", "2005-0001-BHS"},
            )
            self.assertEqual(set(audit["start_date"]), {"2005-08-25"})
            self.assertEqual(len(grouped), 1)
            self.assertEqual(grouped.iloc[0]["country_record_count"], 2)

    def test_pipeline_keeps_ambiguous_matches_out_of_grouped_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            emdat_path = root / "emdat.xlsx"
            ibtracs_path = root / "ibtracs.csv"
            pd.DataFrame([emdat_row()]).to_excel(emdat_path, sheet_name="EM-DAT Data", index=False)
            write_ibtracs(ibtracs_path, [
                ["2005235N12345", "2005", "NA", "KATRINA", "2005-08-23 00:00:00"],
                ["2005235N12345", "2005", "NA", "KATRINA", "2005-08-31 00:00:00"],
                ["2005236N12346", "2005", "NA", "KATRINA", "2005-08-24 00:00:00"],
                ["2005236N12346", "2005", "NA", "KATRINA", "2005-08-30 00:00:00"],
            ])
            audit, grouped = run_matching_pipeline(emdat_path, ibtracs_path)
            self.assertEqual(audit.iloc[0]["match_status"], "AMBIGUOUS")
            self.assertTrue(grouped.empty)


if __name__ == "__main__":
    unittest.main()
