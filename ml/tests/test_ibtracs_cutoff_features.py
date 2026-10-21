"""Artificial-fixture tests for time-safe IBTrACS cyclone features."""

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml.preprocessing.build_ibtracs_cutoff_features import (
    BASE_FEATURE_COLUMNS,
    IBTRACS_CATEGORICAL_FEATURES,
    IBTRACS_NUMERIC_FEATURES,
    TARGET_COLUMNS,
    TrackObservation,
    build_feature_table,
    extract_event_features,
    join_matches_to_processed,
    load_ibtracs_observations,
)
from ml.training.ibtracs_cyclone_validation_experiment import load_experiment_rows


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def processed_row(dis_no: str, split: str, affected: str, priority: str) -> dict:
    return {
        "dis_no": dis_no,
        "temporal_split": split,
        "disaster_type": "Storm",
        "disaster_subtype": "Tropical cyclone",
        "iso": dis_no[-3:],
        "region": "Americas",
        "subregion": "Caribbean",
        "season": "Summer",
        "start_month": 8,
        "magnitude": 100.0,
        "magnitude_available": 1,
        "affected_range": affected,
        "impact_priority": priority,
    }


def match_row(
    dis_no: str,
    sid: str = "2005235N12345",
    split: str = "train",
    start_date: str = "2005-08-25",
) -> dict:
    return {
        "event_country_id": dis_no,
        "dis_no": dis_no,
        "start_date": start_date,
        "split": split,
        "matched_sid": sid,
        "is_high_confidence": True,
    }


def event_row(**overrides: object) -> pd.Series:
    values: dict[str, object] = {
        "event_country_id": "2005-0001-USA",
        "matched_sid": "2005235N12345",
        "start_date": "2005-08-25",
        "t0_utc": utc("2005-08-25T00:00:00"),
        "temporal_split": "train",
        **processed_row("2005-0001-USA", "train", "high", "medium"),
    }
    values.pop("dis_no")
    values.update(overrides)
    return pd.Series(values)


def observation(
    observed_at: str,
    latitude: float | None = 20.0,
    longitude: float | None = -70.0,
    wind: float | None = 60.0,
) -> TrackObservation:
    return TrackObservation(
        sid="2005235N12345",
        observed_at=utc(observed_at),
        latitude=latitude,
        longitude=longitude,
        wmo_wind=wind,
        wmo_pressure=980.0,
        distance_to_land=150.0,
        wmo_agency="hurdat_atl",
    )


def complete_feature_row(
    event_country_id: str,
    sid: str,
    split: str,
) -> dict:
    base = processed_row(event_country_id, split, "low", "medium")
    base.pop("dis_no")
    base.pop("temporal_split")
    row = {
        "event_country_id": event_country_id,
        "matched_sid": sid,
        "temporal_split": split,
        **base,
        "ibtracs_wmo_agency": "hurdat_atl",
    }
    for column in IBTRACS_NUMERIC_FEATURES:
        row[column] = 1.0
    return row


class TestLinkage(unittest.TestCase):
    def test_multi_country_rows_keep_distinct_labels_and_one_sid(self) -> None:
        matches = pd.DataFrame(
            [
                match_row("2005-0001-USA"),
                match_row("2005-0001-BHS"),
            ]
        )
        processed = pd.DataFrame(
            [
                processed_row("2005-0001-USA", "train", "high", "high"),
                processed_row("2005-0001-BHS", "train", "medium", "low"),
            ]
        )
        joined = join_matches_to_processed(matches, processed)
        self.assertEqual(len(joined), 2)
        self.assertEqual(joined["matched_sid"].nunique(), 1)
        self.assertEqual(set(joined["affected_range"]), {"high", "medium"})

    def test_linkage_rejects_missing_duplicate_and_split_mismatch(self) -> None:
        matches = pd.DataFrame([match_row("2005-0001-USA")])
        missing = pd.DataFrame(
            [processed_row("2005-0002-USA", "train", "low", "low")]
        )
        with self.assertRaisesRegex(ValueError, "absent"):
            join_matches_to_processed(matches, missing)
        duplicate = pd.DataFrame(
            [
                processed_row("2005-0001-USA", "train", "low", "low"),
                processed_row("2005-0001-USA", "train", "high", "high"),
            ]
        )
        with self.assertRaises(Exception):
            join_matches_to_processed(matches, duplicate)
        mismatch = pd.DataFrame(
            [processed_row("2005-0001-USA", "validation", "low", "low")]
        )
        with self.assertRaisesRegex(ValueError, "splits disagree"):
            join_matches_to_processed(matches, mismatch)

    def test_cross_split_physical_sid_is_rejected(self) -> None:
        matches = pd.DataFrame(
            [
                match_row("2019-0001-USA", split="train"),
                match_row("2020-0001-BHS", split="validation"),
            ]
        )
        processed = pd.DataFrame(
            [
                processed_row("2019-0001-USA", "train", "low", "low"),
                processed_row("2020-0001-BHS", "validation", "high", "high"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "spans temporal splits"):
            join_matches_to_processed(matches, processed)

    def test_missing_or_invalid_complete_start_date_is_rejected(self) -> None:
        processed = pd.DataFrame(
            [processed_row("2005-0001-USA", "train", "low", "low")]
        )
        for invalid in ("", "2005-08", "2005-02-30"):
            with self.subTest(invalid=invalid):
                matches = pd.DataFrame(
                    [match_row("2005-0001-USA", start_date=invalid)]
                )
                with self.assertRaisesRegex(ValueError, "start_date"):
                    join_matches_to_processed(matches, processed)


class TestCutoffFeatures(unittest.TestCase):
    def test_future_observation_is_never_used(self) -> None:
        observations = [
            observation("2005-08-24T18:00:00", wind=60.0),
            observation("2005-08-25T00:00:00", wind=70.0),
            observation("2005-08-25T06:00:00", wind=150.0),
        ]
        features, reason = extract_event_features(event_row(), observations)
        self.assertEqual(reason, "eligible")
        self.assertIsNotNone(features)
        self.assertEqual(features["ibtracs_wmo_wind_knots"], 70.0)
        self.assertEqual(features["ibtracs_storm_age_hours"], 6.0)

    def test_zero_one_and_two_observation_behavior(self) -> None:
        no_features, reason = extract_event_features(
            event_row(), [observation("2005-08-25T06:00:00")]
        )
        self.assertIsNone(no_features)
        self.assertEqual(reason, "no_observation_at_or_before_t0")

        one, _ = extract_event_features(
            event_row(), [observation("2005-08-24T18:00:00")]
        )
        self.assertEqual(one["ibtracs_motion_available"], 0)
        self.assertIsNone(one["ibtracs_motion_speed_kmh"])

        two, _ = extract_event_features(
            event_row(),
            [
                observation("2005-08-24T12:00:00", longitude=-71.0),
                observation("2005-08-24T18:00:00", longitude=-70.0),
            ],
        )
        self.assertEqual(two["ibtracs_motion_available"], 1)
        self.assertGreater(two["ibtracs_motion_speed_kmh"], 0)

    def test_duplicate_track_timestamp_blocks_all_country_rows_for_sid(self) -> None:
        events = pd.DataFrame(
            [
                event_row().to_dict(),
                event_row(event_country_id="2005-0001-BHS").to_dict(),
            ]
        )
        features, reasons = build_feature_table(
            events,
            {
                "2005235N12345": [
                    observation("2005-08-24T18:00:00", wind=50.0),
                    observation("2005-08-24T18:00:00", wind=60.0),
                ]
            },
        )
        self.assertTrue(features.empty)
        self.assertEqual(
            reasons["duplicate_track_timestamps_at_or_before_t0"], 2
        )

    def test_duplicate_after_t0_does_not_change_cutoff_features(self) -> None:
        features, reason = extract_event_features(
            event_row(),
            [
                observation("2005-08-24T18:00:00", wind=50.0),
                observation("2005-08-25T06:00:00", wind=100.0),
                observation("2005-08-25T06:00:00", wind=120.0),
            ],
        )
        self.assertEqual(reason, "eligible")
        self.assertEqual(features["ibtracs_wmo_wind_knots"], 50.0)

    def test_ibtracs_loader_preserves_duplicate_times_for_cutoff_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ibtracs.csv"
            headers = [
                "SID",
                "ISO_TIME",
                "LAT",
                "LON",
                "WMO_WIND",
                "WMO_PRES",
                "DIST2LAND",
                "WMO_AGENCY",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                writer.writerow(["", "UTC", "degrees_north", "degrees_east", "kts", "mb", "km", ""])
                for wind in ("50", "60"):
                    writer.writerow(
                        [
                            "2005235N12345",
                            "2005-08-24 18:00:00",
                            "20",
                            "-70",
                            wind,
                            "980",
                            "150",
                            "hurdat_atl",
                        ]
                    )
            observations, metadata = load_ibtracs_observations(
                path, {"2005235N12345"}
            )
            self.assertEqual(len(observations["2005235N12345"]), 2)
            self.assertEqual(metadata["malformed_selected_iso_time_rows"], 0)

    def test_ibtracs_loader_rejects_malformed_selected_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ibtracs.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "SID",
                        "ISO_TIME",
                        "LAT",
                        "LON",
                        "WMO_WIND",
                        "WMO_PRES",
                        "DIST2LAND",
                        "WMO_AGENCY",
                    ]
                )
                writer.writerow(["", "UTC", "", "", "kts", "mb", "km", ""])
                writer.writerow(
                    ["2005235N12345", "not-a-time", "20", "-70", "50", "980", "150", "agency"]
                )
            with self.assertRaisesRegex(ValueError, "cutoff cannot be enforced"):
                load_ibtracs_observations(path, {"2005235N12345"})


class TestExperimentInputGuards(unittest.TestCase):
    def test_feature_table_rejects_test_split_and_cross_split_sid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            test_row = complete_feature_row(
                "2023-0001-USA", "2023001N12345", "test"
            )
            pd.DataFrame([test_row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "forbidden split"):
                load_experiment_rows(path)

            rows = [
                complete_feature_row("2019-0001-USA", "2019001N12345", "train"),
                complete_feature_row(
                    "2020-0001-BHS", "2019001N12345", "validation"
                ),
            ]
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "spans temporal splits"):
                load_experiment_rows(path)


if __name__ == "__main__":
    unittest.main()
