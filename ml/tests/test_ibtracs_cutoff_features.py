"""Artificial-fixture tests for time-safe IBTrACS cyclone features."""

from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

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
    load_matching_audit,
)
from ml.training.ibtracs_cyclone_validation_experiment import (
    artifact_metadata,
    calculate_metrics,
    load_experiment_input,
    load_experiment_rows,
    prepare_target_cohort,
    run_experiment,
    run_target,
    source_code_metadata,
    summarize_target_cohort,
)


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
    agency: str = "hurdat_atl",
) -> TrackObservation:
    return TrackObservation(
        sid="2005235N12345",
        observed_at=utc(observed_at),
        latitude=latitude,
        longitude=longitude,
        wmo_wind=wind,
        wmo_pressure=980.0,
        distance_to_land=150.0,
        wmo_agency=agency,
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
        "start_date": f"{event_country_id[:4]}-08-25",
        "temporal_split": split,
        **base,
        "ibtracs_wmo_agency": "hurdat_atl",
    }
    for column in IBTRACS_NUMERIC_FEATURES:
        row[column] = 1.0
    return row


class TestLinkage(unittest.TestCase):
    def test_matching_audit_rejects_duplicate_or_inconsistent_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matches.csv"
            duplicate = pd.DataFrame(
                [match_row("2005-0001-USA"), match_row("2005-0001-USA")]
            )
            duplicate.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate event_country_id"):
                load_matching_audit(path)

            inconsistent = pd.DataFrame([match_row("2005-0001-USA")])
            inconsistent.loc[0, "dis_no"] = "2005-0002-USA"
            inconsistent.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "event_country_id and dis_no"):
                load_matching_audit(path)

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
    def test_absent_sid_observations_are_ineligible_and_counted(self) -> None:
        features, reasons = build_feature_table(
            pd.DataFrame([event_row().to_dict()]),
            {},
        )
        self.assertTrue(features.empty)
        self.assertEqual(reasons["no_observation_at_or_before_t0"], 1)

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

    def test_missing_latest_wind_and_agency_are_not_backfilled(self) -> None:
        features, reason = extract_event_features(
            event_row(),
            [
                observation(
                    "2005-08-24T12:00:00", wind=90.0, agency="hurdat_atl"
                ),
                observation(
                    "2005-08-24T18:00:00", wind=None, agency=""
                ),
            ],
        )
        self.assertEqual(reason, "eligible")
        self.assertIsNone(features["ibtracs_wmo_wind_knots"])
        self.assertEqual(features["ibtracs_wmo_wind_available"], 0)
        self.assertEqual(features["ibtracs_wmo_agency"], "Unknown")
        self.assertEqual(features["ibtracs_wmo_agency_available"], 0)

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

    def test_ibtracs_loader_normalizes_explicit_offset_to_utc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ibtracs.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "SID", "ISO_TIME", "LAT", "LON", "WMO_WIND",
                        "WMO_PRES", "DIST2LAND", "WMO_AGENCY", "UNUSED",
                    ]
                )
                writer.writerow(["", "UTC", "", "", "kts", "mb", "km", "", ""])
                # The omitted UNUSED trailing field must not discard a row
                # that still contains every required field.
                writer.writerow(
                    [
                        "2005235N12345", "2005-08-25T05:30:00+05:30",
                        "20", "-70", "50", "980", "150", "agency",
                    ]
                )
            observations, _ = load_ibtracs_observations(
                path, {"2005235N12345"}
            )
            self.assertEqual(
                observations["2005235N12345"][0].observed_at,
                utc("2005-08-25T00:00:00"),
            )

    def test_ibtracs_loader_rejects_selected_truncated_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ibtracs.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "SID", "ISO_TIME", "LAT", "LON", "WMO_WIND",
                        "WMO_PRES", "DIST2LAND", "WMO_AGENCY",
                    ]
                )
                writer.writerow(["", "UTC", "", "", "kts", "mb", "km", ""])
                writer.writerow(["2005235N12345", "2005-08-24 18:00:00"])
            with self.assertRaisesRegex(ValueError, "missing required trailing"):
                load_ibtracs_observations(path, {"2005235N12345"})


class TestExperimentInputGuards(unittest.TestCase):
    def test_artifact_metadata_hashes_exact_bytes_without_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            content = b"event_country_id,matched_sid\nexample,sid\n"
            path.write_bytes(content)
            metadata = artifact_metadata(
                path, reported_path="v2_ibtracs_cyclone_features.csv"
            )
            self.assertEqual(
                metadata,
                {
                    "path": "v2_ibtracs_cyclone_features.csv",
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
            )
            self.assertNotIn(directory, str(metadata))

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

    def test_feature_table_rejects_split_year_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            row = complete_feature_row(
                "2019-0001-USA", "2019001N12345", "train"
            )
            row["start_date"] = "2020-01-01"
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "fixed temporal split"):
                load_experiment_rows(path)

    def test_feature_table_requires_canonical_complete_start_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            for invalid in ("", "2019-8-5", "2019-02-30"):
                with self.subTest(invalid=invalid):
                    row = complete_feature_row(
                        "2019-0001-USA", "2019001N12345", "train"
                    )
                    row["start_date"] = invalid
                    pd.DataFrame([row]).to_csv(path, index=False)
                    with self.assertRaisesRegex(ValueError, "canonical"):
                        load_experiment_rows(path)

    def test_feature_table_rejects_blank_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            row = complete_feature_row(
                "2019-0001-USA", "2019001N12345", "train"
            )
            row["event_country_id"] = "   "
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "event_country_id"):
                load_experiment_rows(path)

            row = complete_feature_row(
                "2019-0001-USA", "2019001N12345", "train"
            )
            row["matched_sid"] = "   "
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "matched_sid"):
                load_experiment_rows(path)

    def test_same_physical_storm_can_have_multiple_country_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            rows = [
                complete_feature_row(
                    "2019-0001-USA", "2019001N12345", "train"
                ),
                complete_feature_row(
                    "2019-0001-BHS", "2019001N12345", "train"
                ),
                complete_feature_row(
                    "2020-0001-USA", "2020001N12345", "validation"
                ),
            ]
            pd.DataFrame(rows).to_csv(path, index=False)
            loaded = load_experiment_rows(path)
            self.assertEqual(len(loaded), 3)
            train_sids = loaded.loc[
                loaded["temporal_split"].eq("train"), "matched_sid"
            ]
            self.assertEqual(
                int(train_sids.nunique()),
                1,
            )

    def test_input_loader_returns_hash_of_validated_feature_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            rows = [
                complete_feature_row(
                    "2019-0001-USA", "2019001N12345", "train"
                ),
                complete_feature_row(
                    "2020-0001-USA", "2020001N12345", "validation"
                ),
            ]
            pd.DataFrame(rows).to_csv(path, index=False)
            frame, metadata = load_experiment_input(path)
            self.assertEqual(len(frame), 2)
            self.assertEqual(metadata["path"], "features.csv")
            self.assertEqual(metadata["size_bytes"], path.stat().st_size)
            self.assertEqual(
                metadata["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_input_loader_rejects_file_changed_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            row = complete_feature_row(
                "2019-0001-USA", "2019001N12345", "train"
            )
            pd.DataFrame([row]).to_csv(path, index=False)
            before = {"path": "features.csv", "size_bytes": 1, "sha256": "a"}
            after = {"path": "features.csv", "size_bytes": 2, "sha256": "b"}
            module = "ml.training.ibtracs_cyclone_validation_experiment"
            with patch(
                f"{module}.artifact_metadata", side_effect=[before, after]
            ):
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    load_experiment_input(path)

    def test_source_metadata_hashes_the_two_reproducibility_sources(self) -> None:
        metadata = source_code_metadata()
        self.assertEqual(
            set(metadata),
            {"cyclone_validation_experiment", "ibtracs_cutoff_feature_builder"},
        )
        for source in metadata.values():
            self.assertEqual(len(source["sha256"]), 64)
            self.assertGreater(source["size_bytes"], 0)
            self.assertFalse(Path(source["path"]).is_absolute())


class TestCycloneValidationReliability(unittest.TestCase):
    @staticmethod
    def cohort_frame() -> pd.DataFrame:
        specifications = [
            ("2018-0001-USA", "2018001N12345", "train", "high", "low"),
            ("2018-0002-USA", "2018002N12345", "train", "low", "medium"),
            ("2019-0003-USA", "2019003N12345", "train", "medium", "high"),
            ("2019-0004-USA", "2019004N12345", "train", None, "low"),
            ("2020-0001-USA", "2020001N12345", "validation", "high", "low"),
            ("2021-0002-USA", "2021002N12345", "validation", "low", "medium"),
            ("2022-0003-USA", "2022003N12345", "validation", "medium", "high"),
            ("2022-0004-USA", "2022004N12345", "validation", None, "low"),
        ]
        rows = []
        for event_id, sid, split, affected, priority in specifications:
            row = complete_feature_row(event_id, sid, split)
            row["affected_range"] = affected
            row["impact_priority"] = priority
            rows.append(row)
        rows[1]["ibtracs_wmo_wind_knots"] = float("nan")
        rows[4]["ibtracs_wmo_agency"] = ""
        return pd.DataFrame(rows)

    def test_target_cohort_uses_target_specific_label_availability(self) -> None:
        frame = self.cohort_frame()
        affected_train, affected_validation, _, _, _ = prepare_target_cohort(
            frame, "affected_range"
        )
        priority_train, priority_validation, _, _, _ = prepare_target_cohort(
            frame, "impact_priority"
        )
        self.assertEqual((len(affected_train), len(affected_validation)), (3, 3))
        self.assertEqual((len(priority_train), len(priority_validation)), (4, 4))
        self.assertFalse(
            affected_train["ibtracs_wmo_wind_knots"].notna().all()
        )

    def test_cohort_summary_audits_imbalance_missingness_and_records(self) -> None:
        frame = self.cohort_frame()
        train, validation, _, _, classes = prepare_target_cohort(
            frame, "impact_priority"
        )
        summary = summarize_target_cohort(
            frame, "impact_priority", train, validation, classes
        )
        self.assertEqual(
            summary["class_distribution"]["train"],
            {"high": 1, "low": 2, "medium": 1},
        )
        self.assertEqual(
            summary["excluded_missing_target"],
            {"train": 0, "validation": 0},
        )
        self.assertEqual(
            summary["missing_features"]["train"]["ibtracs_wmo_wind_knots"],
            1,
        )
        self.assertEqual(
            summary["missing_features"]["validation"]["ibtracs_wmo_agency"],
            1,
        )
        reversed_summary = summarize_target_cohort(
            frame,
            "impact_priority",
            train.iloc[::-1],
            validation.iloc[::-1],
            classes,
        )
        self.assertEqual(
            summary["record_fingerprints"],
            reversed_summary["record_fingerprints"],
        )

    def test_metrics_include_all_required_values_and_orientation(self) -> None:
        metrics = calculate_metrics(
            pd.Series(["high", "low", "medium", "medium"]),
            pd.Series(["high", "medium", "medium", "low"]).to_numpy(),
            ["high", "low", "medium"],
        )
        self.assertEqual(metrics["validation_sample_count"], 4)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 0.5)
        self.assertAlmostEqual(metrics["weighted_f1"], 0.5)
        self.assertEqual(
            metrics["confusion_matrix"],
            [[1, 0, 0], [0, 0, 1], [0, 1, 1]],
        )
        self.assertEqual(
            metrics["confusion_matrix_orientation"],
            "rows=actual, columns=predicted",
        )
        self.assertEqual(metrics["per_class"]["medium"]["support"], 2)
        self.assertAlmostEqual(metrics["per_class"]["medium"]["precision"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["medium"]["recall"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["medium"]["f1"], 0.5)

    def test_metrics_reject_mismatched_lengths_and_unknown_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal nonempty"):
            calculate_metrics(
                pd.Series(["high"]),
                pd.Series([], dtype=str).to_numpy(),
                ["high", "low", "medium"],
            )
        with self.assertRaisesRegex(ValueError, "outside class_order"):
            calculate_metrics(
                pd.Series(["high"]),
                pd.Series(["unknown"]).to_numpy(),
                ["high", "low", "medium"],
            )

    def test_all_model_results_record_the_same_target_cohort(self) -> None:
        class StubEstimator:
            def fit(self, features: object, labels: pd.Series) -> "StubEstimator":
                return self

            def predict(self, features: object) -> list[str]:
                return ["high"] * len(features)  # type: ignore[arg-type]

        frame = self.cohort_frame()
        module = "ml.training.ibtracs_cyclone_validation_experiment"
        with patch(f"{module}.DummyClassifier", return_value=StubEstimator()), patch(
            f"{module}.build_rf_pipeline", return_value=StubEstimator()
        ):
            result = run_target(frame, "impact_priority")
        expected_fingerprints = result["cohort"]["record_fingerprints"]
        for name, model in result["models"].items():
            self.assertEqual(model["training_sample_count"], 4)
            self.assertEqual(model["validation_sample_count"], 4)
            self.assertEqual(model["cohort_fingerprints"], expected_fingerprints)
            if name == "dummy_prior":
                self.assertEqual(model["feature_columns"], [])
                self.assertEqual(model["preprocessing"], "none")
            else:
                self.assertEqual(
                    model["feature_columns"],
                    model["categorical_feature_columns"]
                    + model["numeric_feature_columns"],
                )
                self.assertEqual(
                    model["preprocessing"]["fit_scope"],
                    "training rows for this target only",
                )

    def test_experiment_report_records_reproducibility_metadata(self) -> None:
        feature_metadata = {
            "path": "features.csv",
            "size_bytes": 123,
            "sha256": "a" * 64,
        }
        source_metadata = {
            "cyclone_validation_experiment": {
                "path": "ml/training/example.py",
                "size_bytes": 456,
                "sha256": "b" * 64,
            }
        }
        module = "ml.training.ibtracs_cyclone_validation_experiment"
        with patch(f"{module}.run_target", return_value={"verified": True}), patch(
            f"{module}.source_code_metadata", return_value=source_metadata
        ):
            report = run_experiment(self.cohort_frame(), feature_metadata)
        self.assertEqual(report["report_schema_version"], 2)
        self.assertEqual(
            report["input_artifacts"]["feature_table"], feature_metadata
        )
        self.assertEqual(report["source_code_at_execution"], source_metadata)
        self.assertEqual(
            report["development_splits"],
            {"train": "2000-2019", "validation": "2020-2022"},
        )
        self.assertIn("2023-2025", report["test_split"])
        self.assertEqual(set(report["random_seeds"].values()), {42})
        self.assertEqual(
            set(report["targets"]), {"affected_range", "impact_priority"}
        )


if __name__ == "__main__":
    unittest.main()
