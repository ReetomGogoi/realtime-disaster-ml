"""Regression tests for the validation-only cyclone feature-block ablation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml.preprocessing.build_ibtracs_cutoff_features import (
    IBTRACS_NUMERIC_FEATURES,
)
from ml.training.ibtracs_feature_block_ablation import (
    CONFIGURATION_A,
    CONFIGURATION_B,
    CONFIGURATION_C,
    CONFIGURATIONS,
    WMO_INTENSITY_NUMERIC_FEATURES,
    _draw_grouped_indices,
    _group_positions,
    grouped_bootstrap_macro_f1_differences,
    run_target_ablation,
    write_new_result,
)
from ml.training.ibtracs_cyclone_validation_experiment import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    ENRICHED_CATEGORICAL_FEATURES,
    ENRICHED_NUMERIC_FEATURES,
    load_experiment_input,
)


def feature_row(
    event_id: str,
    sid: str,
    split: str,
    affected_range: str | None,
    impact_priority: str | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "event_country_id": event_id,
        "matched_sid": sid,
        "start_date": f"{event_id[:4]}-08-25",
        "temporal_split": split,
        "disaster_type": "Storm",
        "disaster_subtype": "Tropical cyclone",
        "iso": event_id[-3:],
        "region": "Americas",
        "subregion": "Caribbean",
        "season": "JJA",
        "start_month": 8,
        "magnitude": 100.0,
        "magnitude_available": 1,
        "affected_range": affected_range,
        "impact_priority": impact_priority,
        "ibtracs_wmo_agency": "hurdat_atl",
    }
    for column in IBTRACS_NUMERIC_FEATURES:
        row[column] = 1.0
    return row


def complete_cohort() -> pd.DataFrame:
    specifications = [
        ("2017-0001-USA", "2017001N12345", "train", "high", "medium"),
        ("2018-0002-USA", "2018002N12345", "train", "low", "high"),
        ("2019-0003-USA", "2019003N12345", "train", "medium", "low"),
        ("2019-0004-BHS", "2019004N12345", "train", None, "medium"),
        ("2020-0001-USA", "2020001N12345", "validation", "high", "medium"),
        ("2020-0001-BHS", "2020001N12345", "validation", "low", "high"),
        ("2021-0002-USA", "2021002N12345", "validation", "medium", "low"),
        ("2022-0003-USA", "2022003N12345", "validation", None, "medium"),
    ]
    rows = [feature_row(*specification) for specification in specifications]
    rows[0]["ibtracs_wmo_wind_knots"] = np.nan
    rows[4]["ibtracs_wmo_pressure_mb"] = np.nan
    return pd.DataFrame(rows)


class TestFeatureBlockDefinitions(unittest.TestCase):
    def test_exact_three_preregistered_feature_sets(self) -> None:
        self.assertEqual(
            list(CONFIGURATIONS),
            [CONFIGURATION_A, CONFIGURATION_B, CONFIGURATION_C],
        )
        a_categorical, a_numeric = CONFIGURATIONS[CONFIGURATION_A]
        b_categorical, b_numeric = CONFIGURATIONS[CONFIGURATION_B]
        c_categorical, c_numeric = CONFIGURATIONS[CONFIGURATION_C]

        self.assertEqual(a_categorical, BASE_CATEGORICAL_FEATURES)
        self.assertEqual(a_numeric, BASE_NUMERIC_FEATURES)
        self.assertEqual(
            b_categorical,
            (*BASE_CATEGORICAL_FEATURES, "ibtracs_wmo_agency"),
        )
        self.assertEqual(
            b_numeric,
            (*BASE_NUMERIC_FEATURES, *WMO_INTENSITY_NUMERIC_FEATURES),
        )
        self.assertEqual(c_categorical, ENRICHED_CATEGORICAL_FEATURES)
        self.assertEqual(c_numeric, ENRICHED_NUMERIC_FEATURES)
        self.assertNotIn("ibtracs_latitude", b_numeric)
        self.assertNotIn("ibtracs_motion_speed_kmh", b_numeric)

    def test_feature_input_guard_still_rejects_test_period(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            pd.DataFrame(
                [
                    feature_row(
                        "2023-0001-USA",
                        "2023001N12345",
                        "test",
                        "high",
                        "high",
                    )
                ]
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "forbidden split"):
                load_experiment_input(path)

    def test_feature_input_guard_still_rejects_temporal_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            row = feature_row(
                "2020-0001-USA",
                "2020001N12345",
                "train",
                "high",
                "high",
            )
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "fixed temporal split"):
                load_experiment_input(path)


class TestGroupedBootstrap(unittest.TestCase):
    def test_group_draw_keeps_all_country_rows_for_each_sampled_sid(self) -> None:
        sids = np.asarray(["storm-a", "storm-a", "storm-b", "storm-c"])
        indices = _draw_grouped_indices(
            _group_positions(sids), np.random.default_rng(8)
        )
        first_country_count = int(np.count_nonzero(indices == 0))
        second_country_count = int(np.count_nonzero(indices == 1))
        self.assertGreater(first_country_count, 0)
        self.assertEqual(first_country_count, second_country_count)

    def test_bootstrap_is_paired_deterministic_and_reports_all_differences(self) -> None:
        actual = ["high", "low", "medium", "high", "low"]
        sids = ["storm-a", "storm-a", "storm-b", "storm-c", "storm-c"]
        predictions = {
            CONFIGURATION_A: actual,
            CONFIGURATION_B: ["high", "medium", "medium", "low", "low"],
            CONFIGURATION_C: ["low", "low", "high", "high", "medium"],
        }
        first = grouped_bootstrap_macro_f1_differences(
            actual,
            sids,
            predictions,
            ["high", "low", "medium"],
            iterations=100,
            seed=17,
        )
        second = grouped_bootstrap_macro_f1_differences(
            actual,
            sids,
            predictions,
            ["high", "low", "medium"],
            iterations=100,
            seed=17,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["validation_physical_sids"], 3)
        self.assertTrue(first["same_resample_for_all_configurations"])
        self.assertEqual(
            set(first["comparisons"]),
            {"B_minus_A", "C_minus_A", "C_minus_B"},
        )
        for comparison in first["comparisons"].values():
            self.assertLessEqual(
                comparison["confidence_interval_lower"],
                comparison["confidence_interval_upper"],
            )

    def test_bootstrap_rejects_missing_sid_and_incomplete_predictions(self) -> None:
        predictions = {
            CONFIGURATION_A: ["high", "low"],
            CONFIGURATION_B: ["high", "low"],
            CONFIGURATION_C: ["high", "low"],
        }
        with self.assertRaisesRegex(ValueError, "nonempty physical SID"):
            grouped_bootstrap_macro_f1_differences(
                ["high", "low"],
                ["storm-a", ""],
                predictions,
                ["high", "low", "medium"],
                iterations=5,
            )
        with self.assertRaisesRegex(ValueError, "exactly configurations"):
            grouped_bootstrap_macro_f1_differences(
                ["high", "low"],
                ["storm-a", "storm-b"],
                {CONFIGURATION_A: ["high", "low"]},
                ["high", "low", "medium"],
                iterations=5,
            )


class TestAblationCohorts(unittest.TestCase):
    def test_three_models_share_cohort_and_missing_features_do_not_drop_rows(self) -> None:
        class StubEstimator:
            def fit(self, features: pd.DataFrame, labels: pd.Series) -> "StubEstimator":
                self.training_indices = features.index.tolist()
                return self

            def predict(self, features: pd.DataFrame) -> list[str]:
                self.validation_indices = features.index.tolist()
                return ["high"] * len(features)

        stubs: list[StubEstimator] = []

        def build_stub(*args: object) -> StubEstimator:
            stub = StubEstimator()
            stubs.append(stub)
            return stub

        module = "ml.training.ibtracs_feature_block_ablation"
        with patch(f"{module}.build_rf_pipeline", side_effect=build_stub):
            result = run_target_ablation(
                complete_cohort(),
                "impact_priority",
                bootstrap_iterations=20,
                bootstrap_seed=4,
            )

        self.assertEqual(len(stubs), 3)
        self.assertTrue(
            all(stub.training_indices == [0, 1, 2, 3] for stub in stubs)
        )
        self.assertTrue(
            all(stub.validation_indices == [4, 5, 6, 7] for stub in stubs)
        )
        fingerprints = result["cohort"]["record_fingerprints"]
        for model in result["models"].values():
            self.assertEqual(model["training_sample_count"], 4)
            self.assertEqual(model["validation_sample_count"], 4)
            self.assertEqual(model["cohort_fingerprints"], fingerprints)
        self.assertEqual(
            result["cohort"]["missing_features"]["train"][
                "ibtracs_wmo_wind_knots"
            ],
            1,
        )

    def test_result_writer_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_new_result(path, {"first": True})
            with self.assertRaises(FileExistsError):
                write_new_result(path, {"second": True})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"first": True},
            )


if __name__ == "__main__":
    unittest.main()
