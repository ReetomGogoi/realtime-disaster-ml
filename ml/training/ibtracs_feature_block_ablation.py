"""Run one validation-only IBTrACS cyclone feature-block ablation.

The experiment compares exactly three preregistered Random Forest feature
sets on one target-specific cohort: the existing nine EM-DAT features, those
nine plus the WMO intensity/agency block, and the full cutoff-safe IBTrACS
feature set. Training uses 2000-2019 rows and scoring uses 2020-2022 rows.
The generated feature input must contain no 2023-2025 test rows.

This script does not tune configurations, select a winner, save models, or
replace any earlier experiment result.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score

from ml.preprocessing.build_ibtracs_cutoff_features import TARGET_COLUMNS
from ml.training.ibtracs_cyclone_validation_experiment import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    ENRICHED_CATEGORICAL_FEATURES,
    ENRICHED_NUMERIC_FEATURES,
    RF_PARAMETERS,
    artifact_metadata,
    build_rf_pipeline,
    calculate_metrics,
    load_experiment_input,
    prepare_target_cohort,
    summarize_target_cohort,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT / "datasets" / "processed" / "v2_ibtracs_cyclone_features.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "v2_ibtracs_feature_block_ablation.json"
)

CONFIGURATION_A = "A_nine_feature_rf"
CONFIGURATION_B = "B_nine_plus_wmo_intensity_rf"
CONFIGURATION_C = "C_nine_plus_full_ibtracs_rf"

WMO_INTENSITY_CATEGORICAL_FEATURES = ("ibtracs_wmo_agency",)
WMO_INTENSITY_NUMERIC_FEATURES = (
    "ibtracs_wmo_wind_knots",
    "ibtracs_wmo_wind_available",
    "ibtracs_wmo_pressure_mb",
    "ibtracs_wmo_pressure_available",
    "ibtracs_wmo_agency_available",
)

CONFIGURATIONS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    CONFIGURATION_A: (BASE_CATEGORICAL_FEATURES, BASE_NUMERIC_FEATURES),
    CONFIGURATION_B: (
        (*BASE_CATEGORICAL_FEATURES, *WMO_INTENSITY_CATEGORICAL_FEATURES),
        (*BASE_NUMERIC_FEATURES, *WMO_INTENSITY_NUMERIC_FEATURES),
    ),
    CONFIGURATION_C: (
        ENRICHED_CATEGORICAL_FEATURES,
        ENRICHED_NUMERIC_FEATURES,
    ),
}

PAIRWISE_COMPARISONS = {
    "B_minus_A": (CONFIGURATION_B, CONFIGURATION_A),
    "C_minus_A": (CONFIGURATION_C, CONFIGURATION_A),
    "C_minus_B": (CONFIGURATION_C, CONFIGURATION_B),
}

BOOTSTRAP_SEED = 20261023
BOOTSTRAP_ITERATIONS = 2000
CONFIDENCE_LEVEL = 0.95
REPORT_SCHEMA_VERSION = 1


def _as_string_array(values: Sequence[str] | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype=str)


def _validate_bootstrap_inputs(
    actual: np.ndarray,
    sids: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    classes: Sequence[str],
) -> None:
    if len(actual) == 0 or len(actual) != len(sids):
        raise ValueError("bootstrap requires equal nonempty labels and SID arrays")
    if any(not sid.strip() for sid in sids):
        raise ValueError("bootstrap requires a nonempty physical SID for every row")
    expected = set(classes)
    if set(actual) - expected:
        raise ValueError("bootstrap labels fall outside the fixed class order")
    if set(predictions) != set(CONFIGURATIONS):
        raise ValueError(
            "bootstrap predictions must contain exactly configurations A, B, C"
        )
    for name, predicted in predictions.items():
        if len(predicted) != len(actual):
            raise ValueError(f"{name}: prediction length differs from validation cohort")
        if set(predicted) - expected:
            raise ValueError(f"{name}: predictions fall outside the fixed class order")
    if len(np.unique(sids)) < 2:
        raise ValueError("grouped bootstrap requires at least two physical SIDs")


def _group_positions(sids: np.ndarray) -> dict[str, np.ndarray]:
    return {
        sid: np.flatnonzero(sids == sid)
        for sid in sorted(np.unique(sids).tolist())
    }


def _draw_grouped_indices(
    groups: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample SIDs with replacement and retain every country row per draw."""
    sid_values = np.asarray(list(groups), dtype=str)
    sampled_sids = rng.choice(sid_values, size=len(sid_values), replace=True)
    return np.concatenate([groups[str(sid)] for sid in sampled_sids])


def _macro_f1(
    actual: np.ndarray,
    predicted: np.ndarray,
    classes: Sequence[str],
) -> float:
    return float(
        f1_score(
            actual,
            predicted,
            labels=list(classes),
            average="macro",
            zero_division=0,
        )
    )


def grouped_bootstrap_macro_f1_differences(
    actual: Sequence[str] | pd.Series,
    sids: Sequence[str] | pd.Series,
    predictions: Mapping[str, Sequence[str] | np.ndarray],
    classes: Sequence[str],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Return percentile CIs from paired physical-SID cluster resampling.

    Each replicate samples validation SIDs with replacement. Every country row
    belonging to a sampled SID is retained, and the same sampled row indices
    are used for A, B, and C. Macro F1 always uses the fixed training class
    order, including when a bootstrap replicate omits a class.
    """
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap confidence level must be between zero and one")

    actual_array = _as_string_array(actual)
    sid_array = _as_string_array(sids)
    prediction_arrays = {
        name: _as_string_array(values) for name, values in predictions.items()
    }
    _validate_bootstrap_inputs(
        actual_array, sid_array, prediction_arrays, classes
    )
    groups = _group_positions(sid_array)
    observed_scores = {
        name: _macro_f1(actual_array, predicted, classes)
        for name, predicted in prediction_arrays.items()
    }
    replicate_differences = {
        comparison: np.empty(iterations, dtype=float)
        for comparison in PAIRWISE_COMPARISONS
    }
    rng = np.random.default_rng(seed)
    for iteration in range(iterations):
        indices = _draw_grouped_indices(groups, rng)
        sampled_actual = actual_array[indices]
        sampled_scores = {
            name: _macro_f1(sampled_actual, predicted[indices], classes)
            for name, predicted in prediction_arrays.items()
        }
        for comparison, (minuend, subtrahend) in PAIRWISE_COMPARISONS.items():
            replicate_differences[comparison][iteration] = (
                sampled_scores[minuend] - sampled_scores[subtrahend]
            )

    tail_probability = (1.0 - confidence_level) / 2.0
    comparisons = {}
    for comparison, (minuend, subtrahend) in PAIRWISE_COMPARISONS.items():
        lower, upper = np.quantile(
            replicate_differences[comparison],
            [tail_probability, 1.0 - tail_probability],
        )
        comparisons[comparison] = {
            "minuend": minuend,
            "subtrahend": subtrahend,
            "observed_macro_f1_difference": float(
                observed_scores[minuend] - observed_scores[subtrahend]
            ),
            "confidence_interval_lower": float(lower),
            "confidence_interval_upper": float(upper),
        }
    return {
        "method": "paired percentile bootstrap clustered by physical SID",
        "sampling_unit": "matched_sid",
        "country_rows_within_sampled_sid": "retained together",
        "same_resample_for_all_configurations": True,
        "iterations": iterations,
        "seed": seed,
        "confidence_level": confidence_level,
        "validation_physical_sids": len(groups),
        "fixed_class_order": list(classes),
        "comparisons": comparisons,
    }


def _preprocessing_description() -> dict[str, Any]:
    return {
        "categorical_imputer": "most_frequent",
        "categorical_encoder": {
            "type": "OneHotEncoder",
            "handle_unknown": "ignore",
            "sparse_output": False,
        },
        "numeric_imputer": "median",
        "fit_scope": "training rows for this target only",
    }


def run_target_ablation(
    frame: pd.DataFrame,
    target: str,
    *,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    train, validation, train_labels, validation_labels, classes = (
        prepare_target_cohort(frame, target)
    )
    cohort = summarize_target_cohort(frame, target, train, validation, classes)
    fingerprints = cohort["record_fingerprints"]
    models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, (categorical, numeric) in CONFIGURATIONS.items():
        pipeline = build_rf_pipeline(categorical, numeric)
        feature_columns = [*categorical, *numeric]
        pipeline.fit(train.loc[:, feature_columns], train_labels)
        predicted = np.asarray(
            pipeline.predict(validation.loc[:, feature_columns]), dtype=str
        )
        predictions[name] = predicted
        models[name] = {
            "parameters": RF_PARAMETERS,
            "feature_columns": feature_columns,
            "categorical_feature_columns": list(categorical),
            "numeric_feature_columns": list(numeric),
            "preprocessing": _preprocessing_description(),
            "training_sample_count": int(len(train)),
            "cohort_fingerprints": fingerprints,
            **calculate_metrics(validation_labels, predicted, classes),
        }

    if any(
        model["training_sample_count"] != len(train)
        or model["validation_sample_count"] != len(validation)
        or model["cohort_fingerprints"] != fingerprints
        for model in models.values()
    ):
        raise RuntimeError(f"{target}: model cohorts diverged")

    bootstrap = grouped_bootstrap_macro_f1_differences(
        validation_labels,
        validation["matched_sid"],
        predictions,
        classes,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    return {
        "cohort": cohort,
        "models": models,
        "paired_macro_f1_differences": bootstrap,
    }


def run_ablation(
    frame: pd.DataFrame,
    feature_table_metadata: dict[str, Any],
    *,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    source_files = {
        "feature_block_ablation": Path(__file__).resolve(),
        "base_experiment": (
            PROJECT_ROOT
            / "ml"
            / "training"
            / "ibtracs_cyclone_validation_experiment.py"
        ),
        "cutoff_feature_builder": (
            PROJECT_ROOT
            / "ml"
            / "preprocessing"
            / "build_ibtracs_cutoff_features.py"
        ),
    }
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment": "validation-only IBTrACS cyclone feature-block ablation",
        "configurations": {
            name: {
                "categorical_feature_columns": list(categorical),
                "numeric_feature_columns": list(numeric),
            }
            for name, (categorical, numeric) in CONFIGURATIONS.items()
        },
        "cutoff_policy": "00:00 UTC on EM-DAT start_date; ISO_TIME <= T0",
        "development_splits": {
            "train": "2000-2019",
            "validation": "2020-2022",
        },
        "test_split": "2023-2025 not present, accessed, evaluated, or reported",
        "selection_policy": (
            "descriptive comparison only; no automatic feature-set selection"
        ),
        "scientific_limitation": (
            "retrospective EM-DAT magnitude availability at T0 is unresolved"
        ),
        "rf_parameters": RF_PARAMETERS,
        "input_artifacts": {"feature_table": feature_table_metadata},
        "source_code_at_execution": {
            name: artifact_metadata(
                path,
                reported_path=path.relative_to(PROJECT_ROOT).as_posix(),
            )
            for name, path in source_files.items()
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "targets": {
            target: run_target_ablation(
                frame,
                target,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            )
            for target in TARGET_COLUMNS
        },
    }


def write_new_result(path: Path, result: Mapping[str, Any]) -> None:
    """Write a new result atomically with respect to pre-existing paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Local feature table does not exist: {args.input}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite local result: {args.output}")
    frame, feature_table_metadata = load_experiment_input(args.input)
    result = run_ablation(frame, feature_table_metadata)
    write_new_result(args.output, result)
    print(f"Saved local validation result: {args.output}")
    for target, target_result in result["targets"].items():
        print(f"\n{target}")
        for name, metrics in target_result["models"].items():
            print(
                f"  {name}: accuracy={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}, "
                f"weighted_f1={metrics['weighted_f1']:.4f}, "
                f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
            )


if __name__ == "__main__":
    main()
