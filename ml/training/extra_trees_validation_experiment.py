"""Validation-only Extra Trees Model-v2 experiment for EM-DAT classifiers.

Three preregistered balanced ExtraTreesClassifier pipelines are fitted on
labeled training rows and compared on labeled validation rows using macro F1.
Rows outside the train and validation splits are discarded during CSV reading
and are never retained, counted, evaluated, or reported. Existing models and
validation result files are read-only; no fitted Extra Trees model is saved.

Usage from any directory:

    python ml/training/extra_trees_validation_experiment.py
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "datasets" / "processed" / "emdat_ml_dataset.csv"
RF_METRICS_PATH = PROJECT_ROOT / "docs" / "model_evaluation" / "validation_metrics.json"
XGBOOST_METRICS_PATH = (
    PROJECT_ROOT
    / "docs"
    / "model_evaluation"
    / "xgboost_model_v2_validation.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "model_evaluation"
    / "extra_trees_model_v2_validation.json"
)

CATEGORICAL_FEATURES = (
    "disaster_type",
    "disaster_subtype",
    "iso",
    "region",
    "subregion",
    "season",
)
NUMERIC_FEATURES = ("start_month", "magnitude", "magnitude_available")
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMNS = ("affected_range", "impact_priority")
EXPECTED_CLASSES = {"high", "low", "medium"}
DEVELOPMENT_SPLITS = {"train", "validation"}
COMMON_EXTRA_TREES_PARAMETERS: dict[str, Any] = {
    "criterion": "gini",
    "class_weight": "balanced",
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1,
}

# Fixed before validation scoring. Configuration order is the only tie-breaker
# if two configurations have exactly equal validation macro F1.
CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "sqrt_leaf1",
        "n_estimators": 500,
        "max_features": "sqrt",
        "max_depth": None,
        "min_samples_leaf": 1,
    },
    {
        "name": "sqrt_leaf2",
        "n_estimators": 500,
        "max_features": "sqrt",
        "max_depth": None,
        "min_samples_leaf": 2,
    },
    {
        "name": "wide_leaf2",
        "n_estimators": 500,
        "max_features": 0.75,
        "max_depth": None,
        "min_samples_leaf": 2,
    },
)


def build_preprocessor() -> ColumnTransformer:
    """Reproduce the existing fitted-on-training-only preprocessing design."""
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        [
            ("cat", categorical, list(CATEGORICAL_FEATURES)),
            ("num", numeric, list(NUMERIC_FEATURES)),
        ]
    )


def build_pipeline(configuration: dict[str, Any]) -> Pipeline:
    """Create an unsaved Extra Trees pipeline for one fixed configuration."""
    configuration_parameters = {
        key: value for key, value in configuration.items() if key != "name"
    }
    classifier = ExtraTreesClassifier(
        **COMMON_EXTRA_TREES_PARAMETERS,
        **configuration_parameters,
    )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def load_development_rows() -> pd.DataFrame:
    """Retain only train/validation rows and required model columns."""
    required_columns = ["temporal_split", *FEATURE_COLUMNS, *TARGET_COLUMNS]
    retained_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(DATASET_PATH, usecols=required_columns, chunksize=50_000):
        retained = chunk.loc[chunk["temporal_split"].isin(DEVELOPMENT_SPLITS)]
        if not retained.empty:
            retained_chunks.append(retained.copy())
    if not retained_chunks:
        raise ValueError("No train or validation rows found")
    return pd.concat(retained_chunks, ignore_index=True)


def load_random_forest_references() -> dict[str, dict[str, Any]]:
    """Read the existing Random Forest validation summary."""
    with RF_METRICS_PATH.open(encoding="utf-8") as handle:
        saved = json.load(handle)

    references: dict[str, dict[str, Any]] = {}
    for target in TARGET_COLUMNS:
        try:
            dataset_info = saved[target]["dataset_info"]
            metrics = saved[target]["random_forest"]
            references[target] = {
                "validation_sample_count": int(dataset_info["val_samples"]),
                "class_order": list(dataset_info["classes"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Malformed Random Forest validation reference for {target}"
            ) from error
    return references


def load_xgboost_references() -> dict[str, dict[str, Any]]:
    """Read each target's saved best XGBoost validation result."""
    with XGBOOST_METRICS_PATH.open(encoding="utf-8") as handle:
        saved = json.load(handle)

    references: dict[str, dict[str, Any]] = {}
    for target in TARGET_COLUMNS:
        try:
            target_result = saved["targets"][target]
            best_name = str(target_result["selection"]["best_configuration"])
            best = target_result["experiments"][best_name]
            dataset_info = target_result["dataset_info"]
            references[target] = {
                "configuration": best_name,
                "validation_sample_count": int(best["validation_sample_count"]),
                "class_order": list(best["class_order"]),
                "accuracy": float(best["accuracy"]),
                "macro_f1": float(best["macro_f1"]),
                "weighted_f1": float(best["weighted_f1"]),
            }
            if references[target]["validation_sample_count"] != int(
                dataset_info["validation_samples"]
            ):
                raise ValueError("XGBoost validation counts disagree internally")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Malformed XGBoost validation reference for {target}"
            ) from error
    return references


def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    report = classification_report(
        actual,
        predicted,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    return {
        "validation_sample_count": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual,
                predicted,
                labels=classes,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in classes
        },
        "class_order": classes,
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=classes
        ).tolist(),
        "confusion_matrix_orientation": "rows=actual, columns=predicted",
    }


def validate_reference(
    target: str,
    reference_name: str,
    reference: dict[str, Any],
    validation_count: int,
    classes: list[str],
) -> None:
    if reference["validation_sample_count"] != validation_count:
        raise ValueError(
            f"{target}: validation count does not match {reference_name} reference"
        )
    if set(reference["class_order"]) != set(classes):
        raise ValueError(f"{target}: class set does not match {reference_name} reference")


def run_target_experiments(
    development: pd.DataFrame,
    target: str,
    random_forest_reference: dict[str, Any],
    xgboost_reference: dict[str, Any],
) -> dict[str, Any]:
    train_rows = development.loc[
        development["temporal_split"].eq("train") & development[target].notna()
    ]
    validation_rows = development.loc[
        development["temporal_split"].eq("validation") & development[target].notna()
    ]
    if train_rows.empty or validation_rows.empty:
        raise ValueError(f"{target}: labeled train and validation rows are required")

    train_features = train_rows.loc[:, FEATURE_COLUMNS]
    validation_features = validation_rows.loc[:, FEATURE_COLUMNS]
    train_labels = train_rows[target].astype(str)
    validation_labels = validation_rows[target].astype(str)
    classes = sorted(train_labels.unique().tolist())
    if len(classes) != 3 or set(classes) != EXPECTED_CLASSES:
        raise ValueError(f"{target}: training classes are not exactly high, low, medium")
    unseen_validation_labels = set(validation_labels.unique()) - set(classes)
    if unseen_validation_labels:
        raise ValueError(
            f"{target}: validation contains unseen labels {sorted(unseen_validation_labels)}"
        )

    validate_reference(
        target,
        "Random Forest",
        random_forest_reference,
        len(validation_rows),
        classes,
    )
    validate_reference(
        target,
        "XGBoost",
        xgboost_reference,
        len(validation_rows),
        classes,
    )

    experiments: dict[str, dict[str, Any]] = {}
    for configuration in CONFIGURATIONS:
        name = str(configuration["name"])
        pipeline = build_pipeline(configuration)
        pipeline.fit(train_features, train_labels)
        predictions = np.asarray(pipeline.predict(validation_features), dtype=str)
        if set(predictions) - set(classes):
            raise ValueError(f"{target}/{name}: classifier returned an invalid class")
        experiments[name] = {
            "parameters": {
                **COMMON_EXTRA_TREES_PARAMETERS,
                **{
                    key: value
                    for key, value in configuration.items()
                    if key != "name"
                },
            },
            **calculate_metrics(validation_labels, predictions, classes),
        }

    best_name = max(
        experiments,
        key=lambda name: experiments[name]["macro_f1"],
    )
    best = experiments[best_name]
    rf_difference = best["macro_f1"] - random_forest_reference["macro_f1"]
    xgb_difference = best["macro_f1"] - xgboost_reference["macro_f1"]
    return {
        "dataset_info": {
            "train_samples": int(len(train_rows)),
            "validation_samples": int(len(validation_rows)),
            "classes": classes,
        },
        "experiments": experiments,
        "selection": {
            "criterion": "highest validation macro_f1 only",
            "tie_breaker": "first preregistered configuration",
            "best_configuration": best_name,
            "best_validation_macro_f1": best["macro_f1"],
        },
        "references": {
            "random_forest": random_forest_reference,
            "xgboost_best": xgboost_reference,
        },
        "comparison": {
            "macro_f1_difference_extra_trees_minus_random_forest": float(
                rf_difference
            ),
            "macro_f1_difference_extra_trees_minus_xgboost": float(
                xgb_difference
            ),
            "extra_trees_higher_than_random_forest": bool(rf_difference > 0),
            "extra_trees_higher_than_xgboost": bool(xgb_difference > 0),
        },
    }


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {OUTPUT_PATH}")
    required_inputs = (DATASET_PATH, RF_METRICS_PATH, XGBOOST_METRICS_PATH)
    missing_inputs = [path for path in required_inputs if not path.is_file()]
    if missing_inputs:
        rendered = ", ".join(str(path) for path in missing_inputs)
        raise FileNotFoundError(f"Required validation input missing: {rendered}")

    random_forest_references = load_random_forest_references()
    xgboost_references = load_xgboost_references()
    development = load_development_rows()
    target_results = {
        target: run_target_experiments(
            development,
            target,
            random_forest_references[target],
            xgboost_references[target],
        )
        for target in TARGET_COLUMNS
    }
    output = {
        "experiment": "validation-only Extra Trees Model-v2 comparison",
        "selection_policy": "per target, highest validation macro_f1 only",
        "feature_columns": list(FEATURE_COLUMNS),
        "preregistered_configuration_order": [
            configuration["name"] for configuration in CONFIGURATIONS
        ],
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "targets": target_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"Saved validation-only Extra Trees results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
