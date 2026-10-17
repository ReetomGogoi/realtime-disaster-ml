"""Validation-only XGBoost Model-v2 experiment for the EM-DAT classifiers.

This script fits three preregistered configurations on labeled training rows,
selects by validation macro F1, and writes a separate result file. It does not
load, modify, or save the existing Random Forest artifacts. Rows outside the
train and validation splits are discarded while the processed CSV is read and
are never retained, counted, evaluated, or reported.

Usage from any directory:

    python ml/training/xgboost_validation_experiment.py
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
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "datasets" / "processed" / "emdat_ml_dataset.csv"
RF_METRICS_PATH = PROJECT_ROOT / "docs" / "model_evaluation" / "validation_metrics.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "model_evaluation"
    / "xgboost_model_v2_validation.json"
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
COMMON_XGBOOST_PARAMETERS: dict[str, Any] = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "booster": "gbtree",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

# Fixed before any validation scoring. The list order is the sole tie-breaker
# when two configurations have exactly equal validation macro F1.
CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "shallow_depth2",
        "n_estimators": 200,
        "max_depth": 2,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "reference_depth3",
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "regularized_depth4",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
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


def build_pipeline(configuration: dict[str, Any], class_count: int) -> Pipeline:
    """Create an unsaved XGBoost pipeline for one preregistered configuration."""
    classifier_parameters = {
        key: value for key, value in configuration.items() if key != "name"
    }
    classifier = XGBClassifier(
        **COMMON_XGBOOST_PARAMETERS,
        **classifier_parameters,
        num_class=class_count,
    )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def load_development_rows() -> pd.DataFrame:
    """Retain only train/validation rows and only required model columns."""
    required_columns = ["temporal_split", *FEATURE_COLUMNS, *TARGET_COLUMNS]
    retained_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(DATASET_PATH, usecols=required_columns, chunksize=50_000):
        retained = chunk.loc[chunk["temporal_split"].isin(DEVELOPMENT_SPLITS)]
        if not retained.empty:
            retained_chunks.append(retained.copy())
    if not retained_chunks:
        raise ValueError("No train or validation rows found")
    return pd.concat(retained_chunks, ignore_index=True)


def load_random_forest_reference() -> dict[str, dict[str, Any]]:
    """Read the existing saved validation reference without changing it."""
    with RF_METRICS_PATH.open(encoding="utf-8") as handle:
        saved = json.load(handle)

    reference: dict[str, dict[str, Any]] = {}
    for target in TARGET_COLUMNS:
        try:
            dataset_info = saved[target]["dataset_info"]
            metrics = saved[target]["random_forest"]
            reference[target] = {
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
    return reference


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
    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in classes
    }
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
        "per_class": per_class,
        "class_order": classes,
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=classes
        ).tolist(),
        "confusion_matrix_orientation": "rows=actual, columns=predicted",
    }


def run_target_experiments(
    development: pd.DataFrame,
    target: str,
    random_forest_reference: dict[str, Any],
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
    if random_forest_reference["validation_sample_count"] != len(validation_rows):
        raise ValueError(f"{target}: validation count does not match saved RF reference")
    if set(random_forest_reference["class_order"]) != set(classes):
        raise ValueError(f"{target}: class set does not match saved RF reference")

    class_to_id = {label: index for index, label in enumerate(classes)}
    encoded_train_labels = train_labels.map(class_to_id).to_numpy(dtype=int)
    sample_weights = compute_sample_weight(class_weight="balanced", y=train_labels)

    experiments: dict[str, dict[str, Any]] = {}
    for configuration in CONFIGURATIONS:
        name = str(configuration["name"])
        pipeline = build_pipeline(configuration, class_count=len(classes))
        pipeline.fit(
            train_features,
            encoded_train_labels,
            classifier__sample_weight=sample_weights,
        )
        encoded_predictions = np.asarray(
            pipeline.predict(validation_features), dtype=int
        )
        if np.any(encoded_predictions < 0) or np.any(encoded_predictions >= len(classes)):
            raise ValueError(f"{target}/{name}: classifier returned an invalid class ID")
        predictions = np.asarray(
            [classes[class_id] for class_id in encoded_predictions], dtype=str
        )

        experiments[name] = {
            "parameters": {
                **COMMON_XGBOOST_PARAMETERS,
                **{
                    key: value
                    for key, value in configuration.items()
                    if key != "name"
                },
                "num_class": len(classes),
            },
            **calculate_metrics(validation_labels, predictions, classes),
        }

    best_name = max(
        experiments,
        key=lambda name: experiments[name]["macro_f1"],
    )
    best = experiments[best_name]
    macro_difference = best["macro_f1"] - random_forest_reference["macro_f1"]
    return {
        "dataset_info": {
            "train_samples": int(len(train_rows)),
            "validation_samples": int(len(validation_rows)),
            "classes": classes,
            "sample_weighting": "sklearn balanced weights derived from this target's training labels only",
        },
        "experiments": experiments,
        "selection": {
            "criterion": "highest validation macro_f1 only",
            "tie_breaker": "first preregistered configuration",
            "best_configuration": best_name,
            "best_validation_macro_f1": best["macro_f1"],
        },
        "random_forest_reference": random_forest_reference,
        "comparison": {
            "macro_f1_difference_xgboost_minus_random_forest": float(macro_difference),
            "xgboost_has_higher_macro_f1": bool(macro_difference > 0),
        },
    }


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {OUTPUT_PATH}")
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {DATASET_PATH}")
    if not RF_METRICS_PATH.is_file():
        raise FileNotFoundError(
            f"Random Forest validation metrics not found: {RF_METRICS_PATH}"
        )

    random_forest_references = load_random_forest_reference()
    development = load_development_rows()
    target_results = {
        target: run_target_experiments(
            development,
            target,
            random_forest_references[target],
        )
        for target in TARGET_COLUMNS
    }
    output = {
        "experiment": "validation-only XGBoost Model-v2 comparison",
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
            "xgboost": xgboost.__version__,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "targets": target_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"Saved validation-only XGBoost results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
