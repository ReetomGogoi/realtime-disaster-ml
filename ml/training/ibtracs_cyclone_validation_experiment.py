"""Compare validation-only cyclone models on one identical matched cohort.

The experiment fits a prior dummy classifier, the existing nine-feature
Random Forest, and the identical Random Forest plus time-safe IBTrACS cutoff
features.  It uses train rows only for fitting and 2020-2022 validation rows
only for scoring.  It never reads, evaluates, or reports the test split and
does not save fitted models.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.preprocessing.build_ibtracs_cutoff_features import (
    BASE_FEATURE_COLUMNS,
    IBTRACS_CATEGORICAL_FEATURES,
    IBTRACS_NUMERIC_FEATURES,
    TARGET_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT / "datasets" / "processed" / "v2_ibtracs_cyclone_features.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "v2_ibtracs_cyclone_validation.json"
)

BASE_CATEGORICAL_FEATURES = BASE_FEATURE_COLUMNS[:6]
BASE_NUMERIC_FEATURES = BASE_FEATURE_COLUMNS[6:]
ENRICHED_CATEGORICAL_FEATURES = (
    *BASE_CATEGORICAL_FEATURES,
    *IBTRACS_CATEGORICAL_FEATURES,
)
ENRICHED_NUMERIC_FEATURES = (*BASE_NUMERIC_FEATURES, *IBTRACS_NUMERIC_FEATURES)
EXPECTED_CLASSES = frozenset({"high", "low", "medium"})
RF_PARAMETERS: dict[str, Any] = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "max_depth": None,
    "min_samples_leaf": 1,
}


def build_preprocessor(
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
) -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        [
            ("cat", categorical, list(categorical_features)),
            ("num", numeric, list(numeric_features)),
        ]
    )


def build_rf_pipeline(
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(categorical_features, numeric_features),
            ),
            ("classifier", RandomForestClassifier(**RF_PARAMETERS)),
        ]
    )


def load_experiment_rows(path: Path) -> pd.DataFrame:
    required = [
        "event_country_id",
        "matched_sid",
        "temporal_split",
        *BASE_FEATURE_COLUMNS,
        *TARGET_COLUMNS,
        *IBTRACS_CATEGORICAL_FEATURES,
        *IBTRACS_NUMERIC_FEATURES,
    ]
    frame = pd.read_csv(path, usecols=required)
    splits = set(frame["temporal_split"].dropna().astype(str))
    if not splits or not splits <= {"train", "validation"}:
        raise ValueError(f"feature table contains forbidden split(s): {sorted(splits)}")
    if frame["event_country_id"].isna().any() or frame[
        "event_country_id"
    ].duplicated().any():
        raise ValueError("event_country_id must be nonmissing and unique")
    if frame["matched_sid"].isna().any():
        raise ValueError("matched_sid must be nonmissing")
    split_counts = frame.groupby("matched_sid")["temporal_split"].nunique()
    cross_split = sorted(split_counts[split_counts > 1].index.astype(str))
    if cross_split:
        raise ValueError(f"physical storm SID spans temporal splits: {cross_split}")
    return frame


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
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(actual, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                actual, predicted, labels=classes, average="weighted", zero_division=0
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


def run_target(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    train = frame.loc[
        frame["temporal_split"].eq("train") & frame[target].notna()
    ].copy()
    validation = frame.loc[
        frame["temporal_split"].eq("validation") & frame[target].notna()
    ].copy()
    if train.empty or validation.empty:
        raise ValueError(f"{target}: labeled train and validation rows are required")
    train_labels = train[target].astype(str)
    validation_labels = validation[target].astype(str)
    classes = sorted(train_labels.unique())
    if set(classes) != EXPECTED_CLASSES:
        raise ValueError(f"{target}: training classes must be high, low, medium")
    unseen = set(validation_labels.unique()) - set(classes)
    if unseen:
        raise ValueError(f"{target}: validation has unseen labels {sorted(unseen)}")

    models: dict[str, Any] = {}
    dummy = DummyClassifier(strategy="prior", random_state=42)
    dummy.fit(np.zeros((len(train), 1)), train_labels)
    dummy_predictions = np.asarray(
        dummy.predict(np.zeros((len(validation), 1))), dtype=str
    )
    models["dummy_prior"] = {
        "parameters": {"strategy": "prior", "random_state": 42},
        **calculate_metrics(validation_labels, dummy_predictions, classes),
    }

    configurations = {
        "random_forest_nine_features": (
            BASE_CATEGORICAL_FEATURES,
            BASE_NUMERIC_FEATURES,
        ),
        "random_forest_nine_plus_ibtracs": (
            ENRICHED_CATEGORICAL_FEATURES,
            ENRICHED_NUMERIC_FEATURES,
        ),
    }
    for name, (categorical, numeric) in configurations.items():
        pipeline = build_rf_pipeline(categorical, numeric)
        feature_columns = [*categorical, *numeric]
        pipeline.fit(train.loc[:, feature_columns], train_labels)
        predictions = np.asarray(
            pipeline.predict(validation.loc[:, feature_columns]), dtype=str
        )
        models[name] = {
            "parameters": RF_PARAMETERS,
            "feature_columns": feature_columns,
            **calculate_metrics(validation_labels, predictions, classes),
        }

    enriched_delta = (
        models["random_forest_nine_plus_ibtracs"]["macro_f1"]
        - models["random_forest_nine_features"]["macro_f1"]
    )
    return {
        "cohort": {
            "train_event_country_rows": int(len(train)),
            "validation_event_country_rows": int(len(validation)),
            "train_physical_sids": int(train["matched_sid"].nunique()),
            "validation_physical_sids": int(validation["matched_sid"].nunique()),
            "class_order": classes,
        },
        "models": models,
        "comparison": {
            "macro_f1_delta_enriched_minus_nine_feature_rf": float(
                enriched_delta
            ),
            "enriched_rf_has_higher_macro_f1": bool(enriched_delta > 0),
        },
    }


def run_experiment(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "experiment": "validation-only time-safe IBTrACS cyclone comparison",
        "cutoff_policy": "00:00 UTC on EM-DAT start_date; ISO_TIME <= T0",
        "development_splits": {"train": "2000-2019", "validation": "2020-2022"},
        "test_split": "2023-2025 not present, accessed, evaluated, or reported",
        "model_selection": "none; fixed preregistered configurations",
        "rf_parameters": RF_PARAMETERS,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "targets": {target: run_target(frame, target) for target in TARGET_COLUMNS},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Local feature table does not exist: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite local result: {args.output}")
    frame = load_experiment_rows(args.input)
    results = run_experiment(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    print(f"Saved local validation result: {args.output}")
    for target, target_result in results["targets"].items():
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
