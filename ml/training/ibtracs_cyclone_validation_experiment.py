"""Compare validation-only cyclone models on one identical matched cohort.

The experiment fits a prior dummy classifier, the existing nine-feature
Random Forest, and the identical Random Forest plus time-safe IBTrACS cutoff
features.  It uses train rows only for fitting and 2020-2022 validation rows
only for scoring.  It never reads, evaluates, or reports the test split and
does not save fitted models.
"""

from __future__ import annotations

import argparse
import hashlib
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
REPORT_SCHEMA_VERSION = 2
SOURCE_FILES = {
    "cyclone_validation_experiment": Path(__file__).resolve(),
    "ibtracs_cutoff_feature_builder": (
        PROJECT_ROOT / "ml" / "preprocessing" / "build_ibtracs_cutoff_features.py"
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_metadata(path: Path, *, reported_path: str) -> dict[str, Any]:
    """Describe exact file bytes without exposing a machine-specific path."""
    return {
        "path": reported_path,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def load_experiment_input(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the feature table and reject a file changed during the read."""
    before = artifact_metadata(path, reported_path=path.name)
    frame = load_experiment_rows(path)
    after = artifact_metadata(path, reported_path=path.name)
    if before != after:
        raise RuntimeError("feature table changed while it was being loaded")
    return frame, before


def source_code_metadata() -> dict[str, dict[str, Any]]:
    return {
        name: artifact_metadata(
            path,
            reported_path=path.relative_to(PROJECT_ROOT).as_posix(),
        )
        for name, path in SOURCE_FILES.items()
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
        "start_date",
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
    event_ids = frame["event_country_id"].fillna("").astype(str).str.strip()
    if event_ids.eq("").any() or event_ids.duplicated().any():
        raise ValueError("event_country_id must be nonmissing and unique")
    sids = frame["matched_sid"].fillna("").astype(str).str.strip()
    if sids.eq("").any():
        raise ValueError("matched_sid must be nonmissing")
    frame["event_country_id"] = event_ids
    frame["matched_sid"] = sids
    canonical_dates = frame["start_date"].fillna("").astype(str).str.strip()
    start_dates = pd.to_datetime(
        canonical_dates, format="%Y-%m-%d", errors="coerce", utc=True
    )
    has_noncanonical_date = (
        ~canonical_dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    ).any()
    if has_noncanonical_date or start_dates.isna().any():
        raise ValueError("start_date must be a complete canonical YYYY-MM-DD date")
    years = start_dates.dt.year
    valid_years = (
        (frame["temporal_split"].eq("train") & years.between(2000, 2019))
        | (
            frame["temporal_split"].eq("validation")
            & years.between(2020, 2022)
        )
    )
    if not valid_years.all():
        raise ValueError("start_date year disagrees with the fixed temporal split")
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
    actual_values = np.asarray(actual, dtype=str)
    predicted_values = np.asarray(predicted, dtype=str)
    if len(actual_values) == 0 or len(actual_values) != len(predicted_values):
        raise ValueError("metrics require equal nonempty actual and predicted arrays")
    expected = set(classes)
    unexpected_actual = set(actual_values) - expected
    unexpected_predicted = set(predicted_values) - expected
    if unexpected_actual or unexpected_predicted:
        raise ValueError(
            "metrics received labels outside class_order: "
            f"actual={sorted(unexpected_actual)}, "
            f"predicted={sorted(unexpected_predicted)}"
        )
    report = classification_report(
        actual_values,
        predicted_values,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    return {
        "validation_sample_count": int(len(actual_values)),
        "accuracy": float(accuracy_score(actual_values, predicted_values)),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual_values, predicted_values)
        ),
        "macro_f1": float(
            f1_score(
                actual_values,
                predicted_values,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual_values,
                predicted_values,
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
            actual_values, predicted_values, labels=classes
        ).tolist(),
        "confusion_matrix_orientation": "rows=actual, columns=predicted",
    }


def prepare_target_cohort(
    frame: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Select one shared target cohort before any model-specific processing."""
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unsupported target: {target}")
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
    return train, validation, train_labels, validation_labels, classes


def _record_fingerprint(rows: pd.DataFrame) -> str:
    identifiers = sorted(rows["event_country_id"].astype(str))
    return hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()


def _missing_counts(rows: pd.DataFrame, columns: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in columns:
        missing = rows[column].isna()
        if pd.api.types.is_object_dtype(rows[column]):
            missing = missing | rows[column].fillna("").astype(str).str.strip().eq("")
        counts[column] = int(missing.sum())
    return counts


def summarize_target_cohort(
    frame: pd.DataFrame,
    target: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    classes: list[str],
) -> dict[str, Any]:
    all_features = [
        *BASE_CATEGORICAL_FEATURES,
        *BASE_NUMERIC_FEATURES,
        *IBTRACS_CATEGORICAL_FEATURES,
        *IBTRACS_NUMERIC_FEATURES,
    ]
    return {
        "train_event_country_rows": int(len(train)),
        "validation_event_country_rows": int(len(validation)),
        "train_physical_sids": int(train["matched_sid"].nunique()),
        "validation_physical_sids": int(validation["matched_sid"].nunique()),
        "excluded_missing_target": {
            "train": int(
                (frame["temporal_split"].eq("train") & frame[target].isna()).sum()
            ),
            "validation": int(
                (
                    frame["temporal_split"].eq("validation")
                    & frame[target].isna()
                ).sum()
            ),
        },
        "class_distribution": {
            "train": {
                label: int(train[target].astype(str).eq(label).sum())
                for label in classes
            },
            "validation": {
                label: int(validation[target].astype(str).eq(label).sum())
                for label in classes
            },
        },
        "missing_features": {
            "train": _missing_counts(train, all_features),
            "validation": _missing_counts(validation, all_features),
        },
        "record_fingerprints": {
            "train_event_country_ids_sha256": _record_fingerprint(train),
            "validation_event_country_ids_sha256": _record_fingerprint(validation),
        },
        "class_order": classes,
    }


def run_target(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    train, validation, train_labels, validation_labels, classes = (
        prepare_target_cohort(frame, target)
    )
    cohort = summarize_target_cohort(frame, target, train, validation, classes)
    cohort_fingerprints = cohort["record_fingerprints"]

    models: dict[str, Any] = {}
    dummy = DummyClassifier(strategy="prior", random_state=42)
    dummy.fit(np.zeros((len(train), 1)), train_labels)
    dummy_predictions = np.asarray(
        dummy.predict(np.zeros((len(validation), 1))), dtype=str
    )
    models["dummy_prior"] = {
        "parameters": {"strategy": "prior", "random_state": 42},
        "feature_columns": [],
        "preprocessing": "none",
        "training_sample_count": int(len(train)),
        "cohort_fingerprints": cohort_fingerprints,
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
            "categorical_feature_columns": list(categorical),
            "numeric_feature_columns": list(numeric),
            "preprocessing": {
                "categorical_imputer": "most_frequent",
                "categorical_encoder": {
                    "type": "OneHotEncoder",
                    "handle_unknown": "ignore",
                    "sparse_output": False,
                },
                "numeric_imputer": "median",
                "fit_scope": "training rows for this target only",
            },
            "training_sample_count": int(len(train)),
            "cohort_fingerprints": cohort_fingerprints,
            **calculate_metrics(validation_labels, predictions, classes),
        }

    if any(
        model["training_sample_count"] != len(train)
        or model["validation_sample_count"] != len(validation)
        or model["cohort_fingerprints"] != cohort_fingerprints
        for model in models.values()
    ):
        raise RuntimeError(f"{target}: model cohorts diverged")

    enriched_delta = (
        models["random_forest_nine_plus_ibtracs"]["macro_f1"]
        - models["random_forest_nine_features"]["macro_f1"]
    )
    return {
        "cohort": cohort,
        "models": models,
        "comparison": {
            "macro_f1_delta_enriched_minus_nine_feature_rf": float(
                enriched_delta
            ),
            "enriched_rf_has_higher_macro_f1": bool(enriched_delta > 0),
        },
    }


def run_experiment(
    frame: pd.DataFrame,
    feature_table_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment": "validation-only time-safe IBTrACS cyclone comparison",
        "cutoff_policy": "00:00 UTC on EM-DAT start_date; ISO_TIME <= T0",
        "development_splits": {"train": "2000-2019", "validation": "2020-2022"},
        "test_split": "2023-2025 not present, accessed, evaluated, or reported",
        "model_selection": "none; fixed preregistered configurations",
        "rf_parameters": RF_PARAMETERS,
        "random_seeds": {
            "dummy_prior": 42,
            "random_forest_nine_features": 42,
            "random_forest_nine_plus_ibtracs": 42,
        },
        "input_artifacts": {"feature_table": feature_table_metadata},
        "source_code_at_execution": source_code_metadata(),
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
    frame, feature_table_metadata = load_experiment_input(args.input)
    results = run_experiment(frame, feature_table_metadata)
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
