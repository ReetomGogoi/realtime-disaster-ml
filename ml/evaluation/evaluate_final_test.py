"""One-time final evaluation of the two selected, fitted EM-DAT pipelines.

This script never trains or refits a model. It validates both trusted local
artifacts before reading dataset rows, evaluates only labeled rows in the
reserved temporal test split, and refuses to overwrite final-test outputs.

Run from any directory only when the project is ready to disclose final test
performance:

    python ml/evaluation/evaluate_final_test.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "datasets" / "processed" / "emdat_ml_dataset.csv"
MODEL_PATHS = {
    "affected_range": PROJECT_ROOT / "backend" / "models" / "affected_range_model.joblib",
    "impact_priority": PROJECT_ROOT / "backend" / "models" / "impact_priority_model.joblib",
}
EVALUATION_DIR = PROJECT_ROOT / "docs" / "model_evaluation"
METRICS_PATH = EVALUATION_DIR / "final_test_metrics.json"
CONFUSION_MATRIX_PATHS = {
    "affected_range": EVALUATION_DIR / "affected_range_final_test_confusion_matrix.png",
    "impact_priority": EVALUATION_DIR / "impact_priority_final_test_confusion_matrix.png",
}
OUTPUT_PATHS = (METRICS_PATH, *CONFUSION_MATRIX_PATHS.values())

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
TARGET_COLUMNS = tuple(MODEL_PATHS)
EXPECTED_CLASSES = {"high", "low", "medium"}
REQUIRED_DATASET_COLUMNS = ("temporal_split", *FEATURE_COLUMNS, *TARGET_COLUMNS)


def _project_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _require_inputs() -> None:
    missing = [path for path in (DATASET_PATH, *MODEL_PATHS.values()) if not path.is_file()]
    if missing:
        rendered = ", ".join(_project_relative(path) for path in missing)
        raise FileNotFoundError(f"Required final-evaluation input missing: {rendered}")


def _refuse_existing_outputs() -> None:
    existing = [path for path in OUTPUT_PATHS if path.exists()]
    if existing:
        rendered = ", ".join(_project_relative(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite final-test output: {rendered}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pipeline(pipeline: Any, target: str) -> Pipeline:
    if not isinstance(pipeline, Pipeline):
        raise ValueError(f"{target}: artifact is not an sklearn Pipeline")
    check_is_fitted(pipeline)

    feature_names = tuple(map(str, pipeline.feature_names_in_))
    if feature_names != FEATURE_COLUMNS:
        raise ValueError(
            f"{target}: feature_names_in_ does not match the exact nine-feature contract"
        )
    if list(pipeline.named_steps) != ["preprocessor", "classifier"]:
        raise ValueError(f"{target}: unexpected pipeline steps")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(preprocessor, ColumnTransformer):
        raise ValueError(f"{target}: preprocessor is not a ColumnTransformer")
    if not isinstance(classifier, RandomForestClassifier):
        raise ValueError(f"{target}: classifier is not a RandomForestClassifier")
    check_is_fitted(preprocessor)
    check_is_fitted(classifier)

    parameters = {
        "n_estimators": 300,
        "class_weight": "balanced",
        "max_depth": None,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1,
    }
    actual_parameters = classifier.get_params(deep=False)
    for name, expected in parameters.items():
        if actual_parameters.get(name) != expected:
            raise ValueError(
                f"{target}: classifier parameter {name!r} is "
                f"{actual_parameters.get(name)!r}, expected {expected!r}"
            )

    classes = tuple(map(str, classifier.classes_))
    if len(classes) != 3 or set(classes) != EXPECTED_CLASSES:
        raise ValueError(f"{target}: classifier classes are not exactly high, low, medium")

    transformer_columns = {
        name: tuple(columns)
        for name, _, columns in preprocessor.transformers_
        if name in {"cat", "num"}
    }
    expected_transformer_columns = {
        "cat": CATEGORICAL_FEATURES,
        "num": NUMERIC_FEATURES,
    }
    if transformer_columns != expected_transformer_columns:
        raise ValueError(f"{target}: fitted transformer feature groups are unexpected")

    categorical = preprocessor.named_transformers_["cat"]
    numeric = preprocessor.named_transformers_["num"]
    if not isinstance(categorical, Pipeline) or not isinstance(numeric, Pipeline):
        raise ValueError(f"{target}: unexpected preprocessing pipeline structure")
    categorical_imputer = categorical.named_steps.get("imputer")
    encoder = categorical.named_steps.get("onehot")
    numeric_imputer = numeric.named_steps.get("imputer")
    if (
        not isinstance(categorical_imputer, SimpleImputer)
        or categorical_imputer.strategy != "most_frequent"
    ):
        raise ValueError(f"{target}: unexpected categorical imputation")
    if not isinstance(encoder, OneHotEncoder) or encoder.handle_unknown != "ignore":
        raise ValueError(f"{target}: unexpected categorical encoding")
    if not isinstance(numeric_imputer, SimpleImputer) or numeric_imputer.strategy != "median":
        raise ValueError(f"{target}: unexpected numeric imputation")

    return pipeline


def _load_and_validate_models() -> dict[str, Pipeline]:
    pipelines: dict[str, Pipeline] = {}
    for target, path in MODEL_PATHS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            artifact = joblib.load(path)
        pipelines[target] = _validate_pipeline(artifact, target)
    return pipelines


def _validate_dataset_header() -> None:
    header = pd.read_csv(DATASET_PATH, nrows=0)
    missing = [column for column in REQUIRED_DATASET_COLUMNS if column not in header.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def _evaluate_target(
    test_rows: pd.DataFrame,
    target: str,
    pipeline: Pipeline,
) -> tuple[dict[str, Any], np.ndarray]:
    labeled_rows = test_rows.loc[test_rows[target].notna(), [*FEATURE_COLUMNS, target]]
    if labeled_rows.empty:
        raise ValueError(f"{target}: no labeled final-test rows")

    features = labeled_rows.loc[:, FEATURE_COLUMNS]
    actual = labeled_rows[target].astype(str)
    class_order = [str(label) for label in pipeline.named_steps["classifier"].classes_]
    unexpected_actual = set(actual.unique()) - set(class_order)
    if unexpected_actual:
        raise ValueError(f"{target}: unexpected final-test labels: {sorted(unexpected_actual)}")

    predicted = pipeline.predict(features)
    predicted_strings = np.asarray(predicted, dtype=str)
    unexpected_predictions = set(predicted_strings) - set(class_order)
    if unexpected_predictions:
        raise ValueError(f"{target}: model produced unexpected labels: {sorted(unexpected_predictions)}")

    report = classification_report(
        actual,
        predicted_strings,
        labels=class_order,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted_strings, labels=class_order)
    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in class_order
    }
    result = {
        "test_sample_count": int(len(labeled_rows)),
        "accuracy": float(accuracy_score(actual, predicted_strings)),
        "macro_f1": float(
            f1_score(
                actual,
                predicted_strings,
                labels=class_order,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual,
                predicted_strings,
                labels=class_order,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "class_order": class_order,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_orientation": "rows=actual, columns=predicted",
    }
    return result, matrix


def _render_confusion_matrix(target: str, result: dict[str, Any], matrix: np.ndarray) -> bytes:
    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=result["class_order"],
    )
    display.plot(cmap="Blues", ax=axis, values_format="d")
    axis.set_title(f"Confusion Matrix - {target} (Final Test)", fontsize=12, fontweight="bold")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=300)
    plt.close(figure)
    return buffer.getvalue()


def _write_outputs_exclusively(payloads: dict[Path, bytes]) -> None:
    """Publish complete in-memory outputs without replacing an existing file."""
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    _refuse_existing_outputs()
    created: list[Path] = []
    try:
        for destination, payload in payloads.items():
            with destination.open("xb") as handle:
                created.append(destination)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    """Run the single final-test evaluation without changing fitted artifacts."""
    _require_inputs()
    _refuse_existing_outputs()

    model_hashes = {
        f"{target}_model": _sha256(path) for target, path in MODEL_PATHS.items()
    }
    pipelines = _load_and_validate_models()
    hashes = {"dataset": _sha256(DATASET_PATH), **model_hashes}
    _validate_dataset_header()

    # Dataset rows are loaded only after both saved pipelines pass preflight.
    dataset = pd.read_csv(DATASET_PATH, usecols=list(REQUIRED_DATASET_COLUMNS))
    test_rows = dataset.loc[dataset["temporal_split"].eq("test")]
    if test_rows.empty:
        raise ValueError("Dataset contains no temporal_split == 'test' rows")

    results: dict[str, dict[str, Any]] = {}
    matrices: dict[str, np.ndarray] = {}
    for target in TARGET_COLUMNS:
        results[target], matrices[target] = _evaluate_target(test_rows, target, pipelines[target])

    metrics = {
        "evaluation": "final untouched temporal test split",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(FEATURE_COLUMNS),
        "artifacts": {
            "dataset": {
                "path": _project_relative(DATASET_PATH),
                "sha256": hashes["dataset"],
            },
            "models": {
                target: {
                    "path": _project_relative(path),
                    "sha256": hashes[f"{target}_model"],
                }
                for target, path in MODEL_PATHS.items()
            },
        },
        "software_versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "targets": results,
    }

    # Both target metrics and both PNGs are completed in memory before any
    # final-test output path is created.
    payloads = {
        METRICS_PATH: (json.dumps(metrics, indent=2) + "\n").encode("utf-8"),
        **{
            CONFUSION_MATRIX_PATHS[target]: _render_confusion_matrix(
                target, results[target], matrices[target]
            )
            for target in TARGET_COLUMNS
        },
    }
    _write_outputs_exclusively(payloads)

    print(f"Saved final test metrics: {_project_relative(METRICS_PATH)}")
    for target, path in CONFUSION_MATRIX_PATHS.items():
        print(f"Saved {target} confusion matrix: {_project_relative(path)}")


if __name__ == "__main__":
    main()
