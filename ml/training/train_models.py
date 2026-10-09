"""
EM-DAT Disaster Risk ML Model Training Script.

Trains two separate RandomForestClassifier pipelines on EM-DAT preprocessed data:
1. affected_range   (low / medium / high)
2. impact_priority  (low / medium / high)

Features used (strictly 9 input columns):
- Categorical: disaster_type, disaster_subtype, iso, region, subregion, season
- Numeric: start_month, magnitude, magnitude_available

Data splits:
- Training: temporal_split == 'train'
- Validation: temporal_split == 'validation'
- Test split (temporal_split == 'test') is NOT evaluated at this stage.

Preprocessing:
- Categorical: SimpleImputer(strategy="most_frequent") + OneHotEncoder(handle_unknown="ignore")
- Numeric: SimpleImputer(strategy="median")

Classifier:
- RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)

Outputs:
- Provisional pipelines saved to backend/models/
- Validation metrics JSON & confusion matrix PNGs saved to docs/model_evaluation/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Ensure UTF-8 output formatting on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Path & Directory Configurations
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "datasets" / "processed" / "emdat_ml_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "backend" / "models"
EVAL_DIR = PROJECT_ROOT / "docs" / "model_evaluation"

# ---------------------------------------------------------------------------
# Input Features & Target Labels Definitions
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "disaster_type",
    "disaster_subtype",
    "iso",
    "region",
    "subregion",
    "season",
]

NUMERIC_FEATURES = [
    "start_month",
    "magnitude",
    "magnitude_available",
]

FEATURE_COLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET_COLUMNS = ["affected_range", "impact_priority"]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def build_preprocessing_pipeline() -> ColumnTransformer:
    """Builds scikit-learn ColumnTransformer for categorical and numeric features."""
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numeric_transformer, NUMERIC_FEATURES),
        ]
    )

    return preprocessor


def build_rf_pipeline() -> Pipeline:
    """Creates a full Pipeline with preprocessing and RandomForestClassifier."""
    preprocessor = build_preprocessing_pipeline()
    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def save_confusion_matrix_plot(
    cm: np.ndarray,
    classes: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Generates and saves a confusion matrix visualization as a PNG file."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def train_and_evaluate_models() -> dict[str, Any]:
    """
    Main training workflow for affected_range and impact_priority models.
    
    Returns:
        dict containing validation metrics for both models and baseline comparison.
    """
    # Ensure output directories exist automatically
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found at: {DATASET_PATH}")

    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    # Filter splits
    df_train_raw = df[df["temporal_split"] == "train"].copy()
    df_val_raw = df[df["temporal_split"] == "validation"].copy()

    print(f"Total training split rows: {len(df_train_raw):,}")
    print(f"Total validation split rows: {len(df_val_raw):,}\n")

    overall_metrics: dict[str, Any] = {}

    for target in TARGET_COLUMNS:
        print("=" * 70)
        print(f" TRAINING MODEL FOR TARGET: {target} ")
        print("=" * 70)

        # Remove rows where target label is missing for this model
        train_mask = df_train_raw[target].notna()
        val_mask = df_val_raw[target].notna()

        df_train = df_train_raw[train_mask]
        df_val = df_val_raw[val_mask]

        X_train, y_train = df_train[FEATURE_COLS], df_train[target].astype(str)
        X_val, y_val = df_val[FEATURE_COLS], df_val[target].astype(str)

        print(f"Clean training samples ({target}): {len(X_train):,}")
        print(f"Clean validation samples ({target}): {len(X_val):,}")

        # Unique target classes in training data
        classes = sorted(y_train.unique().tolist())

        # -------------------------------------------------------------------
        # 1. Baseline Model: DummyClassifier (most_frequent)
        # -------------------------------------------------------------------
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        y_val_dummy_pred = dummy.predict(X_val)

        dummy_acc = float(accuracy_score(y_val, y_val_dummy_pred))
        dummy_macro_f1 = float(f1_score(y_val, y_val_dummy_pred, average="macro", zero_division=0))
        dummy_weighted_f1 = float(f1_score(y_val, y_val_dummy_pred, average="weighted", zero_division=0))

        print("\n--- Baseline Model (Dummy: Most Frequent) ---")
        print(f"Validation Accuracy    : {dummy_acc:.4f}")
        print(f"Validation Macro F1   : {dummy_macro_f1:.4f}")
        print(f"Validation Weighted F1: {dummy_weighted_f1:.4f}")

        # -------------------------------------------------------------------
        # 2. Main Model: RandomForestClassifier Pipeline
        # -------------------------------------------------------------------
        rf_pipeline = build_rf_pipeline()
        rf_pipeline.fit(X_train, y_train)
        y_val_pred = rf_pipeline.predict(X_val)

        rf_acc = float(accuracy_score(y_val, y_val_pred))
        rf_macro_f1 = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
        rf_weighted_f1 = float(f1_score(y_val, y_val_pred, average="weighted", zero_division=0))

        clf_report_str = classification_report(y_val, y_val_pred, digits=4)
        clf_report_dict = classification_report(y_val, y_val_pred, output_dict=True)
        cm = confusion_matrix(y_val, y_val_pred, labels=classes)

        print("\n--- RandomForestClassifier Model ---")
        print(f"Validation Accuracy    : {rf_acc:.4f}")
        print(f"Validation Macro F1   : {rf_macro_f1:.4f}")
        print(f"Validation Weighted F1: {rf_weighted_f1:.4f}")
        print("\nClassification Report:")
        print(clf_report_str)
        print("Confusion Matrix:")
        print(cm)

        # -------------------------------------------------------------------
        # 3. Save Artifacts
        # -------------------------------------------------------------------
        # Save trained pipeline model
        model_path = MODELS_DIR / f"{target}_model.joblib"
        joblib.dump(rf_pipeline, model_path)
        print(f"\n[OK] Saved pipeline model to: {model_path}")

        # Save confusion matrix plot PNG
        cm_png_path = EVAL_DIR / f"{target}_confusion_matrix.png"
        save_confusion_matrix_plot(
            cm=cm,
            classes=classes,
            title=f"Confusion Matrix - {target} (Validation)",
            output_path=cm_png_path,
        )
        print(f"[OK] Saved confusion matrix plot to: {cm_png_path}")

        # Store metric dict
        overall_metrics[target] = {
            "dataset_info": {
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "classes": classes,
            },
            "baseline_dummy_most_frequent": {
                "accuracy": dummy_acc,
                "macro_f1": dummy_macro_f1,
                "weighted_f1": dummy_weighted_f1,
            },
            "random_forest": {
                "accuracy": rf_acc,
                "macro_f1": rf_macro_f1,
                "weighted_f1": rf_weighted_f1,
                "classification_report": clf_report_dict,
                "confusion_matrix": cm.tolist(),
            },
        }

    # -----------------------------------------------------------------------
    # Save Validation Metrics JSON
    # -----------------------------------------------------------------------
    metrics_json_path = EVAL_DIR / "validation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)
    print(f"\n[OK] Saved evaluation metrics to: {metrics_json_path}\n")

    return overall_metrics


if __name__ == "__main__":
    train_and_evaluate_models()
