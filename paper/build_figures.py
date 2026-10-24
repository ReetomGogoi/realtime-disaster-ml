"""Build manuscript figures only from committed, development-period evidence.

This script deliberately does not open the reserved final-test artifact or any
event-level dataset. It parses the committed preprocessing report and validation
JSON files. Run from the repository root with ``python paper/build_figures.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
REPORT = ROOT / "docs" / "emdat_feature_report.txt"
EVAL = ROOT / "docs" / "model_evaluation"
CLASSES = ["high", "low", "medium"]


def load(name: str) -> dict:
    return json.loads((EVAL / name).read_text(encoding="utf-8-sig"))


def report_count(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Missing report evidence for pattern: {pattern}")
    return int(match.group(1).replace(",", ""))


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def dataset_flow(text: str) -> None:
    labels = ["Raw export", "Natural hazards", "Supported types", "2000–2025"]
    values = [
        report_count(r"Raw rows loaded.*: ([\d,]+)", text),
        report_count(r"Natural disaster rows\s*: ([\d,]+)", text),
        report_count(r"After keeping supported types\s*: ([\d,]+)", text),
        report_count(r"After year filter 2000-2025\s*: ([\d,]+)", text),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    colors = ["#19324d", "#22577a", "#2d7da0", "#38a3a5"]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 180, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center")
    ax.set_xlabel("Country-event records")
    ax.set_title("EM-DAT dataset construction recorded by the preprocessing report")
    ax.set_xlim(0, max(values) * 1.13)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "figure_1_dataset_construction.png")


def temporal_split(text: str) -> None:
    labels = ["Training\n2000–2019", "Validation\n2020–2022", "Reserved\n2023–2025"]
    values = [
        report_count(r"train\s+([\d,]+)\s+\(", text),
        report_count(r"validation\s+([\d,]+)\s+\(", text),
        report_count(r"test\s+([\d,]+)\s+\(", text),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    bars = ax.bar(labels, values, color=["#22577a", "#38a3a5", "#8d99ae"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 85, f"{value:,}", ha="center")
    ax.set_ylabel("Country-event records before target-specific exclusion")
    ax.set_title("Chronological partition used by the development protocol")
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "figure_2_temporal_split.png")


def best(experiment: dict, target: str) -> dict:
    candidates = experiment["targets"][target]["experiments"]
    return max(candidates.values(), key=lambda row: row["macro_f1"])


def model_comparison() -> None:
    original = load("validation_metrics.json")
    xgb = load("xgboost_model_v2_validation.json")
    extra = load("extra_trees_model_v2_validation.json")
    logistic = load("logistic_regression_model_v2_validation.json")
    models = ["RF", "XGBoost", "ExtraTrees", "Logistic"]
    targets = ["affected_range", "impact_priority"]
    scores = {
        target: [
            original[target]["random_forest"]["macro_f1"],
            best(xgb, target)["macro_f1"],
            best(extra, target)["macro_f1"],
            best(logistic, target)["macro_f1"],
        ]
        for target in targets
    }
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, scores[targets[0]], width, label="Affected range", color="#2d7da0")
    ax.bar(x + width / 2, scores[targets[1]], width, label="Impact priority", color="#ef8354")
    ax.set_xticks(x, models)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("Validation macro F1")
    ax.set_title("All-hazard development comparison on the 2020–2022 validation period")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "figure_3_all_hazard_model_comparison.png")


def confusion_figures() -> None:
    results = load("validation_metrics.json")
    for number, target in enumerate(["affected_range", "impact_priority"], start=4):
        matrix = np.asarray(results[target]["random_forest"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5.2, 4.5))
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(3):
            for column in range(3):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center",
                        color="white" if matrix[row, column] > matrix.max() / 2 else "#17202a")
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_title(f"Random Forest validation confusion matrix: {target.replace('_', ' ')}")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        save(fig, f"figure_{number}_{target}_confusion_matrix.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = REPORT.read_text(encoding="utf-8-sig")
    dataset_flow(text)
    temporal_split(text)
    model_comparison()
    confusion_figures()
    print(f"Wrote 5 verified figures to {OUT}")


if __name__ == "__main__":
    main()
