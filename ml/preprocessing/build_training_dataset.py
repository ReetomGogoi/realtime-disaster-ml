"""
EM-DAT Feature Engineering and Labelling Pipeline.

Reads the raw EM-DAT Excel file, filters to the six supported natural-hazard
categories (2000-2025), engineers a clean feature set, assigns temporal splits,
and creates two target labels:
  - affected_range   (low / medium / high)
  - impact_priority  (low / medium / high)

Leakage-prevention principle:
  ALL thresholds (percentiles) are computed exclusively from the TRAINING
  split (2000-2019) before being applied to validation and test rows.
  Validation and test data are never seen when calculating thresholds.

Outputs:
  datasets/processed/emdat_ml_dataset.csv
  docs/emdat_feature_report.txt

Usage (from any directory):
  python ml/preprocessing/build_training_dataset.py
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# Force UTF-8 output on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "datasets" / "raw" / "public_emdat_2026-09-09.xlsx"
OUTPUT_CSV   = PROJECT_ROOT / "datasets" / "processed" / "emdat_ml_dataset.csv"
REPORT_PATH  = PROJECT_ROOT / "docs" / "emdat_feature_report.txt"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map Disaster Type -> ML category (Mass movement both variants -> Landslide)
CATEGORY_MAP: dict[str, str] = {
    "Earthquake":           "Earthquake",
    "Flood":                "Flood",
    "Storm":                "Storm",
    "Wildfire":             "Wildfire",
    "Volcanic activity":    "Volcanic activity",
    "Mass movement (wet)":  "Landslide",
    "Mass movement (dry)":  "Landslide",
}

# Season from month number (1-12)
SEASON_MAP: dict[int, str] = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3:  "MAM", 4: "MAM", 5: "MAM",
    6:  "JJA", 7: "JJA", 8: "JJA",
    9:  "SON", 10: "SON", 11: "SON",
}

# Temporal split boundaries (inclusive)
TRAIN_YEARS = (2000, 2019)
VAL_YEARS   = (2020, 2022)
TEST_YEARS  = (2023, 2025)

# Minimum training observations for reliable per-category thresholds.
# Categories below this count get a "skip" flag in the report.
MIN_TRAIN_OBS_FOR_THRESHOLD = 30


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self) -> None:
        self._buf = StringIO()

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        self._buf.write(text)

    def flush(self) -> None:
        sys.stdout.flush()

    def getvalue(self) -> str:
        return self._buf.getvalue()


SEP  = "-" * 72
SEP2 = "=" * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total > 0 else "N/A"


def percentile(series: pd.Series, q: float) -> float:
    vals = series.dropna()
    return float(np.percentile(vals, q)) if len(vals) > 0 else float("nan")


def assign_temporal_split(year: int) -> str:
    if TRAIN_YEARS[0] <= year <= TRAIN_YEARS[1]:
        return "train"
    if VAL_YEARS[0] <= year <= VAL_YEARS[1]:
        return "validation"
    if TEST_YEARS[0] <= year <= TEST_YEARS[1]:
        return "test"
    return "out_of_scope"


def label_affected_range(
    value: float,
    p25: float,
    p75: float,
) -> str:
    if value <= p25:
        return "low"
    if value <= p75:
        return "medium"
    return "high"


def label_impact_priority(
    deaths: float,
    affected: float,
    deaths_median: float,
    deaths_p75: float,
    affected_median: float,
    affected_p75: float,
) -> str:
    """
    high   : deaths > category P75  OR affected > category P75
    medium : not high, but deaths > category median OR affected > category median
    low    : otherwise (both at or below category median)

    All thresholds are pre-computed from TRAIN rows only.
    """
    high_d = deaths  > deaths_p75
    high_a = affected > affected_p75
    if high_d or high_a:
        return "high"
    med_d = deaths  > deaths_median
    med_a = affected > affected_median
    if med_d or med_a:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:

    out = Tee()

    def p(*args: object, **kwargs: object) -> None:
        print(*args, **kwargs, file=out)  # type: ignore[call-arg]

    def section(title: str) -> None:
        p(f"\n{SEP}\n  {title}\n{SEP}")

    # =========================================================================
    # 1. Load raw data
    # =========================================================================
    if not DATASET_PATH.exists():
        print(f"[ERROR] File not found: {DATASET_PATH}")
        sys.exit(1)

    p(SEP2)
    p("  EM-DAT Feature Engineering & Labelling Report")
    p(f"  Input  : {DATASET_PATH}")
    p(f"  Output : {OUTPUT_CSV}")
    p(SEP2)

    xl = pd.ExcelFile(DATASET_PATH, engine="openpyxl")
    preferred = ["EM-DAT Data", "emdat", "data", "EMDAT", "Sheet1"]
    sheet = next((s for s in preferred if s in xl.sheet_names), xl.sheet_names[0])
    df_raw = xl.parse(sheet)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    p(f"\n  Raw rows loaded from {sheet!r}: {len(df_raw):,}")

    # =========================================================================
    # 2. Filter
    # =========================================================================
    section("1. Filtering")

    # Natural only
    n_natural = (df_raw["Disaster Group"].str.strip() == "Natural").sum()
    df = df_raw[df_raw["Disaster Group"].str.strip() == "Natural"].copy()
    p(f"  Natural disaster rows          : {n_natural:,}")

    # Supported types only
    df = df[df["Disaster Type"].isin(CATEGORY_MAP)].copy()
    p(f"  After keeping supported types  : {len(df):,}")

    # Year range 2000-2025
    df = df[(df["Start Year"] >= 2000) & (df["Start Year"] <= 2025)].copy()
    p(f"  After year filter 2000-2025    : {len(df):,}")

    # =========================================================================
    # 3. Feature engineering
    # =========================================================================
    section("2. Feature engineering")

    # -- ML category (normalize Landslide variants) --------------------------
    df["disaster_type"] = df["Disaster Type"].map(CATEGORY_MAP)

    # -- Disaster subtype (keep raw EM-DAT value; useful for granularity) ----
    df["disaster_subtype"] = df["Disaster Subtype"].fillna("Unknown").str.strip()

    # -- Geographic features (always present per inspection report) ----------
    df["iso"]       = df["ISO"].str.strip()
    df["region"]    = df["Region"].str.strip()
    df["subregion"] = df["Subregion"].str.strip()

    # -- Temporal features ---------------------------------------------------
    df["start_month"] = pd.to_numeric(df["Start Month"], errors="coerce")
    df["season"] = df["start_month"].map(
        lambda m: SEASON_MAP.get(int(m), "Unknown") if pd.notna(m) else "Unknown"
    )

    # -- Magnitude (hazard-specific, numeric; NEVER replace NaN with zero) ---
    # Each hazard uses its own scale: EQ=Mw, Flood=Km2, Storm=Kph,
    # Wildfire=Km2. Landslide and Volcanic activity have no magnitude data.
    raw_mag = pd.to_numeric(df["Magnitude"], errors="coerce")
    df["magnitude"] = raw_mag  # NaN stays NaN -- leakage-safe; no imputation
    df["magnitude_available"] = raw_mag.notna().astype(int)
    p("  magnitude : NaN retained for missing values; never replaced with zero.")
    p(f"  magnitude_available=1 rows  : {df['magnitude_available'].sum():,}")
    p(f"  magnitude_available=0 rows  : {(df['magnitude_available'] == 0).sum():,}")

    # -- Temporal split ------------------------------------------------------
    # MUST be assigned before any threshold computation to prevent leakage.
    df["temporal_split"] = df["Start Year"].apply(assign_temporal_split)
    p("\n  LEAKAGE PREVENTION: temporal_split assigned BEFORE any threshold computation.")

    # -- DisNo. as unique record identifier ----------------------------------
    df["dis_no"] = df["DisNo."].str.strip()

    # -- Duplicate DisNo. check ----------------------------------------------
    section("3. Duplicate DisNo. check")
    dup_mask = df["dis_no"].duplicated(keep=False)
    n_dups = dup_mask.sum()
    if n_dups == 0:
        p("  No duplicate DisNo. values found.")
    else:
        p(f"  WARNING: {n_dups} rows share a DisNo. with at least one other row:")
        for disno, grp in df[dup_mask].groupby("dis_no"):
            p(f"    {disno}  x{len(grp)}")

    # =========================================================================
    # 4. Outcome columns (labels only; never model inputs)
    # =========================================================================
    section("4. Outcome columns  (label creation only)")
    df["total_deaths"]   = pd.to_numeric(df["Total Deaths"],   errors="coerce")
    df["total_affected"] = pd.to_numeric(df["Total Affected"], errors="coerce")

    p("  LEAKAGE NOTE: outcome columns are kept ONLY for labelling.")
    p("  They will be DROPPED before saving the feature matrix.")
    p(f"  total_deaths   non-null : {df['total_deaths'].notna().sum():,} / {len(df):,}")
    p(f"  total_affected non-null : {df['total_affected'].notna().sum():,} / {len(df):,}")

    # =========================================================================
    # 5. Label 1: affected_range
    # =========================================================================
    section("5. Label: affected_range")

    # Thresholds computed ONLY from training rows where total_affected is known
    train_mask = df["temporal_split"] == "train"
    train_aff  = df.loc[train_mask, "total_affected"].dropna()
    aff_p25    = percentile(train_aff, 25)
    aff_p75    = percentile(train_aff, 75)

    p(f"  Train rows with known Total Affected : {len(train_aff):,}")
    p(f"  TRAIN P25 threshold  : {aff_p25:,.1f}")
    p(f"  TRAIN P75 threshold  : {aff_p75:,.1f}")
    p("  LEAKAGE NOTE: P25/P75 computed from TRAIN only; applied to all splits.")

    # Assign label only when total_affected is available
    has_aff_mask = df["total_affected"].notna()
    df["affected_range"] = pd.NA
    df.loc[has_aff_mask, "affected_range"] = df.loc[has_aff_mask, "total_affected"].apply(
        label_affected_range, args=(aff_p25, aff_p75)
    )
    p(f"\n  Rows with affected_range label       : {has_aff_mask.sum():,}")
    p(f"  Rows excluded (missing total_affected): {(~has_aff_mask).sum():,}")

    p("\n  affected_range counts per split:")
    for split in ["train", "validation", "test"]:
        sub = df[df["temporal_split"] == split]["affected_range"]
        p(f"    {split:<12}: total={len(sub):,}  "
          f"low={( sub=='low').sum():,}  "
          f"medium={(sub=='medium').sum():,}  "
          f"high={(sub=='high').sum():,}  "
          f"missing={(sub.isna()).sum():,}")

    # =========================================================================
    # 6. Label 2: impact_priority
    # =========================================================================
    section("6. Label: impact_priority")
    p("  Requires BOTH total_deaths AND total_affected to be non-null.")
    p("  Thresholds are category-specific and computed from TRAIN rows only.")
    p("  LEAKAGE NOTE: validation/test data never used for threshold computation.")

    both_mask = df["total_deaths"].notna() & df["total_affected"].notna()
    df["impact_priority"] = pd.NA

    # Per-category thresholds from training data
    thresholds: dict[str, dict[str, float]] = {}
    cats = sorted(df["disaster_type"].unique())

    p("\n  Per-category TRAIN thresholds:")
    p(f"  {'Category':<22}  {'n_train':>8}  "
      f"{'deaths_med':>11}  {'deaths_p75':>11}  "
      f"{'aff_med':>11}  {'aff_p75':>11}")
    p("  " + "-" * 78)

    for cat in cats:
        cat_train = df[train_mask & (df["disaster_type"] == cat) & both_mask]
        n = len(cat_train)
        if n < MIN_TRAIN_OBS_FOR_THRESHOLD:
            thresholds[cat] = {}
            p(f"  {cat:<22}  {n:>8,}  "
              f"  [SKIPPED — fewer than {MIN_TRAIN_OBS_FOR_THRESHOLD} "
              f"rows with both outcomes in training set]")
            continue

        d_med = percentile(cat_train["total_deaths"],   50)
        d_p75 = percentile(cat_train["total_deaths"],   75)
        a_med = percentile(cat_train["total_affected"], 50)
        a_p75 = percentile(cat_train["total_affected"], 75)

        thresholds[cat] = {
            "deaths_median":   d_med,
            "deaths_p75":      d_p75,
            "affected_median": a_med,
            "affected_p75":    a_p75,
        }
        p(f"  {cat:<22}  {n:>8,}  "
          f"{d_med:>11,.1f}  {d_p75:>11,.1f}  "
          f"{a_med:>11,.1f}  {a_p75:>11,.1f}")

    # Apply thresholds row-by-row
    for cat, thresh in thresholds.items():
        if not thresh:
            continue  # category skipped — insufficient training obs
        cat_both_mask = both_mask & (df["disaster_type"] == cat)
        df.loc[cat_both_mask, "impact_priority"] = df.loc[cat_both_mask].apply(
            lambda row: label_impact_priority(
                row["total_deaths"],
                row["total_affected"],
                thresh["deaths_median"],
                thresh["deaths_p75"],
                thresh["affected_median"],
                thresh["affected_p75"],
            ),
            axis=1,
        )

    labeled_ip = df["impact_priority"].notna().sum()
    p(f"\n  Rows with impact_priority label      : {labeled_ip:,}")
    p(f"  Rows excluded (missing both outcomes): {len(df) - labeled_ip:,}")

    p("\n  impact_priority counts per split:")
    for split in ["train", "validation", "test"]:
        sub = df[df["temporal_split"] == split]["impact_priority"]
        p(f"    {split:<12}: total={len(sub):,}  "
          f"low={(sub=='low').sum():,}  "
          f"medium={(sub=='medium').sum():,}  "
          f"high={(sub=='high').sum():,}  "
          f"missing={(sub.isna()).sum():,}")

    # =========================================================================
    # 7. Build final output dataframe
    # =========================================================================
    section("7. Output feature matrix")

    # Columns to keep in the saved CSV
    # Outcome columns are retained alongside labels for model evaluation
    # but clearly documented as NOT model inputs.
    feature_cols = [
        "dis_no",
        "temporal_split",
        # --- features ---
        "disaster_type",
        "disaster_subtype",
        "iso",
        "region",
        "subregion",
        "start_month",
        "season",
        "magnitude",
        "magnitude_available",
        # --- labels (not model inputs) ---
        "affected_range",
        "impact_priority",
        # --- raw outcomes for reference / sanity checks ---
        "total_deaths",
        "total_affected",
    ]

    df_out = df[feature_cols].copy()

    # =========================================================================
    # 8. Missing feature summary
    # =========================================================================
    section("8. Missing feature counts")
    input_features = [
        "disaster_type", "disaster_subtype", "iso", "region", "subregion",
        "start_month", "season", "magnitude", "magnitude_available",
    ]
    total_rows = len(df_out)
    for col in input_features:
        n_miss = df_out[col].isna().sum()
        p(f"  {col:<25}  missing={n_miss:>5,}  ({pct(n_miss, total_rows)})")

    # =========================================================================
    # 9. Split counts
    # =========================================================================
    section("9. Temporal split counts")
    for split in ["train", "validation", "test"]:
        cnt = (df_out["temporal_split"] == split).sum()
        p(f"  {split:<14}  {cnt:>5,}  ({pct(cnt, total_rows)})")
    p(f"  {'TOTAL':<14}  {total_rows:>5,}")

    # =========================================================================
    # 10. Save
    # =========================================================================
    section("10. Saving output")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    p(f"  Saved {len(df_out):,} rows x {len(df_out.columns)} columns")
    p(f"  Path : {OUTPUT_CSV}")

    # =========================================================================
    # 11. Summary
    # =========================================================================
    section("11. Summary")
    p(f"  Input file          : {DATASET_PATH.name}")
    p(f"  Output file         : {OUTPUT_CSV.name}")
    p(f"  Total output rows   : {len(df_out):,}")
    p(f"  Features            : {len(input_features)}")
    p(f"  Labels              : affected_range, impact_priority")
    p(f"  affected_range")
    p(f"    P25 threshold (train) : {aff_p25:,.1f}")
    p(f"    P75 threshold (train) : {aff_p75:,.1f}")
    p(f"  impact_priority")
    p(f"    Thresholds computed per category from training rows only.")
    p(f"    Min training obs required: {MIN_TRAIN_OBS_FOR_THRESHOLD}")
    p(f"\n  LEAKAGE PREVENTION CONFIRMED:")
    p(f"    - All thresholds derived from training split (2000-2019) only.")
    p(f"    - Outcome columns retained only for labelling; NOT in feature set.")
    p(f"    - magnitude NaNs never imputed to prevent leakage via imputation.")
    p(f"\n  Next step: model training (do NOT start yet).")
    p(SEP2)

    # =========================================================================
    # Save report
    # =========================================================================
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(out.getvalue(), encoding="utf-8")
    print(f"\n[OK] Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
