"""
EM-DAT Supported Hazards Analysis Script.

Analyses only the six natural-hazard categories that the ML model will support.
2026 records are excluded from all model-development statistics (incomplete year).
The raw Excel file is never modified.

Usage (from any directory):
    python ml/preprocessing/analyze_supported_hazards.py

Output is printed to stdout AND saved to docs/emdat_supported_hazards_report.txt.
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

# Force UTF-8 on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
DATASET_PATH  = PROJECT_ROOT / "datasets" / "raw" / "public_emdat_2026-09-09.xlsx"
REPORT_PATH   = PROJECT_ROOT / "docs" / "emdat_supported_hazards_report.txt"

# ---------------------------------------------------------------------------
# Category mapping: EM-DAT Disaster Type -> canonical ML label
# Wildcard subtypes for Storm handle "Tropical cyclone".
# Landslide covers both Mass movement (wet) and Mass movement (dry).
# ---------------------------------------------------------------------------
CATEGORY_MAP: dict[str, str] = {
    "Flood":                  "Flood",
    "Storm":                  "Storm",
    "Earthquake":             "Earthquake",
    "Mass movement (wet)":    "Landslide",
    "Mass movement (dry)":    "Landslide",
    "Wildfire":               "Wildfire",
    "Volcanic activity":      "Volcanic activity",
}

SUPPORTED_TYPES = set(CATEGORY_MAP.keys())

OUTCOME_COLS = {
    "Total Deaths":                    "deaths",
    "Total Affected":                  "affected",
    "Total Damage, Adjusted ('000 US$)": "damage_adj",
}

PERCENTILE_LABELS = [0, 25, 50, 75, 90, 95, 100]

SEP  = "-" * 72
SEP2 = "=" * 72


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


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total > 0 else "N/A"


def fmt_percentiles(series) -> str:  # type: ignore[type-arg]
    import numpy as np
    vals = series.dropna()
    if len(vals) == 0:
        return "  (no non-missing values)"
    qs = np.percentile(vals, PERCENTILE_LABELS)
    parts = [f"    P{p:>3}: {v:>14,.1f}" for p, v in zip(PERCENTILE_LABELS, qs)]
    parts.append(f"    n  : {len(vals):>14,}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import pandas as pd
    import numpy as np

    out = Tee()

    def p(*args: object, **kwargs: object) -> None:  # type: ignore[type-arg]
        print(*args, **kwargs, file=out)  # type: ignore[call-arg]

    def section(title: str) -> None:
        p(f"\n{SEP}\n  {title}\n{SEP}")

    # ── Load ────────────────────────────────────────────────────────────────
    if not DATASET_PATH.exists():
        print(f"[ERROR] File not found: {DATASET_PATH}")
        sys.exit(1)

    xl = pd.ExcelFile(DATASET_PATH, engine="openpyxl")
    preferred = ["EM-DAT Data", "emdat", "data", "EMDAT", "Sheet1"]
    sheet = next((s for s in preferred if s in xl.sheet_names), xl.sheet_names[0])
    df_raw = xl.parse(sheet)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # ── Header ──────────────────────────────────────────────────────────────
    p(SEP2)
    p("  EM-DAT Supported Hazards Analysis")
    p(f"  File  : {DATASET_PATH}")
    p(f"  Sheet : {sheet!r}")
    p(f"  Total rows in file: {len(df_raw):,}")
    p(SEP2)

    # ── Filter: Natural disasters only ──────────────────────────────────────
    section("1. Base filtering")
    natural = df_raw[df_raw["Disaster Group"].str.strip() == "Natural"].copy()
    p(f"  Natural disaster rows              : {len(natural):,}")
    techno  = df_raw[df_raw["Disaster Group"].str.strip() == "Technological"]
    p(f"  Technological rows excluded        : {len(techno):,}")
    other   = len(df_raw) - len(natural) - len(techno)
    p(f"  Other / unclassified excluded      : {other:,}")

    # ── Filter: Supported types only ────────────────────────────────────────
    supported = natural[natural["Disaster Type"].isin(SUPPORTED_TYPES)].copy()
    supported["ml_category"] = supported["Disaster Type"].map(CATEGORY_MAP)

    unsupported_natural = natural[~natural["Disaster Type"].isin(SUPPORTED_TYPES)]
    p(f"\n  Supported-type rows                : {len(supported):,}")
    p(f"  Unsupported natural-hazard types   : {len(unsupported_natural):,}")
    p("    (breakdown):")
    for dtype, cnt in unsupported_natural["Disaster Type"].value_counts().items():
        p(f"      {str(dtype):<45} {cnt:>5,}")

    # ── Exclude 2026 for model development ──────────────────────────────────
    section("2. Year filter  (2026 excluded — incomplete year)")
    all_years = supported["Start Year"].value_counts().sort_index()
    p("  Records by year (all supported, including 2026):")
    for yr, cnt in all_years.items():
        p(f"    {yr}  {cnt:>5,}")

    df = supported[supported["Start Year"] < 2026].copy()
    count_2026 = len(supported) - len(df)
    p(f"\n  Rows kept (Start Year < 2026)      : {len(df):,}")
    p(f"  Rows excluded (2026)               : {count_2026:,}")

    # ── Per-category row counts ─────────────────────────────────────────────
    section("3. Row counts per ML category")
    cat_counts = df["ml_category"].value_counts().sort_values(ascending=False)
    total_df = len(df)
    for cat, cnt in cat_counts.items():
        p(f"  {str(cat):<22}  {cnt:>5,}   ({pct(cnt, total_df)})")
    p(f"  {'TOTAL':<22}  {total_df:>5,}")

    # ── Disaster Subtype breakdown per ML category ───────────────────────────
    section("4. Disaster Subtype breakdown within each ML category")
    for cat in sorted(df["ml_category"].unique()):
        sub_df = df[df["ml_category"] == cat]
        p(f"\n  [{cat}]")
        for subtype, cnt in sub_df["Disaster Subtype"].value_counts().items():
            p(f"    {str(subtype):<45}  {cnt:>5,}   ({pct(cnt, len(sub_df))})")

    # ── Outcome missingness per category ────────────────────────────────────
    section("5. Outcome missingness per ML category")
    p(f"  {'Category':<22}  {'n':>6}  "
      f"{'Deaths missing':>16}  {'Affected missing':>17}  "
      f"{'Damage Adj missing':>19}")
    p("  " + "-" * 82)
    for cat in sorted(df["ml_category"].unique()):
        sub_df = df[df["ml_category"] == cat]
        n = len(sub_df)
        d_miss  = sub_df["Total Deaths"].isna().sum()
        a_miss  = sub_df["Total Affected"].isna().sum()
        dm_miss = sub_df["Total Damage, Adjusted ('000 US$)"].isna().sum()
        p(f"  {cat:<22}  {n:>6,}  "
          f"{pct(d_miss, n):>16}  "
          f"{pct(a_miss, n):>17}  "
          f"{pct(dm_miss, n):>19}")

    # ── Overall outcome availability ────────────────────────────────────────
    section("6. Overall outcome availability  (Deaths & Total Affected)")
    has_deaths   = df["Total Deaths"].notna()
    has_affected = df["Total Affected"].notna()
    both_avail    = (has_deaths & has_affected).sum()
    only_deaths   = (has_deaths & ~has_affected).sum()
    only_affected = (~has_deaths & has_affected).sum()
    neither       = (~has_deaths & ~has_affected).sum()
    p(f"  Both Deaths AND Total Affected     : {both_avail:>5,}   ({pct(both_avail, total_df)})")
    p(f"  Only Deaths available              : {only_deaths:>5,}   ({pct(only_deaths, total_df)})")
    p(f"  Only Total Affected available      : {only_affected:>5,}   ({pct(only_affected, total_df)})")
    p(f"  Neither available                  : {neither:>5,}   ({pct(neither, total_df)})")
    p(f"\n  NOTE: Missing outcomes are treated as UNKNOWN, never as zero.")

    # ── Outcome distribution percentiles ────────────────────────────────────
    section("7. Outcome distribution  (non-missing values only)")

    for col_name, label in [("Total Deaths", "Total Deaths"),
                             ("Total Affected", "Total Affected")]:
        p(f"\n  {label}  (all supported categories, 2000-2025, non-missing):")
        p(fmt_percentiles(df[col_name]))

    p("\n  Per-category Total Deaths percentiles:")
    for cat in sorted(df["ml_category"].unique()):
        sub = df[df["ml_category"] == cat]["Total Deaths"]
        n_nonmiss = sub.notna().sum()
        p(f"\n    [{cat}]  n_nonmissing={n_nonmiss:,}")
        p(fmt_percentiles(sub))

    p("\n  Per-category Total Affected percentiles:")
    for cat in sorted(df["ml_category"].unique()):
        sub = df[df["ml_category"] == cat]["Total Affected"]
        n_nonmiss = sub.notna().sum()
        p(f"\n    [{cat}]  n_nonmissing={n_nonmiss:,}")
        p(fmt_percentiles(sub))

    # ── Magnitude analysis ──────────────────────────────────────────────────
    section("8. Magnitude & Magnitude Scale per ML category")
    for cat in sorted(df["ml_category"].unique()):
        sub_df = df[df["ml_category"] == cat]
        n      = len(sub_df)
        mag_series = sub_df["Magnitude"]
        scale_series = sub_df["Magnitude Scale"].fillna("(missing)")
        mag_miss = mag_series.isna().sum()
        p(f"\n  [{cat}]  total rows={n:,}  magnitude_missing={mag_miss:,} ({pct(mag_miss, n)})")
        scale_counts = scale_series.value_counts()
        for sc, sc_cnt in scale_counts.items():
            p(f"    Scale: {str(sc):<25}  rows={sc_cnt:>4,} ({pct(sc_cnt, n)})")
            # Percentiles for this scale, non-missing magnitudes only
            mask = (sub_df["Magnitude Scale"].fillna("(missing)") == sc)
            sc_mag = sub_df.loc[mask, "Magnitude"]
            sc_nonmiss = sc_mag.notna().sum()
            if sc_nonmiss > 0 and sc != "(missing)":
                p(fmt_percentiles(sc_mag))

    # ── Temporal splits ─────────────────────────────────────────────────────
    section("9. Temporal splits for potential train/val/test")
    splits = [
        ("2000-2019  (train candidate)", (df["Start Year"] >= 2000) & (df["Start Year"] <= 2019)),
        ("2020-2022  (val candidate)",   (df["Start Year"] >= 2020) & (df["Start Year"] <= 2022)),
        ("2023-2025  (test candidate)",  (df["Start Year"] >= 2023) & (df["Start Year"] <= 2025)),
    ]
    for label, mask in splits:
        cnt = mask.sum()
        p(f"  {label}  :  {cnt:>5,}  ({pct(cnt, total_df)})")

    p(f"\n  Per-category breakdown by split:")
    for cat in sorted(df["ml_category"].unique()):
        sub_df = df[df["ml_category"] == cat]
        n_cat  = len(sub_df)
        p(f"\n    [{cat}]  total={n_cat:,}")
        for label, mask in splits:
            cat_mask = mask & (df["ml_category"] == cat)
            cnt = cat_mask.sum()
            p(f"      {label}  :  {cnt:>5,}  ({pct(cnt, n_cat)})")

    # ── Duplicate DisNo. check ───────────────────────────────────────────────
    section("10. Duplicate DisNo. check  (within supported set, 2000-2025)")
    disno_counts = df["DisNo."].value_counts()
    dupes = disno_counts[disno_counts > 1]
    if len(dupes) == 0:
        p("  No duplicate DisNo. values found.")
    else:
        p(f"  {len(dupes)} DisNo. values appear more than once:")
        for disno, cnt in dupes.items():
            p(f"    {disno}  x{cnt}")

    # ── Summary ─────────────────────────────────────────────────────────────
    section("11. Summary")
    p(f"  File                   : {DATASET_PATH.name}")
    p(f"  Sheet                  : {sheet!r}")
    p(f"  Total rows in file     : {len(df_raw):,}")
    p(f"  Natural, supported     : {len(supported):,}  (including 2026)")
    p(f"  Model-dev rows (excl 2026) : {len(df):,}")
    p(f"  Year range             : {int(df['Start Year'].min())} - {int(df['Start Year'].max())}")
    p(f"  Supported categories   : {', '.join(sorted(df['ml_category'].unique()))}")
    p(f"\n  DO NOT train a model yet.  Next step: feature engineering and labelling.")
    p(SEP2)

    # ── Save ─────────────────────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(out.getvalue(), encoding="utf-8")
    print(f"\n[OK] Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
