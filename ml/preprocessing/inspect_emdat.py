"""
EM-DAT dataset inspection script.

Usage (from any directory):
    python ml/preprocessing/inspect_emdat.py

Output is printed to stdout AND saved to docs/emdat_inspection_report.txt.
The original Excel file is never modified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from io import StringIO

# Force stdout to UTF-8 on Windows (avoids cp1252 UnicodeEncodeError for
# any Unicode characters in Excel column names or cell values).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# ── Resolve project root (two levels up from this file) ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "datasets" / "raw" / "public_emdat_2026-09-09.xlsx"
REPORT_PATH  = PROJECT_ROOT / "docs" / "emdat_inspection_report.txt"

# ── Candidate feature / outcome column fragments (case-insensitive) ──────
OUTCOME_FRAGMENTS = [
    "total deaths", "no injured", "no affected", "no homeless",
    "total affected", "reconstruction costs", "insured damage",
    "total damage", "cpi", "entry date", "last update",
]
INPUT_CANDIDATE_FRAGMENTS = [
    "disaster group", "disaster subgroup", "disaster type", "disaster subtype",
    "event name", "country", "iso", "region", "continent",
    "location", "origin", "associated types", "magnitude",
    "start year", "start month", "start day",
    "end year", "end month", "end day",
]


# ── Helpers ───────────────────────────────────────────────────────────────

class Tee:
    """Write to both stdout and a StringIO buffer simultaneously."""
    def __init__(self):
        self._buf = StringIO()

    def write(self, text: str):
        sys.stdout.write(text)
        self._buf.write(text)

    def flush(self):
        sys.stdout.flush()

    def getvalue(self) -> str:
        return self._buf.getvalue()


def sep(char: str = "-", width: int = 72) -> str:
    return char * width


def section(title: str) -> str:
    return f"\n{sep()}\n  {title}\n{sep()}\n"


def col_matches(col: str, fragments: list[str]) -> bool:
    col_lower = col.lower()
    return any(f in col_lower for f in fragments)


# -- Main ------------------------------------------------------------------

def main() -> None:
    import pandas as pd

    out = Tee()

    def p(*args, **kwargs):
        print(*args, **kwargs, file=out)  # type: ignore[call-arg]

    p("=" * 72)
    p("  EM-DAT Dataset Inspection Report")
    p(f"  File : {DATASET_PATH}")
    p("=" * 72)

    # -- Validate file exists ---------------------------------------------
    if not DATASET_PATH.exists():
        p(f"\n[ERROR] File not found: {DATASET_PATH}")
        p("Make sure the dataset is placed at datasets/raw/public_emdat_2026-09-09.xlsx")
        sys.exit(1)

    # -- List all sheet names ---------------------------------------------
    p(section("1. Sheet names"))
    try:
        xl = pd.ExcelFile(DATASET_PATH, engine="openpyxl")
    except Exception as exc:
        p(f"[ERROR] Cannot open Excel file: {exc}")
        sys.exit(1)

    for i, name in enumerate(xl.sheet_names):
        p(f"  [{i}] {name!r}")

    # -- Select the data sheet --------------------------------------------
    # EM-DAT public export typically has a sheet called "EM-DAT Data" or similar.
    # Fall back to the first sheet if exact name not found.
    preferred = ["EM-DAT Data", "emdat", "data", "EMDAT", "Sheet1"]
    data_sheet: str | None = None
    for candidate in preferred:
        if candidate in xl.sheet_names:
            data_sheet = candidate
            break
    if data_sheet is None:
        data_sheet = xl.sheet_names[0]  # best-effort fallback

    p(f"\n  → Loading sheet: {data_sheet!r}")

    try:
        df = xl.parse(data_sheet, dtype=str)  # read everything as str first
    except Exception as exc:
        p(f"[ERROR] Cannot parse sheet {data_sheet!r}: {exc}")
        sys.exit(1)

    # Strip leading/trailing whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # -- Shape ------------------------------------------------------------
    p(section("2. Dataset shape"))
    p(f"  Rows    : {df.shape[0]:,}")
    p(f"  Columns : {df.shape[1]}")

    # -- Column names -----------------------------------------------------
    p(section("3. Exact column names"))
    for i, col in enumerate(df.columns):
        p(f"  {i:>3}. {col}")

    # -- Re-parse with proper types for numeric analysis ------------------
    df_typed = xl.parse(data_sheet)
    df_typed.columns = [str(c).strip() for c in df_typed.columns]

    # -- Data types -------------------------------------------------------
    p(section("4. Column data types (pandas inferred)"))
    for col, dtype in df_typed.dtypes.items():
        p(f"  {col:<50}  {dtype}")

    # -- Year range -------------------------------------------------------
    p(section("5. Year range"))
    year_col_candidates = [c for c in df_typed.columns if "start year" in c.lower()]
    if year_col_candidates:
        year_col = year_col_candidates[0]
        years = pd.to_numeric(df_typed[year_col], errors="coerce").dropna()
        p(f"  Column  : {year_col!r}")
        p(f"  Min year: {int(years.min())}")
        p(f"  Max year: {int(years.max())}")
        p(f"  Total unique years: {years.nunique()}")
    else:
        p("  [WARN] No 'Start Year' column found.")

    # -- Categorical distributions -----------------------------------------
    p(section("6. Categorical value counts"))
    cat_targets = {
        "Disaster Group":   "disaster group",
        "Disaster Type":    "disaster type",
        "Disaster Subtype": "disaster subtype",
    }
    for label, fragment in cat_targets.items():
        matched = [c for c in df_typed.columns if fragment in c.lower()]
        if not matched:
            p(f"\n  {label}: column not found")
            continue
        col = matched[0]
        counts = df_typed[col].fillna("(missing)").value_counts()
        p(f"\n  {label}  [{col!r}]  — {len(counts)} unique values")
        for val, cnt in counts.items():
            pct = cnt / len(df_typed) * 100
            p(f"    {str(val):<45} {cnt:>6,}  ({pct:5.1f}%)")

    # -- Missing-value percentages ----------------------------------------
    p(section("7. Missing-value percentages (all columns)"))
    n = len(df_typed)
    missing = (df_typed.isnull().sum() / n * 100).sort_values(ascending=False)
    for col, pct in missing.items():
        bar = "#" * int(pct / 5)
        p(f"  {str(col):<50}  {pct:5.1f}%  {bar}")

    # -- Feature role classification --------------------------------------
    p(section("8. Column role classification"))

    p("\n  OUTCOME columns (must NOT become model inputs):")
    outcome_cols = [c for c in df_typed.columns if col_matches(c, OUTCOME_FRAGMENTS)]
    if outcome_cols:
        for c in outcome_cols:
            miss_pct = df_typed[c].isnull().sum() / n * 100
            p(f"    [OUTCOME]  {c:<50}  missing={miss_pct:.1f}%")
    else:
        p("    (none matched — check column names manually)")

    p("\n  INPUT CANDIDATE columns (potential model features):")
    input_cols = [c for c in df_typed.columns if col_matches(c, INPUT_CANDIDATE_FRAGMENTS)]
    if input_cols:
        for c in input_cols:
            miss_pct = df_typed[c].isnull().sum() / n * 100
            n_unique = df_typed[c].nunique(dropna=True)
            p(f"    [INPUT?]   {c:<50}  missing={miss_pct:4.1f}%  unique={n_unique}")
    else:
        p("    (none matched — check column names manually)")

    p("\n  UNCLASSIFIED columns (review manually):")
    classified = set(outcome_cols) | set(input_cols)
    unclassified = [c for c in df_typed.columns if c not in classified]
    for c in unclassified:
        p(f"    [REVIEW]   {c}")

    # ── Summary ──────────────────────────────────────────────────────────
    p(section("9. Inspection summary"))
    p(f"  Sheet loaded      : {data_sheet!r}")
    p(f"  Rows              : {df.shape[0]:,}")
    p(f"  Columns           : {df.shape[1]}")
    p(f"  Outcome columns   : {len(outcome_cols)}")
    p(f"  Input candidates  : {len(input_cols)}")
    p(f"  Unclassified      : {len(unclassified)}")
    p(f"\n  Next step: feature engineering and labelling (do NOT train yet).")
    p(sep("═"))

    # ── Save report ──────────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(out.getvalue(), encoding="utf-8")
    print(f"\n[OK] Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
