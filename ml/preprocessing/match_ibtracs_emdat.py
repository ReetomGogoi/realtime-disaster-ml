"""Conservative EM-DAT to IBTrACS tropical-cyclone matching for Model V2.

This matching-stage research tool does not train models or load human-impact
outcomes. High-confidence matches require a supported unique IBTrACS SID,
name agreement, direct event/track date overlap, known compatible geography,
and one temporal split per physical cyclone. Country/date-only matching is
unsupported because IBTrACS does not identify affected countries.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMDAT_PATH = PROJECT_ROOT / "datasets" / "raw" / "public_emdat_2026-09-09.xlsx"
IBTRACS_PATH = PROJECT_ROOT / "datasets" / "raw" / "ibtracs" / "ibtracs.since1980.list.v04r01.csv"
LOG_CSV = PROJECT_ROOT / "docs" / "v2_match_pilot_log.csv"
PHYSICAL_STORM_OUTPUT = PROJECT_ROOT / "datasets" / "processed" / "v2_physical_storm_matches.csv"

PILOT_YEAR_START = 2000
PILOT_YEAR_END = 2022
TRAIN_END = 2019
GUARDED_TEST_YEARS = {2023, 2024, 2025}
JW_THRESHOLD = 0.92

# IBTrACS SID: YYYY + three-digit day-of-year + hemisphere + five digits.
SUPPORTED_SID_RE = re.compile(r"^\d{7}[NS]\d{5}$")

# Conservative exact EM-DAT subregion names. Unknown geography is rejected.
BASIN_SUBREGION: dict[str, frozenset[str]] = {
    "NA": frozenset({"caribbean", "central america", "northern america"}),
    "EP": frozenset({"central america", "northern america", "polynesia"}),
    "WP": frozenset({"eastern asia", "south-eastern asia", "micronesia", "melanesia", "polynesia"}),
    "NI": frozenset({"southern asia", "south-eastern asia", "western asia"}),
    "SI": frozenset({"eastern africa", "southern africa", "south-eastern asia", "australia and new zealand"}),
    "SP": frozenset({"australia and new zealand", "melanesia", "micronesia", "polynesia"}),
    "SA": frozenset({"south america"}),
}

EMDAT_MATCH_COLUMNS = [
    "DisNo.", "Disaster Group", "Disaster Type", "Disaster Subtype", "Event Name",
    "ISO", "Country", "Subregion", "External IDs", "Start Year", "Start Month",
    "Start Day", "End Year", "End Month", "End Day",
]


def guard_test_period(df: pd.DataFrame, year_col: str = "Start Year") -> None:
    """Reject development data containing years after the validation period."""
    if year_col not in df.columns:
        raise ValueError(f"Missing required year column: {year_col}")
    years = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
    violations = years[(years > PILOT_YEAR_END) | years.isin(GUARDED_TEST_YEARS)]
    if not violations.empty:
        raise ValueError(
            "LEAKAGE GUARD VIOLATION: development data contains years "
            f"{sorted(violations.unique())}; allowed range is {PILOT_YEAR_START}-{PILOT_YEAR_END}."
        )


def normalise_name(raw: str | None) -> str:
    """Normalize a cyclone name without inventing missing identity."""
    if raw is None or pd.isna(raw):
        return ""
    value = str(raw).upper()
    value = re.sub(r"\[.*?]", " ", value)
    value = re.sub(r"\(.*?\)", " ", value)
    prefixes = (
        "SUPER TYPHOON", "SUPER CYCLONE", "TROPICAL CYCLONE", "TROPICAL STORM",
        "CYCLONIC STORM", "HURRICANE", "TYPHOON", "CYCLONE", "STORM", "STS", "TS", "TC",
    )
    for prefix in prefixes:
        value = re.sub(r"\b" + re.escape(prefix) + r"\b", " ", value)
    value = re.sub(r"['\"\-_,.]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalise_geography(raw: str | None) -> str:
    """Conservatively normalize whitespace and case in a geography label."""
    if raw is None or pd.isna(raw):
        return ""
    return re.sub(r"\s+", " ", str(raw).strip().casefold())


def jaro_winkler(s1: str, s2: str) -> float:
    """Calculate Jaro-Winkler similarity between two strings."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    match_dist = max(max(len1, len2) // 2 - 1, 0)
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = transpositions = 0
    for i in range(len1):
        for j in range(max(0, i - match_dist), min(i + match_dist + 1, len2)):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] != s2[i]:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def is_supported_sid(sid: str) -> bool:
    """Return whether a SID has the documented IBTrACS physical-storm format."""
    return bool(SUPPORTED_SID_RE.fullmatch(str(sid).strip()))


def basin_matches_subregion(basin: str, subregion: str) -> bool:
    """Require a known basin and an exact supported EM-DAT subregion."""
    basin_key = str(basin).strip().upper()
    subregion_key = normalise_geography(subregion)
    if not basin_key or not subregion_key or basin_key not in BASIN_SUBREGION:
        return False
    return subregion_key in BASIN_SUBREGION[basin_key]


def safe_date(year: Any, month: Any, day: Any) -> date | None:
    """Build a complete calendar date, returning None for missing/invalid input."""
    try:
        if pd.isna(year) or pd.isna(month) or pd.isna(day):
            return None
        values = [float(year), float(month), float(day)]
        if not all(math.isfinite(value) and value.is_integer() for value in values):
            return None
        return date(*(int(value) for value in values))
    except (ValueError, TypeError, OverflowError):
        return None


def _audit_date_component(value: Any) -> int | str:
    """Preserve a date component in a CSV-safe audit representation."""
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if math.isfinite(numeric) and numeric.is_integer():
        return int(numeric)
    return str(value).strip()


def validate_event_country_identifiers(data: pd.DataFrame) -> pd.Series:
    """Return stripped EM-DAT identifiers after fail-closed uniqueness checks."""
    if "DisNo." not in data.columns:
        raise ValueError("EM-DAT data missing required event-country identifier column: DisNo.")
    identifiers = data["DisNo."].map(
        lambda value: "" if value is None or pd.isna(value) else str(value).strip()
    )
    missing_rows = identifiers.index[identifiers.eq("")].tolist()
    if missing_rows:
        raise ValueError(
            "Eligible EM-DAT country records have missing DisNo. identifiers "
            f"at source row index(es): {missing_rows}"
        )
    duplicated = sorted(identifiers[identifiers.duplicated(keep=False)].unique())
    if duplicated:
        raise ValueError(
            "Eligible EM-DAT country records have duplicate DisNo. identifiers: "
            f"{duplicated}"
        )
    return identifiers


def parse_ibtracs_date(value: str) -> date | None:
    """Parse the date part of an IBTrACS ISO_TIME value."""
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (ValueError, TypeError):
        return None


@dataclass
class IBTrACSStorm:
    """One physical IBTrACS cyclone assembled from track observations."""

    sid: str
    season: str
    names: set[str] = field(default_factory=set)
    basins: set[str] = field(default_factory=set)
    first_obs: date | None = None
    last_obs: date | None = None
    obs_count: int = 0

    def add_observation(self, basin: str, name: str, iso_time: str) -> None:
        basin = str(basin).strip().upper()
        if basin:
            self.basins.add(basin)
        name = str(name).strip()
        normalized = normalise_name(name)
        if normalized and normalized not in {"NOT NAMED", "UNNAMED", "NO NAME"}:
            self.names.add(name)
        observed = parse_ibtracs_date(iso_time)
        if observed is not None:
            if self.first_obs is None or observed < self.first_obs:
                self.first_obs = observed
            if self.last_obs is None or observed > self.last_obs:
                self.last_obs = observed
        self.obs_count += 1

    @property
    def normalized_names(self) -> set[str]:
        return {name for raw in self.names if (name := normalise_name(raw))}

    @property
    def display_name(self) -> str:
        return "; ".join(sorted(self.names))

    @property
    def display_basin(self) -> str:
        return "; ".join(sorted(self.basins))


def storm_overlaps_event(storm: IBTrACSStorm, start_date: date | None, end_date: date | None) -> bool:
    """Require direct overlap between an event interval and observed storm track."""
    if start_date is None or storm.first_obs is None or storm.last_obs is None:
        return False
    effective_end = end_date or start_date
    if effective_end < start_date:
        return False
    return storm.first_obs <= effective_end and storm.last_obs >= start_date


def storm_matches_geography(storm: IBTrACSStorm, subregion: str) -> bool:
    """Require at least one known compatible basin for the EM-DAT subregion."""
    return bool(normalise_geography(subregion)) and any(
        basin_matches_subregion(basin, subregion) for basin in storm.basins
    )


def load_ibtracs_index(path: Path) -> dict[str, IBTrACSStorm]:
    """Stream IBTrACS CSV into a physical-storm index for development years."""
    storms: dict[str, IBTrACSStorm] = {}
    required = {"SID", "SEASON", "BASIN", "NAME", "ISO_TIME"}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            headers = [value.strip() for value in next(reader)]
            next(reader)  # IBTrACS unit row
        except StopIteration as exc:
            raise ValueError("IBTrACS CSV is missing its header or units row") from exc
        columns = {name: index for index, name in enumerate(headers)}
        missing = required.difference(columns)
        if missing:
            raise ValueError(f"IBTrACS CSV missing required columns: {sorted(missing)}")
        for row in reader:
            if len(row) < len(headers):
                continue
            sid = row[columns["SID"]].strip()
            season = row[columns["SEASON"]].strip()
            if not is_supported_sid(sid):
                continue
            try:
                season_year = int(season)
            except ValueError:
                continue
            if not (PILOT_YEAR_START - 1 <= season_year <= PILOT_YEAR_END + 1):
                continue
            storm = storms.setdefault(sid, IBTrACSStorm(sid=sid, season=season))
            if storm.season != season:
                raise ValueError(f"IBTrACS SID {sid} has conflicting seasons")
            storm.add_observation(row[columns["BASIN"]], row[columns["NAME"]], row[columns["ISO_TIME"]])
    return storms


def build_ibtracs_indexes(
    storms: dict[str, IBTrACSStorm],
) -> tuple[dict[tuple[str, str], list[IBTrACSStorm]], dict[str, list[IBTrACSStorm]]]:
    """Build exact-name/season and season indexes from supported storms."""
    name_season: dict[tuple[str, str], list[IBTrACSStorm]] = defaultdict(list)
    season: dict[str, list[IBTrACSStorm]] = defaultdict(list)
    for storm in storms.values():
        if not is_supported_sid(storm.sid):
            continue
        season[storm.season].append(storm)
        for name in storm.normalized_names:
            name_season[(name, storm.season)].append(storm)
    return dict(name_season), dict(season)


def _candidate_seasons(season: str) -> tuple[str, ...]:
    try:
        value = int(season)
    except (TypeError, ValueError):
        return ()
    return str(value), str(value - 1), str(value + 1)


def _unique_supported_candidates(candidates: Iterable[IBTrACSStorm]) -> list[IBTrACSStorm]:
    by_sid = {storm.sid: storm for storm in candidates if is_supported_sid(storm.sid)}
    return [by_sid[sid] for sid in sorted(by_sid)]


def try_t2_exact(
    norm_name: str,
    season: str,
    subregion: str,
    start_date: date | None,
    end_date: date | None,
    name_season_idx: dict[tuple[str, str], list[IBTrACSStorm]],
) -> list[IBTrACSStorm]:
    """Match exact name with verified geography and direct track-date overlap."""
    if not norm_name or not normalise_geography(subregion) or start_date is None:
        return []
    candidates: list[IBTrACSStorm] = []
    for candidate_season in _candidate_seasons(season):
        candidates.extend(name_season_idx.get((norm_name, candidate_season), []))
    return _unique_supported_candidates(
        storm for storm in candidates
        if storm_matches_geography(storm, subregion) and storm_overlaps_event(storm, start_date, end_date)
    )


def try_t3_fuzzy(
    norm_name: str,
    season: str,
    subregion: str,
    start_date: date | None,
    end_date: date | None,
    season_idx: dict[str, list[IBTrACSStorm]],
) -> tuple[list[IBTrACSStorm], float]:
    """Match fuzzy name with verified geography and direct track-date overlap."""
    if len(norm_name) < 3 or not normalise_geography(subregion) or start_date is None:
        return [], 0.0
    scored: list[tuple[float, IBTrACSStorm]] = []
    for candidate_season in _candidate_seasons(season):
        for storm in season_idx.get(candidate_season, []):
            if not is_supported_sid(storm.sid) or not storm_matches_geography(storm, subregion):
                continue
            if not storm_overlaps_event(storm, start_date, end_date):
                continue
            score = max((jaro_winkler(norm_name, candidate) for candidate in storm.normalized_names), default=0.0)
            if score >= JW_THRESHOLD:
                scored.append((score, storm))
    if not scored:
        return [], 0.0
    best_score = max(score for score, _ in scored)
    best = _unique_supported_candidates(
        storm for score, storm in scored if abs(score - best_score) < 1e-12
    )
    return best, best_score


def try_t4_country_date(
    iso: str,
    start_date: date | None,
    end_date: date | None,
    season: str,
    season_idx: dict[str, list[IBTrACSStorm]],
) -> list[IBTrACSStorm]:
    """Return no matches because IBTrACS has no affected-country relation."""
    del iso, start_date, end_date, season, season_idx
    return []


def load_emdat_tc(path: Path) -> pd.DataFrame:
    """Load only non-outcome fields for eligible 2000-2022 tropical cyclones."""
    preferred = ["EM-DAT Data", "emdat", "data", "EMDAT", "Sheet1"]
    # ExcelFile/openpyxl retains a handle to the workbook until explicitly
    # closed. Windows prevents TemporaryDirectory cleanup (and file moves or
    # replacement) while that handle is open, so keep parsing inside the
    # context manager and release it before processing the DataFrame.
    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        sheet = next(
            (candidate for candidate in preferred if candidate in workbook.sheet_names),
            workbook.sheet_names[0],
        )
        source = workbook.parse(
            sheet,
            usecols=lambda name: str(name).strip() in EMDAT_MATCH_COLUMNS,
        )
    source.columns = [str(column).strip() for column in source.columns]
    missing = set(EMDAT_MATCH_COLUMNS).difference(source.columns)
    if missing:
        raise ValueError(f"EM-DAT workbook missing required matching columns: {sorted(missing)}")
    years = pd.to_numeric(source["Start Year"], errors="coerce")
    eligible = (
        source["Disaster Group"].fillna("").astype(str).str.strip().eq("Natural")
        & source["Disaster Type"].fillna("").astype(str).str.strip().eq("Storm")
        & source["Disaster Subtype"].fillna("").astype(str).str.strip().eq("Tropical cyclone")
        & years.between(PILOT_YEAR_START, PILOT_YEAR_END)
    )
    data = source.loc[eligible].copy()
    guard_test_period(data)
    data["_norm_name"] = data["Event Name"].apply(normalise_name)
    for raw, prepared in [
        ("Start Year", "_start_year"), ("Start Month", "_start_month"), ("Start Day", "_start_day"),
        ("End Year", "_end_year"), ("End Month", "_end_month"), ("End Day", "_end_day"),
    ]:
        data[prepared] = pd.to_numeric(data[raw], errors="coerce")
    data["_iso"] = data["ISO"].fillna("").astype(str).str.strip().str.upper()
    data["_subregion"] = data["Subregion"].apply(normalise_geography)
    data["_split"] = data["_start_year"].map(lambda value: "train" if value <= TRAIN_END else "validation")
    data["_event_country_id"] = validate_event_country_identifiers(data)
    return data.reset_index(drop=True)


def match_emdat_record(
    row: pd.Series,
    name_season_idx: dict[tuple[str, str], list[IBTrACSStorm]],
    season_idx: dict[str, list[IBTrACSStorm]],
) -> dict[str, Any]:
    """Match one prepared EM-DAT row using the audited T2/T3 rules."""
    dis_no = str(row.get("DisNo.", "")).strip()
    event_country_id = str(row.get("_event_country_id", dis_no)).strip()
    norm_name = normalise_name(row.get("_norm_name", row.get("Event Name")))
    year_value = row.get("_start_year", row.get("Start Year"))
    season = str(int(year_value)) if pd.notna(year_value) else ""
    subregion = normalise_geography(row.get("_subregion", row.get("Subregion")))
    iso = str(row.get("_iso", row.get("ISO", ""))).strip().upper()
    start_date = safe_date(row.get("_start_year", row.get("Start Year")), row.get("_start_month", row.get("Start Month")), row.get("_start_day", row.get("Start Day")))
    end_date = safe_date(row.get("_end_year", row.get("End Year")), row.get("_end_month", row.get("End Month")), row.get("_end_day", row.get("End Day")))
    result: dict[str, Any] = {
        "event_country_id": event_country_id, "dis_no": dis_no,
        "dis_event_id": "-".join(dis_no.split("-")[:2]),
        "event_name": str(row.get("Event Name", "")).strip(), "norm_name": norm_name,
        "start_year": _audit_date_component(row.get("_start_year", row.get("Start Year"))),
        "start_month": _audit_date_component(row.get("_start_month", row.get("Start Month"))),
        "start_day": _audit_date_component(row.get("_start_day", row.get("Start Day"))),
        "start_date": start_date.isoformat() if start_date is not None else "",
        "country": str(row.get("Country", "")).strip(), "iso": iso,
        "subregion": subregion, "split": str(row.get("_split", "")).strip(),
        "external_ids": str(row.get("External IDs", "")).strip(), "match_status": "NO_MATCH",
        "match_tier": "", "match_score": "", "matched_sid": "", "matched_name": "",
        "matched_basin": "", "matched_season": "", "candidate_count": 0,
        "reject_reason": "", "is_high_confidence": False,
    }
    if not event_country_id:
        result["reject_reason"] = "missing_event_country_id"
        return result
    if event_country_id != dis_no:
        result["reject_reason"] = "inconsistent_event_country_id"
        return result
    if not iso or not subregion:
        result["reject_reason"] = "missing_geography"
        return result
    if start_date is None:
        result["reject_reason"] = "missing_or_invalid_start_date"
        return result
    if not norm_name:
        result["reject_reason"] = "missing_event_name"
        return result

    t2 = try_t2_exact(norm_name, season, subregion, start_date, end_date, name_season_idx)
    if len(t2) > 1:
        result.update(match_status="AMBIGUOUS", match_tier="T2_exact", match_score="1.000", candidate_count=len(t2), reject_reason="multiple_supported_sids")
        return result
    if len(t2) == 1:
        storm = t2[0]
        result.update(match_status="ACCEPTED_HIGH_CONFIDENCE", match_tier="T2_exact", match_score="1.000", matched_sid=storm.sid, matched_name=storm.display_name, matched_basin=storm.display_basin, matched_season=storm.season, candidate_count=1, is_high_confidence=True)
        return result

    t3, score = try_t3_fuzzy(norm_name, season, subregion, start_date, end_date, season_idx)
    if len(t3) > 1:
        result.update(match_status="AMBIGUOUS", match_tier="T3_fuzzy", match_score=f"{score:.3f}", candidate_count=len(t3), reject_reason="multiple_supported_sids")
        return result
    if len(t3) == 1:
        storm = t3[0]
        result.update(match_status="ACCEPTED_HIGH_CONFIDENCE", match_tier="T3_fuzzy", match_score=f"{score:.3f}", matched_sid=storm.sid, matched_name=storm.display_name, matched_basin=storm.display_basin, matched_season=storm.season, candidate_count=1, is_high_confidence=True)
        return result
    result["reject_reason"] = "no_unique_supported_name_date_geography_match"
    return result


def group_by_physical_storm(matched_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and group accepted country records to one row per physical SID."""
    required = {"matched_sid", "matched_name", "matched_basin", "matched_season", "start_year", "split", "country", "iso", "event_country_id", "dis_no", "dis_event_id", "match_tier"}
    missing = required.difference(matched_df.columns)
    if missing:
        raise ValueError(f"Matched records missing required columns: {sorted(missing)}")
    output_columns = ["matched_sid", "storm_name", "basin", "matched_season", "start_year_min", "start_year_max", "split", "country_record_count", "countries", "isos", "event_country_ids", "dis_nos", "dis_event_ids", "tiers_used"]
    if matched_df.empty:
        return pd.DataFrame(columns=output_columns)
    event_country_ids = matched_df["event_country_id"].fillna("").astype(str).str.strip()
    dis_nos = matched_df["dis_no"].fillna("").astype(str).str.strip()
    if event_country_ids.eq("").any() or dis_nos.eq("").any():
        raise ValueError("Matched records contain missing event-country identifiers")
    if not event_country_ids.equals(dis_nos):
        raise ValueError("Matched records contain inconsistent event_country_id and dis_no values")
    duplicated_ids = sorted(
        event_country_ids[event_country_ids.duplicated(keep=False)].unique()
    )
    if duplicated_ids:
        raise ValueError(
            "Matched records contain duplicate event-country identifiers: "
            f"{duplicated_ids}"
        )
    invalid = sorted(sid for sid in matched_df["matched_sid"].astype(str).unique() if not is_supported_sid(sid))
    if invalid:
        raise ValueError(f"Unsupported or missing IBTrACS SID(s): {invalid}")
    split_counts = matched_df.groupby("matched_sid")["split"].nunique(dropna=False)
    cross_split = sorted(split_counts[split_counts != 1].index.astype(str))
    if cross_split:
        raise ValueError(f"Physical storm SID spans temporal splits: {cross_split}")
    grouped = matched_df.groupby("matched_sid", sort=True).agg(
        storm_name=("matched_name", "first"), basin=("matched_basin", "first"),
        matched_season=("matched_season", "first"), start_year_min=("start_year", "min"),
        start_year_max=("start_year", "max"), split=("split", "first"),
        country_record_count=("country", "size"), countries=("country", lambda values: "; ".join(sorted(set(values)))),
        isos=("iso", lambda values: "; ".join(sorted(set(values)))), event_country_ids=("event_country_id", lambda values: "; ".join(sorted(set(values)))),
        dis_nos=("dis_no", lambda values: "; ".join(sorted(set(values)))),
        dis_event_ids=("dis_event_id", lambda values: "; ".join(sorted(set(values)))), tiers_used=("match_tier", lambda values: "; ".join(sorted(set(values)))),
    )
    return grouped.reset_index()[output_columns]


def run_matching_pipeline(emdat_path: Path = EMDAT_PATH, ibtracs_path: Path = IBTRACS_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the country-level audit and mandatory physical-storm grouping."""
    emdat = load_emdat_tc(emdat_path)
    storms = load_ibtracs_index(ibtracs_path)
    name_season_idx, season_idx = build_ibtracs_indexes(storms)
    audit = pd.DataFrame([match_emdat_record(row, name_season_idx, season_idx) for _, row in emdat.iterrows()])
    if audit.empty:
        accepted = pd.DataFrame(columns=["matched_sid", "matched_name", "matched_basin", "matched_season", "start_year", "split", "country", "iso", "event_country_id", "dis_no", "dis_event_id", "match_tier"])
    else:
        accepted = audit[audit["is_high_confidence"]].copy()
    return audit, group_by_physical_storm(accepted)


def main() -> None:
    """Run the local matching audit without training or loading outcomes."""
    audit, physical_storms = run_matching_pipeline()
    accepted = int(audit["is_high_confidence"].sum()) if not audit.empty else 0
    print("=" * 72)
    print("Version 2 IBTrACS-EM-DAT conservative matching audit")
    print(f"EM-DAT country records evaluated: {len(audit):,}")
    print(f"High-confidence country records: {accepted:,}")
    print(f"Unique physical cyclones after mandatory grouping: {len(physical_storms):,}")
    print("Counts are computed from local source files; no fixed target count is assumed.")
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(LOG_CSV, index=False, encoding="utf-8")
    PHYSICAL_STORM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    physical_storms.to_csv(PHYSICAL_STORM_OUTPUT, index=False, encoding="utf-8")
    print(f"Saved country-level audit: {LOG_CSV}")
    print(f"Saved physical-storm matches: {PHYSICAL_STORM_OUTPUT}")
    print("No model was trained. Target aggregation and time-safe features remain pending.")


if __name__ == "__main__":
    main()
