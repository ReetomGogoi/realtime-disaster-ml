"""Build a validation-only, time-safe EM-DAT/IBTrACS cyclone feature table.

The builder consumes the reviewed country-level matching audit, the existing
processed EM-DAT model table, and authentic IBTrACS observations.  It retains
only train (2000-2019) and validation (2020-2022) records.  For every accepted
event-country match, T0 is 00:00 UTC on its complete EM-DAT start date and no
IBTrACS observation after T0 is used.

Generated event-level data and aggregate audits are local artifacts under
datasets/processed, which is ignored by Git.  This module does not train or
save a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ml.preprocessing.match_ibtracs_emdat import is_supported_sid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCH_AUDIT = PROJECT_ROOT / "docs" / "v2_match_pilot_log.csv"
DEFAULT_PROCESSED_EMDAT = (
    PROJECT_ROOT / "datasets" / "processed" / "emdat_ml_dataset.csv"
)
DEFAULT_IBTRACS = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "ibtracs"
    / "ibtracs.since1980.list.v04r01.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "datasets" / "processed" / "v2_ibtracs_cyclone_features.csv"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT / "datasets" / "processed" / "v2_ibtracs_cutoff_audit.json"
)

DEVELOPMENT_SPLITS = frozenset({"train", "validation"})
TARGET_COLUMNS = ("affected_range", "impact_priority")
BASE_FEATURE_COLUMNS = (
    "disaster_type",
    "disaster_subtype",
    "iso",
    "region",
    "subregion",
    "season",
    "start_month",
    "magnitude",
    "magnitude_available",
)
IBTRACS_REQUIRED_COLUMNS = (
    "SID",
    "ISO_TIME",
    "LAT",
    "LON",
    "WMO_WIND",
    "WMO_PRES",
    "DIST2LAND",
    "WMO_AGENCY",
)
IBTRACS_CATEGORICAL_FEATURES = ("ibtracs_wmo_agency",)
IBTRACS_NUMERIC_FEATURES = (
    "ibtracs_wmo_wind_knots",
    "ibtracs_wmo_wind_available",
    "ibtracs_wmo_pressure_mb",
    "ibtracs_wmo_pressure_available",
    "ibtracs_latitude",
    "ibtracs_longitude",
    "ibtracs_location_available",
    "ibtracs_distance_to_land_km",
    "ibtracs_distance_to_land_available",
    "ibtracs_wmo_agency_available",
    "ibtracs_storm_age_hours",
    "ibtracs_motion_speed_kmh",
    "ibtracs_motion_bearing_sin",
    "ibtracs_motion_bearing_cos",
    "ibtracs_motion_available",
)

MATCH_REQUIRED_COLUMNS = (
    "event_country_id",
    "dis_no",
    "start_date",
    "split",
    "matched_sid",
    "is_high_confidence",
)
PROCESSED_REQUIRED_COLUMNS = (
    "dis_no",
    "temporal_split",
    *BASE_FEATURE_COLUMNS,
    *TARGET_COLUMNS,
)


@dataclass(frozen=True)
class TrackObservation:
    sid: str
    observed_at: datetime
    latitude: float | None
    longitude: float | None
    wmo_wind: float | None
    wmo_pressure: float | None
    distance_to_land: float | None
    wmo_agency: str


def _clean_identifier(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _require_unique_nonempty(frame: pd.DataFrame, column: str, source: str) -> None:
    values = frame[column].map(_clean_identifier)
    if values.eq("").any():
        raise ValueError(f"{source}: missing {column}")
    duplicates = sorted(values[values.duplicated(keep=False)].unique())
    if duplicates:
        raise ValueError(f"{source}: duplicate {column} values: {duplicates}")
    frame[column] = values


def _parse_high_confidence(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1"}


def _parse_start_date(value: Any) -> datetime:
    text = _clean_identifier(value)
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError(f"Invalid or incomplete canonical start_date: {text!r}") from error
    return parsed.replace(tzinfo=timezone.utc)


def _parse_iso_time(value: str) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # The IBTrACS ISO_TIME column is documented as UTC even though its CSV
        # representation is commonly timezone-free.
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _finite_number(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _bounded(value: str, minimum: float, maximum: float) -> float | None:
    number = _finite_number(value)
    if number is None or not minimum <= number <= maximum:
        return None
    return number


def _nonnegative(value: str, *, allow_zero: bool = True) -> float | None:
    number = _finite_number(value)
    if number is None or number < 0 or (not allow_zero and number == 0):
        return None
    return number


def load_matching_audit(path: Path) -> pd.DataFrame:
    """Load the reviewed country matching audit without accepting test rows."""
    frame = pd.read_csv(path, dtype=str, usecols=list(MATCH_REQUIRED_COLUMNS))
    _require_unique_nonempty(frame, "event_country_id", "matching audit")
    dis_nos = frame["dis_no"].map(_clean_identifier)
    if not frame["event_country_id"].equals(dis_nos):
        raise ValueError("matching audit: event_country_id and dis_no disagree")
    normalized_splits = frame["split"].map(_clean_identifier)
    if normalized_splits.eq("").any():
        raise ValueError("matching audit contains a missing temporal split")
    splits = set(normalized_splits)
    if not splits <= DEVELOPMENT_SPLITS:
        raise ValueError(f"matching audit contains forbidden split(s): {sorted(splits)}")
    frame["split"] = normalized_splits
    confidence_values = frame["is_high_confidence"].map(_clean_identifier).str.casefold()
    if not set(confidence_values) <= {"true", "false", "1", "0"}:
        raise ValueError("matching audit contains an invalid high-confidence flag")
    frame["is_high_confidence"] = frame["is_high_confidence"].map(
        _parse_high_confidence
    )
    return frame


def load_processed_development(path: Path) -> pd.DataFrame:
    """Read required columns and immediately discard all non-development rows."""
    retained: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=list(PROCESSED_REQUIRED_COLUMNS),
        chunksize=50_000,
    ):
        development = chunk.loc[
            chunk["temporal_split"].isin(DEVELOPMENT_SPLITS)
        ].copy()
        if not development.empty:
            retained.append(development)
    if not retained:
        raise ValueError("processed EM-DAT contains no train/validation rows")
    frame = pd.concat(retained, ignore_index=True)
    _require_unique_nonempty(frame, "dis_no", "processed EM-DAT development rows")
    return frame


def join_matches_to_processed(
    matches: pd.DataFrame,
    processed: pd.DataFrame,
) -> pd.DataFrame:
    """Validate all IDs/splits, then return accepted event-country rows."""
    linkage = matches.merge(
        processed,
        left_on="event_country_id",
        right_on="dis_no",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_match", "_processed"),
    )
    if not linkage["_merge"].eq("both").all():
        missing = linkage.loc[
            linkage["_merge"].ne("both"), "event_country_id"
        ].tolist()
        raise ValueError(f"matching IDs absent from processed EM-DAT: {missing}")
    if not linkage["split"].eq(linkage["temporal_split"]).all():
        raise ValueError("matching and processed temporal splits disagree")

    accepted = linkage.loc[linkage["is_high_confidence"]].copy()
    if accepted.empty:
        raise ValueError("no high-confidence event-country matches")
    if not accepted["matched_sid"].map(is_supported_sid).all():
        raise ValueError("accepted records contain an unsupported IBTrACS SID")
    cross_split = accepted.groupby("matched_sid")["split"].nunique(dropna=False)
    cross_split = sorted(cross_split[cross_split != 1].index.astype(str))
    if cross_split:
        raise ValueError(f"physical storm SID spans temporal splits: {cross_split}")

    accepted["t0_utc"] = accepted["start_date"].map(_parse_start_date)
    columns = [
        "event_country_id",
        "matched_sid",
        "start_date",
        "t0_utc",
        "temporal_split",
        *BASE_FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    ]
    return accepted.loc[:, columns].reset_index(drop=True)


def load_ibtracs_observations(
    path: Path,
    required_sids: Iterable[str],
) -> tuple[dict[str, list[TrackObservation]], dict[str, Any]]:
    """Stream only matched physical storms from the authentic IBTrACS CSV."""
    wanted = set(required_sids)
    observations: dict[str, list[TrackObservation]] = defaultdict(list)
    malformed_times = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            headers = [value.strip() for value in next(reader)]
            units = [value.strip() for value in next(reader)]
        except StopIteration as error:
            raise ValueError("IBTrACS is missing its header or units row") from error
        columns = {name: index for index, name in enumerate(headers)}
        missing = set(IBTRACS_REQUIRED_COLUMNS).difference(columns)
        if missing:
            raise ValueError(f"IBTrACS missing required columns: {sorted(missing)}")
        unit_map = {
            name: units[index] if index < len(units) else ""
            for name, index in columns.items()
            if name in IBTRACS_REQUIRED_COLUMNS
        }
        for row in reader:
            if len(row) < len(headers):
                continue
            sid = row[columns["SID"]].strip()
            if sid not in wanted:
                continue
            observed_at = _parse_iso_time(row[columns["ISO_TIME"]])
            if observed_at is None:
                malformed_times += 1
                continue
            observations[sid].append(
                TrackObservation(
                    sid=sid,
                    observed_at=observed_at,
                    latitude=_bounded(row[columns["LAT"]], -90.0, 90.0),
                    longitude=_bounded(row[columns["LON"]], -180.0, 180.0),
                    wmo_wind=_nonnegative(row[columns["WMO_WIND"]]),
                    wmo_pressure=_nonnegative(
                        row[columns["WMO_PRES"]], allow_zero=False
                    ),
                    distance_to_land=_nonnegative(row[columns["DIST2LAND"]]),
                    wmo_agency=row[columns["WMO_AGENCY"]].strip(),
                )
            )

    for sid_observations in observations.values():
        sid_observations.sort(key=lambda observation: observation.observed_at)
    if malformed_times:
        raise ValueError(
            "Matched IBTrACS rows contain malformed ISO_TIME values; "
            "the T0 cutoff cannot be enforced safely"
        )
    metadata = {
        "required_column_units_as_read": unit_map,
        "malformed_selected_iso_time_rows": malformed_times,
        "matched_sids_missing_from_ibtracs": sorted(wanted - set(observations)),
        "duplicate_timestamp_policy": "audited per event using ISO_TIME <= T0 only",
    }
    return dict(observations), metadata


def _motion(
    previous: TrackObservation,
    current: TrackObservation,
) -> tuple[float, float, float] | None:
    if None in (
        previous.latitude,
        previous.longitude,
        current.latitude,
        current.longitude,
    ):
        return None
    elapsed_hours = (current.observed_at - previous.observed_at).total_seconds() / 3600
    if elapsed_hours <= 0:
        return None
    lat1 = math.radians(float(previous.latitude))
    lat2 = math.radians(float(current.latitude))
    delta_lat = lat2 - lat1
    delta_lon = math.radians(float(current.longitude) - float(previous.longitude))
    haversine_a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    distance_km = 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(haversine_a)))
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2
    ) * math.cos(delta_lon)
    bearing = math.atan2(y, x)
    return distance_km / elapsed_hours, math.sin(bearing), math.cos(bearing)


def extract_event_features(
    event: pd.Series,
    observations: list[TrackObservation],
) -> tuple[dict[str, Any] | None, str]:
    """Extract only observations at or before this event-country cutoff."""
    t0 = event["t0_utc"]
    past = [observation for observation in observations if observation.observed_at <= t0]
    if not past:
        return None, "no_observation_at_or_before_t0"
    past_times = [observation.observed_at for observation in past]
    if len(past_times) != len(set(past_times)):
        return None, "duplicate_track_timestamps_at_or_before_t0"
    latest = past[-1]
    location_available = latest.latitude is not None and latest.longitude is not None
    agency = latest.wmo_agency.strip()

    location_history = [
        observation
        for observation in past
        if observation.latitude is not None and observation.longitude is not None
    ]
    motion = (
        _motion(location_history[-2], location_history[-1])
        if len(location_history) >= 2
        else None
    )
    row = {
        "event_country_id": event["event_country_id"],
        "matched_sid": event["matched_sid"],
        "start_date": event["start_date"],
        "temporal_split": event["temporal_split"],
        **{column: event[column] for column in BASE_FEATURE_COLUMNS},
        **{column: event[column] for column in TARGET_COLUMNS},
        "ibtracs_wmo_agency": agency if agency else "Unknown",
        "ibtracs_wmo_wind_knots": latest.wmo_wind,
        "ibtracs_wmo_wind_available": int(latest.wmo_wind is not None),
        "ibtracs_wmo_pressure_mb": latest.wmo_pressure,
        "ibtracs_wmo_pressure_available": int(latest.wmo_pressure is not None),
        "ibtracs_latitude": latest.latitude,
        "ibtracs_longitude": latest.longitude,
        "ibtracs_location_available": int(location_available),
        "ibtracs_distance_to_land_km": latest.distance_to_land,
        "ibtracs_distance_to_land_available": int(
            latest.distance_to_land is not None
        ),
        "ibtracs_wmo_agency_available": int(bool(agency)),
        "ibtracs_storm_age_hours": (
            t0 - past[0].observed_at
        ).total_seconds()
        / 3600,
        "ibtracs_motion_speed_kmh": motion[0] if motion else None,
        "ibtracs_motion_bearing_sin": motion[1] if motion else None,
        "ibtracs_motion_bearing_cos": motion[2] if motion else None,
        "ibtracs_motion_available": int(motion is not None),
    }
    return row, "eligible"


def build_feature_table(
    events: pd.DataFrame,
    observations: dict[str, list[TrackObservation]],
) -> tuple[pd.DataFrame, Counter[str]]:
    """Build one row per eligible event-country and count exclusion reasons."""
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for _, event in events.iterrows():
        sid = str(event["matched_sid"])
        row, reason = extract_event_features(event, observations.get(sid, []))
        reasons[reason] += 1
        if row is not None:
            rows.append(row)
    expected_columns = [
        "event_country_id",
        "matched_sid",
        "start_date",
        "temporal_split",
        *BASE_FEATURE_COLUMNS,
        *TARGET_COLUMNS,
        *IBTRACS_CATEGORICAL_FEATURES,
        *IBTRACS_NUMERIC_FEATURES,
    ]
    return pd.DataFrame(rows, columns=expected_columns), reasons


def build_audit(
    matches: pd.DataFrame,
    accepted: pd.DataFrame,
    features: pd.DataFrame,
    reasons: Counter[str],
    ibtracs_metadata: dict[str, Any],
) -> dict[str, Any]:
    availability: dict[str, Any] = {}
    flag_columns = [
        column for column in IBTRACS_NUMERIC_FEATURES if column.endswith("_available")
    ]
    for split in sorted(DEVELOPMENT_SPLITS):
        split_rows = features.loc[features["temporal_split"].eq(split)]
        availability[split] = {
            "event_country_rows": int(len(split_rows)),
            "physical_sids": int(split_rows["matched_sid"].nunique()),
            "available_counts": {
                column: int(split_rows[column].sum()) for column in flag_columns
            },
            "missing_counts": {
                column: int(split_rows[column].isna().sum())
                for column in (
                    "ibtracs_wmo_wind_knots",
                    "ibtracs_wmo_pressure_mb",
                    "ibtracs_latitude",
                    "ibtracs_longitude",
                    "ibtracs_distance_to_land_km",
                    "ibtracs_motion_speed_kmh",
                )
            },
        }
    return {
        "policy": {
            "cutoff": "00:00 UTC on complete EM-DAT start_date",
            "observation_rule": "IBTrACS ISO_TIME <= T0 only",
            "future_interpolation": False,
            "development_splits": ["train", "validation"],
            "test_years": "2023-2025 excluded",
        },
        "counts": {
            "matching_audit_event_country_rows": int(len(matches)),
            "high_confidence_event_country_rows": int(len(accepted)),
            "high_confidence_physical_sids": int(accepted["matched_sid"].nunique()),
            "eligible_event_country_rows": int(len(features)),
            "eligible_physical_sids": int(features["matched_sid"].nunique()),
            "eligibility_reasons": dict(sorted(reasons.items())),
        },
        "integrity": {
            "duplicate_output_event_country_ids": int(
                features["event_country_id"].duplicated().sum()
            ),
            "cross_split_output_sids": int(
                (features.groupby("matched_sid")["temporal_split"].nunique() > 1).sum()
            ),
        },
        "availability_by_split": availability,
        "ibtracs_input_audit": ibtracs_metadata,
    }


def run_builder(
    match_audit_path: Path,
    processed_emdat_path: Path,
    ibtracs_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    matches = load_matching_audit(match_audit_path)
    processed = load_processed_development(processed_emdat_path)
    accepted = join_matches_to_processed(matches, processed)
    observations, metadata = load_ibtracs_observations(
        ibtracs_path, accepted["matched_sid"].unique()
    )
    features, reasons = build_feature_table(accepted, observations)
    if features.empty:
        raise ValueError("no event-country rows are eligible at the cutoff")
    if features["event_country_id"].duplicated().any():
        raise ValueError("feature output contains duplicate event_country_id values")
    if (features.groupby("matched_sid")["temporal_split"].nunique() > 1).any():
        raise ValueError("feature output contains a physical SID in both splits")
    audit = build_audit(matches, accepted, features, reasons, metadata)
    return features, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-audit", type=Path, default=DEFAULT_MATCH_AUDIT)
    parser.add_argument("--processed-emdat", type=Path, default=DEFAULT_PROCESSED_EMDAT)
    parser.add_argument("--ibtracs", type=Path, default=DEFAULT_IBTRACS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.matching_audit, args.processed_emdat, args.ibtracs):
        if not path.is_file():
            raise FileNotFoundError(f"Required local input does not exist: {path}")
    if not args.overwrite:
        existing = [path for path in (args.output, args.audit_output) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite local output(s): {existing}")

    features, audit = run_builder(
        args.matching_audit, args.processed_emdat, args.ibtracs
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False, encoding="utf-8")
    with args.audit_output.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")
    print(f"Saved local feature table: {args.output}")
    print(f"Saved aggregate cutoff audit: {args.audit_output}")
    print(json.dumps(audit["counts"], indent=2))


if __name__ == "__main__":
    main()
