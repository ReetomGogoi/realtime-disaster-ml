"""Guarded, provisional ML enrichment; unresolved live mappings block inference.

Only trusted, locally provisioned joblib artifacts may be placed at the fixed
paths below: joblib deserialization can execute code. No training or downloads
occur here. Returned probabilities are model class probabilities, not verified
operational confidence.
"""

from __future__ import annotations

import csv
import logging
import math
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = (
    "disaster_type", "disaster_subtype", "iso", "region", "subregion", "season",
)
NUMERIC_FEATURES = ("start_month", "magnitude", "magnitude_available")
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGETS = ("affected_range", "impact_priority")
_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_COUNTRY_CROSSWALK = (
    Path(__file__).resolve().parents[2] / "docs" / "emdat_country_geography_crosswalk.csv"
)
_HAZARDS = {
    "Earthquake": "Earthquake", "Flood": "Flood", "Landslide": "Landslide",
    "Wildfire": "Wildfire", "Tropical Cyclone": "Storm",
    "Volcanic Event": "Volcanic activity",
}
_SEASONS = {
    12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON",
}


@dataclass
class MappingResult:
    features: dict[str, Any]
    status: str
    reasons: list[str]


@dataclass
class ModelState:
    pipeline: Any = None
    reasons: list[str] = field(default_factory=list)
    supported_hazards: frozenset[str] = frozenset()


def _value(event: Any, name: str) -> Any:
    return event.get(name) if isinstance(event, Mapping) else getattr(event, name, None)


def _normalize_country(country: str) -> str:
    # Preserve punctuation and accents; do not fuzzy-match or transliterate.
    return " ".join(unicodedata.normalize("NFC", country).split()).casefold()


@lru_cache(maxsize=1)
def _load_country_geography() -> tuple[dict[str, set[tuple[str, str, str]]], str | None]:
    """Read the authoritative historical snapshot once; never replace its codes.

    No aliases are currently approved. Historical/territory codes are retained
    rather than rewritten using a current sovereign-country or ISO library.
    Restart the application after replacing the reviewed crosswalk snapshot.
    """
    try:
        index: dict[str, set[tuple[str, str, str]]] = {}
        with _COUNTRY_CROSSWALK.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            columns = {"Country", "ISO", "region", "subregion"}
            if reader.fieldnames is None or set(reader.fieldnames) != columns or len(reader.fieldnames) != 4:
                raise ValueError("Unexpected geography crosswalk columns")
            for row in reader:
                if set(row) != columns or any(not isinstance(v, str) or not v.strip() for v in row.values()):
                    raise ValueError("Incomplete geography crosswalk row")
                key = _normalize_country(row["Country"])
                values = tuple(row[name].strip() for name in ("ISO", "region", "subregion"))
                index.setdefault(key, set()).add(values)
        if not index:
            raise ValueError("Empty geography crosswalk")
        # An ISO with conflicting geography makes every associated name unsafe.
        by_iso: dict[str, set[tuple[str, str]]] = {}
        for tuples in index.values():
            for iso, region, subregion in tuples:
                by_iso.setdefault(iso, set()).add((region, subregion))
        for tuples in index.values():
            if any(len(by_iso[iso]) != 1 for iso, _, _ in tuples):
                tuples.clear()
        return index, None
    except (OSError, UnicodeError, csv.Error, ValueError):
        logger.warning("Historical geography crosswalk unavailable or invalid")
        return {}, "Historical geography crosswalk unavailable or invalid; geography unresolved."


def resolve_country_geography(country: Any) -> tuple[dict[str, str] | None, str | None]:
    """Return the entire unique historical tuple or no geographic features.

    Matching a country identity does not validate a GDACS event's country-impact
    scope. In particular, no multi-country record is reduced to a first country.
    """
    if not isinstance(country, str) or not country.strip():
        return None, "Country missing or not a single country-text value; geography unresolved."
    index, error = _load_country_geography()
    if error:
        return None, error
    key = _normalize_country(country)
    if key in index:
        tuples = index[key]
        if len(tuples) != 1:
            return None, "Ambiguous or conflicting historical country mapping; geography unresolved."
        iso, region, subregion = next(iter(tuples))
        return {"iso": iso, "region": region, "subregion": subregion}, None
    # Check exact names first: valid names may contain commas or 'and'. Do not
    # split a list to choose one entry, or invent aliases for unknown strings.
    if re.search(r"[,;/|\r\n]|\s+and\s+|&", country, flags=re.IGNORECASE):
        return None, "Country text is multi-country or not one exact verified name; geography unresolved."
    return None, "Country has no exact verified historical match; geography unresolved."


def _country_with_valid_scope(event: Any) -> tuple[str | None, str | None]:
    """Return one preserved source country only when its scope is unambiguous."""
    raw = _value(event, "raw_payload")
    if not isinstance(raw, Mapping) or "gdacs_country_occurrences" not in raw:
        return None, "GDACS event-to-country impact scope remains unvalidated."
    occurrences = raw["gdacs_country_occurrences"]
    if not isinstance(occurrences, list):
        return None, "GDACS country occurrence metadata is invalid; geography unresolved."
    if not occurrences:
        return None, "No GDACS country occurrences were preserved; geography unresolved."
    if len(occurrences) > 1:
        return None, "Multiple GDACS country occurrences make event-to-country scope unresolved."
    country = occurrences[0]
    if not isinstance(country, str) or not country.strip():
        return None, "GDACS country occurrence missing or not a single country-text value; geography unresolved."
    return country, None


def _start_date_with_valid_provenance(
    event: Any, hazard: str | None,
) -> tuple[datetime | None, str | None]:
    """Return one UTC source start date only for provisionally safe hazards."""
    raw = _value(event, "raw_payload")
    if not isinstance(raw, Mapping) or "gdacs_fromdate_occurrences" not in raw:
        return None, "GDACS fromdate occurrence provenance unavailable; month/season unresolved."
    occurrences = raw["gdacs_fromdate_occurrences"]
    if not isinstance(occurrences, list):
        return None, "GDACS fromdate occurrence metadata is invalid; month/season unresolved."
    if not occurrences:
        return None, "No GDACS fromdate occurrences were preserved; month/season unresolved."
    if len(occurrences) > 1:
        return None, "Multiple GDACS fromdate occurrences make date provenance unresolved."
    value = occurrences[0]
    if not isinstance(value, str) or not value.strip():
        return None, "GDACS fromdate occurrence is empty or invalid; month/season unresolved."
    try:
        date = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None, "GDACS fromdate timestamp is malformed; month/season unresolved."
    if date.tzinfo is None or date.utcoffset() is None:
        return None, "GDACS fromdate timestamp has no timezone; month/season unresolved."
    if date.utcoffset().total_seconds() != 0:
        return None, "GDACS fromdate timestamp is not GMT/UTC; month/season unresolved."
    if hazard not in {"Earthquake", "Landslide"}:
        return None, "GDACS fromdate mapping is unsupported for this hazard; month/season unresolved."
    return date.astimezone(timezone.utc), None


def map_event_features(event: Any) -> MappingResult:
    """Build candidates without inventing geographic, subtype or magnitude data.

    Geography is populated only from a unique authoritative historical tuple.
    Subtype and magnitude policy remain unresolved; candidate values and
    Unknown/NaN placeholders do not authorize inference.
    """
    source_type = _value(event, "disaster_type")
    hazard = _HAZARDS.get(source_type) if isinstance(source_type, str) else None
    features = {
        "disaster_type": hazard,
        "disaster_subtype": "Unknown",
        "iso": None,
        "region": None,
        "subregion": None,
        "start_month": math.nan,
        "season": "Unknown",
        "magnitude": math.nan,
        "magnitude_available": 0,
    }
    reasons = []
    if _value(event, "source") != "GDACS":
        reasons.append("Only GDACS source records are covered by this mapping proposal.")
    if hazard is None:
        reasons.append("Missing or unsupported GDACS hazard category.")
    scoped_country, scope_reason = _country_with_valid_scope(event)
    if scope_reason is not None:
        reasons.append(scope_reason)
    else:
        geography, geography_reason = resolve_country_geography(scoped_country)
        if geography is not None:
            features.update(geography)
        else:
            reasons.append(geography_reason)
    reasons.extend([
        "No validated disaster_subtype mapping; Unknown is a placeholder only.",
        "No validated hazard-specific magnitude/unit or missing-magnitude policy.",
    ])

    # Use only the provenance-preserved source field; never any timestamp fallback.
    date, date_reason = _start_date_with_valid_provenance(event, hazard)
    if date is not None:
        month = date.month
        features["start_month"] = month
        features["season"] = _SEASONS[month]
    else:
        reasons.append(date_reason)

    status = "unsupported_hazard" if source_type and hazard is None else "mapping_incomplete"
    return MappingResult(features, status, reasons)


def _validate_pipeline(pipeline: Any) -> frozenset[str]:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_is_fitted

    if not isinstance(pipeline, Pipeline):
        raise ValueError("Artifact is not a pipeline")
    if list(pipeline.feature_names_in_) != list(FEATURES):
        raise ValueError("Unexpected historical feature schema")
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(preprocessor, ColumnTransformer) or not isinstance(classifier, RandomForestClassifier):
        raise ValueError("Unexpected pipeline components")
    check_is_fitted(classifier)
    if set(map(str, classifier.classes_)) != {"high", "low", "medium"}:
        raise ValueError("Unexpected target classes")
    columns = {name: list(cols) for name, _, cols in preprocessor.transformers_ if name in {"cat", "num"}}
    if columns != {"cat": list(CATEGORICAL_FEATURES), "num": list(NUMERIC_FEATURES)}:
        raise ValueError("Unexpected transformer feature groups")
    cat = preprocessor.named_transformers_["cat"]
    num = preprocessor.named_transformers_["num"]
    if (cat.named_steps["imputer"].strategy != "most_frequent"
            or cat.named_steps["onehot"].handle_unknown != "ignore"
            or num.named_steps["imputer"].strategy != "median"):
        raise ValueError("Unexpected preprocessing settings")
    return frozenset(map(str, cat.named_steps["onehot"].categories_[0]))


def load_model_bundle() -> dict[str, ModelState]:
    """Load once during lifespan; absence/failure disables only the affected target.

    Fixed paths are not accepted from clients. Version-mismatch warnings reject
    an artifact rather than silently using a cross-version sklearn pickle.
    Deployment must separately establish trusted provenance/package versions.
    """
    bundle = {}
    for target in TARGETS:
        path = _MODELS_DIR / f"{target}_model.joblib"
        if not path.is_file():
            bundle[target] = ModelState(reasons=[f"Model artifact missing: {path.name}."])
            continue
        try:
            import joblib
            from sklearn.exceptions import InconsistentVersionWarning

            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                pipeline = joblib.load(path)
            hazards = _validate_pipeline(pipeline)
            bundle[target] = ModelState(pipeline=pipeline, supported_hazards=hazards)
        except Exception:
            logger.exception("Unable to load/validate the %s model", target)
            bundle[target] = ModelState(reasons=["Model artifact could not be loaded or validated."])
    return bundle


def predict_event(event: Any, bundle: Mapping[str, ModelState] | None) -> dict[str, dict[str, Any]]:
    """Never invoke a pipeline until both mapping and target eligibility pass."""
    mapped = map_event_features(event)
    results = {}
    for target in TARGETS:
        state = (bundle or {}).get(target, ModelState(reasons=["Model bundle unavailable."]))
        result = {"status": "mapping_incomplete", "reasons": list(mapped.reasons),
                  "predicted_label": None, "probabilities": None}
        if mapped.status == "unsupported_hazard":
            result["status"] = "unsupported_hazard"
        elif state.pipeline is None:
            result["status"] = "model_unavailable"
            result["reasons"].extend(state.reasons)
        elif mapped.features["disaster_type"] not in state.supported_hazards:
            result["status"] = "unsupported_for_target"
            result["reasons"].append("Hazard category absent from this target's fitted training vocabulary.")
        elif mapped.status == "mapped" and not mapped.reasons:
            try:
                import pandas as pd

                frame = pd.DataFrame([mapped.features], columns=list(FEATURES))
                label = str(state.pipeline.predict(frame)[0])
                scores = state.pipeline.predict_proba(frame)[0]
                classes = state.pipeline.named_steps["classifier"].classes_
                probabilities = {str(name): float(score) for name, score in zip(classes, scores, strict=True)}
                if (label not in probabilities
                        or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values())
                        or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6)):
                    raise ValueError("Invalid model output")
                result.update(status="predicted", reasons=[], predicted_label=label, probabilities=probabilities)
            except Exception:
                logger.exception("Prediction failed for %s", target)
                result.update(status="inference_error", reasons=["Model prediction could not be produced."])
        results[target] = result
    return results
