"""Guarded, provisional ML enrichment; unresolved live mappings block inference.

Only trusted, locally provisioned joblib artifacts may be placed at the fixed
paths below: joblib deserialization can execute code. No training or downloads
occur here. Returned probabilities are model class probabilities, not verified
operational confidence.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
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


def map_event_features(event: Any) -> MappingResult:
    """Build candidates without inventing geographic, subtype or magnitude data.

    Current ingestion has no validated geography/subtype/magnitude crosswalk.
    Consequently even a supported hazard remains mapping_incomplete in this
    first stage. Unknown/NaN placeholders are not authorization to predict.
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
    reasons.extend([
        "No validated disaster_subtype mapping; Unknown is a placeholder only.",
        "No verified country-to-ISO lookup or country-event scope mapping.",
        "No verified region or subregion enrichment.",
        "No validated hazard-specific magnitude/unit or missing-magnitude policy.",
    ])

    # Inspect the preserved source start date, never event_time's pubDate fallback.
    raw = _value(event, "raw_payload")
    start = raw.get("fromdate") if isinstance(raw, Mapping) else None
    try:
        date = parsedate_to_datetime(start) if isinstance(start, str) else None
        if date is None or date.tzinfo is None or date.utcoffset() is None:
            raise ValueError("No timezone-aware source start date")
        month = date.astimezone(timezone.utc).month
        features["start_month"] = month
        features["season"] = _SEASONS[month]
        reasons.append("Source start-date correspondence to EM-DAT country records is unvalidated.")
    except (TypeError, ValueError, OverflowError):
        reasons.append("Verified source start date unavailable; month/season unresolved.")

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
