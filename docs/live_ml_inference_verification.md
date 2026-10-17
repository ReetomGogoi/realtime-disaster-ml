# Guarded live ML inference framework: local verification

## Evidence and scope

Current implementation reference: `de68f5f4bbb7555787d92a986d8e7ee6747cb4fb` (`Validate GDACS start date for inference`). The original local framework verification was performed against `6b67b769819fb23262f06eec3b08a378efd5314b`; later guarded geography, country-scope and date-provenance changes were verified by offline backend tests.

The local execution results below were supplied by the project owner in the documentation request. They record that local verification, not a new execution or independent reproduction in Codex Cloud. The current code was inspected for consistency with the reported behavior. The local joblib artifacts are not present in this cloud checkout, so successful loading and the fitted hazard vocabularies were not independently rechecked here.

## Reported local verification results

| Check | Local result |
|---|---|
| Backend startup | FastAPI starts successfully locally. |
| Health endpoint | `GET /health` returns `status: "ok"`. |
| Model loading | Both local artifacts, `backend/models/affected_range_model.joblib` and `backend/models/impact_priority_model.joblib`, load successfully. |
| Unsupported hazard | Unsupported hazard code `DR` returns `unsupported_hazard`. |
| Tropical cyclone normalization | The parsed label `Tropical Cyclone` is normalized to the candidate historical category `Storm`. With the locally loaded models, the event proceeds to `mapping_incomplete`, rather than `unsupported_hazard`. |
| Incomplete mapping | Events with a recognized candidate category but incomplete feature mapping return null `predicted_label` and null `probabilities`. |
| Live prediction output | No live predictions were produced because required mappings remain unresolved. |
| Prediction persistence | No database persistence of predictions occurred. |
| Training and final evaluation | No training or test-split evaluation occurred. |

### Reported fitted hazard coverage

| Target | Supported hazard categories |
|---|---|
| `affected_range` | Earthquake, Flood, Landslide, Storm, Volcanic activity, Wildfire |
| `impact_priority` | Earthquake, Flood, Landslide, Storm, Wildfire |

Here, “supported” refers to the loaded pipeline's fitted hazard vocabulary used by the target-coverage guard. It does not establish validated live feature compatibility or predictive performance. In particular, Volcanic activity is not included in the reported priority-model coverage.

## Current feature-mapping status

| Historical model feature | Current live-mapping status |
|---|---|
| `disaster_type` | Partially verified. Direct candidates exist for Earthquake, Flood, Landslide and Wildfire. Tropical Cyclone → Storm and Volcanic Event → Volcanic activity remain controlled category normalizations rather than proof of identical event scope. |
| `disaster_subtype` | Unresolved. GDACS ingestion has no verified field or crosswalk for the EM-DAT subtype. `Unknown` remains a blocked placeholder. |
| `iso`, `region`, `subregion` | Conditionally verified. Exactly one nonempty preserved GDACS country occurrence must match one unique tuple in the EM-DAT country-geography crosswalk. Missing, multiple, repeated, ambiguous, unmatched and legacy cases remain unresolved; all three values are populated atomically. |
| `start_month`, `season` | Conditionally verified only for Earthquake and Landslide. Exactly one nonempty preserved `gdacs:fromdate` must parse with an explicit zero UTC/GMT offset. Month and season are populated together. Missing, repeated, conflicting, malformed, timezone-free and non-UTC values are blocked. Flood, Storm, Volcanic activity and Wildfire remain unresolved for these features. |
| `magnitude`, `magnitude_available` | Unresolved. No validated hazard-specific GDACS quantity, unit, scale or live missingness policy has been established. `NaN` and availability `0` remain blocked placeholders and do not authorize inference. |

The mapper does not silently supply guessed geography, a training-mode category, zero magnitude, publication time, `event_time`, ingestion time or current time as a verified model input. Candidate feature construction does not authorize prediction.

## Final feasibility decision

**Current live ML inference is NO-GO.** No supported hazard currently has a complete, scientifically defensible nine-feature live vector. Even Earthquake and Landslide, which have conditional geography and date paths, still lack verified `disaster_subtype`, `magnitude` and `magnitude_available` mappings. The implementation therefore continues to return blocked status with null prediction outputs.

The saved Random Forest pipelines and validation reports remain historical validation models and historical validation evidence. They do not establish performance on live GDACS-derived inputs. GDACS remains a live-event monitoring prototype; completed real-time impact prediction is not claimed.

If live impact prediction becomes a project requirement, the scientifically defensible path is to redesign and retrain a GDACS-native model using only features that are reliably available at prediction time, followed by a new leakage-safe validation plan. Existing historical validation results must remain unchanged and must not be presented as validation of that future model.

## Implementation consistency checks

- [Startup and health](../backend/app/main.py): lifespan initializes the model bundle in `app.state`; `/health` returns status `ok`.
- [Guarded mapping, loading and prediction](../backend/app/ml_inference.py): fixed local artifact paths, proposed cyclone/volcanic category normalization, fitted hazard coverage checks, explicit blocking reasons, and nullable blocked outputs.
- [Event API](../backend/app/api/routes/events.py): opt-in enrichment on both event GET routes using `include_predictions=true`; ordinary requests retain their existing behavior. The enrichment path does not insert or commit prediction records.
- [Response schemas](../backend/app/schemas/events.py): separate target statuses, reasons, nullable labels and nullable probability dictionaries. Probabilities, if later produced, are model class probabilities, not verified operational confidence.

Successful local artifact loading verifies framework readiness to hold the supplied models. It does **not** establish successful live ML inference: unresolved subtype and magnitude requirements still block every supported hazard, and additional scope/date/target guards block ineligible events. No code or manuscript changes, model execution, training, or test-split evaluation were performed while preparing this record.
