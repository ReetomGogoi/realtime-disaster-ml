# Guarded live ML inference framework: local verification

## Evidence and scope

Implementation reference: `6b67b769819fb23262f06eec3b08a378efd5314b` (`Add guarded live ML inference framework`).

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

## Unresolved mappings

The implementation continues to report unresolved requirements for:

- Disaster subtype: no validated subtype mapping; `Unknown` remains a placeholder.
- ISO: no verified country-to-ISO lookup or country-event scope mapping.
- Region and subregion: no verified geographic enrichment.
- Magnitude: no validated hazard-specific magnitude/unit or missing-magnitude policy.
- Start date: correspondence between the source event-start date and historical EM-DAT country records remains unvalidated. If an appropriate source date is unavailable, start month and season remain unresolved.

The mapper does not silently supply guessed geography, a training-mode category, zero magnitude, or publication time as a verified event start. Candidate feature construction does not authorize prediction.

## Implementation consistency checks

- [Startup and health](../backend/app/main.py): lifespan initializes the model bundle in `app.state`; `/health` returns status `ok`.
- [Guarded mapping, loading and prediction](../backend/app/ml_inference.py): fixed local artifact paths, proposed cyclone/volcanic category normalization, fitted hazard coverage checks, explicit blocking reasons, and nullable blocked outputs.
- [Event API](../backend/app/api/routes/events.py): opt-in enrichment on both event GET routes using `include_predictions=true`; ordinary requests retain their existing behavior. The enrichment path does not insert or commit prediction records.
- [Response schemas](../backend/app/schemas/events.py): separate target statuses, reasons, nullable labels and nullable probability dictionaries. Probabilities, if later produced, are model class probabilities, not verified operational confidence.

Successful local artifact loading verifies framework readiness to hold the supplied models. It does **not** establish successful live ML inference: unresolved mapping requirements still block predictions. No code or manuscript changes, model execution, training, or test-split evaluation were performed while preparing this record.
