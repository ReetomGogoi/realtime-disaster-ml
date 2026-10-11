# Proposed GDACS-to-EM-DAT feature mapping

**Status:** design proposal only; not an implemented or validated live inference contract.

**Evidence basis:** repository revision `9e726a05e0b31e488873861c241103f4f7511ebe`, inspected on 2026-10-11. This analysis reads ingestion, storage, response-schema, historical preprocessing, and training definitions. It does not fetch a fresh GDACS feed, inspect historical data records, load models, run inference or training, or evaluate the reserved test split. Backend code and manuscripts are unchanged.

## 1. Verified implementation and historical contract

The ingestion module requests `https://www.gdacs.org/xml/rss_7d.xml`. Its normalized event dictionary contains:

| Current field | Extraction or meaning |
|---|---|
| `source` | Constant `GDACS`. |
| `source_event_id` | Usually `gdacs:eventtype` plus `gdacs:eventid`; otherwise event ID, link, or title fallback. Not a historical model feature. |
| `title`, `source_url` | RSS `title` and `link`. |
| `disaster_type` | Uppercased `gdacs:eventtype` translated through the ingestion code's six-entry map; unrecognized codes pass through as their code text. |
| `country` | Text from `gdacs:country`; no country-code or geographic classification conversion. |
| `latitude`, `longitude` | `geo:Point/geo:lat` and `geo:Point/geo:long`, falling back to `georss:point`. |
| `event_time` | Parsed `gdacs:fromdate`, falling back to RSS `pubDate`, converted to UTC; `None` if neither parses. |
| `raw_payload` | Direct child elements reduced to local tag names and `child.text`. |
| `fetched_at` | Ingestion time in UTC, not event start time. |

`Event` persists these fields and adds database `id` and `created_at`. There are no dedicated stored fields for `disaster_subtype`, `iso`, `region`, `subregion`, magnitude, or magnitude units. The public event response excludes `raw_payload`.

**Raw-payload limitation:** this is not a complete XML archive. The parser drops attributes, namespace distinctions, and nested content; repeated local tag names overwrite earlier values. A value or unit present only in an XML attribute cannot be recovered from this representation. Possible extra feed fields are not treated as verified candidates merely because raw payload storage exists.

The historical preprocessing constructs exactly nine predictors:

- Categorical: `disaster_type`, `disaster_subtype`, `iso`, `region`, `subregion`, `season`.
- Numeric: `start_month`, `magnitude`, `magnitude_available`.

Training uses most-frequent categorical imputation, one-hot encoding with unknown categories ignored, and numeric median imputation. These are verified software behaviors, not evidence that an incomplete live record is scientifically suitable for prediction. No change to that preprocessing is proposed here.

## 2. Feature-by-feature proposal

**Status meanings:** direct = same candidate category label without further derivation; derived = transformation with a known rule; incomplete = a candidate exists but required semantics or enrichment are unresolved; unavailable = no verified supplying field in the current normalized ingestion. All statuses describe the inspected implementation, not every capability GDACS may offer elsewhere.

| Historical feature and definition | GDACS candidate / path | Mapping status | Category or representation mismatch | Scientific or implementation risk | Proposed fallback when unavailable |
|---|---|---|---|---|---|
| `disaster_type`: EM-DAT type normalized using `CATEGORY_MAP` | `gdacs:eventtype` → stored `disaster_type` | **Direct** candidate for Earthquake, Flood, Wildfire and Landslide labels; **derived/incomplete** for cyclone and volcanic name normalization | `Tropical Cyclone` versus `Storm`; `Volcanic Event` versus `Volcanic activity`. Historical Landslide merges wet/dry mass movements. Unknown GDACS codes are not supported historical types. | Matching a label does not validate scope or reporting equivalence. GDACS cyclones cover a subset of historical Storm. Identical Landslide labels do not establish equivalent inclusion rules. | Preserve the source label, mark mapping unresolved, and withhold a model prediction for unknown/unsupported types. Do not replace a missing hazard with the training-mode hazard. |
| `disaster_subtype`: stripped EM-DAT `Disaster Subtype`, with missing values filled as `Unknown` | No dedicated parsed/stored GDACS field; `disaster_type` identifies a broad hazard only | **Unavailable** in normalized ingestion | No verified subtype crosswalk. In particular, a cyclone type code alone does not verify an exact historical subtype string. | Guessing subtype from title or type would create an unsupported category. Encoder acceptance of an unseen string is not semantic validation. | Represent the missing subtype as `Unknown`, consistent with historical construction, and mark subtype resolution incomplete. Verify training-vocabulary handling from training metadata before any future use; `Unknown` may be unseen and encoded as all zeros. Do not infer a subtype from prose. |
| `iso`: stripped historical `ISO` value | `gdacs:country` → stored `country`, as an input to a future controlled country-name-to-code lookup | **Derived/incomplete**; no direct normalized ISO field | Country-name text is not the historical ISO code string. Aliases, territories and ambiguous or multiple-country text need explicit handling. | A GDACS event is not necessarily a single EM-DAT country-event record. An event point, especially offshore, does not establish which country suffered the modeled outcomes. | Keep unresolved ISO missing and withhold country-specific prediction. Do not assign the training-mode country or use the nearest country to the coordinates as an automatic substitute. Resolve country scope before any lookup. |
| `region`: stripped historical `Region` value | No parsed GDACS region; future lookup from a verified country/ISO mapping into an EM-DAT-compatible geography crosswalk | **Derived/incomplete**; unavailable directly | No current crosswalk or verification of exact historical region labels/version | Generic continent names or a different geographic hierarchy may not match training categories. A correct country code does not by itself implement region enrichment. | Leave unresolved and withhold prediction until the same geographic hierarchy and exact label are established. Do not silently use the training-mode region. |
| `subregion`: stripped historical `Subregion` value | No parsed GDACS subregion; same future verified country/ISO geography lookup | **Derived/incomplete**; unavailable directly | Exact historical subregion labels and treatment of territories are not mapped | Independent guesses can produce inconsistent `iso`/`region`/`subregion` combinations. Coordinates alone are not an affected-country assignment. | Leave unresolved and withhold prediction until enrichment is verified. Do not substitute region for subregion or silently use the training mode. |
| `start_month`: numeric historical `Start Month` | `gdacs:fromdate` → `event_time` → month, provided the timestamp is verified as the relevant event start | **Derived/incomplete** | Current `event_time` may instead come from `pubDate`; the normalized field does not retain which source succeeded. UTC conversion can change the month near a boundary. | Publication/update month can differ from start month; one multi-country or prolonged event may not have the same onset date as each historical country record. | Use a missing numeric value (`NaN`) in a proposed feature record when start time cannot be established, with an explicit mapping flag. Do not silently use publication, fetch or current time. Withhold prediction pending date-policy validation; later use of fitted numeric imputation would need documented acceptance. |
| `season`: historical calendar grouping from start month, with `Unknown` for missing/unmapped month | Verified event-start month derived above → historical `SEASON_MAP` | **Derived**, conditional on validated start month; otherwise **incomplete** | Exact categories are `DJF`, `MAM`, `JJA`, `SON`, not local wet/dry or hemisphere-adjusted seasons | A wrong start month propagates into season. These features must refer to the same verified date. Calendar grouping does not measure local climate. | Use `Unknown` when start month is unavailable, matching historical construction. Do not derive season from an imputed month or publication time. Validate date provenance before future prediction. |
| `magnitude`: numeric EM-DAT `Magnitude`, with missing values retained before pipeline fitting | **No verified normalized supplying field.** Further review of original GDACS XML or documented event detail would be needed to identify any numerical measure and its attributes/units. | **Unavailable** in normalized ingestion; prospective mapping **incomplete** | Historical magnitude is hazard-dependent. No verified GDACS numeric-field/scale crosswalk exists in this inspection. Alert score, alert color, and event ID are not established substitutes. | Same number can mean different physical quantities, units, magnitude scales, aggregation windows or country/event scope. Early measurements may be revised; a historical final value may not exist at prediction time. Flattened raw payload may lose units stored as attributes. | Preserve as `NaN`, never zero, when no compatible value is established; pair with availability 0 and document the reason. The fitted pipeline would median-impute, but that behavior does not validate the live mapping or its missingness pattern. Withhold use until an explicit hazard-specific magnitude/missing-data policy is reviewed. |
| `magnitude_available`: historical `1` when numeric magnitude is non-missing, otherwise `0` | Derived from the accepted mapped `magnitude`, not from hazard type, coordinates or timestamp | **Derived**, but **incomplete** until magnitude eligibility is defined | A number being present in GDACS is not proof that it is compatible with historical magnitude. Feed absence, parser loss and unit incompatibility are different reasons for no mapped value. | Encoding every live record as 0 because ingestion lacks magnitude could introduce a different feature distribution. Rejected incompatible values are not the same mechanism as historical missingness. | Use 0 only when the proposed mapped magnitude is missing; use 1 only for a verified compatible numeric value. Record absence versus incompatibility separately outside the nine-feature vector. Do not treat availability 0 as permission to infer. |

## 3. Supported candidate category normalization

These examples follow the ingestion type map and historical preprocessing type map. They are proposed label normalization, not an implemented crosswalk or proof of outcome equivalence.

| GDACS code | Current stored label | Proposed historical model label | Qualification |
|---|---|---|---|
| `EQ` | Earthquake | Earthquake | Same category-name candidate; event scope still requires validation. |
| `FL` | Flood | Flood | Does not supply a flood subtype or comparable magnitude. |
| `TC` | Tropical Cyclone | Storm | A narrower GDACS type maps to the broader historical category. This does not imply all historical storms are tropical cyclones. No exact subtype is assigned here. |
| `VO` | Volcanic Event | Volcanic activity | Name normalization only. Historical priority labeling skips Volcanic activity because it has fewer than 30 complete-outcome training observations; a matching type label cannot establish priority-model coverage. |
| `WF` | Wildfire | Wildfire | Same label does not validate burned-area units or temporal scope. |
| `LS` | Landslide | Landslide | Historical construction merges `Mass movement (wet)` and `Mass movement (dry)`; equivalent GDACS inclusion scope has not been established. |

An unrecognized event code is not assigned to a supported category. No tsunami, drought, storm subtype, flood subtype, or volcanic subtype mapping is inferred from the inspected code.

## 4. Magnitude evidence and unresolved questions

The historical script copies the numeric `Magnitude` column without converting units or including a magnitude-scale feature. Its comments describe earthquake magnitude as Mw, flood and wildfire magnitude as km², and storm magnitude as kph; they also describe Landslide and Volcanic activity as lacking magnitude data. These comments document the intended interpretation, **not independent verification of every raw historical value or any GDACS quantity**. Magnitude-scale documentation and actual source metadata must govern a future mapping.

No specific GDACS magnitude element or attribute is asserted here because the current ingestion does not extract one. Any future candidate must establish:

1. Original field, units/scale and parsing requirements, including attributes currently discarded.
2. Compatibility with the historical quantity for that hazard: for example, wind speed is not an alert severity index, and exposed population is not flooded area.
3. Country-event versus whole-event scope and comparable measurement/aggregation window.
4. Availability at the declared prediction time, with revision provenance.
5. A documented response to missing or incompatible measurements without fabricating values or changing the nine-feature schema.

## 5. Proposed fallback and acceptance policy

The existing pipeline's imputation is a software fallback. This proposal distinguishes that from a scientific acceptance decision:

- Preserve source observations and mapping provenance. Do not overwrite a GDACS source label with an EM-DAT label in storage merely to satisfy the model.
- Missing hazard or unresolved country/geography: retain the monitoring event and withhold prediction; categorical mode imputation must not silently invent its identity.
- Missing subtype or season: `Unknown` follows historical dataset construction, but is not necessarily a category learned by a saved encoder.
- Missing start month or magnitude: retain `NaN` in the proposed model-input record; never replace missing magnitude with zero. Numeric imputation, if later approved for live use, must use the existing fitted training transformer, not a statistic recomputed from live data.
- Construct all nine features together. Check country/geographic consistency, date/season consistency, and magnitude/availability consistency before a future inference step.
- Treat a GDACS event affecting several countries as an unresolved unit-of-analysis question. Do not assume its single stored `country` string represents every affected country or assign a modeled priority to the whole event using one country's identity.
- Retain the origin of event start time, original measurement units, mapping version, and missing/rejected-field reasons in proposed provenance metadata outside the nine predictor columns. These are proposed requirements, not currently implemented fields.

The required next evidence is a reviewed source-field specification and representative GDACS XML examples, an EM-DAT-compatible geography and subtype vocabulary based on training information, and a hazard-specific quantity/timing audit. This document claims no validated live-feature mapping, operational readiness, inference completion, or final predictive performance. None of these design checks requires the reserved test split.

## 6. Repository evidence

- [GDACS ingestion](../backend/app/ingestion/gdacs.py): `_DISASTER_TYPE_MAP`, `_parse_event_time`, `_parse_coordinates`, `_item_to_dict`, feed URL.
- [Stored event fields](../backend/app/database/models.py): `Event` columns and uniqueness constraint.
- [Public event schema](../backend/app/schemas/events.py): `EventOut` fields and exclusion of raw payload.
- [Event routes](../backend/app/api/routes/events.py): persistence/updating of normalized fields and responses.
- [Historical preprocessing](../ml/preprocessing/build_training_dataset.py): `CATEGORY_MAP`, `SEASON_MAP`, feature construction, magnitude handling and target coverage.
- [Training pipeline](../ml/training/train_models.py): exact feature lists and fitted imputation/encoding definitions.

This is a code-based proposal. No external GDACS schema, fresh feed content, geographical lookup table, subtype taxonomy, or magnitude conversion has been claimed as verified by this inspection.
