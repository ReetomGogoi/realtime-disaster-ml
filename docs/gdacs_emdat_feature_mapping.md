# Proposed GDACS-to-EM-DAT feature mapping

**Status:** guarded mapping analysis with conditionally verified geography and date components. It is not a complete live inference contract. Final feasibility decision: **NO-GO for current live ML inference**.

**Evidence basis:** repository revision `de68f5f4bbb7555787d92a986d8e7ee6747cb4fb`, inspected after the guarded geography, country-scope and start-date provenance changes. This analysis reads ingestion, storage, response-schema, historical preprocessing, training definitions and saved validation reports. It does not run live inference or training, alter historical results, or evaluate the reserved test split.

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
| `raw_payload` | Direct child elements reduced to local tag names and `child.text`. Separate `gdacs_country_occurrences` and `gdacs_fromdate_occurrences` lists preserve every direct namespaced occurrence in source order. |
| `fetched_at` | Ingestion time in UTC, not event start time. |

`Event` persists these fields and adds database `id` and `created_at`. There are no dedicated stored fields for `disaster_subtype`, `iso`, `region`, `subregion`, magnitude, or magnitude units. The public event response excludes `raw_payload`.

**Raw-payload limitation:** this is not a complete XML archive. The general flattened fields still drop attributes, namespace distinctions and nested content, and repeated local tag names overwrite earlier values. Country and `fromdate` are exceptions because their direct namespaced occurrences are also preserved in dedicated ordered lists. A value or unit present only in an XML attribute cannot be recovered from this representation. Possible extra feed fields are not treated as verified candidates merely because raw payload storage exists.

The historical preprocessing constructs exactly nine predictors:

- Categorical: `disaster_type`, `disaster_subtype`, `iso`, `region`, `subregion`, `season`.
- Numeric: `start_month`, `magnitude`, `magnitude_available`.

Training uses most-frequent categorical imputation, one-hot encoding with unknown categories ignored, and numeric median imputation. These are verified software behaviors, not evidence that an incomplete live record is scientifically suitable for prediction. No change to that preprocessing is proposed here.

## 2. Feature-by-feature proposal

**Status meanings:** direct = same candidate category label without further derivation; derived = transformation with a known rule; conditionally verified = implemented and accepted only when every stated provenance and scope guard passes; incomplete = a candidate exists but required semantics or enrichment are unresolved; unavailable = no verified supplying field in the current normalized ingestion. All statuses describe the inspected implementation, not every capability GDACS may offer elsewhere.

| Historical feature and definition | GDACS candidate / path | Mapping status | Category or representation mismatch | Scientific or implementation risk | Proposed fallback when unavailable |
|---|---|---|---|---|---|
| `disaster_type`: EM-DAT type normalized using `CATEGORY_MAP` | `gdacs:eventtype` → stored `disaster_type` | **Direct** candidate for Earthquake, Flood, Wildfire and Landslide labels; **derived/incomplete** for cyclone and volcanic name normalization | `Tropical Cyclone` versus `Storm`; `Volcanic Event` versus `Volcanic activity`. Historical Landslide merges wet/dry mass movements. Unknown GDACS codes are not supported historical types. | Matching a label does not validate scope or reporting equivalence. GDACS cyclones cover a subset of historical Storm. Identical Landslide labels do not establish equivalent inclusion rules. | Preserve the source label, mark mapping unresolved, and withhold a model prediction for unknown/unsupported types. Do not replace a missing hazard with the training-mode hazard. |
| `disaster_subtype`: stripped EM-DAT `Disaster Subtype`, with missing values filled as `Unknown` | No dedicated parsed/stored GDACS field; `disaster_type` identifies a broad hazard only | **Unavailable** in normalized ingestion | No verified subtype crosswalk. In particular, a cyclone type code alone does not verify an exact historical subtype string. | Guessing subtype from title or type would create an unsupported category. Encoder acceptance of an unseen string is not semantic validation. | Represent the missing subtype as `Unknown`, consistent with historical construction, and mark subtype resolution incomplete. Verify training-vocabulary handling from training metadata before any future use; `Unknown` may be unseen and encoded as all zeros. Do not infer a subtype from prose. |
| `iso`: stripped historical `ISO` value | Exactly one nonempty `raw_payload["gdacs_country_occurrences"]` entry → conservative exact country-name normalization → unique row in `docs/emdat_country_geography_crosswalk.csv` | **Derived and conditionally verified**; no direct normalized ISO field | Country-name text is not the historical ISO code string. Missing, repeated, multiple, ambiguous, unmatched, malformed and legacy provenance cases are rejected. Territories are not reassigned to administering countries. | A GDACS event is not necessarily a single EM-DAT country-event record. Accepting the display `event.country`, coordinates or a deduplicated list would weaken source-scope evidence. | Leave ISO, region and subregion unresolved together and withhold prediction. Do not fall back to `event.country`, a training-mode country or coordinates. |
| `region`: stripped historical `Region` value | Same unique EM-DAT country-geography crosswalk row used for ISO | **Derived and conditionally verified** with ISO and subregion | The crosswalk supplies the exact historical tuple; the mapper does not translate through a separate generic continent hierarchy. | Independent lookup or partial assignment could create a tuple not observed in the historical geography contract. | Populate only as part of the unique atomic ISO/region/subregion tuple; otherwise leave all three unresolved. |
| `subregion`: stripped historical `Subregion` value | Same unique EM-DAT country-geography crosswalk row used for ISO | **Derived and conditionally verified** with ISO and region | Territory and historical-code treatment is limited to the reviewed crosswalk; no administrative-country substitution is allowed. | Independent guesses can produce inconsistent `iso`/`region`/`subregion` combinations. Coordinates do not establish affected-country scope. | Populate only as part of the unique atomic ISO/region/subregion tuple; otherwise leave all three unresolved. |
| `start_month`: numeric historical `Start Month` | Exactly one nonempty `raw_payload["gdacs_fromdate_occurrences"]` entry, parsed with an explicit zero UTC/GMT offset, then converted to month | **Derived and conditionally verified only for Earthquake and Landslide**; unresolved for Flood, Storm, Volcanic activity and Wildfire | `event_time` may fall back to `pubDate` and is therefore forbidden as a model-feature source. Missing, repeated, multiple, malformed, timezone-free and non-zero-offset provenance is rejected. | Long-duration and evolving hazards need separate temporal semantics. Even an accepted source timestamp does not remove subtype, geography, magnitude, target-coverage or model-availability guards. | Populate `start_month` and `season` together only when every provenance and hazard guard passes. Otherwise leave both unresolved; never use `event_time`, flattened `raw_payload["fromdate"]`, `pubDate`, fetch/ingestion time or current time. |
| `season`: historical calendar grouping from start month, with `Unknown` for missing/unmapped month | Accepted `start_month` above → historical `SEASON_MAP` | **Derived and conditionally verified only for Earthquake and Landslide**, atomically with `start_month`; unresolved otherwise | Exact categories are `DJF`, `MAM`, `JJA`, `SON`, not local wet/dry or hemisphere-adjusted seasons. | A wrong start month propagates into season. Calendar grouping does not measure local climate, and deriving it from a rejected or imputed date would create false provenance. | Derive only from an accepted start month. Otherwise leave both date features unresolved; do not use `Unknown` to bypass the date guard for live prediction. |
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
- Geography is accepted only when exactly one nonempty preserved GDACS country occurrence resolves to one unique tuple in the EM-DAT country-geography crosswalk. Populate ISO, region and subregion atomically; otherwise retain the monitoring event and withhold prediction.
- Missing subtype remains unresolved. `Unknown` follows historical dataset construction as a placeholder, but does not establish a scientifically compatible live value and must not bypass the prediction guard.
- Accept `start_month` and derive `season` only for Earthquake and Landslide when exactly one nonempty preserved `gdacs:fromdate` parses with an explicit zero UTC/GMT offset. Populate the pair atomically. Never fall back to `event_time`, the flattened `fromdate`, `pubDate`, fetch/ingestion time or current time.
- Magnitude and `magnitude_available` remain unresolved. Never replace missing magnitude with zero or treat fitted numeric imputation as evidence that the live input is scientifically compatible.
- Construct all nine features together. Check country/geographic consistency, date/season consistency, and magnitude/availability consistency before a future inference step.
- Treat zero, repeated or multiple GDACS country occurrences as unresolved scope. Do not infer or deduplicate country scope, use the display `country` as a fallback, or assign a modeled priority to a whole event using one country's identity.
- Retain the origin of event start time, original measurement units, mapping version, and missing/rejected-field reasons in proposed provenance metadata outside the nine predictor columns. These are proposed requirements, not currently implemented fields.

## 6. Final feasibility decision

**NO-GO for current live ML inference.** No supported hazard currently produces a complete, scientifically defensible nine-feature vector. Geography has a guarded conditional path, and Earthquake and Landslide additionally have a guarded conditional date path, but `disaster_subtype`, `magnitude` and `magnitude_available` remain unresolved for every supported hazard. Other scope, target-coverage and model-availability guards also continue to apply.

The saved Random Forest models remain historical validation models. Their saved validation results describe the historical temporal validation design and are not evidence of performance on live GDACS-derived features. GDACS remains a live monitoring prototype, and this project does not claim completed real-time impact prediction or operational readiness.

A future GDACS-native prediction model would require redesign and retraining around fields that can be obtained reliably at prediction time, followed by its own leakage-safe validation. This future work must not alter or be presented as an extension of the existing historical validation results.

## 7. Repository evidence

- [GDACS ingestion](../backend/app/ingestion/gdacs.py): `_DISASTER_TYPE_MAP`, `_parse_event_time`, `_parse_coordinates`, `_item_to_dict`, feed URL.
- [Stored event fields](../backend/app/database/models.py): `Event` columns and uniqueness constraint.
- [Public event schema](../backend/app/schemas/events.py): `EventOut` fields and exclusion of raw payload.
- [Event routes](../backend/app/api/routes/events.py): persistence/updating of normalized fields and responses.
- [Historical preprocessing](../ml/preprocessing/build_training_dataset.py): `CATEGORY_MAP`, `SEASON_MAP`, feature construction, magnitude handling and target coverage.
- [Training pipeline](../ml/training/train_models.py): exact feature lists and fitted imputation/encoding definitions.
- [EM-DAT country-geography crosswalk](emdat_country_geography_crosswalk.csv): authoritative project lookup used by guarded geography enrichment.
- [Saved validation reports](model_evaluation/): historical train/validation evidence; the reserved test split remains unused for this decision.

This is a code-and-repository-evidence mapping record. No fresh GDACS feed, live inference, subtype taxonomy or magnitude conversion has been claimed as verified by this update.
