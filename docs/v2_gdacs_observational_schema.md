# GDACS cyclone observational schema 1.0

Contract identifier: `gdacs-cyclone-observations/1.0`. The extractor is a local,
offline research tool. Production ingestion and inference are unchanged. Future
historical construction and live adapters must use this same contract or declare
a new version; it is not a fitted model feature vector.

## Acceptance and reproducibility

Only parsed version-1 collector archives from the fixed official RSS source with
2xx HTTP status are accepted. Before extraction the collector auditor checks
response byte hash/size and every re-extracted cyclone item against archived
XML and item metadata. Corrupt, incomplete, failed and malformed archives are
rejected and listed in the extraction audit. An invalid snapshot does not
prevent extracting other valid snapshots. Response hashes and metadata hashes
identify accepted inputs. These checks detect accidental corruption; they do
not authenticate local timestamps or defend against coordinated modification
of metadata and response bytes.

Outputs retain one row per observed cyclone item per retrieval. Identical item
hashes are flagged, not dropped. Reused event/version IDs do not collapse revised
content. Sorting snapshot directories makes rerunning the same inputs stable;
the duplicate flag describes extraction order, not chronological first sighting.
No publication or modification timestamp establishes original availability.
`retrieved_at_utc` means only that the archived version was observed by that
local retrieval; local clock accuracy remains an assumption.

## Record and field contract

Record-level provenance: schema version, snapshot ID, source RSS URL, retrieval
UTC time, response SHA-256, serialized-item SHA-256 and duplicate-content flag.
`fields` entries each contain:

- `source_path`: namespace-expanded XML path; `attribute` identifies the source
  attribute when applicable;
- `occurrences`: original text and all attributes for each matching element;
- `value`: parsed value or null; `unit`: original unit or null (coordinate units
  are explicit degrees from the documented coordinate representation);
- `status`: `available`, `missing`, `malformed`, or `ambiguous`;
- `failure`: parsing/missingness explanation when applicable.

`available` means parseable in this snapshot, not scientifically validated or
available at an earlier prediction cutoff. A numeric zero is available zero;
empty/missing attributes stay null, invalid/nonfinite numbers are malformed,
and repeated scalar occurrences are ambiguous even if identical. No unit
conversion or wind averaging-period normalization is performed. Unknown or
absent units do not become verified units.

| Field | Source | Interpretation |
|---|---|---|
| event_id / episode_id / version | gdacs:eventid / episodeid / version | Opaque strings; no invented IDs |
| report_url | RSS link | Original report link, never fetched |
| publication_time | RSS pubDate | Source-reported publication timestamp |
| modification_time / added_time | gdacs:datemodified / dateadded | Separate source-reported timestamps |
| event_start_time / event_end_time | gdacs:fromdate / todate | Source event timing, not EM-DAT onset equivalence |
| severity | gdacs:severity @value, @unit | Original severity quantity; no automatic EM-DAT/IBTrACS mapping |
| population_exposure | gdacs:population @value, @unit | Source-modeled exposure; preserve `Pop74` literally |
| vulnerability | gdacs:vulnerability @value, @unit if present | Source scale; absent unit remains null |
| alert_level / alert_score | gdacs:alertlevel / alertscore | Source category/score, not an outcome label |
| episode_alert_score | gdacs:episodealertscore | Source episode score |
| latitude / longitude | geo:Point geo:lat/long; GeoRSS fallback only when both absent | Finite degrees within geographic bounds |

An incomplete coordinate pair is flagged unusable as a pair. Malformed values
are not silently replaced from another representation. Original timestamp text
is retained; valid timezone-aware times are normalized to UTC. Timezone-free
values are malformed. Observation/forecast valid time remains null/unverified:
none of the event or publication times is substituted for it.

`country_occurrences` and `iso3_occurrences` preserve source order, repetition
and original text. They neither establish affected-country labels nor guarantee
country-to-ISO correspondence. `numeric_attribute_candidates` preserves every
nested element carrying a `value` attribute with its expanded tag, text and
attributes; these are unvalidated source candidates, not approved predictors.

`Pop74` is retained with its source value and text. The project has not
independently validated its complete operational definition. Never rename it
Total Affected, treat it as observed human impact, or use it as a ground-truth
label. No historical EM-DAT/IBTrACS predictor equivalence is asserted.

## Windows PowerShell

```powershell
cd D:\ML\realtime-disaster-ml
git fetch origin
git switch review/v2-gdacs-snapshot-features
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s ml/tests -p "test_*.py" -v
python -m ml.collection.gdacs_features --snapshots "datasets\raw\gdacs_snapshots"
```

Default new output directory: `datasets/processed/gdacs_cyclone_extraction/`.
It contains `features.jsonl` and `audit.json`; both are already Git-ignored.
Existing output directories are refused. Review rejection records and field
status counts before using any row. For another intentional extraction use a
new `--output` directory; never force-add outputs to Git.

This tool reads archived inputs only: no network, model fitting, target changes,
reserved test datasets or production access. Collection begun now cannot recover
old alert versions. Coverage, measurement definitions, historical availability,
country scope and outcome linkage remain prerequisites for future ML work.
