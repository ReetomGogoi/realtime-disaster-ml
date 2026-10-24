# Local GDACS cyclone snapshot collection

This standalone research collector performs exactly one official RSS request.
It does not import database/startup code, alter deployed ingestion, run models,
or schedule recurring requests. The production parser flattens XML and the
Event model identifies an event rather than maintaining an alert-version
archive; neither is an adequate archival format. This tool uses the same
source URL and 30-second timeout while preserving the response separately.

Each retrieval creates a new UUID directory under
`datasets/raw/gdacs_snapshots/` (already Git-ignored). It contains:

- `response.bin`: exact response entity bytes received (identity encoding
  requested), including HTTP error bodies when available;
- `metadata.json`: request-start and completed-retrieval UTC times, fixed source
  URL, HTTP status, allowlisted provenance headers, response hash/size, outcome,
  and cyclone item index;
- `cyclone_*.xml`: complete serialized TC item subtrees retaining expanded
  namespaces, attributes, repeated fields and nested elements.

The original response bytes are authoritative: item serialization can rename
namespace prefixes and alter whitespace/lexical formatting. It is not an exact
original byte slice. All original timestamp fields remain in the response and
item XML. Publication, added, modified, start and end occurrences are indexed
separately as unmodified text. Observation/forecast valid time is left unresolved;
no generic timestamp is silently substituted for it.

Identical retrievals remain separate observations with identical response hashes.
Revised alert content is retained even when event/episode identifiers are reused.
The tool does not assume identifiers uniquely establish a version. Missing
identifiers or attributes do not discard TC items. Malformed XML/unsupported
DTD declarations and non-2xx responses retain bytes and explicit failure outcomes;
transport failures retain metadata without inventing response bytes. Partial
storage failures raise an error; do not treat an incomplete directory as a
successful snapshot. The auditor checks response integrity, re-extracted item
metadata and XML. Integrity success does not imply successful upstream collection;
inspect `outcome` too.

Only provenance response headers are retained; cookie/auth headers are excluded.
Do not put credentials in command arguments or archive paths. Future credential
files must stay in ignored `.env` files, never source control. Do not force-add
archives or event-level data. No raw datasets or historical records are needed
for tests or collection; this command retrieves the current feed only.

## Windows PowerShell

```powershell
cd D:\ML\realtime-disaster-ml
git fetch origin
git switch review/v2-gdacs-local-snapshot-collector
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s ml/tests -p "test_*.py" -v
python -m ml.collection.gdacs_snapshots collect
```

The collector prints the newly created directory. Audit that exact directory
with the separate offline command (replace the UUID):

```powershell
python -m ml.collection.gdacs_snapshots audit "datasets\raw\gdacs_snapshots\<UUID printed by collect>"
```

A current alert may retain an old `pubDate` while its content has been revised.
Retrieval time establishes only when this collector observed the current bytes;
it does not establish first public availability. Collection beginning now cannot
recover earlier alert versions. Even archived observation timestamps do not prove
that values were published at those times. The feed may contain forecasts or
summaries requiring source-specific valid-time interpretation.

These snapshots are not yet sufficient for training. Historical coverage,
publication/valid-time semantics, missingness, country scope, outcome linkage,
and leakage-safe evaluation must be demonstrated first. GDACS event dates and
country lists are not assumed equivalent to EM-DAT dates and affected countries.
The collector performs no reserved 2023–2025 dataset access or model evaluation.
