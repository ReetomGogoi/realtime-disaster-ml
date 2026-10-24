# Local two-part demonstration

The opt-in demo displays archived GDACS observations, separate from a manual
retrospective historical-model form. Exposure (including Pop74) is source-modeled
exposure, not observed Total Affected. Retrieval time does not establish original
availability. The UI shows source, response hash, timestamps and missing values.

The form accepts exactly the original nine predictors and never copies values
from the GDACS cards. Use one authorized train/validation historical record only;
never reserved test records. No training or live GDACS mapping occurs.

Trusted artifacts must already exist at
`backend/models/affected_range_model.joblib` and
`backend/models/impact_priority_model.joblib`. Existing lifespan loading validates
schema, fitted components and sklearn compatibility. Absent or rejected models
return model_unavailable with null labels. Never load untrusted joblib files.
Probabilities are model class probabilities, not operational confidence.

Backend routes are mounted only with ENABLE_LOCAL_ML_DEMO=1 and no VERCEL flag.
Each request additionally checks loopback client and flags. Run only on 127.0.0.1;
do not expose through proxies or tunnels. Frontend rendering requires Vite dev
mode, VITE_ENABLE_LOCAL_ML_DEMO=1 and `?demo=1`. The ordinary monitoring page is
unchanged at `/`; the demo is not rendered by production builds.

## Windows PowerShell

Repository preparation:

```powershell
cd D:\ML\realtime-disaster-ml
git fetch origin
git switch review/v2-local-demonstration
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

The observation API reads only
`datasets/processed/gdacs_cyclone_extraction/features.jsonl`.
If missing it returns unavailable. Use the existing verified extraction. To
create it when the output directory does not already exist:

```powershell
python -m ml.collection.gdacs_features --snapshots datasets\raw\gdacs_snapshots
```

The API validates schema, not the authenticity of generated JSON; retain and
verify the source snapshots. Loading the demo performs no network requests.

Backend terminal, with a separate ignored demo database protecting the existing SQLite:

```powershell
cd D:\ML\realtime-disaster-ml\backend
..\.venv\Scripts\Activate.ps1
$env:ENABLE_LOCAL_ML_DEMO = "1"
Remove-Item Env:VERCEL -ErrorAction SilentlyContinue
$env:DATABASE_URL = "sqlite+pysqlite:///./data/local-demo-dashboard.db"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend terminal:

```powershell
cd D:\ML\realtime-disaster-ml\frontend
npm ci
$env:VITE_ENABLE_LOCAL_ML_DEMO = "1"
$env:VITE_API_URL = "http://127.0.0.1:8000"
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173/?demo=1`; click Load local cyclone observations.
Paste an authorized historical nine-feature JSON record into the separate form.
Required keys: disaster_type, disaster_subtype, iso, region, subregion, season,
start_month, magnitude, magnitude_available. Missing magnitude is JSON null with
availability 0; month and season must agree. No example prediction is embedded.

API: GET `/health`, GET `/local-demo/cyclones`, POST
`/local-demo/historical-predict`. Predictions are computed without persistence.

## Checks and limits

```powershell
cd D:\ML\realtime-disaster-ml
$env:PYTHONPATH = "backend"
python -m unittest discover -s backend/tests -p "test_*.py" -v
python -m unittest discover -s ml/tests -p "test_*.py" -v
cd frontend
node --test tests/local-demo.test.mjs
npm run build
npm run lint
```

Tests use isolated APIs, synthetic fixtures, mocked model outputs, and React
server rendering. They do not verify the owner's five real records or artifacts,
or replace a browser walkthrough. Historical model loading must succeed locally
before actual predictions can be demonstrated. No accuracy claim is added.

Still unvalidated: live feature availability, country-impact scope, historical
source equivalence and operational predictive value. This is human-reviewed
research, not an official warning/dispatch system or completed live ML inference.

## Dashboard redesign

The review branch `review/disaster-intelligence-dashboard` adds Dashboard,
Explorer, Cyclone Intelligence, ML Prediction Lab, Model Performance and About
navigation. Use that branch instead of the original demonstration branch in the
setup commands above. The same local environment flags and ports apply.

The normal URL `/` is now the complete dashboard; `?demo=1` opens Cyclone
Intelligence directly. Research pages remain disabled outside explicitly opted-in
Vite development mode. Model Performance reads only the saved aggregate
`docs/model_evaluation/validation_metrics.json` through a guarded local endpoint.
Those results cover the historical all-hazard cohort, not the matched cyclone
subset; the page labels that distinction. No final-test files are read.

The dashboard loads saved events in pages of 500, computes actual counts, and
uses the existing map, sync route and popup reports. Marker colors represent
hazard categories. Alert severity is explicitly unavailable on ordinary saved
records because that API does not provide it. Actual source alert levels are
shown only on local cyclone observation cards. No country list or chart is
fabricated. Search and filters operate on all loaded saved events.

Frontend rendering checks: `node --test tests/local-demo.test.mjs`.
Optional browser integration check (requires Playwright Chromium and the two
local servers above): `python frontend/tests/dashboard_browser.py`. It uses the
actual API, avoids live synchronization, checks unavailable states when local
files/models are absent, and saves screenshots outside the repository.

## Latest cyclone refresh versus local archives

Branch `review/cyclone-latest-and-archive` adds two Cyclone Intelligence tabs.
Latest GDACS Cyclones uses read-only GET `/cyclones/latest` and the existing
manual POST `/sync-live-events` route. The endpoint reports the last successful
SyncLog completion separately from each observation's retrieval, publication,
modification and event timestamps. Only rows tagged as belonging to that
successful refresh are included. Older rows remain in the database and Explorer.
Failed refreshes retain the previous successful view. Empty successful refreshes
show no cyclone rows. Untagged legacy records may require another refresh.

Ingestion now retains a namespace-aware cyclone observation summary in
raw_payload, including numeric attributes and original units. This is not a full
XML archive or predictor mapping. Missing/invalid/ambiguous fields stay
unavailable. Exposure, including Pop74, is source-modelled exposure, not observed
EM-DAT Total Affected. The public endpoint exposes only an allowlisted summary;
existing event responses and database tables are unchanged. Intra-feed duplicate
records retain the existing first-occurrence policy.

Archived Cyclone Observations reads the verified local extraction through the
existing loopback/environment-gated research endpoint. It remains disabled in
production. Refreshing saved events neither collects nor modifies archive
snapshots. Latest and archive observations have separate caches. Collecting a
new versioned snapshot still requires the separate explicit collector command.

Use `git switch review/cyclone-latest-and-archive` after `git fetch origin`, then
run the backend/frontend commands above. Open Cyclone Intelligence and click
Refresh GDACS in the Latest tab. This makes a real official-feed request on your
machine. Monitoring is refreshed near-real-time data, not continuous streaming
or validated live ML inference.
