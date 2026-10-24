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

Backend terminal, with ephemeral demo storage protecting the existing SQLite:

```powershell
cd D:\ML\realtime-disaster-ml\backend
..\.venv\Scripts\Activate.ps1
$env:ENABLE_LOCAL_ML_DEMO = "1"
Remove-Item Env:VERCEL -ErrorAction SilentlyContinue
$env:DATABASE_URL = "sqlite+pysqlite:///:memory:"
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
