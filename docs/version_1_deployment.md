# Version 1 deployment and verification

## Scope

Version 1 is deployed at <https://realtime-disaster-ml.vercel.app>. It is a live disaster-monitoring prototype composed of a React/Vite frontend and a FastAPI backend on Vercel. The deployed backend stores events in Neon PostgreSQL; local development uses SQLite by default.

The monitoring data comes from the official GDACS seven-day RSS feed. The frontend plots events with React Leaflet on OpenStreetMap tiles. It displays saved-event and mapped-event counts, event markers, event details in a popup, and a link to the corresponding official GDACS report when the source URL is present.

This deployment does not provide live ML impact predictions. Historical ML models were trained locally, but the GDACS-to-model feature mapping and live inference path are not sufficiently complete or validated. The interface therefore reports that live predictions are unavailable. The detailed feasibility decision is recorded in [Guarded live ML inference framework: local verification](live_ml_inference_verification.md).

## Verified production snapshot

The following is a dated snapshot, not a permanent event count or an uptime claim:

| Verification item | Observed result on 21 October 2026 |
|---|---|
| Production application | <https://realtime-disaster-ml.vercel.app> loaded successfully. |
| Backend health | `GET /health` returned HTTP 200 with `status: "ok"`. |
| Saved events | The API and frontend reported 391 saved events. |
| Mapped events | All 391 returned events had coordinates, and the frontend reported 391 events on the map. |
| Event popup | A marker popup displayed the stored event details. |
| Source link | The popup's **View official GDACS report** link was present and linked to the stored GDACS source URL. |

GDACS is a changing seven-day feed. A later manual refresh can change both the saved-event count and which records appear in the current feed, so `391` must be cited only as the 21 October 2026 snapshot.

## Deployment configuration

The root [`vercel.json`](../vercel.json) defines two Vercel services:

- `frontend`: rooted at `frontend/`, built as a Vite application, with an SPA fallback to `index.html`.
- `backend`: rooted at `backend/`, built as FastAPI with the entrypoint `app.main:app`.

Root rewrites send `/health`, `/events`, `/events/{event_id}`, `/sync-live-events`, `/openapi.json`, `/docs`, and `/redoc` to the backend. The final catch-all sends other paths to the frontend. This lets the production frontend use same-origin API requests.

The frontend API client follows this order:

1. Use `VITE_API_URL` when it is explicitly configured at build time.
2. In local Vite development, default to `http://127.0.0.1:8000`.
3. In production, default to an empty base URL, producing same-origin paths such as `/events`.

FastAPI permits the local Vite origins `http://localhost:5173` and `http://127.0.0.1:5173`. Production uses the same Vercel origin and routing, so browser requests do not require a separate cross-origin production API URL in the default deployment.

## Database configuration

The backend selects its database from the environment:

- On Vercel, `DATABASE_URL` is required and must identify PostgreSQL. The deployed database is Neon PostgreSQL.
- Outside Vercel, when `DATABASE_URL` is absent, the backend creates or uses `backend/data/disasters.db` through SQLite.
- A supplied `postgresql://` or `postgresql+psycopg://` URL is normalized to the installed `psycopg2` SQLAlchemy driver without changing its remaining URL content.
- Vercel supplies `VERCEL=1`; it is used by the backend to reject a missing or non-PostgreSQL hosted database configuration.

SQLite database files are ignored by Git and remain local. PostgreSQL schema changes for an existing database must be applied explicitly when required: SQLAlchemy's startup `create_all()` creates missing tables but does not alter existing column types. The repository includes the idempotent country-column widening migration at [`backend/migrations/20261020_widen_events_country_to_text.sql`](../backend/migrations/20261020_widen_events_country_to_text.sql).

## Environment variables

Configure names only through the deployment environment; never commit their values.

| Variable | Deployment use | Required? |
|---|---|---|
| `DATABASE_URL` | Neon PostgreSQL connection URL used by FastAPI. | Required on Vercel. |
| `VERCEL` | Platform marker checked by the backend. Vercel supplies it automatically. | Platform-provided. |
| `VITE_API_URL` | Optional explicit frontend API base URL. Leave unset for the verified same-origin Vercel routing. | Optional. |

No environment variable enables live ML inference. Model loading is guarded, and missing model artifacts do not cause backend startup to fail.

## GDACS refresh behavior

Refresh is manual in Version 1. Selecting **Refresh Live Events** sends `POST /sync-live-events`. The backend then:

1. fetches `https://www.gdacs.org/xml/rss_7d.xml`;
2. parses GDACS events;
3. identifies records by `(source, source_event_id)`;
4. inserts new records and refreshes mutable fields on existing records; and
5. commits the event changes and a synchronization log.

There is no scheduled task, background worker, or automatic periodic synchronization in the inspected Version 1 code. Opening the frontend performs `GET /events`; it does not automatically refresh the GDACS feed.

## Basic verification

After deployment, perform read-only checks first:

```bash
curl -i https://realtime-disaster-ml.vercel.app/health
curl -i "https://realtime-disaster-ml.vercel.app/events?limit=500"
```

Expected health behavior is HTTP 200 with `status` equal to `ok`. For the event response, verify that `total`, `limit`, `offset`, and `events` are present; the event count is expected to vary over time.

Then verify the browser interface:

1. Open <https://realtime-disaster-ml.vercel.app>.
2. Confirm the saved-event count and map render without an API error.
3. Select a marker and verify its popup fields.
4. Follow **View official GDACS report** and confirm it opens the event's stored source URL.
5. Confirm the interface says live ML predictions are not yet available.

Use **Refresh Live Events** only when a write to the configured event database is intended. A successful response reports fetched, inserted, updated, and skipped counts. This action is the deployment's manual ingestion control, not evidence of automated background synchronization or completed live ML prediction.

## Version 1 boundaries

- The deployment monitors and maps GDACS events; it is not an official warning or dispatch system.
- Historical model training and saved evaluation evidence are separate from the live monitoring path.
- The deployment does not claim validated GDACS-to-EM-DAT feature equivalence or completed live inference.
- The dated event counts above do not establish service uptime, data completeness, or future feed size.
