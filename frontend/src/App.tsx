import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchEvents, syncLiveEvents, type SyncResult } from "./api/client";
import { hasCoords } from "./utils/mapUtils";
import DisasterMap from "./components/DisasterMap";
import MapLegend from "./components/MapLegend";
import { useState } from "react";
import LocalDemo from "./components/LocalDemo";
import { API_BASE } from "./api/client";

function MonitoringApp() {
  const queryClient = useQueryClient();
  const [syncMsg, setSyncMsg] = useState<string | null>(null);

  // ── Data fetching ────────────────────────────────────────────────────────
  const {
    data,
    isLoading,
    isError,
    error,
    isFetching,
  } = useQuery({
    queryKey: ["events"],
    queryFn: () => fetchEvents(500),
    staleTime: 60_000,
    retry: 2,
  });

  // ── Sync mutation ─────────────────────────────────────────────────────────
  const { mutate: triggerSync, isPending: isSyncing } = useMutation({
    mutationFn: syncLiveEvents,
    onSuccess: (result: SyncResult) => {
      setSyncMsg(
        `✓ Synced — fetched ${result.fetched_count}, ` +
        `inserted ${result.inserted_count}, ` +
        `updated ${result.updated_count}, ` +
        `skipped ${result.skipped_count}`
      );
      queryClient.invalidateQueries({ queryKey: ["events"] });
      setTimeout(() => setSyncMsg(null), 8000);
    },
    onError: (err: Error) => {
      setSyncMsg(`✗ Sync failed: ${err.message}`);
      setTimeout(() => setSyncMsg(null), 8000);
    },
  });

  // ── Derived counts ────────────────────────────────────────────────────────
  const events      = data?.events ?? [];
  const totalSaved  = data?.total  ?? 0;
  const mappable    = events.filter(hasCoords).length;

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="shell">
      {/* ── Header ────────────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-left">
          <div className="header-icon">🌍</div>
          <div>
            <h1 className="header-title">Realtime Disaster Monitor</h1>
            <p className="header-sub">Powered by Official GDACS Feed · OpenStreetMap</p>
          </div>
        </div>

        <div className="header-right">
          <button
            className={`btn-sync ${isSyncing ? "btn-sync--loading" : ""}`}
            onClick={() => triggerSync()}
            disabled={isSyncing || isFetching}
          >
            {isSyncing ? (
              <><span className="spinner" /> Syncing…</>
            ) : (
              <>⟳ Refresh Live Events</>
            )}
          </button>
        </div>
      </header>

      {/* ── Stat bar ──────────────────────────────────────────────────── */}
      <div className="statbar">
        <div className="stat-card">
          <div className="stat-value">{isLoading ? "—" : totalSaved.toLocaleString()}</div>
          <div className="stat-label">Total Saved Events</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{isLoading ? "—" : mappable.toLocaleString()}</div>
          <div className="stat-label">Events on Map</div>
        </div>
        <div className="stat-card">
          <div className="stat-value stat-source">GDACS</div>
          <div className="stat-label">Official Live Source</div>
        </div>
        <div className="stat-card stat-card--ml">
          <div className="stat-value stat-ml">⏳</div>
          <div className="stat-label">LIVE ML PREDICTIONS NOT YET AVAILABLE</div>
        </div>
      </div>

      {/* ── Sync feedback banner ──────────────────────────────────────── */}
      {syncMsg && (
        <div className={`sync-banner ${syncMsg.startsWith("✗") ? "sync-banner--error" : ""}`}>
          {syncMsg}
        </div>
      )}

      {/* ── Map area ──────────────────────────────────────────────────── */}
      <main className="map-area">
        {isLoading && (
          <div className="overlay">
            <div className="spinner spinner--lg" />
            <p>Loading events…</p>
          </div>
        )}

        {isError && (
          <div className="overlay overlay--error">
            <div className="overlay-icon">⚠️</div>
            <p className="overlay-title">Cannot reach the backend</p>
            <p className="overlay-msg">{(error as Error).message}</p>
            <p className="overlay-hint">
              Make sure <code>uvicorn app.main:app --reload</code> is running on port 8000.
            </p>
          </div>
        )}

        {!isLoading && !isError && events.length === 0 && (
          <div className="overlay">
            <div className="overlay-icon">🗂️</div>
            <p className="overlay-title">No events in the database yet</p>
            <p className="overlay-msg">
              Click <strong>Refresh Live Events</strong> to fetch from GDACS.
            </p>
          </div>
        )}

        {!isLoading && !isError && events.length > 0 && (
          <DisasterMap events={events} />
        )}

        {/* Legend always visible once map is shown */}
        {!isLoading && !isError && events.length > 0 && <MapLegend />}
      </main>

      {/* ── Footer ────────────────────────────────────────────────────── */}
      <footer className="footer">
        Map data © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors ·
        Disaster data © <a href="https://www.gdacs.org" target="_blank" rel="noopener noreferrer">GDACS</a>
      </footer>
    </div>
  );
}

export default function App() {
  const enabled = import.meta.env.DEV && import.meta.env.VITE_ENABLE_LOCAL_ML_DEMO === "1";
  return enabled && new URLSearchParams(window.location.search).get("demo") === "1"
    ? <div style={{height:"100vh",overflowY:"auto"}}><LocalDemo apiBase={API_BASE}/></div>
    : <MonitoringApp/>;
}
