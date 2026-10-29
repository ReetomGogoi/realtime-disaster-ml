import { useEffect, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

type ObservedField = {
  value: string | number | null;
  unit: string | null;
  status: string;
};
export type Cyclone = {
  snapshot_id: string;
  retrieved_at_utc: string;
  source_url: string | null;
  response_sha256?: string;
  country_occurrences: (string | null)[];
  fields: Record<string, ObservedField>;
};
const shown = (field?: ObservedField) =>
  field?.status === "available"
    ? `${field.value ?? "Unavailable"} ${field.unit ?? ""}`
    : `Unavailable (${field?.status ?? "missing"})`;
export function ObservationCards({
  records,
  mode = "archive",
}: {
  records: Cyclone[];
  mode?: "archive" | "latest";
}) {
  return (
    <div className="cyclone-grid">
      {records.length === 0 && (
        <div className="empty">
          {mode === "archive"
            ? "No local cyclone observations available. Generate and verify the local extraction first."
            : "No tropical cyclones were saved in the latest successful refresh."}
        </div>
      )}
      {records.map((row, i) => (
        <article className="panel cyclone-card" key={`${row.snapshot_id}-${i}`}>
          <div className="cyclone-card-top">
            <div className="cyclone-symbol" aria-hidden="true"><i /><i /><b>◉</b></div>
            <div>
              <span className="eyebrow">TROPICAL CYCLONE · {mode === "archive" ? "ARCHIVED" : "LATEST REFRESH"}</span>
              <h2>Cyclone {row.fields.event_id?.value ?? "ID unavailable"}</h2>
            </div>
            <span
              className={`badge severity-${String(row.fields.alert_level?.value ?? "").toLowerCase()}`}
            >
              {shown(row.fields.alert_level)}
            </span>
          </div>
          <div className="cyclone-location">
            <span>◌</span><p>{row.country_occurrences.join(", ") || "Country unavailable"}</p>
          </div>
          <div className="cyclone-hero-metrics">
            <div className="observation-value">
              <span>SEVERITY</span>
              <b>{shown(row.fields.severity)}</b>
              <small>GDACS source value · original units</small>
            </div>
            <div className="coordinate-block">
              <span>COORDINATES</span>
              <b>{shown(row.fields.latitude)}</b>
              <small>{shown(row.fields.longitude)}</small>
            </div>
          </div>
          <dl className="detail-grid cyclone-details">
            {[
              ["population_exposure", "Modelled exposure"],
              ["vulnerability", "Vulnerability"],
              ["alert_score", "Alert score"],
              ["version", "Source version"],
              ["episode_id", "Episode"],
              ["publication_time", "Published"],
              ["modification_time", "Modified"],
              ["event_start_time", "Event timing"],
            ].map(([key, label]) => (
              <div key={key}>
                <dt>{label}</dt>
                <dd>{shown(row.fields[key])}</dd>
              </div>
            ))}
          </dl>
          <p className="notice exposure-note">
            Exposure (including Pop74) is source-modelled exposure, not observed
            affected population.
          </p>
          <details className="provenance-block">
            <summary><span>Observation provenance</span><b>View source details +</b></summary>
            <p>Retrieved: {row.retrieved_at_utc}</p>
            <p>Source: {row.source_url ?? "Unavailable"}</p>
            {row.source_url && /^https?:\/\//i.test(row.source_url) && (
              <a
                href={row.source_url}
                target="_blank"
                rel="noopener noreferrer"
              >
                View GDACS source ↗
              </a>
            )}
            <p>
              {mode === "archive" ? "Snapshot" : "Refresh record"}:{" "}
              {row.snapshot_id}
            </p>
            {mode === "archive" && (
              <small>
                Response SHA-256: {row.response_sha256 ?? "Unavailable"}
              </small>
            )}
            <p>
              Retrieval establishes observation of this version, not original
              availability.
            </p>
          </details>
        </article>
      ))}
    </div>
  );
}
async function request(url: string, options?: RequestInit) {
  const response = await fetch(url, options);
  if (!response.ok)
    throw new Error(
      `Request failed (${response.status}). Check local demo settings and input values.`,
    );
  return response.json();
}
export function CycloneIntelligence({
  apiBase,
  enabled,
}: {
  apiBase: string;
  enabled: boolean;
}) {
  const [view, setView] = useState<"latest" | "archive">("latest");
  const client = useQueryClient();
  const latest = useQuery({
    queryKey: ["latest-cyclones", apiBase],
    queryFn: () => request(`${apiBase}/cyclones/latest`),
    retry: false,
  });
  const archive = useQuery({
    queryKey: ["local-cyclones", apiBase],
    queryFn: () => request(`${apiBase}/local-demo/cyclones`),
    enabled: enabled && view === "archive",
    retry: false,
  });
  const refresh = useMutation({
    mutationFn: () =>
      request(`${apiBase}/sync-live-events`, { method: "POST" }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["latest-cyclones"] });
      void client.invalidateQueries({ queryKey: ["dashboard-events"] });
    },
  });
  const query = view === "latest" ? latest : archive;
  return (
    <>
      <div className="cyclone-control-panel">
        <div>
          <span className="eyebrow">OBSERVATION SOURCE</span>
          <h2>Choose an evidence layer</h2>
          <p>Current refreshes and immutable local archives remain separate.</p>
        </div>
        <div className="segmented-control" role="group" aria-label="Cyclone observation source">
          <button
            aria-pressed={view === "latest"}
            onClick={() => setView("latest")}
          >
            <span className="status-dot" /> Latest GDACS Cyclones
          </button>{" "}
          <button
            aria-pressed={view === "archive"}
            onClick={() => setView("archive")}
          >
            <span aria-hidden="true">◫</span> Archived Cyclone Observations
          </button>
        </div>
      </div>
      <div className="evidence-banner">
        <span>i</span>
        <p><b>Observation layer only.</b> Near-real-time after manual refresh, not continuous streaming or validated ML prediction. Population exposure is modelled exposure, not EM-DAT Total Affected.</p>
      </div>
      {view === "archive" && !enabled ? (
        <LocalUnavailable />
      ) : (
        <>
          <div className="section-toolbar cyclone-toolbar">
            <div className="refresh-meta">
              <span>{view === "latest" ? "LAST SUCCESSFUL REFRESH" : "ARCHIVE POLICY"}</span>
              <b>
              {view === "latest"
                ? (latest.data?.last_successful_refresh_utc ?? "Unavailable")
                : "Versioned local archive · unaffected by live refresh"}
              </b>
            </div>
            {view === "latest" ? (
              <button
                disabled={refresh.isPending || latest.isFetching}
                onClick={() => refresh.mutate()}
              >
                {refresh.isPending ? "Refreshing…" : "↻ Refresh GDACS"}
              </button>
            ) : (
              <button
                disabled={archive.isFetching}
                onClick={() => void archive.refetch()}
              >
                ↻ Reload archive
              </button>
            )}
          </div>
          {refresh.isError && view === "latest" && (
            <div role="alert" className="notice">
              Refresh failed. The previous successful refresh remains displayed.{" "}
              {String(refresh.error)}
            </div>
          )}
          <p role="status">
            {query.error
              ? String(query.error)
              : (query.data?.reason ??
                (view === "latest"
                  ? "Saved observations from the latest successful refresh"
                  : "Verified local extraction · archived observations"))}
          </p>
          {query.isLoading ? (
            <Skeleton />
          ) : query.isError && !query.data ? (
            <div className="panel empty" role="alert">Observations could not be loaded. Retry when the service is available.</div>
          ) : (
            <ObservationCards records={query.data?.records ?? []} mode={view} />
          )}
        </>
      )}
    </>
  );
}

export function LocalUnavailable() {
  return (
    <div className="panel empty">
      <span className="empty-icon">◇</span>
      <h2>Local research workspace</h2>
      <p>
        This feature is available only in the opt-in local demonstration.
        Production monitoring remains available in Dashboard and Explorer.
      </p>
      <p>
        Enable the local backend and Vite research flags using the project
        instructions.
      </p>
      <span className="badge">Unavailable in production</span>
    </div>
  );
}
export function Skeleton() {
  return (
    <div className="skeleton-grid" aria-label="Loading data" role="status">
      {[1, 2, 3].map((i) => (
        <div className="skeleton" key={i} />
      ))}
    </div>
  );
}
export function ModelPerformance({
  apiBase,
  enabled,
}: {
  apiBase: string;
  enabled: boolean;
}) {
  type Row = {
    target: string;
    model: string;
    accuracy: number;
    macro_f1: number;
    validation_samples: number;
  };
  const [data, setData] = useState<{
    rows: Row[];
    source?: string;
    cohort?: string;
    sha256?: string;
  }>({ rows: [] });
  const [message, setMessage] = useState("Loading saved validation results…");
  useEffect(() => {
    if (enabled)
      request(`${apiBase}/local-demo/performance`)
        .then((result) => {
          setData(result);
          setMessage(result.reason ?? "Saved validation metrics · 2020–2022");
        })
        .catch((error) => setMessage(String(error)));
  }, [enabled, apiBase]);
  if (!enabled) return <LocalUnavailable />;
  return (
    <>
      <section className="performance-hero">
        <div>
          <span className="eyebrow">SAVED VALIDATION EVIDENCE</span>
          <h2>Model results, with their boundaries visible.</h2>
          <p role="status">{message}</p>
        </div>
        <div className="metric-guide">
          <span><i className="key blue" /> Accuracy</span>
          <span><i className="key cyan" /> Macro F1</span>
          <small>Validation period · 2020–2022</small>
        </div>
      </section>
      <div className="evidence-banner">
        <span>i</span>
        <p><b>Validation evidence only.</b> {data.cohort ?? "Only saved validation outputs are loaded."} No reserved test-period results are displayed. These are saved results, not newly measured scores.</p>
      </div>
      <div className="result-grid">
        {["affected_range", "impact_priority"].map((target, targetIndex) => (
          <article className="panel performance-card" key={target}>
            <div className="performance-card-head">
              <span>0{targetIndex + 1}</span>
              <div><small>CLASSIFICATION TARGET</small><h2>{target.replaceAll("_", " ")}</h2></div>
            </div>
            {data.rows
              .filter((row) => row.target === target)
              .map((row) => (
                <div className="comparison" key={row.model}>
                  <h3>
                    {row.model}
                    <small>{row.validation_samples.toLocaleString()} validation records</small>
                  </h3>
                  {(["accuracy", "macro_f1"] as const).map((metric) => (
                    <div className="bar-row" key={metric}>
                      <span>{metric.replace("_", " ")}</span>
                      <div className="track">
                        <div
                          className={
                            metric === "accuracy" ? "bar blue" : "bar cyan"
                          }
                          style={{ width: `${row[metric] * 100}%` }}
                        />
                      </div>
                      <b>{(row[metric] * 100).toFixed(2)}%</b>
                    </div>
                  ))}
                </div>
              ))}
            {!data.rows.length && (
              <p className="empty">Validation results unavailable.</p>
            )}
          </article>
        ))}
      </div>
      {data.source && (
        <details className="panel provenance-block performance-provenance">
          <summary><span>Saved-result provenance</span><b>Inspect evidence +</b></summary>
          <p>{data.source}</p>
          <small>{data.sha256}</small>
        </details>
      )}
    </>
  );
}
export default function LocalDemo({ apiBase }: { apiBase: string }) {
  return <CycloneIntelligence apiBase={apiBase} enabled />;
}
