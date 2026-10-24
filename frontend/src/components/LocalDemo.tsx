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
          <div className="panel-heading">
            <span className="eyebrow">
              TROPICAL CYCLONE ·{" "}
              {mode === "archive" ? "ARCHIVED" : "LATEST REFRESH"}
            </span>
            <span
              className={`badge severity-${String(row.fields.alert_level?.value ?? "").toLowerCase()}`}
            >
              {shown(row.fields.alert_level)}
            </span>
          </div>
          <h2>Cyclone {row.fields.event_id?.value ?? "ID unavailable"}</h2>
          <p className="muted">
            {row.country_occurrences.join(", ") || "Country unavailable"}
          </p>
          <div className="observation-value">
            {shown(row.fields.severity)}
            <small>GDACS source severity · original units</small>
          </div>
          <dl className="detail-grid">
            {[
              ["population_exposure", "Modelled exposure"],
              ["vulnerability", "Vulnerability"],
              ["alert_score", "Alert score"],
              ["latitude", "Latitude"],
              ["longitude", "Longitude"],
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
          <p className="notice">
            Exposure (including Pop74) is source-modelled exposure, not observed
            affected population.
          </p>
          <details>
            <summary>Observation provenance</summary>
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
      <div className="section-toolbar">
        <div role="group" aria-label="Cyclone observation source">
          <button
            aria-pressed={view === "latest"}
            onClick={() => setView("latest")}
          >
            Latest GDACS Cyclones
          </button>{" "}
          <button
            aria-pressed={view === "archive"}
            onClick={() => setView("archive")}
          >
            Archived Cyclone Observations
          </button>
        </div>
      </div>
      <div className="notice">
        Near-real-time monitoring after manual refresh, not continuous streaming
        or validated ML predictions. Population exposure is modelled exposure,
        not EM-DAT Total Affected.
      </div>
      {view === "archive" && !enabled ? (
        <LocalUnavailable />
      ) : (
        <>
          <div className="section-toolbar">
            <p className="muted">
              {view === "latest"
                ? `Last successful refresh: ${latest.data?.last_successful_refresh_utc ?? "Unavailable"}`
                : "Versioned local archive · unaffected by live refresh"}
            </p>
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
      <p role="status">{message}</p>
      <div className="notice">
        {data.cohort ?? "Only saved validation outputs are loaded."} · No
        reserved test-period results are displayed. These are saved results, not
        newly measured scores.
      </div>
      <div className="result-grid">
        {["affected_range", "impact_priority"].map((target) => (
          <article className="panel" key={target}>
            <h2>{target.replaceAll("_", " ")}</h2>
            <p className="muted">
              Accuracy <span className="key blue" /> Macro F1{" "}
              <span className="key cyan" />
            </p>
            {data.rows
              .filter((row) => row.target === target)
              .map((row) => (
                <div className="comparison" key={row.model}>
                  <h3>
                    {row.model}
                    <small>n = {row.validation_samples}</small>
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
        <details className="panel">
          <summary>Saved-result provenance</summary>
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
