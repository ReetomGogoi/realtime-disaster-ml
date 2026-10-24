import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  API_BASE,
  syncLiveEvents,
  type DisasterEvent,
  type EventListResponse,
} from "./api/client";
import DisasterMap from "./components/DisasterMap";
import MapLegend from "./components/MapLegend";
import {
  CycloneIntelligence,
  ModelPerformance,
  Skeleton,
} from "./components/LocalDemo";
import { formatUtc, hasCoords, markerColor } from "./utils/mapUtils";

const pages = [
  "Dashboard",
  "Disaster Explorer",
  "Cyclone Intelligence",
  "Model Performance",
  "About",
] as const;
const icons = ["◫", "◎", "◉", "▥", "ⓘ"];
type Page = (typeof pages)[number];
const localEnabled =
  import.meta.env.DEV && import.meta.env.VITE_ENABLE_LOCAL_ML_DEMO === "1";
async function loadEvents(): Promise<EventListResponse> {
  const records: DisasterEvent[] = [];
  let total = 0;
  do {
    const response = await fetch(
      `${API_BASE}/events?limit=500&offset=${records.length}`,
    );
    if (!response.ok)
      throw new Error(
        `Event service returned ${response.status}. Check backend availability.`,
      );
    const page: EventListResponse = await response.json();
    total = page.total;
    records.push(...page.events);
    if (page.events.length === 0) break;
  } while (records.length < total);
  return { total, limit: records.length, offset: 0, events: records };
}
export default function App() {
  const [page, setPage] = useState<Page>(
    new URLSearchParams(window.location.search).get("demo") === "1"
      ? "Cyclone Intelligence"
      : "Dashboard",
  );
  const [menu, setMenu] = useState(false),
    [search, setSearch] = useState(""),
    [hazard, setHazard] = useState(""),
    [country, setCountry] = useState("");
  const [selected, setSelected] = useState<DisasterEvent | null>(null),
    [message, setMessage] = useState("");
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ["dashboard-events"],
    queryFn: loadEvents,
    staleTime: 60000,
    retry: 1,
  });
  const sync = useMutation({
    mutationFn: syncLiveEvents,
    onSuccess: (result) => {
      setMessage(
        `Refresh complete: ${result.fetched_count} fetched · ${result.inserted_count} new · ${result.updated_count} updated`,
      );
      void client.invalidateQueries({ queryKey: ["dashboard-events"] });
      void client.invalidateQueries({ queryKey: ["latest-cyclones"] });
    },
    onError: (error) => setMessage(`Refresh failed: ${error.message}`),
  });
  const events = query.data?.events ?? [];
  const filtered = events.filter(
    (event) =>
      (!hazard || event.disaster_type === hazard) &&
      (!country || event.country === country) &&
      `${event.title} ${event.country ?? ""} ${event.source_event_id}`
        .toLowerCase()
        .includes(search.toLowerCase()),
  );
  const types = [
    ...new Set(
      events.map((e) => e.disaster_type).filter((v): v is string => Boolean(v)),
    ),
  ].sort();
  const countries = [
    ...new Set(
      events.map((e) => e.country).filter((v): v is string => Boolean(v)),
    ),
  ].sort();
  const counts = types
    .map((type) => ({
      type,
      count: events.filter((e) => e.disaster_type === type).length,
    }))
    .sort((a, b) => b.count - a.count);
  const recent = [...filtered].sort(
    (a, b) => Date.parse(b.fetched_at) - Date.parse(a.fetched_at),
  );
  const mapped = events.filter(hasCoords).length;
  const total = query.data?.total;
  function navigate(next: Page) {
    setPage(next);
    setMenu(false);
  }
  function eventList(list: DisasterEvent[]) {
    return (
      <div className="event-list">
        {list.map((event) => (
          <button
            className="event-row"
            key={event.id}
            onClick={() => setSelected(event)}
          >
            <span className="hazard-icon" style={{ color: markerColor(event) }}>
              ◎
            </span>
            <span>
              <b>{event.title}</b>
              <small>
                {event.country ?? "Country unavailable"} ·{" "}
                {event.disaster_type ?? "Unknown type"}
              </small>
              <small>Event time: {formatUtc(event.event_time)}</small>
            </span>
            <span className="row-arrow">↗</span>
          </button>
        ))}
        {list.length === 0 && (
          <div className="empty">
            No matching records. Adjust filters or manually refresh the feed.
          </div>
        )}
      </div>
    );
  }
  return (
    <div className="command-center">
      <a className="skip" href="#content">
        Skip to content
      </a>
      <aside className={`sidebar ${menu ? "open" : ""}`}>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            navigate("Dashboard");
          }}
        >
          <span className="brand-mark">◈</span>
          <span>
            DISASTER<span className="brand-light">INTELLIGENCE</span>
          </span>
        </a>
        <div className="nav-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {pages.map((name, index) => (
            <button
              key={name}
              className={page === name ? "nav-item active" : "nav-item"}
              aria-current={page === name ? "page" : undefined}
              onClick={() => navigate(name)}
            >
              <span>{icons[index]}</span>
              {name}
              {index === 2 && <small>ARCHIVE</small>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <span className="status-dot" /> Official GDACS source
          <p>Academic research platform</p>
          <small>
            Human-reviewed decision support.
            <br />
            Not an official warning system.
          </small>
        </div>
      </aside>
      {menu && (
        <button
          className="mobile-shade"
          aria-label="Close navigation"
          onClick={() => setMenu(false)}
        />
      )}
      <div className="workspace">
        <header className="topbar">
          <div>
            <button
              className="menu-toggle"
              aria-label="Toggle navigation"
              onClick={() => setMenu(!menu)}
            >
              ☰
            </button>
            <span className="breadcrumb">
              WORKSPACE <span>/ {page}</span>
            </span>
          </div>
          <div className="topbar-right">
            <span className="connection">
              <span
                className={`status-dot ${query.isError ? "offline" : ""}`}
              />
              {query.isError
                ? "Service unavailable"
                : query.isLoading
                  ? "Connecting"
                  : "Monitoring connected"}
            </span>
            <span className="research-badge">RESEARCH · V2</span>
          </div>
        </header>
        <main id="content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">GLOBAL DISASTER INTELLIGENCE</div>
              <h1>
                {page === "Dashboard"
                  ? "A clearer picture. A better response."
                  : page}
              </h1>
              <p>
                {page === "Dashboard"
                  ? "Monitor official events, explore evidence, and understand the limits of prediction."
                  : page === "Disaster Explorer"
                    ? "Search saved official records. Every event links back to its source."
                    : page === "Cyclone Intelligence"
                      ? "Source observations with transparent units and collection provenance."
                      : page === "Model Performance"
                          ? "Evidence-led evaluation. Saved validation results, clearly scoped."
                          : "The data, methodology, and boundaries behind this project."}
              </p>
            </div>
            {["Dashboard", "Disaster Explorer"].includes(page) && (
              <button
                className="primary"
                disabled={sync.isPending || query.isFetching}
                onClick={() => sync.mutate()}
              >
                {sync.isPending ? "Refreshing…" : "↻ Refresh GDACS"}
              </button>
            )}
          </div>
          {message && (
            <div className="notice" role="status">
              {message}
            </div>
          )}
          {["Dashboard", "Disaster Explorer"].includes(page) && (
            <>
              {query.isError ? (
                <div className="panel empty" role="alert">
                  <h2>Event service unavailable</h2>
                  <p>{query.error.message}</p>
                  <button onClick={() => void query.refetch()}>
                    Try again
                  </button>
                </div>
              ) : query.isLoading ? (
                <Skeleton />
              ) : (
                <>
                  {page === "Dashboard" && (
                    <>
                      <div className="kpi-grid">
                        {[
                          [
                            "Saved events",
                            total ?? "—",
                            "Official records in storage",
                            "◫",
                          ],
                          [
                            "Mapped events",
                            mapped,
                            "Records with coordinates",
                            "◎",
                          ],
                          [
                            "Hazard categories",
                            types.length,
                            "Across loaded saved events",
                            "◈",
                          ],
                          [
                            "Alert severity",
                            "Not supplied",
                            "Saved event API has no alert-level field",
                            "◉",
                          ],
                        ].map(([label, value, note, icon]) => (
                          <article className="kpi" key={label}>
                            <div className="kpi-top">
                              <span>{label}</span>
                              <span>{icon}</span>
                            </div>
                            <strong>{value}</strong>
                            <small>{note}</small>
                          </article>
                        ))}
                      </div>
                      <div className="dashboard-grid">
                        <section className="panel map-panel">
                          <div className="panel-heading">
                            <div>
                              <span className="eyebrow">
                                GEOSPATIAL OVERVIEW
                              </span>
                              <h2>Global event monitor</h2>
                            </div>
                            <span className="badge">{mapped} mapped</span>
                          </div>
                          <div className="map-canvas">
                            <DisasterMap events={events} />
                            <MapLegend />
                          </div>
                          <div className="map-caption">
                            OpenStreetMap · Marker colors indicate hazard type,
                            not alert severity.
                          </div>
                        </section>
                        <section className="panel distribution">
                          <span className="eyebrow">EVENT COMPOSITION</span>
                          <h2>Hazard distribution</h2>
                          <p className="muted">
                            Actual loaded records · not a risk forecast
                          </p>
                          {counts.map(({ type, count }) => (
                            <div className="distribution-row" key={type}>
                              <div>
                                <span>{type}</span>
                                <b>{count}</b>
                              </div>
                              <div className="track">
                                <div
                                  className="bar"
                                  style={{
                                    width: `${events.length ? (count / events.length) * 100 : 0}%`,
                                    background: markerColor({
                                      disaster_type: type,
                                    } as DisasterEvent),
                                  }}
                                />
                              </div>
                            </div>
                          ))}
                          {!events.length && (
                            <p className="empty">
                              Refresh GDACS to populate the monitor.
                            </p>
                          )}
                          <div className="insight">
                            <span>↗</span>
                            <p>
                              <b>Know the distinction.</b> Monitoring shows
                              source observations. Live impact predictions
                              remain unvalidated.
                            </p>
                          </div>
                        </section>
                      </div>
                      <section className="panel">
                        <div className="panel-heading">
                          <div>
                            <span className="eyebrow">
                              LATEST SAVED OBSERVATIONS
                            </span>
                            <h2>Recent disaster records</h2>
                          </div>
                          <button onClick={() => navigate("Disaster Explorer")}>
                            Explore all →
                          </button>
                        </div>
                        {eventList(recent.slice(0, 6))}
                      </section>
                    </>
                  )}
                  {page === "Disaster Explorer" && (
                    <>
                      <div className="panel filters">
                        <label>
                          Search records
                          <input
                            type="search"
                            placeholder="Event, country, or source ID…"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                          />
                        </label>
                        <label>
                          Hazard
                          <select
                            value={hazard}
                            onChange={(e) => setHazard(e.target.value)}
                          >
                            <option value="">All hazards</option>
                            {types.map((type) => (
                              <option key={type}>{type}</option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Country text
                          <select
                            value={country}
                            onChange={(e) => setCountry(e.target.value)}
                          >
                            <option value="">All reported countries</option>
                            {countries.map((value) => (
                              <option key={value}>{value}</option>
                            ))}
                          </select>
                        </label>
                        <button
                          onClick={() => {
                            setSearch("");
                            setHazard("");
                            setCountry("");
                          }}
                        >
                          Reset
                        </button>
                      </div>
                      <div className="section-toolbar">
                        <p>{filtered.length} matching records</p>
                        <span className="badge">
                          Severity unavailable in saved-event API
                        </span>
                      </div>
                      <section className="panel">{eventList(recent)}</section>
                    </>
                  )}
                </>
              )}
            </>
          )}
          {page === "Cyclone Intelligence" && (
            <CycloneIntelligence apiBase={API_BASE} enabled={localEnabled} />
          )}
          {page === "Model Performance" && (
            <ModelPerformance apiBase={API_BASE} enabled={localEnabled} />
          )}
          {page === "About" && (
            <>
              <div className="about-hero panel">
                <span className="eyebrow">
                  REAL-TIME MULTI-DISASTER IMPACT PRIORITY PREDICTION
                </span>
                <h2>
                  Intelligence grounded
                  <br />
                  in transparent evidence.
                </h2>
                <p>
                  We connect official disaster monitoring with historical
                  machine-learning research, while keeping observations,
                  predictions, and validation evidence distinct.
                </p>
              </div>
              <div className="about-grid">
                {[
                  [
                    "GDACS",
                    "Live monitoring",
                    "Official alerts power the saved-event map. Archived cyclone snapshots preserve original response bytes and source attributes.",
                  ],
                  [
                    "EM-DAT",
                    "Historical outcomes",
                    "Country-event records define affected-range and impact-priority research labels. They are not automatically interchangeable with GDACS country lists.",
                  ],
                  [
                    "IBTrACS",
                    "Cyclone research",
                    "Historical cyclone tracks enrich retrospective experiments. Best-track timestamps do not prove real-time publication availability.",
                  ],
                ].map(([name, role, text]) => (
                  <article className="panel" key={name}>
                    <span className="eyebrow">{role}</span>
                    <h2>{name}</h2>
                    <p>{text}</p>
                  </article>
                ))}
              </div>
              <section className="panel">
                <h2>Temporal evaluation protocol</h2>
                <div className="timeline">
                  <div>
                    <b>2000–2019</b>
                    <span>Training only</span>
                  </div>
                  <div>
                    <b>2020–2022</b>
                    <span>Validation</span>
                  </div>
                  <div>
                    <b>2023–2025</b>
                    <span>Reserved · no evaluation in this dashboard</span>
                  </div>
                </div>
                <p>
                  Reserved-period results are not loaded or published here. No
                  model training or final evaluation occurs through this
                  interface.
                </p>
                <div className="notice">
                  Operational feature availability, historical magnitude timing,
                  country-impact scope and live source equivalence remain
                  unverified. This is an academic research platform, not an
                  official warning or dispatch system.
                </div>
              </section>
            </>
          )}
          <footer>
            GDACS source data · OpenStreetMap contributors{" "}
            <span>OBSERVATIONS ≠ VALIDATED LIVE PREDICTIONS</span>
          </footer>
        </main>
      </div>
      {selected && (
        <div className="modal-shade" onClick={() => setSelected(null)}>
          <section
            className="detail-drawer"
            role="dialog"
            aria-modal="true"
            aria-label="Disaster details"
            onKeyDown={(event) => {
              if (event.key === "Escape") setSelected(null);
              if (event.key === "Tab") {
                const elements =
                  event.currentTarget.querySelectorAll<HTMLElement>(
                    "button, a[href]",
                  );
                const first = elements[0],
                  last = elements[elements.length - 1];
                if (event.shiftKey && document.activeElement === first) {
                  event.preventDefault();
                  last?.focus();
                } else if (!event.shiftKey && document.activeElement === last) {
                  event.preventDefault();
                  first?.focus();
                }
              }
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              autoFocus
              className="close"
              aria-label="Close details"
              onClick={() => setSelected(null)}
            >
              ✕
            </button>
            <span className="eyebrow">OFFICIAL SAVED RECORD</span>
            <h2>{selected.title}</h2>
            <span className="badge">
              {selected.disaster_type ?? "Unknown hazard"}
            </span>
            <dl className="detail-grid">
              {[
                ["Country", selected.country],
                ["Event time", formatUtc(selected.event_time)],
                ["Retrieved", formatUtc(selected.fetched_at)],
                ["Source ID", selected.source_event_id],
                [
                  "Coordinates",
                  hasCoords(selected)
                    ? `${selected.latitude}, ${selected.longitude}`
                    : "Unavailable",
                ],
                ["Alert severity", "Not provided by this API"],
              ].map(([key, value]) => (
                <div key={key}>
                  <dt>{key}</dt>
                  <dd>{value ?? "Unavailable"}</dd>
                </div>
              ))}
            </dl>
            {selected.source_url &&
              /^https?:\/\//i.test(selected.source_url) && (
                <a
                  className="button primary"
                  href={selected.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  View official GDACS report ↗
                </a>
              )}
            <div className="notice">
              This record is a monitoring observation. No validated live ML
              prediction is available.
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
