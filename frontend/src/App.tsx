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
import { ModelPerformance, Skeleton } from "./components/LocalDemo";
import { formatUtc, hasCoords, markerColor } from "./utils/mapUtils";
import { PUBLIC_HAZARD_TYPES, isPublicHazard } from "./utils/hazardMeta";

const pages = [
  "Dashboard",
  "Disaster Explorer",
  "Model Performance",
  "About",
] as const;
const icons = ["◫", "◎", "▥", "ⓘ"];
type Page = (typeof pages)[number];
const pageMeta: Record<Page, { index: string; eyebrow: string; title: string; description: string }> = {
  Dashboard: {
    index: "01",
    eyebrow: "GLOBAL OPERATIONS OVERVIEW",
    title: "Three hazards. One clear view.",
    description: "Monitor official Flood, Drought, and Earthquake observations while keeping source evidence separate from prediction.",
  },
  "Disaster Explorer": {
    index: "02",
    eyebrow: "OFFICIAL EVENT CATALOGUE",
    title: "Disaster Explorer",
    description: "Search scoped Flood, Drought, and Earthquake records with source identity, timing, geography, and report links.",
  },
  "Model Performance": {
    index: "03",
    eyebrow: "HISTORICAL VALIDATION EVIDENCE",
    title: "Model Performance",
    description: "Review saved validation results with explicit cohort scope, metric definitions, and evidence boundaries.",
  },
  About: {
    index: "04",
    eyebrow: "METHODS & DATA PROVENANCE",
    title: "Built around evidence.",
    description: "Understand the data sources, temporal protocol, research architecture, and limits of operational use.",
  },
};
const localEnabled =
  import.meta.env.DEV && import.meta.env.VITE_ENABLE_LOCAL_ML_DEMO === "1";
async function loadEvents(): Promise<EventListResponse> {
  const scopedRecords: DisasterEvent[] = [];
  let fetched = 0;
  let total = 0;
  do {
    const response = await fetch(
      `${API_BASE}/events?limit=500&offset=${fetched}`,
    );
    if (!response.ok)
      throw new Error(
        `Event service returned ${response.status}. Check backend availability.`,
      );
    const page: EventListResponse = await response.json();
    total = page.total;
    fetched += page.events.length;
    scopedRecords.push(...page.events.filter((event) => isPublicHazard(event.disaster_type)));
    if (page.events.length === 0) break;
  } while (fetched < total);
  return { total: scopedRecords.length, limit: scopedRecords.length, offset: 0, events: scopedRecords };
}
export default function App() {
  const [page, setPage] = useState<Page>("Dashboard");
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
    onSuccess: () => {
      setMessage("GDACS refresh complete. Public Flood, Drought, and Earthquake records updated.");
      void client.invalidateQueries({ queryKey: ["dashboard-events"] });
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
  const legendEvents = events.filter(
    (event) =>
      (!country || event.country === country) &&
      `${event.title} ${event.country ?? ""} ${event.source_event_id}`
        .toLowerCase()
        .includes(search.toLowerCase()),
  );
  const types = [...PUBLIC_HAZARD_TYPES];
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
  const sourceAlertLevels = events.filter((event) => event.alert_level).length;
  const total = query.data?.total;
  const mappedPercent = events.length ? Math.round((mapped / events.length) * 100) : 0;
  const latestRetrieved = recent[0]?.fetched_at;
  const activeFilterCount = [search, hazard, country].filter(Boolean).length;
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
            <span
              className="hazard-icon"
              style={{ color: markerColor(event), borderColor: `${markerColor(event)}66` }}
            >
              <i style={{ background: markerColor(event) }} />
            </span>
            <span className="event-copy">
              <span className="event-tags">
                <em>{event.disaster_type ?? "Unknown type"}</em>
                <em>{event.country ?? "Country unavailable"}</em>
              </span>
              <b>{event.title}</b>
              <small>{formatUtc(event.event_time)} · {event.source_event_id}</small>
            </span>
            <span className="row-arrow" aria-hidden="true">→</span>
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
          <span className="brand-mark" aria-hidden="true">
            <i />
            <i />
            <b>DI</b>
          </span>
          <span className="brand-copy">
            <b>DISASTER</b><span className="brand-light">INTELLIGENCE</span>
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
              <span className="nav-icon">{icons[index]}</span>
              <span className="nav-copy">
                <b>{name}</b>
                <small>{["Command overview", "Search & investigate", "Validation evidence", "Methods & provenance"][index]}</small>
              </span>
              <span className="nav-index">0{index + 1}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="source-lockup">
            <span className="status-dot" />
            <span><b>GDACS connected</b><small>Official monitoring source</small></span>
          </div>
          <div className="research-card">
            <span>RESEARCH PLATFORM</span>
            <p>Observations for human review.</p>
            <small>Not an official warning or dispatch system.</small>
          </div>
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
          <div className="topbar-context">
            <button
              className="menu-toggle"
              aria-label="Toggle navigation"
              onClick={() => setMenu(!menu)}
            >
              ☰
            </button>
            <span className="breadcrumb"><b>{pageMeta[page].index}</b> / {page}</span>
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
            <span className="topbar-separator" />
            <span className="research-badge">RESEARCH SYSTEM · V2</span>
          </div>
        </header>
        <main id="content">
          <div className="page-heading">
            <div className="heading-copy">
              <div className="eyebrow"><span>{pageMeta[page].index}</span>{pageMeta[page].eyebrow}</div>
              <h1>{pageMeta[page].title}</h1>
              <p>{pageMeta[page].description}</p>
            </div>
            {["Dashboard", "Disaster Explorer"].includes(page) && (
              <div className="heading-actions">
                <div className="last-update">
                  <span>LAST RETRIEVED</span>
                  <b>{latestRetrieved ? formatUtc(latestRetrieved) : "Unavailable"}</b>
                </div>
                <button className="primary" disabled={sync.isPending || query.isFetching} onClick={() => sync.mutate()}>
                  <span aria-hidden="true">↻</span>{sync.isPending ? "Refreshing…" : "Refresh GDACS"}
                </button>
              </div>
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
                      <div className="operations-strip">
                        <span><i className="pulse-dot" /> Monitoring workspace operational</span>
                        <span>{events.length.toLocaleString()} records loaded</span>
                        <span>{mappedPercent}% geocoded</span>
                        <span>Scope: Flood · Drought · Earthquake</span>
                      </div>
                      <div className="kpi-grid">
                        {[
                          [
                            "Saved events",
                            total ?? "—",
                            "Official records in storage",
                            "01",
                            "blue",
                          ],
                          [
                            "Mapped events",
                            mapped,
                            "Records with coordinates",
                            `${mappedPercent}%`,
                            "cyan",
                          ],
                          [
                            "Hazard categories",
                            types.length,
                            "Across loaded saved events",
                            "03",
                            "violet",
                          ],
                          [
                            "Alert severity",
                            sourceAlertLevels || "Unavailable",
                            sourceAlertLevels ? "Records with recognized GDACS alert level" : "No recognized source alert levels",
                            "—",
                            "amber",
                          ],
                        ].map(([label, value, note, icon, accent]) => (
                          <article className={`kpi accent-${accent}`} key={label}>
                            <div className="kpi-top">
                              <span>{label}</span>
                              <span>{icon}</span>
                            </div>
                            <strong>{value}</strong>
                            <div className="kpi-foot"><small>{note}</small><i /></div>
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
                              <h2>Three-hazard event monitor</h2>
                            </div>
                            <div className="map-statuses">
                              <span className="badge"><i className="status-dot" /> LIVE DATA</span>
                              <span className="badge">{mapped.toLocaleString()} MAPPED</span>
                            </div>
                          </div>
                          <div className="map-canvas">
                            <DisasterMap events={filtered} />
                            <MapLegend events={legendEvents} activeHazard={hazard} onHazardChange={setHazard} />
                            <div className="map-hud">
                              <span>VISIBLE LAYER</span>
                              <b>Flood · Drought · Earthquake</b>
                              <small>{filtered.filter(hasCoords).length.toLocaleString()} visible markers · {mappedPercent}% overall coordinate coverage</small>
                            </div>
                          </div>
                          <div className="map-caption">
                            OpenStreetMap · Icon shape identifies hazard type; marker border shows recognized GDACS alert level only.
                          </div>
                        </section>
                        <section className="panel distribution">
                          <div className="distribution-head">
                            <span className="eyebrow">EVENT COMPOSITION</span>
                            <b>{events.length.toLocaleString()}</b>
                          </div>
                          <h2>Hazard distribution</h2>
                          <p className="muted">Actual loaded records · not a risk forecast</p>
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
                      <section className="panel recent-panel">
                        <div className="panel-heading">
                          <div>
                            <span className="eyebrow">
                              LATEST SAVED OBSERVATIONS
                            </span>
                            <h2>Recent disaster records</h2>
                            <p className="muted">Newest saved monitoring observations</p>
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
                      <div className="explorer-summary">
                        <article><span>RESULTS</span><b>{filtered.length.toLocaleString()}</b><small>matching saved records</small></article>
                        <article><span>MAP COVERAGE</span><b>{mappedPercent}%</b><small>{mapped.toLocaleString()} coordinate pairs</small></article>
                        <article><span>FILTERS</span><b>{activeFilterCount || "None"}</b><small>{activeFilterCount ? "active constraints" : "showing complete catalogue"}</small></article>
                      </div>
                      <div className="panel filters">
                        <label>
                          <span>Search records</span>
                          <input
                            type="search"
                            placeholder="Event, country, or source ID…"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                          />
                        </label>
                        <label>
                          <span>Hazard category</span>
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
                          <span>Reported country</span>
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
                          Clear filters
                        </button>
                      </div>
                      <div className="section-toolbar">
                        <p>{filtered.length} matching records</p>
                        <span className="badge">Public scope · Flood, Drought, Earthquake</span>
                      </div>
                      <section className="panel explorer-results">{eventList(recent)}</section>
                    </>
                  )}
                </>
              )}
            </>
          )}
          {page === "Model Performance" && (
            <ModelPerformance apiBase={API_BASE} enabled={localEnabled} />
          )}
          {page === "About" && (
            <>
              <div className="about-hero panel">
                <div>
                  <span className="eyebrow">REAL-TIME DISASTER ML · RESEARCH V2</span>
                  <h2>Intelligence grounded<br />in transparent evidence.</h2>
                  <p>Official monitoring and historical machine-learning research share one interface while their evidence, timing, and claims remain explicitly separate.</p>
                </div>
                <div className="about-seal"><span>HUMAN</span><b>REVIEWED</b><small>DECISION SUPPORT</small></div>
              </div>
              <div className="evidence-banner">
                <span>i</span>
                <p><b>Public monitoring scope:</b> Flood, Drought, and Earthquake. Historical research models and saved validation results were not retrained for this restricted display scope.</p>
              </div>
              <div className="about-grid">
                {[
                  [
                    "GDACS",
                    "Live monitoring",
                    "Official source observations power the public Flood, Drought, and Earthquake map. Other stored hazards remain outside the displayed scope.",
                  ],
                  [
                    "EM-DAT",
                    "Historical outcomes",
                    "Country-event records define affected-range and impact-priority research labels. They are not automatically interchangeable with GDACS country lists.",
                  ],
                  [
                    "IBTrACS",
                    "Retained historical research",
                    "Historical cyclone experiments remain preserved in the repository but are not part of the public monitoring interface.",
                  ],
                ].map(([name, role, text]) => (
                  <article className="panel source-card" key={name}>
                    <span className="source-number">0{["GDACS", "EM-DAT", "IBTrACS"].indexOf(name) + 1}</span>
                    <span className="eyebrow">{role}</span>
                    <h2>{name}</h2>
                    <p>{text}</p>
                  </article>
                ))}
              </div>
              <section className="panel protocol-panel">
                <div className="panel-heading"><div><span className="eyebrow">RESEARCH DESIGN</span><h2>Temporal evaluation protocol</h2></div><span className="badge">LEAKAGE-AWARE</span></div>
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
            <div><b>DISASTER INTELLIGENCE</b><small>GDACS source data · OpenStreetMap contributors</small></div>
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
            <div className="drawer-kicker"><span className="eyebrow">OFFICIAL SAVED RECORD</span><span className="badge">{selected.source}</span></div>
            <h2>{selected.title}</h2>
            <span className="hazard-badge" style={{ color: markerColor(selected), borderColor: `${markerColor(selected)}77` }}>
              <i style={{ background: markerColor(selected) }} />
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
                ["GDACS alert level", selected.alert_level ?? "Unavailable"],
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
