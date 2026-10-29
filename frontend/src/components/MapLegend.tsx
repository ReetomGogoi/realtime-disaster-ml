import { useMemo, useState } from "react";
import type { DisasterEvent } from "../api/client";
import { PUBLIC_HAZARD_TYPES, hazardDefinition, isPublicHazard } from "../utils/hazardMeta";
import { ALERT_COLORS, hasCoords } from "../utils/mapUtils";

type Props = {
  events: DisasterEvent[];
  activeHazard: string;
  onHazardChange: (hazard: string) => void;
};

export default function MapLegend({ events, activeHazard, onHazardChange }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const mappable = useMemo(
    () => events.filter((event) => hasCoords(event) && isPublicHazard(event.disaster_type)),
    [events],
  );
  const categories = useMemo(() => {
    return PUBLIC_HAZARD_TYPES.map((type) => [
      type,
      mappable.filter((event) => event.disaster_type === type).length,
    ] as const);
  }, [mappable]);
  const severityCounts = useMemo(() => ({
    Green: mappable.filter((event) => event.alert_level === "Green").length,
    Orange: mappable.filter((event) => event.alert_level === "Orange").length,
    Red: mappable.filter((event) => event.alert_level === "Red").length,
    unavailable: mappable.filter((event) => !event.alert_level).length,
  }), [mappable]);

  return (
    <aside className={`legend ${collapsed ? "is-collapsed" : ""}`} aria-label="Map legend and hazard filters">
      <div className="legend-title">
        <span><small>MAP LAYERS</small><b>Hazards</b></span>
        <button aria-expanded={!collapsed} aria-label={collapsed ? "Expand map legend" : "Collapse map legend"} onClick={() => setCollapsed((value) => !value)}>
          {collapsed ? "+" : "−"}
        </button>
      </div>
      {!collapsed && (
        <>
          <div className="legend-summary"><span>{mappable.length.toLocaleString()} mapped</span><span>{categories.length} categories</span></div>
          <div className="legend-categories" role="group" aria-label="Filter map by disaster type">
            {categories.map(([type, count]) => {
              const hazard = hazardDefinition(type);
              const active = activeHazard === type;
              return (
                <button
                  key={type}
                  className={active ? "active" : ""}
                  aria-pressed={active}
                  aria-label={`${active ? "Clear" : "Show"} ${type} filter, ${count} mapped events`}
                  onClick={() => onHazardChange(active ? "" : type)}
                >
                  <span className="legend-hazard-icon" style={{ color: hazard.color }} aria-hidden="true"><i /></span>
                  <span>{type}</span>
                  <b>{count}</b>
                </button>
              );
            })}
          </div>
          {activeHazard && <button className="legend-clear" onClick={() => onHazardChange("")}>Show all mapped hazards</button>}
          <div className="severity-legend">
            <span className="severity-title">GDACS ALERT LEVEL <small>marker border</small></span>
            {(Object.keys(ALERT_COLORS) as Array<keyof typeof ALERT_COLORS>).map((level) => (
              <span key={level}><i style={{ borderColor: ALERT_COLORS[level], boxShadow: `0 0 0 2px ${ALERT_COLORS[level]}22` }} />{level}<b>{severityCounts[level]}</b></span>
            ))}
            <span><i className="severity-unknown" />Unavailable<b>{severityCounts.unavailable}</b></span>
            <p>Green, Orange, and Red indicate increasing potential humanitarian impact in GDACS. A neutral border means no recognized source alert level.</p>
          </div>
        </>
      )}
    </aside>
  );
}
