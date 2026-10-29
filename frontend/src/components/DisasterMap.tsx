import { useMemo, useState } from "react";
import L from "leaflet";
import {
  MapContainer,
  Marker,
  Popup,
  ScaleControl,
  TileLayer,
  ZoomControl,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { type DisasterEvent } from "../api/client";
import { HazardIcon } from "./HazardIcon";
import { hazardDefinition, hazardKey, hazardSvgMarkup } from "../utils/hazardMeta";
import { alertColor, formatUtc, hasCoords } from "../utils/mapUtils";

interface Props { events: DisasterEvent[] }
type MappableEvent = DisasterEvent & { latitude: number; longitude: number };
type MarkerGroup = { key: string; latitude: number; longitude: number; events: MappableEvent[] };

function groupEvents(events: DisasterEvent[]): MarkerGroup[] {
  const groups = new Map<string, MarkerGroup>();
  for (const event of events.filter(hasCoords)) {
    const key = `${event.latitude.toFixed(5)}:${event.longitude.toFixed(5)}`;
    const group = groups.get(key);
    if (group) group.events.push(event);
    else groups.set(key, { key, latitude: event.latitude, longitude: event.longitude, events: [event] });
  }
  return [...groups.values()];
}

function groupAppearance(group: MarkerGroup) {
  const hazardKeys = new Set(group.events.map((event) => hazardKey(event.disaster_type)));
  const type = hazardKeys.size === 1 ? group.events[0].disaster_type : null;
  const levels = new Set(group.events.map((event) => event.alert_level).filter(Boolean));
  const candidate = levels.size === 1 ? group.events.find((event) => event.alert_level)?.alert_level : null;
  const alertLevel = candidate && group.events.every((event) => event.alert_level === candidate) ? candidate : null;
  return { hazard: hazardDefinition(type), alertLevel };
}

function markerIcon(group: MarkerGroup, selected: boolean): L.DivIcon {
  const { hazard, alertLevel } = groupAppearance(group);
  const severity = alertLevel ? alertColor({ alert_level: alertLevel } as DisasterEvent) : null;
  const iconType = group.events.every((event) => hazardKey(event.disaster_type) === hazard.key)
    ? group.events[0].disaster_type
    : null;
  return L.divIcon({
    className: "disaster-div-icon",
    iconSize: [46, 46],
    iconAnchor: [23, 23],
    popupAnchor: [0, -22],
    html: `<span class="disaster-marker${selected ? " is-selected" : ""}${severity ? " has-severity" : ""}" style="--hazard-color:${hazard.color};--severity-color:${severity ?? "#66788b"}">${hazardSvgMarkup(iconType)}${group.events.length > 1 ? `<b class="marker-count">${group.events.length}</b>` : ""}</span>`,
  });
}

function EventDetails({ event }: { event: MappableEvent }) {
  const hazard = hazardDefinition(event.disaster_type);
  return (
    <article className="popup-event">
      <div className="popup-event-head">
        <span className="popup-hazard" style={{ color: hazard.color }}><HazardIcon type={event.disaster_type} /></span>
        <div><span>{event.disaster_type ?? "Unknown type"}</span><b>{event.title}</b></div>
      </div>
      <div className="popup-meta">
        <span><small>COUNTRY</small>{event.country ?? "Unavailable"}</span>
        <span><small>EVENT TIME</small>{formatUtc(event.event_time)}</span>
        <span><small>ALERT LEVEL</small>{event.alert_level ? <i className={`alert-value alert-${event.alert_level.toLowerCase()}`}>{event.alert_level}</i> : "Unavailable"}</span>
        <span><small>COORDINATES</small>{event.latitude.toFixed(4)}, {event.longitude.toFixed(4)}</span>
      </div>
      <div className="popup-freshness"><span>Retrieved</span>{formatUtc(event.fetched_at)}</div>
      {event.source_url && (
        <a href={event.source_url} target="_blank" rel="noopener noreferrer" className="popup-link">
          Open official GDACS report <span aria-hidden="true">↗</span>
        </a>
      )}
    </article>
  );
}

export default function DisasterMap({ events }: Props) {
  const groups = useMemo(() => groupEvents(events), [events]);
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <div className="map-shell" role="region" aria-label={`Interactive disaster map with ${events.filter(hasCoords).length} mapped events`}>
      <MapContainer
        center={[20, 15]}
        zoom={2.5}
        minZoom={2}
        maxZoom={18}
        worldCopyJump
        zoomControl={false}
        preferCanvas
        className="premium-disaster-map"
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors'
          maxZoom={19}
        />
        <ZoomControl position="topright" />
        <ScaleControl position="bottomright" imperial={false} />
        {groups.map((group) => {
          const label = group.events.length === 1
            ? `${group.events[0].disaster_type ?? "Unknown disaster"}: ${group.events[0].title}`
            : `${group.events.length} events at this location`;
          return (
            <Marker
              key={group.key}
              position={[group.latitude, group.longitude]}
              icon={markerIcon(group, selected === group.key)}
              keyboard
              title={label}
              alt={label}
              riseOnHover
              eventHandlers={{ click: () => setSelected(group.key), popupopen: () => setSelected(group.key), popupclose: () => setSelected(null) }}
            >
              <Popup maxWidth={380} minWidth={285} className="disaster-popup" closeButton>
                <div className="popup-inner">
                  {group.events.length > 1 && <div className="popup-group-head"><b>{group.events.length} co-located events</b><span>All records remain individually accessible</span></div>}
                  {group.events.map((event) => <EventDetails event={event} key={event.id} />)}
                  <p className="popup-evidence-note">Source observation details. Alert level appears only when supplied and recognized by GDACS.</p>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
      {!groups.length && <div className="map-empty-state"><HazardIcon type={null} size={22} /><span><b>No mapped records</b>Refresh GDACS or adjust the active hazard filter.</span></div>}
    </div>
  );
}
