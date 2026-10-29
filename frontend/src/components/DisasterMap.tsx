import { useEffect, useMemo, useState } from "react";
import L from "leaflet";
import {
  MapContainer,
  Marker,
  Popup,
  ScaleControl,
  TileLayer,
  useMap,
  useMapEvents,
  ZoomControl,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { type DisasterEvent } from "../api/client";
import { HazardIcon } from "./HazardIcon";
import { hazardDefinition, hazardKey, isPublicHazard } from "../utils/hazardMeta";
import { alertColor, formatUtc, hasCoords } from "../utils/mapUtils";

interface Props { events: DisasterEvent[] }
type MappableEvent = DisasterEvent & { latitude: number; longitude: number };
type MarkerGroup = { key: string; latitude: number; longitude: number; events: MappableEvent[]; spread: boolean };

function groupEvents(events: DisasterEvent[], zoom: number): MarkerGroup[] {
  const groups = new Map<string, MarkerGroup>();
  const scale = 256 * 2 ** zoom;
  const cellSize = zoom >= 12 ? 28 : 58;
  for (const item of events) {
    if (!hasCoords(item) || !isPublicHazard(item.disaster_type)) continue;
    const event: MappableEvent = item;
    const sine = Math.max(-0.9999, Math.min(0.9999, Math.sin((event.latitude * Math.PI) / 180)));
    const x = ((event.longitude + 180) / 360) * scale;
    const y = (0.5 - Math.log((1 + sine) / (1 - sine)) / (4 * Math.PI)) * scale;
    const key = `${Math.floor(x / cellSize)}:${Math.floor(y / cellSize)}`;
    const group = groups.get(key);
    if (group) {
      group.events.push(event);
      group.latitude = group.events.reduce((sum, item) => sum + item.latitude, 0) / group.events.length;
      group.longitude = group.events.reduce((sum, item) => sum + item.longitude, 0) / group.events.length;
      group.spread = group.events.some((item) => Math.abs(item.latitude - group.events[0].latitude) > 0.00001 || Math.abs(item.longitude - group.events[0].longitude) > 0.00001);
    } else groups.set(key, { key, latitude: event.latitude, longitude: event.longitude, events: [event], spread: false });
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

function markerIcon(group: MarkerGroup, selected: boolean, expandable: boolean): L.DivIcon {
  if (expandable) {
    return L.divIcon({
      className: "disaster-div-icon",
      iconSize: [42, 42],
      iconAnchor: [21, 21],
      html: `<span class="regional-cluster"><b>${group.events.length}</b><small>events</small></span>`,
    });
  }
  const { hazard, alertLevel } = groupAppearance(group);
  const severity = alertLevel ? alertColor({ alert_level: alertLevel } as DisasterEvent) : null;
  return L.divIcon({
    className: "disaster-div-icon",
    iconSize: [36, 36],
    iconAnchor: [18, 18],
    popupAnchor: [0, -17],
    html: `<span class="disaster-marker${selected ? " is-selected" : ""}${severity ? " has-severity" : ""}" style="--hazard-color:${hazard.color};--severity-color:${severity ?? "#66788b"}"><span class="marker-ripple ripple-one" aria-hidden="true"></span><span class="marker-ripple ripple-two" aria-hidden="true"></span><i class="marker-core-dot" aria-hidden="true"></i>${group.events.length > 1 ? `<b class="marker-count">${group.events.length}</b>` : ""}</span>`,
  });
}

function MapEvents({ events }: { events: DisasterEvent[] }) {
  const map = useMap();
  const [zoom, setZoom] = useState(map.getZoom());
  const [selected, setSelected] = useState<string | null>(null);
  useMapEvents({ zoomend: () => setZoom(map.getZoom()) });
  const groups = useMemo(() => groupEvents(events, zoom), [events, zoom]);

  return groups.map((group) => {
    const expandable = group.events.length > 1 && group.spread && zoom < 18;
    const label = expandable
      ? `${group.events.length} nearby events; activate to zoom in`
      : group.events.length === 1
        ? `${group.events[0].disaster_type}: ${group.events[0].title}`
        : `${group.events.length} events at this location`;
    return (
      <Marker
        key={`${zoom}:${group.key}`}
        position={[group.latitude, group.longitude]}
        icon={markerIcon(group, selected === group.key, expandable)}
        keyboard
        title={label}
        alt={label}
        riseOnHover
        eventHandlers={{
          click: () => {
            if (expandable) {
              const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
              map.setView([group.latitude, group.longitude], Math.min(zoom + 2, 18), { animate: !reduced });
            } else setSelected(group.key);
          },
          popupopen: () => setSelected(group.key),
          popupclose: () => setSelected(null),
        }}
      >
        {!expandable && (
          <Popup maxWidth={380} minWidth={285} className="disaster-popup" closeButton>
            <div className="popup-inner">
              {group.events.length > 1 && <div className="popup-group-head"><b>{group.events.length} co-located events</b><span>All records remain individually accessible</span></div>}
              {group.events.map((event) => <EventDetails event={event} key={event.id} />)}
              <p className="popup-evidence-note">Source observation details. Alert level appears only when supplied and recognized by GDACS.</p>
            </div>
          </Popup>
        )}
      </Marker>
    );
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
  const scopedEvents = useMemo(
    () => events.filter((event) => isPublicHazard(event.disaster_type)),
    [events],
  );
  const mappedCount = scopedEvents.filter(hasCoords).length;
  const denseMap = mappedCount > 80;
  const [motionPreference, setMotionPreference] = useState({ reduced: false, mobile: false });
  const [animationOverride, setAnimationOverride] = useState<boolean | null>(null);

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    const mobile = window.matchMedia("(max-width: 700px)");
    const update = () => setMotionPreference({ reduced: reduced.matches, mobile: mobile.matches });
    update();
    reduced.addEventListener("change", update);
    mobile.addEventListener("change", update);
    return () => {
      reduced.removeEventListener("change", update);
      mobile.removeEventListener("change", update);
    };
  }, []);

  const defaultAnimation = !denseMap && !motionPreference.mobile;
  const animationEnabled = !motionPreference.reduced && (animationOverride ?? defaultAnimation);
  const animationReason = motionPreference.reduced
    ? "Disabled by your reduced-motion preference"
    : animationEnabled
      ? "Selected or hovered markers only"
      : denseMap
        ? "Reduced by default on dense maps"
        : motionPreference.mobile
          ? "Reduced by default on mobile"
          : "Paused by user";

  return (
    <div className={`map-shell${animationEnabled ? " ripples-enabled" : ""}`} role="region" aria-label={`Interactive disaster map with ${mappedCount} mapped events`}>
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
        <MapEvents events={scopedEvents} />
      </MapContainer>
      <div className="map-motion-control">
        <button
          type="button"
          aria-pressed={animationEnabled}
          aria-label={`${animationEnabled ? "Disable" : "Enable"} decorative marker ripple animation`}
          disabled={motionPreference.reduced}
          onClick={() => setAnimationOverride(!animationEnabled)}
        >
          <span aria-hidden="true" className="motion-control-symbol"><i /></span>
          <span><b>Marker ripple {animationEnabled ? "on" : "off"}</b><small>{animationReason}</small></span>
        </button>
        <p>Visual focus effect only — not disaster radius, impact area, or forecast severity.</p>
      </div>
      {!mappedCount && <div className="map-empty-state"><HazardIcon type={null} size={22} /><span><b>No mapped records</b>Refresh GDACS or adjust the active hazard filter.</span></div>}
    </div>
  );
}
