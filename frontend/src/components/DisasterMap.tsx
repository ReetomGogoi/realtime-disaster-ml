import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { type DisasterEvent } from "../api/client";
import { markerColor, formatUtc, hasCoords } from "../utils/mapUtils";

interface Props {
  events: DisasterEvent[];
}

export default function DisasterMap({ events }: Props) {
  const mappable = events.filter(hasCoords);

  return (
    <MapContainer
      center={[20.5937, 78.9629]} // India as default centre
      zoom={3}
      minZoom={2}
      worldCopyJump={true}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors'
        maxZoom={19}
      />

      {mappable.map((ev) => (
        <CircleMarker
          key={ev.id}
          center={[ev.latitude, ev.longitude]}
          radius={7}
          pathOptions={{
            color: markerColor(ev),
            fillColor: markerColor(ev),
            fillOpacity: 0.85,
            weight: 1.5,
          }}
        >
          <Popup maxWidth={320} className="disaster-popup">
            <div className="popup-inner">
              <div className="popup-type" style={{ color: markerColor(ev) }}>
                {ev.disaster_type ?? "Unknown Type"}
              </div>
              <div className="popup-title">{ev.title}</div>

              <table className="popup-table">
                <tbody>
                  <tr>
                    <td>Country</td>
                    <td>{ev.country ?? "—"}</td>
                  </tr>
                  <tr>
                    <td>Event time</td>
                    <td>{formatUtc(ev.event_time)}</td>
                  </tr>
                  <tr>
                    <td>Fetched at</td>
                    <td>{formatUtc(ev.fetched_at)}</td>
                  </tr>
                  <tr>
                    <td>Coordinates</td>
                    <td>
                      {ev.latitude?.toFixed(4)}, {ev.longitude?.toFixed(4)}
                    </td>
                  </tr>
                </tbody>
              </table>

              {ev.source_url && (
                <a
                  href={ev.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="popup-link"
                >
                  View official GDACS report ↗
                </a>
              )}

              <div className="popup-ml-note">
                🤖 Historical ML models have been trained locally. Live impact predictions are not yet enabled because real-time feature mapping and inference validation remain incomplete.
              </div>
            </div>
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}
