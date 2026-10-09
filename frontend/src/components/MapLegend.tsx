import { DISASTER_COLORS, DEFAULT_MARKER_COLOR } from "../utils/mapUtils";

export default function MapLegend() {
  const entries = Object.entries(DISASTER_COLORS);

  return (
    <div className="legend">
      <div className="legend-title">Disaster Types</div>
      {entries.map(([type, color]) => (
        <div key={type} className="legend-item">
          <span className="legend-dot" style={{ background: color }} />
          <span>{type}</span>
        </div>
      ))}
      <div className="legend-item">
        <span className="legend-dot" style={{ background: DEFAULT_MARKER_COLOR }} />
        <span>Other</span>
      </div>
    </div>
  );
}
