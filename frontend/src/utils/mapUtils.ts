import { type DisasterEvent } from "../api/client";

/** Colour for each disaster type marker */
export const DISASTER_COLORS: Record<string, string> = {
  Earthquake:        "#ff4d4d",
  "Tropical Cyclone":"#a855f7",
  Flood:             "#3b82f6",
  "Volcanic Event":  "#f97316",
  Wildfire:          "#ef4444",
  Landslide:         "#a16207",
};

export const DEFAULT_MARKER_COLOR = "#6b7280"; // grey for unknown types

export function markerColor(event: DisasterEvent): string {
  return DISASTER_COLORS[event.disaster_type ?? ""] ?? DEFAULT_MARKER_COLOR;
}

export function formatUtc(iso: string | null): string {
  if (!iso) return "—";
  try {
    // SQLite returns naive ISO strings (no Z or ±HH:MM offset).
    // JavaScript's Date parser treats those as *local* time, which shifts
    // the value by the browser's UTC offset (e.g. IST = +05:30 → -5h30m).
    // Fix: append Z when no offset marker is present so the string is
    // always parsed as UTC.
    const hasOffset = /[Z+\-]\d*:?\d*$/.test(iso.trim());
    const normalized = hasOffset ? iso : `${iso}Z`;
    const d = new Date(normalized);
    if (isNaN(d.getTime())) return iso;           // unrecognised format — show raw
    return d.toUTCString().replace(" GMT", " UTC");
  } catch {
    return iso;
  }
}

export function hasCoords(e: DisasterEvent): e is DisasterEvent & { latitude: number; longitude: number } {
  return e.latitude !== null && e.longitude !== null;
}
