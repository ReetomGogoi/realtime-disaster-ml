import { type DisasterEvent } from "../api/client";
import { hazardDefinition } from "./hazardMeta";

/** Colour for each disaster type marker */
export const DISASTER_COLORS: Record<string, string> = {
  Flood: "#4aa8ff",
  Drought: "#e4b657",
  Earthquake: "#ff824d",
};

export const DEFAULT_MARKER_COLOR = "#6b7280"; // grey for unknown types

export function markerColor(event: DisasterEvent): string {
  return hazardDefinition(event.disaster_type).color ?? DEFAULT_MARKER_COLOR;
}

export const ALERT_COLORS = {
  Green: "#48d39f",
  Orange: "#f5a84b",
  Red: "#ff5d6c",
} as const;

export function alertColor(event: DisasterEvent): string | null {
  return event.alert_level ? ALERT_COLORS[event.alert_level] : null;
}

export function formatUtc(iso: string | null): string {
  if (!iso) return "—";
  try {
    // SQLite returns naive ISO strings (no Z or ±HH:MM offset).
    // JavaScript's Date parser treats those as *local* time, which shifts
    // the value by the browser's UTC offset (e.g. IST = +05:30 → -5h30m).
    // Fix: append Z when no offset marker is present so the string is
    // always parsed as UTC.
    const hasOffset = /[Z+-]\d*:?\d*$/.test(iso.trim());
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
