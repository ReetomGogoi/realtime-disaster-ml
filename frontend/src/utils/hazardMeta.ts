export type HazardKey = "earthquake" | "flood" | "cyclone" | "wildfire" | "drought" | "landslide" | "volcano" | "other";
export type HazardDefinition = { key: HazardKey; label: string; color: string; path: string };

export const HAZARDS: Record<HazardKey, HazardDefinition> = {
  earthquake: { key: "earthquake", label: "Earthquake", color: "#ff6577", path: "M13 1 7.7 9.2l4.2 1.5-4.8 5.1 3.3 1.1L7 23l9.2-9.8-4-1.4L18 4.2 13 1Z" },
  flood: { key: "flood", label: "Flood", color: "#4aa8ff", path: "M2 10c2.2 0 2.2-1.6 4.4-1.6S8.6 10 10.8 10s2.2-1.6 4.4-1.6 2.2 1.6 4.4 1.6M2 15c2.2 0 2.2-1.6 4.4-1.6S8.6 15 10.8 15s2.2-1.6 4.4-1.6 2.2 1.6 4.4 1.6M4 6.2 12 1l8 5.2" },
  cyclone: { key: "cyclone", label: "Tropical Cyclone", color: "#b381ff", path: "M19.8 7.3A8.8 8.8 0 0 0 5 5.8C1.6 8.5 2.2 13.7 6 15.5c2.8 1.3 6.2-.1 6.8-2.8.4-1.8-.8-3.5-2.5-3.8-1.5-.3-2.9.7-3 2.2m-3.1 5.6A8.8 8.8 0 0 0 19 18.2c3.4-2.7 2.8-7.9-1-9.7-2.8-1.3-6.2.1-6.8 2.8-.4 1.8.8 3.5 2.5 3.8 1.5.3 2.9-.7 3-2.2" },
  wildfire: { key: "wildfire", label: "Wildfire", color: "#ff7b45", path: "M13.2 1.5c.7 4-3 5.1-2 8.4.5 1.7 2.2 2.2 3.4.9 1-1.1 1.2-2.8.4-4.1 3.7 2.7 5.3 5.5 3.7 9.5C17.3 24 6 23.7 4 16.4c-1.1-4 1.2-7.4 4.3-10-.2 2.6.6 4.1 1.8 4.2 2.1.1.3-5.3 3.1-9.1Z" },
  drought: { key: "drought", label: "Drought", color: "#e4b657", path: "M3 2h18v7H3zM3 13h7v9H3zM14 13h7v9h-7zM9 9v4m6-4v4M6 17h4m4 1h4" },
  landslide: { key: "landslide", label: "Landslide", color: "#cf9b5b", path: "M2 20 10 4l4 8 2-3 6 11H2Zm9.5-4.5 2.5 2.2 2-1.2 2.5 3.5" },
  volcano: { key: "volcano", label: "Volcanic Activity", color: "#ff9c55", path: "M2 21h20L15.5 8l-2.7 3-2.5-3L2 21Zm8.6-10.3L8 6m5.4 2.7L14 3m2.2 5.7L20 5" },
  other: { key: "other", label: "Other", color: "#8392a5", path: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 5v6m0 4h.01" },
};

export function hazardKey(type: string | null | undefined): HazardKey {
  const value = (type ?? "").trim().toLowerCase();
  if (value === "earthquake") return "earthquake";
  if (value === "flood") return "flood";
  if (["tropical cyclone", "storm", "cyclone"].includes(value)) return "cyclone";
  if (value === "wildfire") return "wildfire";
  if (value === "drought") return "drought";
  if (value === "landslide") return "landslide";
  if (["volcanic event", "volcanic activity", "volcano"].includes(value)) return "volcano";
  return "other";
}

export function hazardDefinition(type: string | null | undefined): HazardDefinition {
  return HAZARDS[hazardKey(type)];
}

export function hazardSvgMarkup(type: string | null | undefined): string {
  const hazard = hazardDefinition(type);
  return `<svg class="hazard-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${hazard.path}"/></svg>`;
}
