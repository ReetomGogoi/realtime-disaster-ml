export const PUBLIC_HAZARD_TYPES = ["Flood", "Drought", "Earthquake"] as const;
export type PublicHazard = (typeof PUBLIC_HAZARD_TYPES)[number];
export type HazardKey = "flood" | "drought" | "earthquake" | "other";
export type HazardDefinition = { key: HazardKey; label: string; color: string; path: string };

export const HAZARDS: Record<HazardKey, HazardDefinition> = {
  flood: {
    key: "flood",
    label: "Flood",
    color: "#4aa8ff",
    path: "M2 9.5c2.2 0 2.2-1.7 4.4-1.7s2.2 1.7 4.4 1.7 2.2-1.7 4.4-1.7 2.2 1.7 4.4 1.7M2 14.5c2.2 0 2.2-1.7 4.4-1.7s2.2 1.7 4.4 1.7 2.2-1.7 4.4-1.7 2.2 1.7 4.4 1.7",
  },
  drought: {
    key: "drought",
    label: "Drought",
    color: "#e5ad43",
    path: "M12 7.5a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9ZM12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8",
  },
  earthquake: {
    key: "earthquake",
    label: "Earthquake",
    color: "#ff824d",
    path: "M2 12h4l2.2-6 3.8 12 3-9 2.2 3H22",
  },
  other: {
    key: "other",
    label: "Unsupported hazard",
    color: "#7d8d9d",
    path: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 5v6m0 4h.01",
  },
};

export function isPublicHazard(type: string | null | undefined): type is PublicHazard {
  return PUBLIC_HAZARD_TYPES.includes(type as PublicHazard);
}

export function hazardKey(type: string | null | undefined): HazardKey {
  if (type === "Flood") return "flood";
  if (type === "Drought") return "drought";
  if (type === "Earthquake") return "earthquake";
  return "other";
}

export function hazardDefinition(type: string | null | undefined): HazardDefinition {
  return HAZARDS[hazardKey(type)];
}

export function hazardSvgMarkup(type: string | null | undefined): string {
  const hazard = hazardDefinition(type);
  return `<svg class="hazard-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${hazard.path}"/></svg>`;
}
