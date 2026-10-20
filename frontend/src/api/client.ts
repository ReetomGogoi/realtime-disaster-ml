// API base URL – override via VITE_API_URL env var if needed
export const API_BASE =
  import.meta.env.VITE_API_URL ??
  (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");

// ── Types ────────────────────────────────────────────────────────────────────

export interface DisasterEvent {
  id: number;
  source: string;
  source_event_id: string;
  title: string;
  disaster_type: string | null;
  country: string | null;
  latitude: number | null;
  longitude: number | null;
  event_time: string | null;
  source_url: string | null;
  fetched_at: string;
  created_at: string;
}

export interface EventListResponse {
  total: number;
  limit: number;
  offset: number;
  events: DisasterEvent[];
}

export interface SyncResult {
  status: string;
  fetched_count: number;
  inserted_count: number;
  updated_count: number;
  skipped_count: number;
}

// ── API functions ─────────────────────────────────────────────────────────────

export async function fetchEvents(limit = 500): Promise<EventListResponse> {
  const res = await fetch(`${API_BASE}/events?limit=${limit}`);
  if (!res.ok) throw new Error(`GET /events failed: ${res.status} ${res.statusText}`);
  return res.json() as Promise<EventListResponse>;
}

export async function syncLiveEvents(): Promise<SyncResult> {
  const res = await fetch(`${API_BASE}/sync-live-events`, { method: "POST" });
  if (!res.ok) throw new Error(`POST /sync-live-events failed: ${res.status} ${res.statusText}`);
  return res.json() as Promise<SyncResult>;
}
