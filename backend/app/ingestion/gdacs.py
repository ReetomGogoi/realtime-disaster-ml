"""
GDACS live-event ingestion.

Fetches the official GDACS 7-day RSS feed, parses each <item> element
using standard-library xml.etree.ElementTree (no third-party XML deps),
and returns a list of plain dicts ready to be persisted as Event rows.

Confirmed namespace URIs (from live feed inspection)
─────────────────────────────────────────────────────
  gdacs   : http://www.gdacs.org
  geo     : http://www.w3.org/2003/01/geo/wgs84_pos#
  georss  : http://www.georss.org/georss
  dc      : http://purl.org/dc/elements/1.1/

Coordinate extraction order (confirmed against live XML)
─────────────────────────────────────────────────────────
  1. geo:Point/geo:lat  and  geo:Point/geo:long
       <geo:Point>
         <geo:lat>17.3</geo:lat>
         <geo:long>-128.3</geo:long>
       </geo:Point>

  2. georss:point  (space-separated "lat lon")
       <georss:point>17.3 -128.3</georss:point>

Event-time extraction order
────────────────────────────
  1. gdacs:fromdate  (RFC-2822, represents when the event started)
  2. pubDate         (RFC-2822, fallback — time the feed item was published)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GDACS_FEED_URL = "https://www.gdacs.org/xml/rss_7d.xml"
FETCH_TIMEOUT_SECONDS = 30

# Fully-qualified namespace URIs (as ElementTree expands them)
_NS = {
    "gdacs":  "http://www.gdacs.org",
    "geo":    "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "georss": "http://www.georss.org/georss",
    "dc":     "http://purl.org/dc/elements/1.1/",
}

# Map GDACS two-letter event codes → human-readable disaster type
_DISASTER_TYPE_MAP: dict[str, str] = {
    "EQ": "Earthquake",
    "TC": "Tropical Cyclone",
    "FL": "Flood",
    "VO": "Volcanic Event",
    "WF": "Wildfire",
    "LS": "Landslide",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(element: ET.Element | None) -> str | None:
    """Return stripped text of an element, or None if missing/empty."""
    if element is None:
        return None
    t = (element.text or "").strip()
    return t if t else None


def _float(element: ET.Element | None) -> float | None:
    """Parse a float from element text; return None on failure."""
    raw = _text(element)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_rfc2822(value: str | None) -> datetime | None:
    """Parse an RFC-2822 date string to a timezone-aware UTC datetime."""
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _parse_event_time(item: ET.Element) -> datetime | None:
    """
    Priority order:
      1. gdacs:fromdate — when the disaster event started (most accurate).
      2. pubDate        — when the feed item was published (fallback).

    Always returns a timezone-aware UTC datetime or None.
    """
    # 1. gdacs:fromdate (RFC-2822)
    from_date = _parse_rfc2822(_text(item.find("gdacs:fromdate", _NS)))
    if from_date is not None:
        return from_date

    # 2. pubDate (RFC-2822)
    pub_date = _parse_rfc2822(_text(item.find("pubDate")))
    if pub_date is not None:
        return pub_date

    return None


def _parse_coordinates(item: ET.Element) -> tuple[float | None, float | None]:
    """
    Extract (latitude, longitude) from the item using two strategies.

    Strategy 1 — nested geo:Point children (confirmed structure in live feed):
        <geo:Point>
          <geo:lat>17.3</geo:lat>
          <geo:long>-128.3</geo:long>
        </geo:Point>
        XPath: geo:Point/geo:lat  and  geo:Point/geo:long

    Strategy 2 — georss:point space-separated fallback:
        <georss:point>17.3 -128.3</georss:point>
        Value format: "latitude longitude"
    """
    # ── Strategy 1: geo:Point/geo:lat and geo:Point/geo:long ────────────
    lat = _float(item.find("geo:Point/geo:lat", _NS))
    lon = _float(item.find("geo:Point/geo:long", _NS))

    if lat is not None and lon is not None:
        return lat, lon

    # ── Strategy 2: georss:point ─────────────────────────────────────────
    georss_pt = _text(item.find("georss:point", _NS))
    if georss_pt:
        parts = georss_pt.split()
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass

    return None, None


def _item_to_dict(item: ET.Element) -> dict[str, Any]:
    """Convert one RSS <item> element into a normalised event dict."""
    # ── Core RSS fields ──────────────────────────────────────────────────
    title = _text(item.find("title")) or "Untitled"
    link  = _text(item.find("link"))

    # ── GDACS-specific fields ────────────────────────────────────────────
    event_type_raw = _text(item.find("gdacs:eventtype", _NS)) or ""
    event_type     = event_type_raw.strip().upper()
    disaster_type  = _DISASTER_TYPE_MAP.get(event_type, event_type or None)

    # Compound source_event_id: "EQ:1234567" — human-readable and unique
    gdacs_event_id = _text(item.find("gdacs:eventid", _NS)) or ""
    if event_type and gdacs_event_id:
        source_event_id = f"{event_type}:{gdacs_event_id}"
    elif gdacs_event_id:
        source_event_id = gdacs_event_id
    else:
        source_event_id = link or title

    country = _text(item.find("gdacs:country", _NS))

    # ── Geographic coordinates ───────────────────────────────────────────
    lat, lon = _parse_coordinates(item)

    # ── Raw payload: every direct child tag as {local_name: text} ────────
    raw_payload: dict[str, Any] = {}
    for child in item:
        tag = child.tag
        if tag.startswith("{"):
            tag = tag.split("}", 1)[1]
        raw_payload[tag] = child.text

    return {
        "source":          "GDACS",
        "source_event_id": source_event_id,
        "title":           title,
        "disaster_type":   disaster_type,
        "country":         country,
        "latitude":        lat,
        "longitude":       lon,
        "event_time":      _parse_event_time(item),
        "source_url":      link,
        "raw_payload":     raw_payload,
        "fetched_at":      datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_gdacs_events() -> list[dict[str, Any]]:
    """
    Asynchronously download the GDACS RSS feed and return a list of
    normalised event dicts (one per <item>).

    Raises:
        httpx.HTTPStatusError  – non-2xx response from GDACS.
        httpx.TimeoutException – request timed out.
        ET.ParseError          – malformed XML in the response.
    """
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS) as client:
        logger.info("Fetching GDACS feed: %s", GDACS_FEED_URL)
        response = await client.get(GDACS_FEED_URL)
        response.raise_for_status()

    root = ET.fromstring(response.text)

    channel = root.find("channel") or root
    items = channel.findall("item")

    logger.info("GDACS feed: %d items found", len(items))

    events: list[dict[str, Any]] = []
    for item in items:
        try:
            events.append(_item_to_dict(item))
        except Exception:
            logger.exception("Failed to parse GDACS item, skipping")

    return events
