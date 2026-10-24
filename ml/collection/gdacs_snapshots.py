"""One-shot local GDACS RSS archival and offline integrity audit.

No database, production ingestion, inference, training, or scheduler is used.
Exact received entity bytes are authoritative; extracted item XML is a
structurally faithful serialization, not an original byte slice.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

# Same official source and timeout as backend/app/ingestion/gdacs.py; do not
# import backend startup/database modules for a standalone offline tool.
SOURCE_URL = 'https://www.gdacs.org/xml/rss_7d.xml'
TIMEOUT = 30
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'datasets/raw/gdacs_snapshots'
G = '{http://www.gdacs.org}'
PROVENANCE_HEADERS = {'date', 'last-modified', 'etag', 'content-type',
                      'content-length', 'content-encoding', 'cache-control', 'age'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def extract_cyclones(body):
    # Reject DTD/entity declarations; retain unsafe/malformed bytes but do not
    # parse them. This collector needs only plain GDACS RSS.
    if b'<!DOCTYPE' in body.upper() or b'<!ENTITY' in body.upper():
        raise ValueError('DTD/entity declarations are not supported')
    root = ET.fromstring(body)
    if root.tag != 'rss' or root.find('channel') is None:
        raise ValueError('Expected RSS channel')
    records = []
    for item in root.findall('./channel/item'):
        types = item.findall(G + 'eventtype')
        if not any((e.text or '').strip().upper() == 'TC' for e in types):
            continue
        xml = ET.tostring(item, encoding='utf-8')
        def occurrences(tag):
            return [e.text for e in item.findall(tag)]
        records.append((xml, {
            'serialized_item_sha256': sha256(xml),
            'event_ids': occurrences(G + 'eventid'),
            'episode_ids': occurrences(G + 'episodeid'),
            'versions': occurrences(G + 'version'),
            'country_occurrences': occurrences(G + 'country'),
            'timestamps_as_reported': {
                'publication': occurrences('pubDate'),
                'added': occurrences(G + 'dateadded'),
                'modified': occurrences(G + 'datemodified'),
                'event_start': occurrences(G + 'fromdate'),
                'event_end': occurrences(G + 'todate'),
            },
            'observation_forecast_valid_time': None,
            'valid_time_status': 'not inferred; consult original XML and source documentation',
            'historical_availability': 'not inferred from pubDate',
        }))
    return records


def fetch_once():
    request = urllib.request.Request(SOURCE_URL, headers={'Accept-Encoding': 'identity'})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            try:
                body = response.read()
            except http.client.IncompleteRead as error:
                return response.status, dict(response.headers), error.partial, 'IncompleteRead'
            return response.status, dict(response.headers), body, None
    except urllib.error.HTTPError as error:
        with error:
            return error.code, dict(error.headers), error.read(), 'HTTPError'
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        # Store only the exception class, never arbitrary credential-bearing text.
        return None, {}, None, type(error).__name__


def collect_once(archive_root=DEFAULT_ROOT, fetch=fetch_once):
    started = now()
    status, headers, body, error = fetch()
    retrieved = now()
    directory = Path(archive_root) / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        'schema_version': 1, 'source_url': SOURCE_URL,
        'request_started_utc': started, 'retrieved_at_utc': retrieved,
        'http_status': status,
        'response_headers': {k.lower(): v for k, v in headers.items()
                             if k.lower() in PROVENANCE_HEADERS},
        'error_type': error, 'response_sha256': None,
        'response_size_bytes': None, 'cyclone_items': [],
        'outcome': 'transport_error' if body is None else 'http_error',
        'availability_policy': 'observed at retrieval only; no historical availability inferred',
    }
    if body is not None:
        (directory / 'response.bin').write_bytes(body)
        metadata.update(response_sha256=sha256(body), response_size_bytes=len(body))
        if error and status is not None and 200 <= status < 300:
            metadata['outcome'] = 'incomplete_response'
        elif status is not None and 200 <= status < 300:
            try:
                records = extract_cyclones(body)
                for index, (xml, record) in enumerate(records):
                    filename = f'cyclone_{index:04d}.xml'
                    (directory / filename).write_bytes(xml)
                    metadata['cyclone_items'].append({'file': filename, **record})
                metadata['outcome'] = 'parsed'
            except (ET.ParseError, ValueError):
                metadata['outcome'] = 'malformed_xml'
    # Every retrieval has its own directory, even for identical responses.
    with (directory / 'metadata.json').open('x', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return directory


def audit_snapshot(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    errors = []
    if metadata['response_sha256'] is not None:
        body = (directory / 'response.bin').read_bytes()
        if sha256(body) != metadata['response_sha256']:
            errors.append('response hash mismatch')
        if len(body) != metadata['response_size_bytes']:
            errors.append('response size mismatch')
        if metadata['outcome'] == 'parsed':
            expected = extract_cyclones(body)
            if len(expected) != len(metadata['cyclone_items']):
                errors.append('item count mismatch')
            for index, (xml, record) in enumerate(expected):
                filename = f'cyclone_{index:04d}.xml'
                if index >= len(metadata['cyclone_items']):
                    continue
                if metadata['cyclone_items'][index] != {'file': filename, **record}:
                    errors.append(f'item metadata mismatch: {filename}')
                if not (directory / filename).is_file() or (directory / filename).read_bytes() != xml:
                    errors.append(f'item XML mismatch: {filename}')
    elif metadata['outcome'] != 'transport_error':
        errors.append('response absent for non-transport outcome')
    return {'integrity_ok': not errors, 'errors': errors, 'metadata': metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collect = commands.add_parser('collect')
    collect.add_argument('--archive-root', type=Path, default=DEFAULT_ROOT)
    audit = commands.add_parser('audit')
    audit.add_argument('snapshot', type=Path)
    args = parser.parse_args()
    if args.command == 'collect':
        path = collect_once(args.archive_root)
        print(path)
        print('Collection outcome:', json.loads((path / 'metadata.json').read_text())['outcome'])
    else:
        result = audit_snapshot(args.snapshot)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result['integrity_ok']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
