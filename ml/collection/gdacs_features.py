"""Offline versioned observational features from verified GDACS archives."""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from ml.collection import gdacs_snapshots as archive

SCHEMA_VERSION = 'gdacs-cyclone-observations/1.0'
G = archive.G
GEO = '{http://www.w3.org/2003/01/geo/wgs84_pos#}'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'datasets/processed/gdacs_cyclone_extraction'


def parse_time(text):
    try:
        value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        value = parsedate_to_datetime(text)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('timezone required')
    return value.astimezone(timezone.utc).isoformat()


def field(item, tag, kind='text', attribute=None, bounds=None, unit=None):
    elements = item.findall(tag)
    occurrences = [{'text': e.text, 'attributes': dict(e.attrib)} for e in elements]
    result = {'source_path': tag, 'attribute': attribute, 'occurrences': occurrences,
              'value': None, 'unit': unit, 'status': 'missing', 'failure': None}
    if not elements:
        return result
    if len(elements) != 1:
        result.update(status='ambiguous', failure='multiple source occurrences')
        return result
    element = elements[0]
    result['unit'] = element.get('unit', unit)
    text = element.get(attribute) if attribute else element.text
    if text is None or not text.strip():
        result['failure'] = 'missing or empty source value'
        return result
    try:
        if kind == 'number':
            value = float(text)
            if not math.isfinite(value) or (bounds and not bounds[0] <= value <= bounds[1]):
                raise ValueError('nonfinite or out-of-range number')
        elif kind == 'time':
            value = parse_time(text.strip())
        else:
            value = text.strip()
        result.update(value=value, status='available')
    except (ValueError, TypeError, OverflowError) as error:
        result.update(status='malformed', failure=str(error))
    return result


def extract_item(item):
    fields = {}
    for name, tag in [('event_id', G+'eventid'), ('episode_id', G+'episodeid'),
                      ('version', G+'version'), ('report_url', 'link'),
                      ('alert_level', G+'alertlevel')]:
        fields[name] = field(item, tag)
    for name, tag in [('publication_time', 'pubDate'), ('modification_time', G+'datemodified'),
                      ('added_time', G+'dateadded'), ('event_start_time', G+'fromdate'),
                      ('event_end_time', G+'todate')]:
        fields[name] = field(item, tag, kind='time')
    for name, tag in [('severity', G+'severity'), ('population_exposure', G+'population'),
                      ('vulnerability', G+'vulnerability')]:
        fields[name] = field(item, tag, kind='number', attribute='value')
    for name, tag in [('alert_score', G+'alertscore'),
                      ('episode_alert_score', G+'episodealertscore')]:
        fields[name] = field(item, tag, kind='number')
    fields['latitude'] = field(item, f'{GEO}Point/{GEO}lat', kind='number', bounds=(-90, 90), unit='degrees_north')
    fields['longitude'] = field(item, f'{GEO}Point/{GEO}long', kind='number', bounds=(-180, 180), unit='degrees_east')
    # Do not silently combine different representations when one is malformed.
    if fields['latitude']['status'] == fields['longitude']['status'] == 'missing':
        points = item.findall('{http://www.georss.org/georss}point')
        if points:
            for name, index, limit in [('latitude', 0, 90), ('longitude', 1, 180)]:
                result = fields[name]
                result['source_path'] = '{http://www.georss.org/georss}point'
                result['occurrences'] = [{'text': p.text, 'attributes': dict(p.attrib)} for p in points]
                try:
                    if len(points) != 1:
                        result.update(status='ambiguous', failure='multiple coordinate occurrences')
                        continue
                    tokens = (points[0].text or '').split()
                    if len(tokens) != 2:
                        raise ValueError('expected two coordinates')
                    value = float(tokens[index])
                    if not math.isfinite(value) or not -limit <= value <= limit:
                        raise ValueError('invalid coordinate')
                    result.update(value=value, status='available')
                except ValueError as error:
                    result.update(status='malformed', failure=str(error))
    return {
        'fields': fields,
        'country_occurrences': [e.text for e in item.findall(G+'country')],
        'iso3_occurrences': [e.text for e in item.findall(G+'iso3')],
        'numeric_attribute_candidates': [
            {'expanded_tag': e.tag, 'attributes': dict(e.attrib), 'text': e.text}
            for e in item.iter() if 'value' in e.attrib
        ],
        'coordinate_pair_available': all(fields[k]['status'] == 'available' for k in ['latitude', 'longitude']),
        'observation_forecast_valid_time': None,
        'valid_time_status': 'unverified; not inferred from event/publication timestamps',
        'population_semantics': 'source-modeled exposure; Pop74 retained verbatim; not observed impact or Total Affected',
    }


def extract_directory(root):
    root = Path(root)
    directories = [root] if (root / 'metadata.json').is_file() else sorted(p for p in root.iterdir() if p.is_dir())
    rows, failures, accepted = [], [], []
    seen = Counter()
    for directory in directories:
        try:
            checked = archive.audit_snapshot(directory)
            if not checked['integrity_ok']:
                raise ValueError('; '.join(checked['errors']))
            metadata = checked['metadata']
            if metadata['schema_version'] != 1 or metadata['outcome'] != 'parsed':
                raise ValueError('snapshot is not a parsed version-1 archive')
            if metadata['source_url'] != archive.SOURCE_URL or not 200 <= metadata['http_status'] < 300:
                raise ValueError('unexpected source or HTTP status')
            retrieved = parse_time(metadata['retrieved_at_utc'])
            accepted.append({'snapshot': directory.name, 'response_sha256': metadata['response_sha256'],
                             'metadata_sha256': archive.sha256((directory / 'metadata.json').read_bytes())})
            for xml, original in archive.extract_cyclones((directory / 'response.bin').read_bytes()):
                key = original['serialized_item_sha256']
                row = extract_item(ET.fromstring(xml))
                row.update(schema_version=SCHEMA_VERSION, snapshot_id=directory.name,
                           source_url=metadata['source_url'], retrieved_at_utc=retrieved,
                           item_sha256=key, response_sha256=metadata['response_sha256'],
                           identical_item_previously_observed=seen[key] > 0,
                           availability_evidence='this archived version observed by retrieval only')
                seen[key] += 1
                rows.append(row)
        except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
            failures.append({'snapshot': directory.name, 'reason': str(error)})
    audit = {'schema_version': SCHEMA_VERSION, 'accepted_snapshots': accepted,
             'rejected_snapshots': failures, 'record_count': len(rows),
             'unique_serialized_items': len(seen),
             'field_status_counts': dict(Counter(
                 name+':'+value['status'] for row in rows for name, value in row['fields'].items())),
             'limitation': 'hashes detect corruption, not authenticity; retrieval metadata is trusted local provenance'}
    return rows, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, default=archive.DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Extraction output directory already exists')
    rows, audit = extract_directory(args.snapshots)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'features.jsonl').open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
