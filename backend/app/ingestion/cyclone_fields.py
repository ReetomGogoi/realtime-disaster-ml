"""Conservative source observations for saved TC records, never model inputs."""
import math
from email.utils import parsedate_to_datetime
from datetime import timezone

G = '{http://www.gdacs.org}'
GEO = '{http://www.w3.org/2003/01/geo/wgs84_pos#}'


def observation_fields(item):
    def read(path, kind='text', attribute=None, bounds=None, unit=None):
        elements = item.findall(path)
        result = {'value': None, 'unit': unit, 'status': 'missing', 'source_path': path,
                  'occurrences': [{'text': e.text, 'attributes': dict(e.attrib)} for e in elements]}
        if len(elements) > 1:
            result['status'] = 'ambiguous'
        elif elements:
            element = elements[0]
            result['unit'] = element.get('unit', unit)
            value = element.get(attribute) if attribute else element.text
            if value and value.strip():
                try:
                    if kind == 'number':
                        value = float(value)
                        if not math.isfinite(value) or (bounds and not bounds[0] <= value <= bounds[1]):
                            raise ValueError()
                    elif kind == 'time':
                        value = parsedate_to_datetime(value)
                        if value.tzinfo is None or value.utcoffset() is None:
                            raise ValueError()
                        value = value.astimezone(timezone.utc).isoformat()
                    else:
                        value = value.strip()
                    result.update(value=value, status='available')
                except (ValueError, TypeError, OverflowError):
                    result['status'] = 'malformed'
        return result
    fields = {}
    for key, tag in [('event_id','eventid'), ('episode_id','episodeid'), ('version','version'), ('alert_level','alertlevel')]:
        fields[key] = read(G+tag)
    for key, tag in [('publication_time','pubDate'),('modification_time',G+'datemodified'),
                     ('event_start_time',G+'fromdate'),('event_end_time',G+'todate')]:
        fields[key] = read(tag, 'time')
    for key, tag in [('severity','severity'),('population_exposure','population'),('vulnerability','vulnerability')]:
        fields[key] = read(G+tag,'number','value',bounds=(0, float('inf')))
    fields['alert_score'] = read(G+'alertscore','number',bounds=(0, float('inf')))
    fields['latitude'] = read(GEO+'Point/'+GEO+'lat','number',bounds=(-90,90),unit='degrees_north')
    fields['longitude'] = read(GEO+'Point/'+GEO+'long','number',bounds=(-180,180),unit='degrees_east')
    # Alternate coordinate representation is only used if both primary values
    # are absent, not to conceal malformed or conflicting primary data.
    if all(fields[key]['status'] == 'missing' for key in ('latitude','longitude')):
        points = item.findall('{http://www.georss.org/georss}point')
        for key, index, limit in [('latitude',0,90),('longitude',1,180)]:
            if len(points) > 1:
                fields[key]['status'] = 'ambiguous'
            elif points:
                fields[key]['source_path'] = '{http://www.georss.org/georss}point'
                try:
                    tokens = (points[0].text or '').split()
                    if len(tokens) != 2:
                        raise ValueError()
                    value = float(tokens[index])
                    if not math.isfinite(value) or not -limit <= value <= limit:
                        raise ValueError()
                    fields[key].update(value=value,status='available')
                except ValueError:
                    fields[key]['status'] = 'malformed'
    return fields
