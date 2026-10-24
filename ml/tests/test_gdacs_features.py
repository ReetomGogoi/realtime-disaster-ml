"""Synthetic archived fixtures only; no external requests."""
import tempfile
import unittest
import xml.etree.ElementTree as ET

from ml.collection import gdacs_features as features
from ml.collection import gdacs_snapshots as archive


def xml(extra='', version='1'):
    return (f'<rss xmlns:g="http://www.gdacs.org" xmlns:geo="http://www.w3.org/2003/01/geo/wgs84_pos#"><channel><item>'
            f'<g:eventtype>TC</g:eventtype><g:eventid>synthetic</g:eventid><g:version>{version}</g:version>'
            '<pubDate>Mon, 01 Jun 2026 00:00:00 GMT</pubDate>'
            '<g:datemodified>Tue, 02 Jun 2026 00:00:00 GMT</g:datemodified>'
            '<g:fromdate>Sun, 31 May 2026 00:00:00 GMT</g:fromdate>'
            '<g:country> A </g:country><g:country>A</g:country>'
            f'{extra}</item></channel></rss>').encode()


def item(extra=''):
    return ET.fromstring(xml(extra)).find('./channel/item')


class TestGdacsFeatures(unittest.TestCase):
    def test_units_zero_and_timestamp_separation(self):
        row = features.extract_item(item('<g:severity unit="km/h" value="0"/>'
            '<g:population unit="Pop74" value="0"/><g:vulnerability value="2"/>'
            '<g:alertscore>1</g:alertscore>'))
        fields = row['fields']
        self.assertEqual(fields['population_exposure']['unit'], 'Pop74')
        self.assertEqual(fields['population_exposure']['value'], 0.)
        self.assertEqual(fields['population_exposure']['status'], 'available')
        self.assertEqual(fields['severity']['unit'], 'km/h')
        self.assertNotEqual(fields['publication_time']['value'], fields['modification_time']['value'])
        self.assertNotEqual(fields['event_start_time']['value'], fields['publication_time']['value'])
        self.assertIsNone(row['observation_forecast_valid_time'])
        self.assertEqual(row['country_occurrences'], [' A ', 'A'])

    def test_missing_malformed_ambiguous(self):
        fields = features.extract_item(item('<g:severity unit="km/h"/>'
            '<g:population value="NaN"/><g:alertscore>1</g:alertscore><g:alertscore>1</g:alertscore>'))['fields']
        self.assertEqual(fields['severity']['status'], 'missing')
        self.assertEqual(fields['severity']['unit'], 'km/h')
        self.assertEqual(fields['population_exposure']['status'], 'malformed')
        self.assertEqual(fields['alert_score']['status'], 'ambiguous')
        self.assertEqual(fields['episode_id']['status'], 'missing')

    def test_coordinate_bounds_and_namespaces(self):
        row = features.extract_item(item('<geo:Point><geo:lat>91</geo:lat><geo:long>0</geo:long></geo:Point>'))
        self.assertEqual(row['fields']['latitude']['status'], 'malformed')
        self.assertEqual(row['fields']['longitude']['value'], 0.)
        self.assertFalse(row['coordinate_pair_available'])
        wrong = features.extract_item(item('<severity value="50" unit="km/h"/>'))
        self.assertEqual(wrong['fields']['severity']['status'], 'missing')

    def test_georss_fallback_and_unknown_unit_preserved(self):
        row = features.extract_item(item('<p:point xmlns:p="http://www.georss.org/georss">-10 170</p:point>'
            '<g:severity value="42" unit="unreviewed-unit"/>'))
        self.assertTrue(row['coordinate_pair_available'])
        self.assertEqual(row['fields']['severity']['unit'], 'unreviewed-unit')

    def test_archive_integrity_revisions_duplicates_and_retrieval(self):
        with tempfile.TemporaryDirectory() as root:
            for body in [xml(), xml(), xml(version='2')]:
                archive.collect_once(root, lambda body=body: (200, {}, body, None))
            rows, audit = features.extract_directory(root)
            self.assertEqual(len(rows), 3)
            self.assertEqual(audit['unique_serialized_items'], 2)
            self.assertEqual(sum(r['identical_item_previously_observed'] for r in rows), 1)
            for row in rows:
                self.assertIn('+00:00', row['retrieved_at_utc'])
                self.assertIn(row['fields']['version']['value'], ['1', '2'])
                self.assertNotEqual(row['retrieved_at_utc'], row['fields']['publication_time']['value'])

    def test_corrupted_archive_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = archive.collect_once(root, lambda: (200, {}, xml(), None))
            (path / 'response.bin').write_bytes(b'corrupted')
            rows, audit = features.extract_directory(root)
            self.assertEqual(rows, [])
            self.assertEqual(len(audit['rejected_snapshots']), 1)

    def test_bad_retrieval_metadata_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            path = archive.collect_once(root, lambda: (200, {}, xml(), None))
            metadata = json.loads((path / 'metadata.json').read_text())
            metadata['retrieved_at_utc'] = '2026-06-01T00:00:00'
            (path / 'metadata.json').write_text(json.dumps(metadata))
            rows, audit = features.extract_directory(root)
            self.assertEqual(rows, [])
            self.assertEqual(len(audit['rejected_snapshots']), 1)


if __name__ == '__main__':
    unittest.main()
