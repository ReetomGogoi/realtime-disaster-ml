"""Offline synthetic RSS only; no live network requests."""
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
from ml.collection import gdacs_snapshots as snapshots


def feed(version='1', extra=''):
    return (f'<rss xmlns:g="http://www.gdacs.org" xmlns:x="urn:fixture"><channel><item>'
            f'<g:eventtype>TC</g:eventtype><g:eventid>example</g:eventid>'
            f'<g:episodeid>{version}</g:episodeid><g:version>{version}</g:version>'
            '<pubDate>Mon, 01 Jun 2026 00:00:00 GMT</pubDate>'
            '<g:country> A </g:country><g:country>A</g:country>'
            f'<x:details><x:wind unit="km/h" value="100" /></x:details>{extra}'
            '</item></channel></rss>').encode()


class TestSnapshots(unittest.TestCase):
    def collect(self, root, body, status=200):
        return snapshots.collect_once(root, lambda: (status, {'ETag': 'example',
            'Set-Cookie': 'must-not-store'}, body, None))

    def test_exact_bytes_namespaces_attributes_and_repeated_countries(self):
        with tempfile.TemporaryDirectory() as root:
            original = feed()
            path = self.collect(root, original)
            self.assertEqual((path / 'response.bin').read_bytes(), original)
            result = snapshots.audit_snapshot(path)
            self.assertTrue(result['integrity_ok'])
            metadata = result['metadata']
            self.assertEqual(metadata['response_headers'], {'etag': 'example'})
            self.assertEqual(metadata['cyclone_items'][0]['country_occurrences'], [' A ', 'A'])
            item = ET.parse(path / 'cyclone_0000.xml').getroot()
            self.assertEqual(item.find('{urn:fixture}details/{urn:fixture}wind').attrib,
                             {'unit': 'km/h', 'value': '100'})

    def test_revisions_and_duplicate_retrievals_never_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [self.collect(root, body) for body in [feed(), feed(), feed('2')]]
            self.assertEqual(len(set(paths)), 3)
            self.assertEqual((paths[0] / 'response.bin').read_bytes(), feed())
            self.assertNotEqual((paths[0] / 'response.bin').read_bytes(),
                                (paths[2] / 'response.bin').read_bytes())
            for path in paths:
                self.assertTrue(snapshots.audit_snapshot(path)['integrity_ok'])

    def test_missing_fields_are_retained_without_inventing_times(self):
        body = b'<rss xmlns:g="http://www.gdacs.org"><channel><item><g:eventtype>TC</g:eventtype><g:severity>Unknown</g:severity></item></channel></rss>'
        record = snapshots.extract_cyclones(body)[0][1]
        self.assertEqual(record['versions'], [])
        self.assertEqual(record['timestamps_as_reported']['publication'], [])
        self.assertIsNone(record['observation_forecast_valid_time'])

    def test_http_malformed_and_transport_failures(self):
        with tempfile.TemporaryDirectory() as root:
            for body, status, outcome in [(b'failure', 502, 'http_error'),
                                          (b'<rss>', 200, 'malformed_xml')]:
                result = snapshots.audit_snapshot(self.collect(root, body, status))
                self.assertTrue(result['integrity_ok'])
                self.assertEqual(result['metadata']['outcome'], outcome)
            path = snapshots.collect_once(root, lambda: (None, {}, None, 'TimeoutError'))
            self.assertTrue(snapshots.audit_snapshot(path)['integrity_ok'])
            self.assertFalse((path / 'response.bin').exists())

    def test_tampering_detected(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.collect(root, feed())
            (path / 'cyclone_0000.xml').write_bytes(b'changed')
            self.assertFalse(snapshots.audit_snapshot(path)['integrity_ok'])

    def test_command_audit_performs_no_network_request(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.collect(root, feed())
            with patch.object(snapshots.urllib.request, 'urlopen', side_effect=AssertionError):
                self.assertTrue(snapshots.audit_snapshot(path)['integrity_ok'])


if __name__ == '__main__':
    unittest.main()
