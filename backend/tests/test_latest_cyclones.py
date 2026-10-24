"""Isolated sync/latest integration; synthetic XML and memory DB only."""
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from app.database.base import Base
from app.database.models import Event
with patch.dict(os.environ, {'DATABASE_URL': 'sqlite+pysqlite:///:memory:'}):
    from app.database.session import get_db
    from app.api.routes.events import router as events_router
    from app.api.routes.cyclones import router as cyclone_router
from app.ingestion.gdacs import _item_to_dict


def cyclone(identifier='1', extra=''):
    return _item_to_dict(ET.fromstring(
        '<item xmlns:g="http://www.gdacs.org" xmlns:geo="http://www.w3.org/2003/01/geo/wgs84_pos#">'
        f'<title>SYNTHETIC cyclone {identifier}</title><g:eventtype>TC</g:eventtype><g:eventid>{identifier}</g:eventid>'
        '<g:country>A</g:country><g:country>A</g:country><g:severity value="0" unit="km/h"/>'
        '<g:population value="0" unit="Pop74"/><g:alertlevel>Green</g:alertlevel>'
        '<geo:Point><geo:lat>10</geo:lat><geo:long>20</geo:long></geo:Point>'
        '<pubDate>Mon, 01 Jun 2026 00:00:00 GMT</pubDate>'
        f'{extra}</item>'))


class TestLatestCyclones(unittest.TestCase):
    def setUp(self):
        self.engine=create_engine('sqlite://',connect_args={'check_same_thread':False},poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.app=FastAPI()
        self.app.include_router(events_router)
        self.app.include_router(cyclone_router)
        def session():
            with Session(self.engine) as db:
                yield db
        self.app.dependency_overrides[get_db]=session
        self.client=TestClient(self.app)
    def tearDown(self):
        self.engine.dispose()
    def sync(self, records):
        with patch('app.api.routes.events.fetch_gdacs_events',new=AsyncMock(return_value=records)):
            result=self.client.post('/sync-live-events')
            self.assertEqual(result.status_code,200,result.text)
    def latest(self):
        response=self.client.get('/cyclones/latest')
        self.assertEqual(response.status_code,200,response.text)
        return response.json()
    def test_missing_success(self):
        self.assertEqual(self.latest()['status'],'unavailable')
    def test_latest_refresh_excludes_stale_records_and_keeps_archive(self):
        with tempfile.TemporaryDirectory() as root:
            sentinel=Path(root)/'archived.xml'
            sentinel.write_bytes(b'original archive')
            self.sync([cyclone('old')])
            self.sync([cyclone('new')])
            latest=self.latest()
            self.assertEqual([r['source_event_id'] for r in latest['records']],['TC:new'])
            self.assertTrue(latest['last_successful_refresh_utc'])
            self.assertTrue(latest['records'][0]['retrieved_at_utc'])
            fields=latest['records'][0]['fields']
            self.assertEqual(fields['severity']['value'],0)
            self.assertEqual(fields['population_exposure']['unit'],'Pop74')
            self.assertEqual(latest['records'][0]['country_occurrences'],['A','A'])
            self.assertEqual(sentinel.read_bytes(),b'original archive')
            with Session(self.engine) as db:
                self.assertEqual(len(db.execute(select(Event)).scalars().all()),2)
    def test_empty_success_and_duplicate_feed(self):
        record=cyclone()
        self.sync([record,record])
        self.assertEqual(len(self.latest()['records']),1)
        self.sync([])
        self.assertEqual(self.latest()['records'],[])
    def test_failed_refresh_retains_last_success(self):
        self.sync([cyclone()])
        before=self.latest()
        with patch('app.api.routes.events.fetch_gdacs_events',new=AsyncMock(side_effect=httpx.ReadTimeout('fixture'))):
            self.assertEqual(self.client.post('/sync-live-events').status_code,504)
        self.assertEqual(self.latest(),before)
    def test_partial_ambiguous_and_invalid_fields(self):
        record=cyclone(extra='<g:severity unit="km/h" value="90"/><g:alertscore>NaN</g:alertscore>')
        self.sync([record])
        fields=self.latest()['records'][0]['fields']
        self.assertEqual(fields['severity']['status'],'ambiguous')
        self.assertIsNone(fields['severity']['value'])
        self.assertEqual(fields['alert_score']['status'],'malformed')
        self.assertEqual(fields['modification_time']['status'],'missing')
    def test_non_cyclone_refresh_does_not_include_old_cyclones(self):
        self.sync([cyclone()])
        earthquake=cyclone('eq')
        earthquake.update(disaster_type='Earthquake',source_event_id='EQ:eq')
        self.sync([earthquake])
        self.assertEqual(self.latest()['records'],[])
    def test_storage_failure_keeps_successful_batch(self):
        self.sync([cyclone()])
        before=self.latest()
        bad=cyclone('invalid')
        bad['title']=None
        with patch('app.api.routes.events.fetch_gdacs_events',new=AsyncMock(return_value=[bad])):
            self.assertEqual(self.client.post('/sync-live-events').status_code,503)
        self.assertEqual(self.latest(),before)
