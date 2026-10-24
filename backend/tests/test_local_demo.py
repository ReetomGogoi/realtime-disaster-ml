"""Isolated router tests; no database startup or model loading."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.routes import local_demo


class TestLocalDemo(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'ENABLE_LOCAL_ML_DEMO':'1', 'VERCEL':''})
        self.env.start()
        app = FastAPI()
        app.include_router(local_demo.router)
        app.state.ml_bundle = {}
        self.app = app
        self.client = TestClient(app)
        self.features = dict(disaster_type='Storm', disaster_subtype='Tropical cyclone',
            iso='USA',region='Americas',subregion='Northern America', season='JJA',
            start_month=8, magnitude=None,magnitude_available=0)
    def tearDown(self):
        self.env.stop()
    def test_missing_output_and_models(self):
        with patch.object(local_demo,'FEATURE_PATH',Path('/nonexistent/features.jsonl')):
            self.assertEqual(self.client.get('/local-demo/cyclones').json()['status'],'unavailable')
        response=self.client.post('/local-demo/historical-predict',json=self.features)
        self.assertEqual(response.status_code,200)
        for value in response.json()['results'].values():
            self.assertEqual(value['status'],'model_unavailable')
            self.assertIsNone(value['predicted_label'])
    def test_provenance_missing_fields_and_schema_rejection(self):
        row=dict(schema_version=local_demo.SCHEMA,snapshot_id='fixture',
            source_url='https://www.gdacs.org/xml/rss_7d.xml',retrieved_at_utc='2026-06-01T00:00:00Z',
            item_sha256='a'*64,response_sha256='b'*64,availability_evidence='observed at retrieval only',
            fields={},country_occurrences=['A','A'],population_semantics='modeled exposure')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'features.jsonl'
            path.write_text(json.dumps(row))
            with patch.object(local_demo,'FEATURE_PATH',path):
                result=self.client.get('/local-demo/cyclones')
                self.assertEqual(result.status_code,200)
                self.assertEqual(result.json()['records'][0]['country_occurrences'],['A','A'])
                row['schema_version']='incorrect'
                path.write_text(json.dumps(row))
                self.assertEqual(self.client.get('/local-demo/cyclones').status_code,422)
    def test_disabled_and_hosted(self):
        with patch.dict(os.environ,{'ENABLE_LOCAL_ML_DEMO':'0'}):
            self.assertEqual(self.client.get('/local-demo/cyclones').status_code,404)
        with patch.dict(os.environ,{'VERCEL':'1'}):
            self.assertEqual(self.client.get('/local-demo/cyclones').status_code,404)
    def test_invalid_historical_input(self):
        invalid={**self.features,'magnitude_available':1}
        self.assertEqual(self.client.post('/local-demo/historical-predict',json=invalid).status_code,422)
        invalid={**self.features,'unverified_gdacs_value':4}
        self.assertEqual(self.client.post('/local-demo/historical-predict',json=invalid).status_code,422)
    def test_trusted_bundle_receives_exact_schema(self):
        class Stub:
            named_steps={'classifier':SimpleNamespace(classes_=['high','low','medium'])}
            def predict(self,frame):
                assert frame.columns.tolist()==list(local_demo.FEATURES)
                return ['low']
            def predict_proba(self,frame): return [[0.1,0.7,0.2]]
        self.app.state.ml_bundle={'affected_range':SimpleNamespace(pipeline=Stub(),supported_hazards={'Storm'})}
        result=self.client.post('/local-demo/historical-predict',json=self.features).json()
        self.assertEqual(result['results']['affected_range']['predicted_label'],'low')

    def test_observation_status_and_timestamp_validation(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            local_demo.ObservationField(source_path='fixture', status='available', value=None)
        with self.assertRaises(ValidationError):
            local_demo.ObservationField(source_path='fixture', status='missing', value=0)

    def test_non_loopback_request_rejected(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):
            local_demo.local_only(SimpleNamespace(client=SimpleNamespace(host='192.0.2.1')))

    def test_performance_only_returns_allowlisted_validation_metrics(self):
        saved = {target: {'dataset_info': {'val_samples': 12},
                 'random_forest': {'accuracy': .5, 'macro_f1': .4},
                 'baseline_dummy_most_frequent': {'accuracy': .3, 'macro_f1': .2}}
                 for target in local_demo.TARGETS}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            folder=root/'docs/model_evaluation'
            folder.mkdir(parents=True)
            (folder/'validation_metrics.json').write_text(json.dumps(saved))
            with patch.object(local_demo,'ROOT',root):
                response=self.client.get('/local-demo/performance')
                self.assertEqual(response.status_code,200)
                self.assertEqual(len(response.json()['rows']),4)
                self.assertEqual(response.json()['rows'][0]['validation_samples'],12)
                self.assertIn('all-hazard',response.json()['cohort'])
                self.assertEqual(len(response.json()['sha256']),64)
                (folder/'validation_metrics.json').write_text('{}')
                self.assertEqual(self.client.get('/local-demo/performance').status_code,422)
