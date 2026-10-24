"""Synthetic fixtures and mocked estimators; no model training."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.tests.test_ibtracs_feature_block_ablation import complete_cohort
from ml.training import cyclone_magnitude_ablation as experiment


class TestMagnitudeAblation(unittest.TestCase):
    def test_feature_exclusion(self):
        self.assertEqual(experiment.CONFIGURATIONS['A_nine_features'],
                         ('start_month', 'magnitude', 'magnitude_available'))
        self.assertEqual(experiment.CONFIGURATIONS['B_without_magnitude'], ('start_month',))

    def test_identity_rejects_changed_ids_labels_sids_and_order(self):
        frame = complete_cohort()
        for column in ['event_country_id', 'affected_range', 'matched_sid']:
            changed = frame.copy()
            changed.loc[0, column] = 'changed'
            with self.assertRaises(ValueError):
                experiment.verify_identity(frame, changed, 'affected_range')
        with self.assertRaises(ValueError):
            experiment.verify_identity(frame, frame.iloc[::-1], 'affected_range')

    def test_both_targets_use_identical_rows_and_parameters(self):
        captured = []
        class Stub:
            def __init__(self, categorical, numeric):
                self.numeric = numeric
                self.named_steps = {'classifier': self}
            def get_params(self):
                return experiment.base.RF_PARAMETERS
            def fit(self, features, labels):
                captured.append((self.numeric, features.copy(), labels.copy()))
                return self
            def predict(self, features):
                return ['high'] * len(features)
        for target, expected_count in [('affected_range', 3), ('impact_priority', 4)]:
            captured.clear()
            with patch.object(experiment.base, 'build_rf_pipeline', side_effect=Stub):
                result = experiment.run_target(complete_cohort(), target, iterations=10)
            self.assertTrue(captured[0][1].index.equals(captured[1][1].index))
            self.assertTrue(captured[0][2].equals(captured[1][2]))
            a, b = result['models'].values()
            self.assertEqual(a['ordered_id_label_sid_sha256'], b['ordered_id_label_sid_sha256'])
            self.assertEqual(a['cohort_fingerprints'], b['cohort_fingerprints'])
            self.assertEqual(a['training']['training_sample_count'], expected_count)
            self.assertEqual(b['validation']['validation_sample_count'], expected_count)
            self.assertNotIn('magnitude', b['feature_columns'])
            self.assertNotIn('magnitude_available', b['feature_columns'])

    def test_temporal_and_sid_isolation(self):
        for mutation, message in [
            (lambda f: f.__setitem__('temporal_split', 'test'), 'forbidden split'),
            (lambda f: f.__setitem__('start_date', '2023-01-01'), 'fixed temporal split'),
            (lambda f: f.__setitem__('matched_sid', 'one-storm'), 'spans temporal splits'),
        ]:
            frame = complete_cohort()
            mutation(frame)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'features.csv'
                frame.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    experiment.base.load_experiment_input(path)

    def test_bootstrap_reproducibility_and_zero_paired_difference(self):
        labels = ['high', 'low', 'medium']
        predictions = {name: labels for name in experiment.CONFIGURATIONS}
        first = experiment.paired_difference(labels, ['a', 'b', 'c'], predictions, labels, 20)
        second = experiment.paired_difference(labels, ['a', 'b', 'c'], predictions, labels, 20)
        self.assertEqual(first, second)
        difference = first['comparisons']['B_minus_A']
        self.assertEqual(difference['confidence_interval_lower'], 0.)
        self.assertEqual(difference['confidence_interval_upper'], 0.)


if __name__ == '__main__':
    unittest.main()
