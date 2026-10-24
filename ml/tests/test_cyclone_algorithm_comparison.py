"""Artificial fixtures and estimator stubs only; no model training."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml.tests.test_ibtracs_feature_block_ablation import complete_cohort
from ml.training import cyclone_algorithm_comparison as comparison
from ml.training import ibtracs_cyclone_validation_experiment as base


class TestAlgorithmComparison(unittest.TestCase):
    def test_reference_and_feature_preprocessing_parity(self):
        rf = comparison.build_pipeline('random_forest')
        reference = base.build_rf_pipeline(base.BASE_CATEGORICAL_FEATURES, base.BASE_NUMERIC_FEATURES)
        self.assertEqual(rf.named_steps['classifier'].get_params(),
                         reference.named_steps['classifier'].get_params())
        extra = comparison.build_pipeline('extra_trees')
        for pipeline in [rf, extra]:
            transformers = pipeline.named_steps['preprocessor'].transformers
            self.assertEqual(transformers[0][2] + transformers[1][2], list(comparison.FEATURES))
            self.assertEqual(transformers[0][1].named_steps['onehot'].handle_unknown, 'ignore')
            self.assertEqual(transformers[1][1].named_steps['imputer'].strategy, 'median')

    def test_preprocessing_training_only_missing_and_unknown_values(self):
        frame = complete_cohort()
        pre = base.build_preprocessor(base.BASE_CATEGORICAL_FEATURES, base.BASE_NUMERIC_FEATURES)
        training = frame.iloc[:4].loc[:, list(comparison.FEATURES)].copy()
        training['magnitude'] = [10., 20., np.nan, 30.]
        validation = frame.iloc[4:].loc[:, list(comparison.FEATURES)].copy()
        validation['magnitude'] = np.nan
        validation['iso'] = 'UNSEEN'
        pre.fit(training)
        self.assertEqual(pre.named_transformers_['num'].named_steps['imputer'].statistics_[1], 20.)
        self.assertTrue(np.isfinite(pre.transform(validation)).all())

    def test_same_records_labels_and_training_weights(self):
        captured = []
        class Stub:
            def __init__(self, name):
                self.name = name
                self.named_steps = {'classifier': self}
            def get_params(self):
                return {'stub': True}
            def fit(self, x, y, **kwargs):
                captured.append((self.name, x.copy(), np.asarray(y), kwargs))
                return self
            def predict(self, x):
                return np.zeros(len(x), dtype=int) if self.name == 'xgboost' else ['high'] * len(x)
        with patch.object(comparison, 'build_pipeline', side_effect=Stub):
            result = comparison.run_target(complete_cohort(), 'affected_range', iterations=10)
        for name, features, labels, kwargs in captured:
            self.assertEqual(features.index.tolist(), [0, 1, 2])
            self.assertEqual(features.columns.tolist(), list(comparison.FEATURES))
            if name == 'xgboost':
                np.testing.assert_array_equal(labels, [0, 1, 2])
                np.testing.assert_array_equal(kwargs['classifier__sample_weight'], [1., 1., 1.])
        for model in result['models'].values():
            self.assertEqual(model['cohort_fingerprints'], result['cohort']['record_fingerprints'])
            self.assertEqual(model['training']['training_sample_count'], 3)
            self.assertEqual(model['validation']['validation_sample_count'], 3)
        self.assertEqual(set(result['paired_macro_f1_differences']['comparisons']),
                         {'extra_trees_minus_random_forest', 'xgboost_minus_random_forest'})

    def test_temporal_and_sid_guards_before_comparison(self):
        frame = complete_cohort()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'features.csv'
            frame.loc[4, 'start_date'] = '2023-01-01'
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'fixed temporal split'):
                base.load_experiment_input(path)
            frame.loc[4, 'start_date'] = '2020-08-25'
            frame.loc[4, 'matched_sid'] = frame.loc[0, 'matched_sid']
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'spans temporal splits'):
                base.load_experiment_input(path)

    def test_paired_identical_predictions_have_zero_interval(self):
        labels = ['high', 'low', 'medium']
        result = comparison.paired_comparisons(labels, ['a', 'b', 'c'],
            {name: labels for name in comparison.ALGORITHMS}, labels, iterations=10)
        for difference in result['comparisons'].values():
            self.assertEqual(difference['observed_macro_f1_difference'], 0.)
            self.assertEqual(difference['confidence_interval_lower'], 0.)
            self.assertEqual(difference['confidence_interval_upper'], 0.)


if __name__ == '__main__':
    unittest.main()
