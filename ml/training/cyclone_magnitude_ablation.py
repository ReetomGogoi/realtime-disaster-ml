"""Controlled historical magnitude-availability ablation; local validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from ml.training import ibtracs_cyclone_validation_experiment as base
from ml.training import ibtracs_feature_block_ablation as bootstrap

CONFIGURATIONS = {
    'A_nine_features': base.BASE_NUMERIC_FEATURES,
    'B_without_magnitude': ('start_month',),
}
DEFAULT_OUTPUT = base.PROJECT_ROOT / 'datasets/processed/v2_cyclone_magnitude_ablation.json'


def identity(rows, target):
    """Ordered row identity includes IDs, labels and physical storm membership."""
    return rows.loc[:, ['event_country_id', target, 'matched_sid']].reset_index(drop=True)


def identity_hash(rows, target):
    payload = identity(rows, target).to_json(orient='values', force_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def verify_identity(expected, actual, target):
    if not identity(expected, target).equals(identity(actual, target)):
        raise ValueError('Configuration row IDs, labels or physical SIDs differ')


def paired_difference(actual, sids, predictions, classes, iterations=2000):
    # The reviewed three-way helper also computes unused differences. Duplicate
    # A into its third slot, then retain only the required paired B-minus-A CI.
    result = bootstrap.grouped_bootstrap_macro_f1_differences(
        actual, sids,
        {bootstrap.CONFIGURATION_A: predictions['A_nine_features'],
         bootstrap.CONFIGURATION_B: predictions['B_without_magnitude'],
         bootstrap.CONFIGURATION_C: predictions['A_nine_features']},
        classes, iterations=iterations, seed=bootstrap.BOOTSTRAP_SEED,
    )
    difference = result.pop('comparisons')['B_minus_A']
    difference.update(minuend='B_without_magnitude', subtrahend='A_nine_features')
    result['comparisons'] = {'B_minus_A': difference}
    return result


def run_target(frame, target, iterations=2000):
    train, validation, y_train, y_validation, classes = base.prepare_target_cohort(frame, target)
    cohort = base.summarize_target_cohort(frame, target, train, validation, classes)
    models, predictions = {}, {}
    hashes = {'train': identity_hash(train, target),
              'validation': identity_hash(validation, target)}
    for name, numeric in CONFIGURATIONS.items():
        # Preserve all rows even when either magnitude field is missing.
        model_train, model_validation = train.copy(), validation.copy()
        verify_identity(train, model_train, target)
        verify_identity(validation, model_validation, target)
        columns = [*base.BASE_CATEGORICAL_FEATURES, *numeric]
        pipeline = base.build_rf_pipeline(base.BASE_CATEGORICAL_FEATURES, numeric)
        pipeline.fit(model_train.loc[:, columns], y_train)
        predicted_train = pipeline.predict(model_train.loc[:, columns])
        predicted_validation = pipeline.predict(model_validation.loc[:, columns])
        predictions[name] = predicted_validation
        training = base.calculate_metrics(y_train, predicted_train, classes)
        training['training_sample_count'] = training.pop('validation_sample_count')
        models[name] = {
            'feature_columns': columns,
            'cohort_fingerprints': cohort['record_fingerprints'],
            'ordered_id_label_sid_sha256': hashes,
            'identity_verified': True,
            'training': training,
            'validation': base.calculate_metrics(y_validation, predicted_validation, classes),
            'effective_classifier_parameters': pipeline.named_steps['classifier'].get_params(),
        }
    return {'cohort': cohort, 'models': models,
            'paired_macro_f1_difference': paired_difference(
                y_validation, validation['matched_sid'], predictions, classes, iterations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=base.DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    frame, metadata = base.load_experiment_input(args.input)
    result = {
        'schema_version': 1,
        'experiment': 'controlled nine-feature versus seven-feature magnitude ablation',
        'selection_policy': 'fixed comparison; no automatic selection or tuning',
        'development_splits': {'train': '2000-2019', 'validation': '2020-2022'},
        'test_policy': '2023-2025 excluded; development-only feature table required',
        'limitation': ('Removing magnitude does not establish operational safety; other '
                       'retrospective fields and actual T0 availability remain unverified'),
        'input_artifacts': {'feature_table': metadata},
        'source_code_at_execution': {
            p.name: base.artifact_metadata(p, reported_path=p.relative_to(base.PROJECT_ROOT).as_posix())
            for p in [Path(__file__).resolve(), Path(base.__file__), Path(bootstrap.__file__)]},
        'software_versions': {'python': platform.python_version(), 'numpy': np.__version__,
                              'pandas': pd.__version__, 'sklearn': base.sklearn.__version__},
        'preprocessing': bootstrap._preprocessing_description(),
        'rf_parameters': base.RF_PARAMETERS,
        'targets': {target: run_target(frame, target) for target in base.TARGET_COLUMNS},
    }
    bootstrap.write_new_result(args.output, result)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
