"""Fixed nine-feature RF/ExtraTrees/XGBoost comparison; local validation only."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from ml.training import ibtracs_cyclone_validation_experiment as base
from ml.training import ibtracs_feature_block_ablation as ablation

FEATURES = (*base.BASE_CATEGORICAL_FEATURES, *base.BASE_NUMERIC_FEATURES)
ALGORITHMS = ('random_forest', 'extra_trees', 'xgboost')
EXTRA_PARAMETERS = {**base.RF_PARAMETERS, 'bootstrap': False, 'max_features': 'sqrt'}
XGB_PARAMETERS = {
    'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss',
    'tree_method': 'hist', 'n_estimators': 300, 'max_depth': 3,
    'learning_rate': 0.05, 'min_child_weight': 1, 'subsample': 0.9,
    'colsample_bytree': 0.9, 'reg_alpha': 0.0, 'reg_lambda': 1.0,
    'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
}
DEFAULT_OUTPUT = base.PROJECT_ROOT / 'datasets/processed/v2_cyclone_algorithm_comparison.json'


def build_pipeline(name):
    if name == 'random_forest':
        return base.build_rf_pipeline(base.BASE_CATEGORICAL_FEATURES, base.BASE_NUMERIC_FEATURES)
    if name == 'extra_trees':
        classifier = ExtraTreesClassifier(**EXTRA_PARAMETERS)
    elif name == 'xgboost':
        from xgboost import XGBClassifier
        classifier = XGBClassifier(**XGB_PARAMETERS)
    else:
        raise ValueError(f'Unknown algorithm: {name}')
    return Pipeline([
        ('preprocessor', base.build_preprocessor(base.BASE_CATEGORICAL_FEATURES, base.BASE_NUMERIC_FEATURES)),
        ('classifier', classifier),
    ])


def paired_comparisons(actual, sids, predictions, classes, iterations=2000):
    # Reuse the reviewed paired cluster resampler without changing the ablation.
    aliases = dict(zip(ALGORITHMS, ablation.CONFIGURATIONS))
    result = ablation.grouped_bootstrap_macro_f1_differences(
        actual, sids, {aliases[k]: predictions[k] for k in ALGORITHMS},
        classes, iterations=iterations, seed=ablation.BOOTSTRAP_SEED,
    )
    comparisons = {}
    for key, name in [('B_minus_A', 'extra_trees_minus_random_forest'),
                      ('C_minus_A', 'xgboost_minus_random_forest')]:
        item = result['comparisons'][key].copy()
        item['minuend'] = name.split('_minus_')[0]
        item['subtrahend'] = 'random_forest'
        comparisons[name] = item
    result['comparisons'] = comparisons
    return result


def run_target(frame, target, iterations=2000):
    train, validation, y_train, y_validation, classes = base.prepare_target_cohort(frame, target)
    cohort = base.summarize_target_cohort(frame, target, train, validation, classes)
    x_train, x_validation = train.loc[:, list(FEATURES)], validation.loc[:, list(FEATURES)]
    class_ids = {label: i for i, label in enumerate(classes)}
    weights = compute_sample_weight('balanced', y_train)
    models, predictions = {}, {}
    for name in ALGORITHMS:
        pipeline = build_pipeline(name)
        if name == 'xgboost':
            pipeline.fit(x_train, y_train.map(class_ids).to_numpy(dtype=int),
                         classifier__sample_weight=weights)
            decode = np.asarray(classes)
            train_prediction = decode[np.asarray(pipeline.predict(x_train), dtype=int)]
            prediction = decode[np.asarray(pipeline.predict(x_validation), dtype=int)]
        else:
            pipeline.fit(x_train, y_train)
            train_prediction = pipeline.predict(x_train)
            prediction = pipeline.predict(x_validation)
        predictions[name] = prediction
        training = base.calculate_metrics(y_train, train_prediction, classes)
        training['training_sample_count'] = training.pop('validation_sample_count')
        models[name] = {
            'feature_columns': list(FEATURES),
            'cohort_fingerprints': cohort['record_fingerprints'],
            'training': training,
            'validation': base.calculate_metrics(y_validation, prediction, classes),
            'effective_classifier_parameters': pipeline.named_steps['classifier'].get_params(),
        }
    return {'cohort': cohort, 'models': models,
            'paired_macro_f1_differences': paired_comparisons(
                y_validation, validation['matched_sid'], predictions, classes, iterations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=base.DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    import xgboost
    frame, metadata = base.load_experiment_input(args.input)
    result = {
        'schema_version': 1,
        'experiment': 'fixed nine-feature cyclone algorithm comparison',
        'selection_policy': 'no tuning, early stopping, or automatic selection',
        'development_splits': {'train': '2000-2019', 'validation': '2020-2022'},
        'test_policy': '2023-2025 excluded; development-only feature table required',
        'cutoff': '00:00 UTC on complete EM-DAT start_date; ISO_TIME <= T0',
        'limitation': 'Retrospective EM-DAT magnitude availability at T0 unresolved',
        'input_artifacts': {'feature_table': metadata},
        'source_code_at_execution': {
            p.name: base.artifact_metadata(p, reported_path=p.relative_to(base.PROJECT_ROOT).as_posix())
            for p in [Path(__file__).resolve(), Path(base.__file__), Path(ablation.__file__)]},
        'software_versions': {'python': platform.python_version(), 'numpy': np.__version__,
                              'pandas': base.pd.__version__, 'sklearn': base.sklearn.__version__,
                              'xgboost': xgboost.__version__},
        'preprocessing': ablation._preprocessing_description(),
        'weighting': 'RF/ExtraTrees class_weight=balanced; XGBoost training-label balanced sample weights',
        'targets': {target: run_target(frame, target) for target in base.TARGET_COLUMNS},
    }
    ablation.write_new_result(args.output, result)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
