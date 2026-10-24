# Results provenance register

Prepared on 25 October 2026 (Asia/Calcutta context) from Git revision
`d9bdd06acbf12bae1bd1fe1863c1c6a1e91538b4`. This register distinguishes
committed evidence, derived summaries, locally reported checkpoints, and missing
evidence. No model was trained and no event-level or reserved-period data was
opened while preparing the paper.

## Admitted quantitative evidence

| Evidence | SHA-256 | Use in manuscript |
|---|---|---|
| `docs/model_evaluation/validation_metrics.json` | `c37e4dbc07bf4268acdffded69d666f00f2cdc7f8dc8265f88538d780d2d50eb` | Dummy/RF metrics, class reports, confusion matrices |
| `docs/model_evaluation/xgboost_model_v2_validation.json` | `1fa167e30a782d23a36bfbf2c3b909bf1b65316dd4db3e6e9e803b114d80560f` | Best saved development XGBoost entry per target |
| `docs/model_evaluation/extra_trees_model_v2_validation.json` | `5077e115f3c21f7af6f140c93bf5cfff265ed89782e529d693a8b6b9fae90c30` | Best saved development ExtraTrees entry per target |
| `docs/model_evaluation/logistic_regression_model_v2_validation.json` | `2d6862e7f14b13da01eaa5ed31ad08053bc8e08c67fb95e4dfaab2fd73807387` | Best saved development logistic entry per target |
| `docs/model_evaluation/min_samples_leaf_comparison.json` | `b8cef2cb20ccbf2068093c9e025d300b0100a5459c82a37b0cc18442562dd3d7` | Single-parameter development context |
| `docs/model_evaluation/max_depth_comparison.json` | `1f6554cb3743a883486c9172ddd89cd35e186a913cf8f047320dccc230a279d7` | Single-parameter development context |
| `docs/model_evaluation/class_weight_comparison.json` | `130fbbd13e1509e20a5aeffd228d932e86d5d198f987dfcaf14a662c9f0054cd` | Single-parameter development context |
| `docs/emdat_feature_report.txt` | `0bc693095d6b98f65272b3f821821d97eec760e1311ce8fb68398df5267635f1` | Dataset counts, target thresholds, missingness, split counts |
| `docs/v2_research_checkpoint_2026-10-22.md` | `439ddfca86aa39a6a685dc3302ad7dbb18b89e6d374f7034831574e111746438` | Explicitly qualified local matching checkpoint |

Balanced accuracy in `tables/verified_validation_metrics.csv` is derived as the
mean of the three saved per-class recalls. No predictions were regenerated.
Figure 1 and Figure 2 parse the committed preprocessing report. Figures 3–5
read the validation JSON files above. `paper/build_figures.py` never opens
`final_test_metrics.json`.

## Source-code provenance

| File | SHA-256 |
|---|---|
| `ml/preprocessing/build_training_dataset.py` | `6e5cbd75793f0d046e4dc08e2035a50da42dda418aedc4f19a898baba5bcad70` |
| `ml/preprocessing/match_ibtracs_emdat.py` | `a7effb3b5739dc5e7468927a109c7ee0acadaa823c2e3329cb1d8d03e60373c9` |
| `ml/preprocessing/build_ibtracs_cutoff_features.py` | `f835d5db289d5d5955eca984c61093c9abc8dc7a4d17a7aa53da5cbf8b662d17` |
| `ml/training/train_models.py` | `f81f734cce5e59d3d9d22ec844f0dc1a78058d087f75cda508f8674ca53fbe4f` |
| `ml/training/ibtracs_cyclone_validation_experiment.py` | `ef20ca2411be7b338db163eed0904bf4632665cb0d54643c0c0a54888d7c966f` |
| `ml/training/ibtracs_feature_block_ablation.py` | `054c1930cc90dd4cde4bdfdd6bcbafd84cde25ddc9816075a51217252b18f4f4` |
| `ml/training/cyclone_algorithm_comparison.py` | `9c5b81010a1a105237a3473c6badbb7b755dd255a3f24b673df7ae40b43d0f02` |
| `ml/training/cyclone_magnitude_ablation.py` | `485bd16f25f4c39ec6ad6131c6a6d2dd10b0c3bb41d66ab73808a613f61b8e01` |

## Cyclone checkpoints pending evidence

The manuscript request supplied six accuracy/macro-F1 pairs from authentic local
runs. They are preserved in
`tables/preliminary_cyclone_metrics_pending_verification.csv`, but the following
Git-ignored artifacts are absent:

- `datasets/processed/v2_ibtracs_cyclone_features.csv`
- `datasets/processed/v2_ibtracs_cutoff_audit.json`
- `datasets/processed/v2_ibtracs_cyclone_validation.json`
- `datasets/processed/v2_ibtracs_feature_block_ablation.json`
- `datasets/processed/v2_cyclone_algorithm_comparison.json`
- `datasets/processed/v2_cyclone_magnitude_ablation.json`

Consequently, the supplied cyclone scores are not admitted as verified
manuscript findings. In particular, no ExtraTrees/XGBoost cyclone result,
confusion matrix, class denominator, or bootstrap confidence interval can be
recovered from this checkout.

## Reserved 2023–2025 evidence

`docs/model_evaluation/final_test_metrics.json` exists in the repository. The
preparation request requires proof that any final-test execution was explicitly
authorized and reproducible before it is described as evaluated. That chain of
custody was not established here. The file was therefore not read into any
table or figure, and no 2023–2025 metric is reported in the manuscript.

## Local evidence-completion commands

After restoring authorized source files on Windows, preserve existing outputs
before rerunning because the scripts refuse to overwrite by default:

```powershell
cd D:\ML\realtime-disaster-ml
git switch research/final-manuscript
.\.venv\Scripts\Activate.ps1

python -m unittest discover -s ml\tests -p "test_*.py" -v
python -m ml.preprocessing.match_ibtracs_emdat
python -m ml.preprocessing.build_ibtracs_cutoff_features
python -m ml.training.ibtracs_cyclone_validation_experiment
python -m ml.training.ibtracs_feature_block_ablation
python -m ml.training.cyclone_algorithm_comparison
python -m ml.training.cyclone_magnitude_ablation
```

Before publication, copy only aggregate JSON/audit outputs into a controlled
review location, calculate SHA-256 hashes, and confirm that no event-level or
licensed data are staged for Git.
