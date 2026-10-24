# Prediction-time magnitude availability ablation

This controlled historical validation experiment compares two fixed Random
Forests on the existing matched cyclone cohort, separately for each target:

- A: the existing nine baseline predictors.
- B: the same predictors excluding `magnitude` and `magnitude_available`.

The same categorical preprocessing, numeric median imputation, RF parameters
and seed are reused. Only training records fit preprocessing. Missing magnitude
does not remove a row from either configuration. Ordered row IDs, labels and
physical SIDs are checked before fitting and hashed in the report alongside the
existing event-ID fingerprints. Training is 2000–2019; validation is 2020–2022.
The development-only feature table loader rejects test records and SIDs crossing
temporal splits. No targets, saved models, or source data are changed.

Each model reports accuracy, macro/weighted F1, balanced accuracy, per-class
precision/recall/F1/support, and confusion matrices for training and validation.
Training scores are resubstitution diagnostics, not generalization estimates.

The B-minus-A macro-F1 interval reuses the reviewed paired physical-SID cluster
bootstrap: 2,000 draws, seed 20261023, 95% percentile interval. A draw samples
validation SIDs with replacement and retains all their country rows, using the
same draw for both configurations. The fixed high/low/medium class order is
retained even when a draw omits a class; zero_division=0. Intervals condition on
the fitted models and do not measure training randomness or correct for repeated
validation comparisons. No feature set is automatically selected.

Removing magnitude does **not** establish operational safety. Other retrospective
fields, event selection, and actual real-time feature availability still require
verification. The result measures sensitivity to these two columns only.

PowerShell, with the authentic local development feature table already present:

```powershell
cd D:\ML\realtime-disaster-ml
git fetch origin
git switch review/v2-magnitude-availability-ablation
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s ml/tests -p "test_*.py" -v
python -m ml.training.cyclone_magnitude_ablation
```

Output: `datasets/processed/v2_cyclone_magnitude_ablation.json` (Git-ignored).
The script refuses to overwrite an existing result and writes only after both
targets finish. No model artifacts or event-level predictions are saved.
