# Fixed cyclone algorithm comparison

Use the existing development-only matched feature table. All algorithms use
exactly the nine baseline features and target-specific cohorts from the reviewed
cyclone experiment. Country labels remain separate; physical SIDs cannot span
training (2000–2019) and validation (2020–2022). The 2023–2025 period is excluded.

The Random Forest reference is unchanged: 300 trees, balanced class weights,
unlimited depth, leaf size 1, seed 42, all CPU cores. ExtraTrees uses those same
settings with bootstrap=False and max_features=sqrt. XGBoost uses 300 histogram
boosting trees, depth 3, learning rate .05, min_child_weight=1, subsample=.9,
colsample_bytree=.9, reg_alpha=0, reg_lambda=1, multiclass soft probabilities,
mlogloss, seed 42 and all cores. It receives balanced sample weights calculated
from training labels only, encoded high=0, low=1, medium=2. No early stopping,
parameter search, or automatic winner selection occurs.

Every pipeline fits the same median numeric imputation and most-frequent
categorical imputation plus one-hot encoding on training rows only. Reports
include full train and validation metrics, cohort fingerprints, input/source
hashes, effective model parameters, and software versions. Training metrics are
resubstitution diagnostics, not generalization estimates.

Paired 95% percentile bootstrap intervals compare ExtraTrees−RF and XGBoost−RF.
The reviewed ablation resampler draws physical SIDs with replacement, retains
all their country rows and uses the same draw for all algorithms: 2,000 draws,
seed 20261023, fixed three-class macro F1 with zero_division=0. These intervals
condition on the fitted models; they do not measure training randomness or
correct for repeated validation experimentation.

Historical EM-DAT magnitude availability at T0 remains unresolved. Results are
historical validation evidence, not proof of prospective operational value.
No new algorithm is assumed superior.

PowerShell (after restoring the authentic local cutoff feature table):

```powershell
cd D:\ML\realtime-disaster-ml
git switch review/v2-cyclone-algorithm-comparison
.\.venv\Scripts\Activate.ps1
python -m pip install xgboost
python -m unittest discover -s ml/tests -p "test_*.py" -v
python -m ml.training.cyclone_algorithm_comparison
```

Output: `datasets/processed/v2_cyclone_algorithm_comparison.json` (Git-ignored).
Existing outputs cannot be overwritten; no models or event-level predictions
are saved. Both target results are computed before the new JSON is written.
