# Version 2 IBTrACS Cyclone Validation Experiment

This workflow compares the existing nine-feature Random Forest with the same
classifier augmented by time-safe IBTrACS observations. It is a local research
experiment. It does not replace saved models or alter Version 1.

## Fixed scientific policy

- One row remains one matched EM-DAT event-country record. Countries sharing a
  physical cyclone retain their own `affected_range` and `impact_priority`
  labels.
- `event_country_id` must equal the unique processed EM-DAT `dis_no`.
- A complete canonical EM-DAT `start_date` is required.
- T0 is 00:00 UTC on `start_date`.
- Only IBTrACS observations with `ISO_TIME <= T0` are eligible.
- The code does not interpolate track observations. A SID with duplicate
  observation timestamps is excluded rather than resolved heuristically.
- Physical `SID` values cannot cross the train/validation boundary.
- Training is 2000-2019 and validation is 2020-2022. The 2023-2025 test period
  is neither emitted by the feature builder nor accepted by the experiment.
- Deaths, affected population, damage, landfall, future extrema, future track,
  and full storm duration are not model inputs.

The existing nine-feature baseline retains the historical EM-DAT `magnitude`
feature so that the comparison reproduces the established feature definition.
Its exact operational availability at T0 is not independently established.
Therefore, this experiment is historical pseudo-prospective validation and is
not evidence of deployable live prediction.

## IBTrACS features

The latest observation at or before T0 supplies:

- WMO wind in knots and an availability flag;
- WMO pressure in millibars and an availability flag;
- latitude and longitude in degrees and an atomic location flag;
- distance to land in kilometres and an availability flag;
- WMO agency and an agency-availability flag; and
- storm age in hours from the first observed track point through T0.

Motion uses the final two valid location observations at or before T0. Speed is
kilometres per hour. Direction is represented by bearing sine and cosine. With
fewer than two valid past locations, motion remains missing and its availability
flag is zero. Missing WMO wind or pressure is never replaced with zero or with
another agency's measurement. Model preprocessing fits median or most-frequent
imputation on training rows only.

WMO wind values can reflect different agency averaging conventions. The model
therefore receives `WMO_AGENCY`, but this first experiment does not attempt an
unverified wind-period conversion.

## Local Windows execution

From PowerShell:

```powershell
cd D:\ML\realtime-disaster-ml
git fetch origin
git switch review/v2-ibtracs-cutoff-experiment

python -m unittest discover -s ml/tests -p "test_*.py" -v
python -m compileall -q ml\preprocessing ml\training ml\tests
git diff --check origin/main...HEAD
```

Confirm these authentic local inputs exist:

```text
datasets/raw/public_emdat_2026-09-09.xlsx
datasets/raw/ibtracs/ibtracs.since1980.list.v04r01.csv
datasets/processed/emdat_ml_dataset.csv
docs/v2_match_pilot_log.csv
```

Regenerate the reviewed match audit if necessary:

```powershell
python -m ml.preprocessing.match_ibtracs_emdat
```

Build the cutoff-safe local feature table and aggregate audit:

```powershell
python -m ml.preprocessing.build_ibtracs_cutoff_features
```

Run the fixed validation comparison:

```powershell
python -m ml.training.ibtracs_cyclone_validation_experiment
```

The local outputs are:

```text
datasets/processed/v2_ibtracs_cyclone_features.csv
datasets/processed/v2_ibtracs_cutoff_audit.json
datasets/processed/v2_ibtracs_cyclone_validation.json
```

`datasets/processed/` is ignored by Git. Do not force-add these files. The
validation JSON reports, separately for each target, accuracy, macro F1,
weighted F1, balanced accuracy, per-class precision/recall/F1/support, class
order, and a confusion matrix whose rows are actual and columns predicted.
It also records the exact feature-table SHA-256 and byte size, SHA-256 values
for the experiment and cutoff-feature source files present when the experiment
runs, software versions, random seeds, model parameters, typed feature lists,
and preprocessing settings. For each target, it records the shared
train/validation event-country counts and identifier fingerprints used by all
three models, unique physical-SID counts, class distributions, excluded
missing-label counts, and per-feature missing counts. Machine-specific absolute
paths are intentionally omitted.

The feature-table hash identifies the generated experiment input exactly. The
source-code hashes identify the files available when validation runs; they do
not prove which earlier checkout produced an existing feature table. They also
do not identify the underlying raw EM-DAT or IBTrACS files. Retain the matching
and cutoff audits with the locally downloaded source files when archiving a
run. The experiment rejects a feature table that changes while it is being
read.

To intentionally replace prior local outputs after reviewing them, add
`--overwrite` to the builder and experiment commands. No fitted model is saved.

## Required review before using results

Review the cutoff audit for eligibility loss, missingness by split, malformed
times, missing SIDs, duplicate timestamps, and feature availability. Compare
all three models on the identical target-specific cohort. Do not compare these
cyclone-subset metrics directly with the saved all-hazard validation metrics as
if they were evaluated on the same records. Do not inspect or use the final
test split when deciding whether the enriched feature set is useful.

## Controlled feature-block ablation

The follow-up ablation compares exactly three fixed configurations:

- **A:** the existing nine EM-DAT features;
- **B:** A plus WMO wind, WMO pressure, their availability indicators,
  `WMO_AGENCY`, and the agency-availability indicator; and
- **C:** A plus the complete cutoff-safe IBTrACS feature set described above.

All three configurations use the same target-specific country records, Random
Forest parameters, seed, and training-only preprocessing. This is a diagnostic
comparison. The script does not automatically select a feature set or update a
saved model.

For each pair of configurations, the report includes the observed validation
macro-F1 difference and a 95% percentile confidence interval from 2,000 paired
bootstrap replicates with seed `20261023`. Each replicate samples physical
`matched_sid` values with replacement, keeps every country row belonging to a
sampled SID together, and applies the same sampled rows to A, B, and C. Macro
F1 always uses the fixed training class order, including when a replicate does
not contain every class. These intervals describe validation-set uncertainty;
they are not additional model-selection folds.

Run the ablation locally from PowerShell after generating the cutoff feature
table:

```powershell
cd D:\ML\realtime-disaster-ml
git fetch origin
git switch review/v2-ibtracs-feature-block-ablation
git pull --ff-only origin review/v2-ibtracs-feature-block-ablation

python -m unittest discover -s ml/tests -p "test_*.py" -v
python -m compileall -q ml\preprocessing ml\training ml\tests
python -m ml.training.ibtracs_feature_block_ablation
```

The new local output is:

```text
datasets/processed/v2_ibtracs_feature_block_ablation.json
```

The output directory is Git-ignored. The script refuses to overwrite an
existing ablation result, never saves a fitted model, and never reads a feature
table containing the 2023-2025 test split. Preserve the old result by renaming
it before an intentional rerun. Historical EM-DAT `magnitude` remains in every
configuration for a controlled comparison; whether it was available at T0 is
still unresolved.
