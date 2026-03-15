# External Transfer Guarded Multi-Model Evaluation

## Purpose

Extend the anti-overfitting guarded protocol to multiple model families while keeping UCI Seismic Bumps as the only supervised target.

## Anti-Overfitting Controls

- External reference is frozen from USGS windows up to `2023-12-31`.
- Variant selection uses repeated development-set CV only (`5` folds x `3` repeats).
- Threshold is selected on inner validation in each fold using F2.
- Holdout test is evaluated once per selected model variant.
- Base hyperparameters are loaded from `artifacts/validation_metrics.csv` and are treated as fixed inputs for this run.

## Models

- `logreg`
- `random_forest`
- `xgboost`

Each model compares:

1. `baseline_guarded`
2. `external_prior_guarded`

## CV Selection Summary

| Model | Selected Variant | CV F2 mean | CV Recall mean | CV AUC mean |
| :--- | :--- | :--- | :--- | :--- |
| `logreg` | `external_prior_guarded` | `0.400` | `0.636` | `0.753` |
| `random_forest` | `baseline_guarded` | `0.372` | `0.663` | `0.758` |
| `xgboost` | `external_prior_guarded` | `0.390` | `0.621` | `0.751` |

## Test-Once Results (Selected Variants)

| Model | Variant | Threshold | Recall | Precision | F2 | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `logreg` | `external_prior_guarded` | `0.56` | `0.577` | `0.190` | `0.410` | `0.729` |
| `random_forest` | `baseline_guarded` | `0.45` | `0.500` | `0.333` | `0.455` | `0.762` |
| `xgboost` | `external_prior_guarded` | `0.51` | `0.462` | `0.185` | `0.355` | `0.738` |

## Interpretation

The guarded protocol remains valid and prevents test-driven variant selection, but external prior features are not consistently beneficial across models.

- `logreg`: selected external prior by CV, but test performance remains in the same range as prior guarded analyses.
- `random_forest`: CV and test both favor baseline.
- `xgboost`: CV selects external prior; current test-once result remains relatively low under F2 objective.

This indicates transfer effect is weak and model-dependent, not a robust improvement yet.

## Artifacts

- `artifacts/external_transfer_guarded_multimodel_cv.csv`
- `artifacts/external_transfer_guarded_multimodel_summary.csv`
- `artifacts/external_transfer_guarded_multimodel_selected.csv`
- `artifacts/external_transfer_guarded_multimodel_test_once.csv`

## Reproduce

This report is kept as an archived experiment summary. The active guarded workflow is:

```bash
python3 scripts/run_external_transfer_lockbox.py
```
