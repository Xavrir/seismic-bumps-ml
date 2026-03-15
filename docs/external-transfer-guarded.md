# External Transfer Guarded Evaluation

## Objective

Reduce overfitting risk while testing external-prior features.

## Guarded Protocol

- Keep UCI Seismic Bumps as the only supervised target dataset.
- Use external USGS windows only as unlabeled reference features.
- Freeze the external reference before model comparison and keep it independent from UCI labels.
- Rank variants by repeated validation only (not test):
  - RepeatedStratifiedKFold: 5 splits x 5 repeats (25 folds per variant)
  - Single inner validation split inside each outer fold for threshold selection
  - Threshold objective: validation F2
- Touch holdout test once, after selecting the variant by CV summary.

## Variants Compared

1. `baseline_guarded`
2. `external_prior_guarded`

## CV Results (25 folds each)

| Variant | Recall mean | Recall std | F2 mean | F2 std | AUC mean | AUC std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline_guarded` | `0.600` | `0.128` | `0.390` | `0.047` | `0.756` | `0.030` |
| `external_prior_guarded` | `0.608` | `0.129` | `0.384` | `0.047` | `0.754` | `0.030` |

Selection by CV (F2 priority): **`baseline_guarded`**.

## Test-Once Result (selected variant only)

- Variant: `baseline_guarded`
- Recall: `0.577`
- Precision: `0.214`
- F2: `0.431`
- ROC-AUC: `0.737`
- Threshold: `0.550`

## Interpretation

The guarded procedure does not support a robust transfer gain from the current external-prior features. Although the external-prior variant has a slightly higher mean recall in CV, it has lower mean F2 and slightly lower AUC. The current evidence supports only this statement: under the chosen protocol and objective, baseline has stronger support than external prior for this dataset.

## Artifacts

- `artifacts/external_transfer_guarded_cv.csv`
- `artifacts/external_transfer_guarded_summary.csv`
- `artifacts/external_transfer_guarded_test_once.csv`
