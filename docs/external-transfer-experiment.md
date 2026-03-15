# External Transfer Experiment

## Goal

Use external USGS mining-event data carefully as an auxiliary prior, without pretending it has the same labels as UCI Seismic Bumps.

## Safe Design

- UCI Seismic Bumps remains the authoritative labeled benchmark.
- External USGS data is used only to build an unlabeled 8-hour reference distribution.
- We convert external windows into a compact feature space aligned with UCI activity and energy patterns.
- We then add prior features such as similarity and percentile scores to UCI rows.

This avoids raw label merging across different mines and domains.

## Experiment Setup

- External source: `artifacts/external/usgs_mining_events_aggregated_8h.csv`
- UCI evaluation: standard stratified 70/15/15 split
- Model family: Logistic Regression with `class_weight='balanced'`
- Threshold selection: validation-set F2 optimization only

## Results

| Variant | Threshold | Test Precision | Test Recall | Test F2 | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline_refit` | `0.55` | `0.214` | `0.577` | `0.431` | `0.737` |
| `external_prior_similarity` | `0.61` | `0.224` | `0.577` | `0.439` | `0.729` |

## Interpretation

- The cautious external prior gave a **small F2 improvement** (`0.431 -> 0.439`).
- Hazardous recall **did not improve**; it stayed at `0.577`.
- Precision improved slightly (`0.214 -> 0.224`), which reduced false alarms a bit.
- Validation F2 was actually lower (`0.412 -> 0.400`).
- ROC-AUC dropped slightly (`0.737 -> 0.729`), so this is not a clear overall win.

## Practical Conclusion

This experiment shows that external data from a different source can be used safely as a supporting signal, but the first simple transfer method is **not enough to materially improve hazard capture**.

The result is useful because it validates a careful methodology:

1. no raw cross-domain label merge,
2. no leakage into UCI test selection,
3. direct comparison against the same UCI holdout.

What it does **not** validate is a robust transfer gain. The evidence only supports a narrow statement: the external prior slightly reduced false alarms at one operating point, but it did not increase hazardous recall and it did not improve ranking quality.

## Next Step

If we continue, the next higher-value approach is to use the paper-based rockburst datasets as structured auxiliary supervision, not just USGS event priors.
