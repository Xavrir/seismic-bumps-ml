# Week-1 Improvement Sprint Report

## Scope
Executed a practical 7-step improvement plan for Seismic Bumps hazardous-shift classification.

## Best Configuration
- Stage: `day1_baseline`
- Model: `logreg`
- Variant: `class_weight_balanced`
- Operating threshold: `0.550`

## Best Metrics (Validation/Test)
- Validation hazardous recall: `0.600`
- Validation hazardous F2: `0.412`
- Test hazardous recall: `0.577`
- Test hazardous F2: `0.431`
- Test ROC-AUC: `0.737`

## Robustness (Repeated Stratified CV, 5x5)
- Recall mean ± std: `0.617 ± 0.077`
- F2 mean ± std: `0.392 ± 0.045`
- AUC mean ± std: `0.758 ± 0.027`

## Decision
- Production readiness verdict: **NO-GO**

Decision rule used:
- GO only if recall >= 0.70, F2 >= 0.48, and CV recall std <= 0.10.

## Notes
- This sprint improves operating decisions and uncertainty estimation.
- If verdict is NO-GO, prioritize more hazardous samples and temporal validation data.
