# Danger Level Definition

The model predicts a probability `p = P(hazardous | features)` for each 8-hour shift.

## Decision Rule

- `dangerous_flag = 1` if `p >= threshold`
- `dangerous_flag = 0` if `p < threshold`

`threshold` is frozen from development CV, not tuned on lockbox.

## Human-Friendly Risk Level

Given `watch_floor = 0.30`:

- `low` if `p < 0.30`
- `watch` if `0.30 <= p < threshold`
- `dangerous` if `p >= threshold`

## Risk Score

- `risk_score = round(100 * p)` (clamped to 0..100)

This provides a continuous score plus a thresholded safety decision.
