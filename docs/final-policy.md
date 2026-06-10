# Final Policy Freeze

## Selected Policy

- Model: `logreg`
- Variant: `baseline_lockbox`
- Operating threshold (from dev-CV mean): `0.512`
- Watch floor: `0.30`
- Danger rule: `dangerous if p >= threshold`

## Why This Policy

- Selection objective: maximize dev-CV hazardous F2 mean.
- Winner CV F2 mean: `0.397` (std `0.038`)
- Winner CV recall mean: `0.643` (std `0.145`)
- Winner CV AUC mean: `0.756` (std `0.039`)

## Cost-Based Operating Point

- Cost matrix: miss (FN) = `10`, false alarm (FP) = `1` (ratio `10:1`)
- Cost-optimal threshold (dev OOF): `0.080`
- Active threshold basis: `f2` → operating threshold `0.080`

## Lockbox Snapshot (confirmatory only)

- Lockbox recall: `0.577` | precision: `0.120` | F2: `0.328` | AUC: `0.735`

## Artifacts

- `artifacts/final_policy/final_policy.json`
- `artifacts/final_policy/final_policy_ranked_candidates.csv`
