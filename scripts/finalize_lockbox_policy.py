"""Freeze one final policy from lockbox guarded CV summary (dev-only selection)."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# --- Cost model (explicit, auditable safety assumption) -------------------
# In a rockburst-warning context a missed hazard (false negative) is far more
# costly than a false alarm (false positive). We state that ratio explicitly so
# the operating threshold follows from a documented policy decision instead of
# being whatever maximized F2 on a tiny validation fold. Default 10:1 mirrors
# the README's safety-first rationale; change COST_FN/COST_FP to re-derive.
COST_FN = 10.0  # cost of missing a hazardous shift
COST_FP = 1.0  # cost of a false alarm
COST_RATIO = COST_FN / COST_FP

# Whether the active operating threshold is the F2-selected one (current,
# backward-compatible default) or the cost-optimal one. Both are always recorded
# in the policy so the choice is transparent; flip to "expected_cost" to adopt it.
THRESHOLD_BASIS = "f2"

ARTIFACTS_DIR = Path("artifacts")
FINAL_POLICY_DIR = ARTIFACTS_DIR / "final_policy"

LOCKBOX_SUMMARY_CSV = ARTIFACTS_DIR / "external_transfer_lockbox_summary.csv"
LOCKBOX_RESULTS_CSV = ARTIFACTS_DIR / "external_transfer_lockbox_results.csv"

POLICY_JSON = FINAL_POLICY_DIR / "final_policy.json"
RANKED_CSV = FINAL_POLICY_DIR / "final_policy_ranked_candidates.csv"
POLICY_REPORT_MD = Path("docs/final-policy.md")


def _load_summary() -> pd.DataFrame:
    if not LOCKBOX_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Missing {LOCKBOX_SUMMARY_CSV}. Run scripts/run_external_transfer_lockbox.py first."
        )
    return pd.read_csv(LOCKBOX_SUMMARY_CSV)


def _rank_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.sort_values(
        ["f2_mean", "recall_mean", "auc_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranked.insert(
        0,
        "rank",
        pd.Series(range(1, len(ranked) + 1), index=ranked.index, dtype="int64"),
    )
    return ranked


def _load_matching_lockbox_row(winner: pd.Series) -> pd.Series | None:
    if not LOCKBOX_RESULTS_CSV.exists():
        return None

    lockbox = pd.read_csv(LOCKBOX_RESULTS_CSV)
    match = lockbox[
        (lockbox["model"] == winner["model"])
        & (lockbox["variant"] == winner["variant"])
    ]
    if match.empty:
        return None
    return match.iloc[0]


def _dev_oof_scores(model_name: str, variant: str, policy_params: dict | None):
    """Out-of-fold CALIBRATED hazardous probabilities on the dev split.

    The deployed bundle calibrates probabilities, which shifts the score scale to
    the true base rate. The operating threshold must be derived on that same
    calibrated scale, so we wrap the classifier in CalibratedClassifierCV here too
    (matching build_final_model_bundle's method selection).

    Imported lazily so freezing the policy from cached CSVs alone (the common
    path) does not require the training stack to be installed.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    from build_final_model_bundle import (
        CALIBRATION_CV,
        ISOTONIC_MIN_POSITIVES,
        MODEL_TO_TYPE,
        _build_classifier,
        _numeric_cols,
        _split_dev_and_lockbox,
    )
    from sklearn.calibration import CalibratedClassifierCV
    from preprocess import build_pipeline

    X_dev, _, y_dev, _ = _split_dev_and_lockbox()
    model_type = MODEL_TO_TYPE[model_name]
    pre = build_pipeline(model_type, numeric_cols=_numeric_cols(X_dev))
    params = dict(policy_params or {})
    params.pop("external_prior", None)
    clf = _build_classifier(model_name, params, y_dev)

    method = "isotonic" if int((y_dev == 1).sum()) >= ISOTONIC_MIN_POSITIVES else "sigmoid"
    calibrated = CalibratedClassifierCV(clf, method=method, cv=CALIBRATION_CV)
    estimator = Pipeline(
        steps=[
            ("preprocessor", pre.named_steps["preprocessor"]),
            ("classifier", calibrated),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(estimator, X_dev, y_dev, cv=cv, method="predict_proba")
    return y_dev.to_numpy(), np.asarray(proba, dtype=float)[:, 1]


def _select_cost_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Threshold minimizing expected cost = COST_FN*FN + COST_FP*FP."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_cost = float("inf")
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = COST_FN * fn + COST_FP * fp
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)
    return best_threshold, best_cost


def _compute_cost_threshold(winner: pd.Series, policy: dict) -> dict | None:
    """Derive calibrated F2 and cost-optimal thresholds; None if stack absent."""
    try:
        from metrics import select_threshold

        y_true, y_score = _dev_oof_scores(
            str(winner["model"]), str(winner["variant"]), policy.get("hyperparams")
        )
    except Exception as exc:  # training deps missing or data unavailable
        print(f"[cost-threshold] skipped: {exc}")
        return None

    cost_threshold, cost = _select_cost_threshold(y_true, y_score)
    calibrated_f2_threshold = float(select_threshold(y_true, y_score, beta=2.0))
    return {
        "calibrated_f2_threshold": calibrated_f2_threshold,
        "cost_matrix": {"fn": COST_FN, "fp": COST_FP, "ratio": COST_RATIO},
        "cost_optimal_threshold": cost_threshold,
        "cost_optimal_expected_cost": cost,
        "cost_selection_scope": "dev-only out-of-fold CALIBRATED scores (5-fold)",
    }


def _build_policy(winner: pd.Series, lockbox_row: pd.Series | None) -> dict:
    f2_threshold = float(winner["threshold_mean"])
    policy = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_source": str(LOCKBOX_SUMMARY_CSV),
        "selection_scope": "dev-only repeated CV summary; lockbox used only as confirmatory snapshot",
        "selection_objective": "max dev-cv hazardous_f2 mean, tie by recall then auc",
        "model": str(winner["model"]),
        "variant": str(winner["variant"]),
        "operating_threshold": f2_threshold,
        "f2_threshold": f2_threshold,
        "threshold_basis": THRESHOLD_BASIS,
        "watch_floor": 0.30,
        "danger_rule": "dangerous if predicted_probability >= operating_threshold",
        "cv_metrics": {
            "f2_mean": float(winner["f2_mean"]),
            "f2_std": float(winner["f2_std"]),
            "recall_mean": float(winner["recall_mean"]),
            "recall_std": float(winner["recall_std"]),
            "auc_mean": float(winner["auc_mean"]),
            "auc_std": float(winner["auc_std"]),
        },
    }

    if lockbox_row is not None:
        policy["lockbox_metrics"] = {
            "hazardous_recall": float(lockbox_row["hazardous_recall"]),
            "hazardous_precision": float(lockbox_row["hazardous_precision"]),
            "hazardous_f2": float(lockbox_row["hazardous_f2"]),
            "roc_auc": float(lockbox_row["roc_auc"]),
            "accuracy": float(lockbox_row["accuracy"]),
        }
        if "hyperparams_json" in lockbox_row.index:
            policy["hyperparams"] = json.loads(str(lockbox_row["hyperparams_json"]))

    cost_info = _compute_cost_threshold(winner, policy)
    if cost_info is not None:
        policy.update(cost_info)
        # The deployed bundle is calibrated, so the operating threshold must live on
        # the calibrated scale. The cached threshold_mean was tuned on uncalibrated
        # balanced scores and no longer applies.
        if THRESHOLD_BASIS == "expected_cost":
            policy["operating_threshold"] = cost_info["cost_optimal_threshold"]
        else:
            policy["operating_threshold"] = cost_info["calibrated_f2_threshold"]
        # Keep watch_floor strictly below the operating threshold.
        if policy["watch_floor"] >= policy["operating_threshold"]:
            policy["watch_floor"] = round(policy["operating_threshold"] / 2, 3)

    return policy


def _lockbox_line(lockbox_row: pd.Series | None) -> str:
    if lockbox_row is None:
        return "- Not available in this freeze run."
    return (
        f"- Lockbox recall: `{lockbox_row['hazardous_recall']:.3f}` | "
        f"precision: `{lockbox_row['hazardous_precision']:.3f}` | "
        f"F2: `{lockbox_row['hazardous_f2']:.3f}` | AUC: `{lockbox_row['roc_auc']:.3f}`"
    )


def _cost_section(policy: dict) -> str:
    if "cost_optimal_threshold" not in policy:
        return "- Cost-optimal threshold not computed in this run (training stack unavailable)."
    cm = policy["cost_matrix"]
    return (
        f"- Cost matrix: miss (FN) = `{cm['fn']:.0f}`, false alarm (FP) = `{cm['fp']:.0f}` "
        f"(ratio `{cm['ratio']:.0f}:1`)\n"
        f"- Cost-optimal threshold (dev OOF): `{policy['cost_optimal_threshold']:.3f}`\n"
        f"- Active threshold basis: `{policy['threshold_basis']}` "
        f"→ operating threshold `{policy['operating_threshold']:.3f}`"
    )


def _write_report(
    winner: pd.Series, lockbox_row: pd.Series | None, policy: dict
) -> None:
    report = f"""# Final Policy Freeze

## Selected Policy

- Model: `{winner["model"]}`
- Variant: `{winner["variant"]}`
- Operating threshold (from dev-CV mean): `{winner["threshold_mean"]:.3f}`
- Watch floor: `0.30`
- Danger rule: `dangerous if p >= threshold`

## Why This Policy

- Selection objective: maximize dev-CV hazardous F2 mean.
- Winner CV F2 mean: `{winner["f2_mean"]:.3f}` (std `{winner["f2_std"]:.3f}`)
- Winner CV recall mean: `{winner["recall_mean"]:.3f}` (std `{winner["recall_std"]:.3f}`)
- Winner CV AUC mean: `{winner["auc_mean"]:.3f}` (std `{winner["auc_std"]:.3f}`)

## Cost-Based Operating Point

{_cost_section(policy)}

## Lockbox Snapshot (confirmatory only)

{_lockbox_line(lockbox_row)}

## Artifacts

- `{POLICY_JSON}`
- `{RANKED_CSV}`
"""
    POLICY_REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    FINAL_POLICY_DIR.mkdir(parents=True, exist_ok=True)

    summary = _load_summary()
    ranked = _rank_candidates(summary)
    ranked.to_csv(RANKED_CSV, index=False)

    winner = ranked.iloc[0]
    lockbox_row = _load_matching_lockbox_row(winner)

    policy = _build_policy(winner, lockbox_row)
    POLICY_JSON.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    _write_report(winner, lockbox_row, policy)

    print("=== FINAL POLICY FROZEN ===")
    print(f"Model={winner['model']} Variant={winner['variant']}")
    print(f"Threshold={winner['threshold_mean']:.3f} WatchFloor=0.30")
    print(f"Saved: {POLICY_JSON}")
    print(f"Saved: {RANKED_CSV}")
    print(f"Saved: {POLICY_REPORT_MD}")


if __name__ == "__main__":
    main()
