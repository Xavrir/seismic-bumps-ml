"""Freeze one final policy from lockbox guarded CV summary (dev-only selection)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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


def _build_policy(winner: pd.Series, lockbox_row: pd.Series | None) -> dict:
    policy = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_source": str(LOCKBOX_SUMMARY_CSV),
        "selection_scope": "dev-only repeated CV summary; lockbox used only as confirmatory snapshot",
        "selection_objective": "max dev-cv hazardous_f2 mean, tie by recall then auc",
        "model": str(winner["model"]),
        "variant": str(winner["variant"]),
        "operating_threshold": float(winner["threshold_mean"]),
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

    if lockbox_row is None:
        return policy

    policy["lockbox_metrics"] = {
        "hazardous_recall": float(lockbox_row["hazardous_recall"]),
        "hazardous_precision": float(lockbox_row["hazardous_precision"]),
        "hazardous_f2": float(lockbox_row["hazardous_f2"]),
        "roc_auc": float(lockbox_row["roc_auc"]),
        "accuracy": float(lockbox_row["accuracy"]),
    }
    if "hyperparams_json" in lockbox_row.index:
        policy["hyperparams"] = json.loads(str(lockbox_row["hyperparams_json"]))
    return policy


def _lockbox_line(lockbox_row: pd.Series | None) -> str:
    if lockbox_row is None:
        return "- Not available in this freeze run."
    return (
        f"- Lockbox recall: `{lockbox_row['hazardous_recall']:.3f}` | "
        f"precision: `{lockbox_row['hazardous_precision']:.3f}` | "
        f"F2: `{lockbox_row['hazardous_f2']:.3f}` | AUC: `{lockbox_row['roc_auc']:.3f}`"
    )


def _write_report(winner: pd.Series, lockbox_row: pd.Series | None) -> None:
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
    _write_report(winner, lockbox_row)

    print("=== FINAL POLICY FROZEN ===")
    print(f"Model={winner['model']} Variant={winner['variant']}")
    print(f"Threshold={winner['threshold_mean']:.3f} WatchFloor=0.30")
    print(f"Saved: {POLICY_JSON}")
    print(f"Saved: {RANKED_CSV}")
    print(f"Saved: {POLICY_REPORT_MD}")


if __name__ == "__main__":
    main()
