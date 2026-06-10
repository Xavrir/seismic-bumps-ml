"""Evaluate calibration quality and report honest metric uncertainty.

This script does NOT change the model or policy. It quantifies how trustworthy
the displayed risk score is (Brier score + expected calibration error, before vs
after calibration) and how uncertain the headline lockbox metrics are given the
tiny hazardous sample (bootstrap 95% confidence intervals). It also renders a
reliability diagram.

Run after scripts/build_final_model_bundle.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from scripts.build_final_model_bundle import (
    MODEL_TO_TYPE,
    _build_classifier,
    _load_model_params,
    _load_policy,
    _numeric_cols,
    _read_policy_fields,
    _split_dev_and_lockbox,
)
from src.app.scoring import load_model_bundle
from src.features.preprocess import build_pipeline
from src.models.metrics import compute_metrics
from src.viz.console_theme import PALETTE, apply_console_theme

RANDOM_STATE = 42
N_BOOTSTRAP = 2000
N_RELIABILITY_BINS = 10

ARTIFACTS_DIR = Path("artifacts")
FINAL_POLICY_DIR = ARTIFACTS_DIR / "final_policy"
METRIC_CIS_CSV = FINAL_POLICY_DIR / "lockbox_metric_cis.csv"
CALIBRATION_SUMMARY_CSV = FINAL_POLICY_DIR / "calibration_summary.csv"
RELIABILITY_PNG = Path("reports/figures/calibration_reliability.png")


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_RELIABILITY_BINS
) -> float:
    """Bin-weighted average gap between confidence and observed frequency."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= low) & (y_prob < high)
        if high == 1.0:
            mask |= y_prob == 1.0
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(y_prob[mask].mean())
        observed = float(y_true[mask].mean())
        ece += (count / total) * abs(confidence - observed)
    return ece


def _bootstrap_metric_cis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_boot: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """Bootstrap 95% CIs for the headline hazardous-class metrics."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    keys = ["hazardous_recall", "hazardous_precision", "hazardous_f2", "roc_auc"]
    samples: dict[str, list[float]] = {key: [] for key in keys}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue  # AUC/recall undefined without both classes
        metrics = compute_metrics(
            yt, (ys >= threshold).astype(int), y_score=ys, threshold=threshold
        )
        for key in keys:
            samples[key].append(float(metrics[key]))

    point = compute_metrics(
        y_true, (y_score >= threshold).astype(int), y_score=y_score, threshold=threshold
    )
    rows = []
    for key in keys:
        draws = np.asarray(samples[key], dtype=float)
        rows.append(
            {
                "metric": key,
                "point_estimate": float(point[key]),
                "ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5)),
                "n_bootstrap": int(len(draws)),
            }
        )
    return pd.DataFrame(rows)


def _reconstruct_uncalibrated_scores(
    X_dev, X_lockbox, y_dev
) -> np.ndarray:
    """Re-fit the uncalibrated base model to get pre-calibration probabilities."""
    policy = _load_policy()
    model_name, variant, _, _ = _read_policy_fields(policy)
    params = _load_model_params(model_name, variant, policy)
    model_type = MODEL_TO_TYPE[model_name]

    pipeline = build_pipeline(model_type, numeric_cols=_numeric_cols(X_dev))
    X_dev_proc = pipeline.fit_transform(X_dev)
    base_model = _build_classifier(model_name, params, y_dev)
    base_model.fit(X_dev_proc, y_dev)
    return np.asarray(
        base_model.predict_proba(pipeline.transform(X_lockbox)), dtype=float
    )[:, 1]


def _plot_reliability(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    cal_scores: np.ndarray,
    threshold: float,
) -> None:
    RELIABILITY_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_curve, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    ax_curve.plot([0, 1], [0, 1], "--", color=PALETTE["faint"], label="Perfectly calibrated")
    for scores, label, color in [
        (raw_scores, "Uncalibrated", PALETTE["red"]),
        (cal_scores, "Calibrated", PALETTE["blue"]),
    ]:
        edges = np.linspace(0.0, 1.0, N_RELIABILITY_BINS + 1)
        xs, ys = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (scores >= low) & (scores < high)
            if high == 1.0:
                mask |= scores == 1.0
            if mask.sum() == 0:
                continue
            xs.append(float(scores[mask].mean()))
            ys.append(float(y_true[mask].mean()))
        ax_curve.plot(xs, ys, "o-", color=color, label=label)
    ax_curve.set_xlabel("Mean predicted probability")
    ax_curve.set_ylabel("Observed hazardous frequency")
    ax_curve.set_title("Reliability diagram (lockbox)")
    ax_curve.legend(loc="upper left")
    ax_curve.set_xlim(0, 1)
    ax_curve.set_ylim(0, 1)

    ax_hist.hist(cal_scores, bins=N_RELIABILITY_BINS, range=(0, 1), color=PALETTE["blue"], alpha=0.85)
    ax_hist.axvline(threshold, color=PALETTE["red"], linestyle="--", label=f"threshold={threshold:.3f}")
    ax_hist.set_xlabel("Calibrated predicted probability")
    ax_hist.set_ylabel("Lockbox shift count")
    ax_hist.set_title("Calibrated score distribution")
    ax_hist.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(RELIABILITY_PNG, dpi=120)
    plt.close(fig)


def main() -> None:
    apply_console_theme()
    FINAL_POLICY_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle()
    threshold = float(bundle["threshold"])

    X_dev, X_lockbox, y_dev, y_lockbox = _split_dev_and_lockbox()
    y_true = y_lockbox.to_numpy()

    cal_scores = np.asarray(
        bundle["model"].predict_proba(bundle["pipeline"].transform(X_lockbox)),
        dtype=float,
    )[:, 1]
    raw_scores = _reconstruct_uncalibrated_scores(X_dev, X_lockbox, y_dev)

    raw_brier = brier_score_loss(y_true, raw_scores)
    cal_brier = brier_score_loss(y_true, cal_scores)
    raw_ece = _expected_calibration_error(y_true, raw_scores)
    cal_ece = _expected_calibration_error(y_true, cal_scores)

    calibration_summary = pd.DataFrame(
        [
            {"stage": "uncalibrated", "brier": raw_brier, "ece": raw_ece},
            {"stage": "calibrated", "brier": cal_brier, "ece": cal_ece},
        ]
    )
    calibration_summary.to_csv(CALIBRATION_SUMMARY_CSV, index=False)

    metric_cis = _bootstrap_metric_cis(y_true, cal_scores, threshold)
    metric_cis.to_csv(METRIC_CIS_CSV, index=False)

    _plot_reliability(y_true, raw_scores, cal_scores, threshold)

    print("=== CALIBRATION EVALUATION ===")
    print(f"Calibration method: {bundle.get('calibration_method', 'unknown')}")
    print(f"Brier: uncalibrated={raw_brier:.4f} -> calibrated={cal_brier:.4f}")
    print(f"ECE:   uncalibrated={raw_ece:.4f} -> calibrated={cal_ece:.4f}")
    print("Lockbox metric 95% CIs:")
    for _, row in metric_cis.iterrows():
        print(
            f"  {row['metric']:<20} {row['point_estimate']:.3f} "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        )
    print(f"Saved: {CALIBRATION_SUMMARY_CSV}")
    print(f"Saved: {METRIC_CIS_CSV}")
    print(f"Saved: {RELIABILITY_PNG}")


if __name__ == "__main__":
    main()
