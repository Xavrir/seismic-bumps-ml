"""Rebuild the in-app diagnostic figures in the dark console theme.

The Streamlit app embeds ``final_policy_diagnostics.png`` and ``danger_level_trend.png``,
but no script in the repo generated them (they were committed as light-mode assets that
clash with the dark UI). This script reconstructs both from the FROZEN artifacts only —
it loads the frozen bundle, replays the dev/lockbox split, and scores the lockbox set. It
never retrains and never writes to ``artifacts/``; it only writes the two PNGs.

Run from the project root:
    python3 scripts/render_console_figures.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_final_model_bundle import _split_dev_and_lockbox
from scoring import load_model_bundle
from risk_levels import to_risk_level
from console_theme import PALETTE, apply_console_theme

POLICY_JSON = Path("artifacts/final_policy/final_policy.json")
FIGURES_DIR = Path("reports/figures")
DIAGNOSTICS_PNG = FIGURES_DIR / "final_policy_diagnostics.png"
TREND_PNG = FIGURES_DIR / "danger_level_trend.png"

LEVEL_ORDER = ["low", "watch", "dangerous"]
LEVEL_COLOR = {
    "low": PALETTE["green"],
    "watch": PALETTE["orange"],
    "dangerous": PALETTE["red"],
}


def _plot_diagnostics(probs, y_true, watch_floor, threshold, lockbox_metrics) -> None:
    fig, (ax_dist, ax_metrics) = plt.subplots(1, 2, figsize=(12, 5))

    ax_dist.hist(
        probs[y_true == 0], bins=30, range=(0, 1), density=True,
        color=PALETTE["blue"], alpha=0.65, label="non-hazardous",
    )
    ax_dist.hist(
        probs[y_true == 1], bins=30, range=(0, 1), density=True,
        color=PALETTE["red"], alpha=0.65, label="hazardous",
    )
    ax_dist.axvline(watch_floor, color=PALETTE["orange"], ls="--", lw=1,
                    label=f"watch floor {watch_floor:.2f}")
    ax_dist.axvline(threshold, color=PALETTE["red"], ls="--", lw=1,
                    label=f"threshold {threshold:.3f}")
    ax_dist.set_xlim(0, 1)
    ax_dist.set_xlabel("calibrated hazard probability")
    ax_dist.set_ylabel("density")
    ax_dist.set_title("Calibrated score by true class (lockbox)")
    ax_dist.legend(fontsize=8, loc="upper right")

    names = ["recall", "precision", "f2", "auc"]
    values = [
        lockbox_metrics["hazardous_recall"],
        lockbox_metrics["hazardous_precision"],
        lockbox_metrics["hazardous_f2"],
        lockbox_metrics["roc_auc"],
    ]
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"]]
    bars = ax_metrics.bar(names, values, color=colors, edgecolor=PALETTE["faint"])
    for bar, value in zip(bars, values):
        ax_metrics.text(
            bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}",
            ha="center", va="bottom", color=PALETTE["text"], fontsize=9,
        )
    ax_metrics.set_ylim(0, 1)
    ax_metrics.set_ylabel("score")
    ax_metrics.set_title("Lockbox hazardous-class metrics")

    fig.tight_layout()
    DIAGNOSTICS_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIAGNOSTICS_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_trend(levels, y_true) -> None:
    levels = np.asarray(levels)
    totals = [int((levels == name).sum()) for name in LEVEL_ORDER]
    hazardous = [int(((levels == name) & (y_true == 1)).sum()) for name in LEVEL_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        LEVEL_ORDER, totals,
        color=[LEVEL_COLOR[name] for name in LEVEL_ORDER],
        edgecolor=PALETTE["faint"], alpha=0.85,
    )
    headroom = max(totals) * 0.01 if max(totals) else 1
    for bar, total, haz in zip(bars, totals, hazardous):
        ax.text(
            bar.get_x() + bar.get_width() / 2, total + headroom,
            f"{total} shifts\n{haz} hazardous",
            ha="center", va="bottom", color=PALETTE["text"], fontsize=8,
        )
    ax.set_ylabel("lockbox shift count")
    ax.set_title("Risk-level allocation (lockbox)")
    ax.set_ylim(0, max(totals) * 1.2 if max(totals) else 1)

    fig.tight_layout()
    TREND_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TREND_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_console_theme()

    policy = json.loads(POLICY_JSON.read_text(encoding="utf-8"))
    bundle = load_model_bundle()
    threshold = float(bundle["threshold"])
    watch_floor = float(bundle.get("watch_floor", 0.04))

    _, X_lockbox, _, y_lockbox = _split_dev_and_lockbox()
    y_true = y_lockbox.to_numpy()
    probs = np.asarray(
        bundle["model"].predict_proba(bundle["pipeline"].transform(X_lockbox)),
        dtype=float,
    )[:, 1]
    levels = [
        to_risk_level(p, threshold=threshold, watch_floor=watch_floor) for p in probs
    ]

    _plot_diagnostics(probs, y_true, watch_floor, threshold, policy["lockbox_metrics"])
    _plot_trend(levels, y_true)

    print(f"Saved: {DIAGNOSTICS_PNG}")
    print(f"Saved: {TREND_PNG}")


if __name__ == "__main__":
    main()
