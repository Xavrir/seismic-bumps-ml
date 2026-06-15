"""
Exploratory Data Analysis (EDA) for the UCI Seismic Bumps dataset.

This script loads the cleaned dataset, quantifies class imbalance, profiles
feature types, visualizes distributions and correlations, and prints a concise
summary.  All figures are saved to reports/figures/ for inclusion in reports.

Run from the project root:
    python3 scripts/run_eda.py
"""

import sys
from pathlib import Path

# Allow imports from project root (e.g. `from load_arff import ...`)
sys.path.insert(0, ".")

# Use non-interactive backend BEFORE importing pyplot — no display available
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from load_arff import load_seismic_bumps

# ---------------------------------------------------------------------------
# ENCODING DECISION: seismic/seismoacoustic/ghazard use ordinal encoding
# {a:0, b:1, c:2, d:3} because these represent severity levels where ordering
# carries meaning.  shift uses binary encoding {N:0, W:1}.
# We do NOT use one-hot encoding for ordinal features here — that would discard
# the severity ordering.
# ---------------------------------------------------------------------------

FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

ORDINAL_COLS = ["seismic", "seismoacoustic", "ghazard"]
NOMINAL_COLS = ["shift"]
CATEGORICAL_COLS = ORDINAL_COLS + NOMINAL_COLS
NUMERIC_COLS = [
    "genergy",
    "gpuls",
    "gdenergy",
    "gdpuls",
    "energy",
    "maxenergy",
    "nbumps",
    "nbumps2",
    "nbumps3",
    "nbumps4",
    "nbumps5",
    "nbumps6",
    "nbumps7",
    "nbumps89",
]
KEY_NUMERIC_COLS = ["energy", "maxenergy", "nbumps", "nbumps2", "nbumps3"]

TARGET = "class"


def _separator(title: str) -> None:
    """Print a scannable section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def section_class_imbalance(df: pd.DataFrame) -> None:
    """Quantify and visualize class imbalance."""
    _separator("SECTION 1: CLASS IMBALANCE")

    counts = df[TARGET].value_counts().sort_index()
    total = len(df)
    for label, count in counts.items():
        tag = "Non-hazardous" if label == 0 else "Hazardous"
        pct = count / total * 100
        print(f"  {tag} ({label}): {count:>5d}  ({pct:5.1f}%)")

    ratio = counts[0] / counts[1]
    print(f"\n  Imbalance ratio (majority / minority): {ratio:.1f} : 1")

    # --- bar chart ---
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Non-hazardous (0)", "Hazardous (1)"],
        [counts[0], counts[1]],
        color=["#4c72b0", "#dd8452"],
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, count in zip(bars, [counts[0], counts[1]]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{count}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution — Seismic Bumps")
    ax.set_ylim(0, counts[0] * 1.12)
    plt.tight_layout()

    path = FIGURE_DIR / "class_distribution.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {path}")


def section_categorical_features(df: pd.DataFrame) -> None:
    """Analyze categorical features broken down by class."""
    _separator("SECTION 2: CATEGORICAL FEATURE ANALYSIS")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, col in enumerate(CATEGORICAL_COLS):
        ct = pd.crosstab(df[col], df[TARGET])
        ct.columns = ["Non-haz (0)", "Hazardous (1)"]
        print(f"  --- {col} ---")
        print(ct.to_string(index=True))
        print()

        # stacked bar chart
        ct.plot(
            kind="bar",
            ax=axes[i],
            color=["#4c72b0", "#dd8452"],
            edgecolor="black",
            linewidth=0.5,
        )
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")
        axes[i].legend(title="class", fontsize=8)
        axes[i].tick_params(axis="x", rotation=0)

    fig.suptitle("Categorical Features by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = FIGURE_DIR / "feature_distributions_categorical.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {path}")


def section_numeric_features(df: pd.DataFrame) -> None:
    """Analyze key numeric features with stats and boxplots by class."""
    _separator("SECTION 3: NUMERIC FEATURE ANALYSIS")

    # Print per-class summary stats for key numeric columns
    for col in KEY_NUMERIC_COLS:
        print(f"  --- {col} ---")
        grouped = df.groupby(TARGET)[col].agg(["mean", "median", "std", "max"])
        grouped.index = ["Non-haz (0)", "Hazardous (1)"]
        print(grouped.to_string())
        print()

    # Identify near-zero columns (>95% zeros)
    print("  Near-zero columns (>95% zeros):")
    for col in NUMERIC_COLS:
        zero_frac = (df[col] == 0).mean()
        if zero_frac > 0.95:
            print(f"    {col}: {zero_frac:.1%} zeros")
    print()

    # --- boxplots ---
    n_cols = len(KEY_NUMERIC_COLS)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))

    for i, col in enumerate(KEY_NUMERIC_COLS):
        sns.boxplot(
            data=df,
            x=TARGET,
            y=col,
            hue=TARGET,
            ax=axes[i],
            palette={0: "#4c72b0", 1: "#dd8452"},
            fliersize=2,
            legend=False,
        )
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("class")

    fig.suptitle("Numeric Features by Class (Boxplots)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = FIGURE_DIR / "feature_distributions_numeric.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {path}")


def section_correlation(df: pd.DataFrame) -> None:
    """Compute and plot Spearman correlation heatmap for numeric features."""
    _separator("SECTION 4: SPEARMAN CORRELATION")

    # Include target in correlation to see what correlates with class
    numeric_df = df[NUMERIC_COLS + [TARGET]]
    corr = numeric_df.corr(method="spearman")

    # Print top correlates with target
    target_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    print("  Top correlations with target (|Spearman rho|):")
    for feat, rho in target_corr.head(7).items():
        sign_rho = corr.loc[feat, TARGET]
        print(f"    {feat:>16s}: {sign_rho:+.3f}")
    print()

    # --- heatmap ---
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Spearman Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = FIGURE_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {path}")


def print_final_summary(df: pd.DataFrame) -> None:
    """Print a concise EDA summary block."""
    _separator("EDA SUMMARY")

    print(f"  Dataset shape: {df.shape}")
    print(f"  Target column: '{TARGET}' (0=non-hazardous, 1=hazardous)")
    print()

    # Column type profile
    print("  Feature types:")
    print(f"    Ordinal categorical : {ORDINAL_COLS}")
    print(f"    Nominal categorical : {NOMINAL_COLS}")
    print(
        f"    Numeric ({len(NUMERIC_COLS)} cols)    : {NUMERIC_COLS[:4]} + {len(NUMERIC_COLS) - 4} more"
    )
    print()

    counts = df[TARGET].value_counts().sort_index()
    print(f"  Class 0 (non-hazardous): {counts[0]}")
    print(f"  Class 1 (hazardous)    : {counts[1]}")
    print(f"  Imbalance ratio        : {counts[0] / counts[1]:.1f} : 1")
    print()

    # Encoding plan
    print("  Encoding plan:")
    print("    seismic / seismoacoustic / ghazard → ordinal {a:0, b:1, c:2, d:3}")
    print("    shift → binary {N:0, W:1}")
    print()

    # Teaching moment
    print("  ─── Why this matters ───")
    print("  We quantify class imbalance BEFORE modelling because a naive classifier")
    print("  that always predicts 'non-hazardous' would score ~93% accuracy — a")
    print("  misleading result. Knowing the 14:1 skew upfront tells us we need")
    print(
        "  class-weighted losses, stratified splits, and recall-focused metrics (F2)."
    )
    print()
    print("  We use Spearman (not Pearson) correlation because many features are")
    print("  ordinal or heavily skewed with outliers. Spearman measures monotonic")
    print("  relationships via ranks, making it robust to non-normality — exactly")
    print("  what we need for bump-count and energy distributions with long tails.")
    print()


def main() -> None:
    """Run the full EDA pipeline."""
    print("Loading seismic bumps dataset...")
    df = load_seismic_bumps()
    print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")

    section_class_imbalance(df)
    section_categorical_features(df)
    section_numeric_features(df)
    section_correlation(df)
    print_final_summary(df)

    print("EDA complete. All figures saved to reports/figures/")


if __name__ == "__main__":
    main()
