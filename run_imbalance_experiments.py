"""
Evaluate imbalance-handling strategies on the top-2 models (logreg, random_forest).

For each model we test four strategies:
  1. default           — no special imbalance handling
  2. class_weight_balanced — class_weight='balanced' on the classifier
  3. smote             — SMOTE oversampling on train (class_weight=None)
  4. smote_balanced    — SMOTE + class_weight='balanced'

Hyperparameters are FIXED (no search) — the goal is to isolate the effect of
each imbalance strategy on the same model architecture.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from splits import make_splits
from preprocess import build_pipeline
from metrics import compute_metrics, select_threshold

RANDOM_STATE = 42
ARTIFACTS_DIR = Path("artifacts")
CSV_PATH = ARTIFACTS_DIR / "imbalance_experiments.csv"

CSV_COLUMNS = [
    "model",
    "variant",
    "imbalance_strategy",
    "seed",
    "threshold",
    "split",
    "hazardous_precision",
    "hazardous_recall",
    "hazardous_f1",
    "hazardous_f2",
    "roc_auc",
    "accuracy",
    "tn",
    "fp",
    "fn",
    "tp",
]

STRATEGIES = ["default", "class_weight_balanced", "smote", "smote_balanced"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_rf_best_params() -> dict:
    """Read RF best hyperparameters from validation_metrics.csv and strip classifier__ prefix."""
    val_df = pd.read_csv(ARTIFACTS_DIR / "validation_metrics.csv")
    rf_row = val_df[val_df["model"] == "random_forest"].iloc[0]
    raw_params = json.loads(rf_row["hyperparams_json"])

    params = {k.replace("classifier__", ""): v for k, v in raw_params.items()}
    params.pop("class_weight", None)

    return params


def _save_row_idempotent(new_row: dict) -> None:
    """Append a row to imbalance_experiments.csv, replacing any matching existing row."""
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=CSV_COLUMNS)  # type: ignore[arg-type]

    mask = (
        (df["model"] == new_row["model"])
        & (df["imbalance_strategy"] == new_row["imbalance_strategy"])
        & (df["split"] == new_row["split"])
    )
    df = df[~mask]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)


def _build_pipeline(model_type: str, classifier, use_smote: bool):
    """
    Build either a sklearn Pipeline or imblearn Pipeline with SMOTE.

    Args:
        model_type: 'logreg' or 'rf' — determines preprocessing (scaling vs passthrough)
        classifier: Configured classifier instance
        use_smote: Whether to include SMOTE in the pipeline

    Returns:
        Fitted-ready pipeline (sklearn or imblearn)
    """
    preprocess_pipe = build_pipeline(model_type)
    preprocessor = preprocess_pipe.named_steps["preprocessor"]

    if use_smote:
        return ImbPipeline(
            [
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("classifier", classifier),
            ]
        )
    else:
        return SkPipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )


def _run_single_experiment(
    model_name: str,
    variant: str,
    model_type: str,
    strategy: str,
    classifier,
    use_smote: bool,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
) -> dict:
    """Train one model+strategy combo, compute val+test metrics, save to CSV."""
    pipeline = _build_pipeline(model_type, classifier, use_smote)
    pipeline.fit(X_train, y_train)

    val_scores = pipeline.predict_proba(X_val)[:, 1]
    test_scores = pipeline.predict_proba(X_test)[:, 1]
    threshold = select_threshold(y_val, val_scores, beta=2.0)

    val_pred = (val_scores >= threshold).astype(int)
    test_pred = (test_scores >= threshold).astype(int)

    val_metrics = compute_metrics(
        y_val, val_pred, y_score=val_scores, threshold=threshold
    )
    test_metrics = compute_metrics(
        y_test, test_pred, y_score=test_scores, threshold=threshold
    )

    val_row = {
        "model": model_name,
        "variant": variant,
        "imbalance_strategy": strategy,
        "seed": RANDOM_STATE,
        "threshold": threshold,
        "split": "validation",
        "hazardous_precision": val_metrics["hazardous_precision"],
        "hazardous_recall": val_metrics["hazardous_recall"],
        "hazardous_f1": val_metrics["hazardous_f1"],
        "hazardous_f2": val_metrics["hazardous_f2"],
        "roc_auc": val_metrics["roc_auc"],
        "accuracy": val_metrics["accuracy"],
        "tn": val_metrics["tn"],
        "fp": val_metrics["fp"],
        "fn": val_metrics["fn"],
        "tp": val_metrics["tp"],
    }
    _save_row_idempotent(val_row)

    test_row = {
        "model": model_name,
        "variant": variant,
        "imbalance_strategy": strategy,
        "seed": RANDOM_STATE,
        "threshold": threshold,
        "split": "test",
        "hazardous_precision": test_metrics["hazardous_precision"],
        "hazardous_recall": test_metrics["hazardous_recall"],
        "hazardous_f1": test_metrics["hazardous_f1"],
        "hazardous_f2": test_metrics["hazardous_f2"],
        "roc_auc": test_metrics["roc_auc"],
        "accuracy": test_metrics["accuracy"],
        "tn": test_metrics["tn"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "tp": test_metrics["tp"],
    }
    _save_row_idempotent(test_row)

    return {"val": val_metrics, "test": test_metrics, "threshold": threshold}


# ---------------------------------------------------------------------------
# Model experiment runners
# ---------------------------------------------------------------------------


def run_logreg_experiments(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Run all 4 imbalance strategies for LogisticRegression."""
    results = {}

    configs = {
        "default": {"class_weight": None, "use_smote": False},
        "class_weight_balanced": {"class_weight": "balanced", "use_smote": False},
        "smote": {"class_weight": None, "use_smote": True},
        "smote_balanced": {"class_weight": "balanced", "use_smote": True},
    }

    for strategy, cfg in configs.items():
        print(f"  Training logreg / {strategy} ...")
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight=cfg["class_weight"],
        )
        result = _run_single_experiment(
            model_name="logreg",
            variant="baseline",
            model_type="logreg",
            strategy=strategy,
            classifier=clf,
            use_smote=cfg["use_smote"],
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
        results[strategy] = result

    return results


def run_rf_experiments(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Run all 4 imbalance strategies for RandomForest using best hyperparams from Task 5."""
    rf_params = _load_rf_best_params()
    print(f"  RF base hyperparams (from Task 5): {rf_params}")

    results = {}

    configs = {
        "default": {"class_weight": None, "use_smote": False},
        "class_weight_balanced": {"class_weight": "balanced", "use_smote": False},
        "smote": {"class_weight": None, "use_smote": True},
        "smote_balanced": {"class_weight": "balanced", "use_smote": True},
    }

    for strategy, cfg in configs.items():
        print(f"  Training random_forest / {strategy} ...")
        clf = RandomForestClassifier(
            **rf_params,
            class_weight=cfg["class_weight"],
            random_state=RANDOM_STATE,
        )
        result = _run_single_experiment(
            model_name="random_forest",
            variant="tuned",
            model_type="rf",
            strategy=strategy,
            classifier=clf,
            use_smote=cfg["use_smote"],
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
        results[strategy] = result

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_comparison_table(model_name: str, results: dict) -> str:
    """Print a before/after comparison table for one model. Returns best strategy name."""
    print(f"\n{'=' * 80}")
    print(f"=== {model_name.upper()} IMBALANCE EXPERIMENTS ===")
    print(f"{'=' * 80}")

    header = (
        f"{'Strategy':<25} | {'Val F2':>7} | {'Val Recall':>10} | "
        f"{'Test F2':>7} | {'Test Recall':>11} | {'Test AUC':>8}"
    )
    print(header)
    print("-" * len(header))

    best_strategy = "default"
    best_val_f2 = -1.0

    for strategy in STRATEGIES:
        r = results[strategy]
        val_f2 = r["val"]["hazardous_f2"]
        val_recall = r["val"]["hazardous_recall"]
        test_f2 = r["test"]["hazardous_f2"]
        test_recall = r["test"]["hazardous_recall"]
        test_auc = r["test"]["roc_auc"]

        print(
            f"{strategy:<25} | {val_f2:>7.3f} | {val_recall:>10.3f} | "
            f"{test_f2:>7.3f} | {test_recall:>11.3f} | {test_auc:>8.3f}"
        )

        if val_f2 > best_val_f2:
            best_val_f2 = val_f2
            best_strategy = strategy

    print(
        f"\nBest strategy for {model_name}: {best_strategy} (Val F2={best_val_f2:.3f})"
    )
    return best_strategy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 80)
    print("IMBALANCE HANDLING EXPERIMENTS")
    print("Models: logreg (rank 1), random_forest (rank 2)")
    print("Strategies: default, class_weight_balanced, smote, smote_balanced")
    print("=" * 80)

    splits = make_splits(random_state=RANDOM_STATE)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]

    print(f"\nTrain: {X_train.shape}, hazardous rate = {y_train.mean():.4f}")
    print(f"Val:   {X_val.shape}, hazardous rate = {y_val.mean():.4f}")
    print(f"Test:  {X_test.shape}, hazardous rate = {y_test.mean():.4f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- LogReg experiments ---
    print("\n--- Logistic Regression ---")
    logreg_results = run_logreg_experiments(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # --- Random Forest experiments ---
    print("\n--- Random Forest ---")
    rf_results = run_rf_experiments(X_train, y_train, X_val, y_val, X_test, y_test)

    # --- Print comparison tables ---
    best_logreg = _print_comparison_table("logreg", logreg_results)
    best_rf = _print_comparison_table("random_forest", rf_results)

    # --- Final teaching moment ---
    print("\n" + "=" * 80)
    print(
        "Teaching: Imbalance handling changes the bias of the model toward the minority class."
    )
    print("class_weight='balanced' is equivalent to reweighting the loss function.")
    print(
        "SMOTE creates synthetic minority samples, effectively increasing the minority proportion."
    )
    print(
        "Neither is universally better — the winning strategy depends on the data and the cost tradeoff."
    )
    print("=" * 80)

    # --- Verify CSV ---
    final_df = pd.read_csv(CSV_PATH)
    print(f"\nArtifact: {CSV_PATH} — {len(final_df)} rows written")
    assert len(final_df) >= 16, (
        f"Expected at least 16 rows (2 models x 4 strategies x 2 splits), got {len(final_df)}"
    )
    print("Verification: PASS (>= 16 rows)")


if __name__ == "__main__":
    main()
