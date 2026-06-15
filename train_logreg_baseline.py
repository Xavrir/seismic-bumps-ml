"""
Train and evaluate a Logistic Regression baseline on the Seismic Bumps dataset.

Logistic Regression is our baseline because it is interpretable
and linear. If a complex model only beats it modestly, the data may not need
much complexity. Starting here gives us a meaningful benchmark.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

from splits import make_splits
from preprocess import build_pipeline
from metrics import compute_metrics, select_threshold

RANDOM_STATE = 42
MODEL_NAME = "logreg"
ARTIFACTS_DIR = Path("artifacts")
FIGURES_DIR = Path("reports/figures")


def _save_metrics_csv(filepath: Path, new_row: dict, model_name: str) -> None:
    """Save a metrics row to CSV, replacing any existing row for this model (idempotent)."""
    if filepath.exists():
        df = pd.read_csv(filepath)
        df = df[df["model"] != model_name]
    else:
        df = pd.DataFrame()

    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_csv(filepath, index=False)


def main():
    # 1. Load splits
    splits = make_splits(random_state=RANDOM_STATE)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]

    # 2. Build and fit preprocessing pipeline on TRAIN ONLY
    pipeline = build_pipeline("logreg")
    X_train_proc = pipeline.fit_transform(X_train)
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)

    # 3. Train LogisticRegression with class_weight='balanced'
    model = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(X_train_proc, y_train)

    # 4. Get probability scores (predict_proba[:,1] for positive class)
    val_scores = model.predict_proba(X_val_proc)[:, 1]
    test_scores = model.predict_proba(X_test_proc)[:, 1]

    # 5. Select optimal threshold on validation set (maximizes F2)
    threshold = select_threshold(y_val, val_scores, beta=2.0)
    print(f"Selected threshold (maximizes F2 on validation): {threshold:.4f}")

    # 6. Apply threshold and compute metrics
    val_pred = (val_scores >= threshold).astype(int)
    test_pred = (test_scores >= threshold).astype(int)

    val_metrics = compute_metrics(
        y_val, val_pred, y_score=val_scores, threshold=threshold
    )
    test_metrics = compute_metrics(
        y_test, test_pred, y_score=test_scores, threshold=threshold
    )

    # 7. Print results clearly
    print()
    print("=== LOGISTIC REGRESSION BASELINE ===")
    print()
    print("--- VALIDATION SET ---")
    print(f"  Threshold:          {threshold:.4f}")
    print(f"  Hazardous Recall:   {val_metrics['hazardous_recall']:.4f}")
    print(f"  Hazardous Precision:{val_metrics['hazardous_precision']:.4f}")
    print(f"  Hazardous F1:       {val_metrics['hazardous_f1']:.4f}")
    print(f"  Hazardous F2:       {val_metrics['hazardous_f2']:.4f}")
    print(f"  ROC-AUC:            {val_metrics['roc_auc']:.4f}")
    print(f"  Accuracy:           {val_metrics['accuracy']:.4f}")
    print(
        f"  Confusion (val):    TN={val_metrics['tn']} FP={val_metrics['fp']}"
        f" FN={val_metrics['fn']} TP={val_metrics['tp']}"
    )
    print()
    print("--- TEST SET (final, one-time report) ---")
    print(f"  Hazardous Recall:   {test_metrics['hazardous_recall']:.4f}")
    print(f"  Hazardous Precision:{test_metrics['hazardous_precision']:.4f}")
    print(f"  Hazardous F1:       {test_metrics['hazardous_f1']:.4f}")
    print(f"  Hazardous F2:       {test_metrics['hazardous_f2']:.4f}")
    print(f"  ROC-AUC:            {test_metrics['roc_auc']:.4f}")
    print(f"  Accuracy:           {test_metrics['accuracy']:.4f}")
    print(
        f"  Confusion (test):   TN={test_metrics['tn']} FP={test_metrics['fp']}"
        f" FN={test_metrics['fn']} TP={test_metrics['tp']}"
    )
    print()
    print(
        "Artifacts saved to artifacts/model_metrics.csv and artifacts/validation_metrics.csv"
    )

    # 8. Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    hyperparams = json.dumps(
        {"class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"}
    )

    # 8a. Test metrics → model_metrics.csv
    test_row = {
        "model": MODEL_NAME,
        "variant": "class_weight_balanced",
        "split": "test",
        "seed": RANDOM_STATE,
        "threshold": threshold,
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
        "hyperparams_json": hyperparams,
    }
    _save_metrics_csv(ARTIFACTS_DIR / "model_metrics.csv", test_row, MODEL_NAME)

    # 8b. Validation metrics → validation_metrics.csv
    val_row = {
        "model": MODEL_NAME,
        "variant": "class_weight_balanced",
        "split": "validation",
        "seed": RANDOM_STATE,
        "threshold": threshold,
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
        "hyperparams_json": hyperparams,
        "selection_rank": 1,
    }
    _save_metrics_csv(ARTIFACTS_DIR / "validation_metrics.csv", val_row, MODEL_NAME)

    # 8c. Confusion matrix figure
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, test_pred, display_labels=["Non-hazardous", "Hazardous"]
    )
    disp.ax_.set_title(f"Logistic Regression — Test Set\n(threshold={threshold:.3f})")
    disp.figure_.savefig(
        FIGURES_DIR / "logreg_confusion_matrix.png", dpi=100, bbox_inches="tight"
    )
    plt.close("all")



if __name__ == "__main__":
    main()
