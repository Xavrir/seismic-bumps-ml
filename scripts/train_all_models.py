"""
Train Random Forest, SVM, and XGBoost on the Seismic Bumps dataset.
Append results to existing CSVs alongside the Logistic Regression baseline,
then print a final 4-model comparison table.

Teaching note: We compare four different classifier families on identical
train/validation/test splits so the comparison is fair. Each model sees the
same rows, uses the same F2-optimised threshold selection, and reports on the
held-out test set exactly once.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, make_scorer, fbeta_score
from xgboost import XGBClassifier

from splits import make_splits
from preprocess import build_pipeline
from metrics import compute_metrics, select_threshold

RANDOM_STATE = 42
ARTIFACTS_DIR = Path("artifacts")
FIGURES_DIR = Path("reports/figures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_metrics_csv(filepath: Path, new_row: dict, model_name: str) -> None:
    """Save a metrics row to CSV, replacing any existing row for this model (idempotent)."""
    if filepath.exists():
        df = pd.read_csv(filepath)
        df = df[df["model"] != model_name]
    else:
        df = pd.DataFrame()

    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_csv(filepath, index=False)


def _make_full_pipeline(model_type: str, classifier) -> Pipeline:
    """Build a preprocessing pipeline and append a classifier step."""
    preprocess_pipe = build_pipeline(model_type)
    # build_pipeline returns Pipeline([('preprocessor', ColumnTransformer(...))])
    # We append a classifier step to make it end-to-end for RandomizedSearchCV
    steps = list(preprocess_pipe.steps) + [("classifier", classifier)]
    return Pipeline(steps)


def _rank_validation_models(filepath: Path) -> None:
    """Re-rank all models in validation_metrics.csv by the selection criteria."""
    df = pd.read_csv(filepath)

    # Simplicity tiebreak: lower number = simpler model
    simplicity_order = {"logreg": 1, "random_forest": 2, "svm": 3, "xgboost": 4}
    df["_simplicity"] = df["model"].map(simplicity_order)

    # Sort: hazardous_f2 DESC, hazardous_recall DESC, roc_auc DESC, simplicity ASC
    df = df.sort_values(
        by=["hazardous_f2", "hazardous_recall", "roc_auc", "_simplicity"],
        ascending=[False, False, False, True],
    )
    df["selection_rank"] = range(1, len(df) + 1)
    df = df.drop(columns=["_simplicity"])
    df.to_csv(filepath, index=False)


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Random Forest with RandomizedSearchCV."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)
    print(
        "Teaching: Random Forest is an ensemble of decision trees that"
        " reduces overfitting through bagging (bootstrap aggregating)."
        " Each tree votes, and the majority wins. It handles non-linear"
        " relationships and is robust to feature scaling."
    )

    classifier = RandomForestClassifier(random_state=RANDOM_STATE)
    pipeline = _make_full_pipeline("rf", classifier)

    param_grid = {
        "classifier__n_estimators": [200, 400, 600],
        "classifier__max_depth": [None, 8, 16, 24],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", 0.5],
        "classifier__class_weight": [None, "balanced"],
    }

    f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)
    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=20,
        cv=5,
        scoring=f2_scorer,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("  Running RandomizedSearchCV (20 iterations, 5-fold CV)...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"  Best params: {search.best_params_}")

    # Threshold selection on validation set
    val_scores = best_model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, val_scores, beta=2.0)
    print(f"  Selected threshold (F2-optimal on validation): {threshold:.4f}")

    # Validation metrics
    val_pred = (val_scores >= threshold).astype(int)
    val_metrics = compute_metrics(
        y_val, val_pred, y_score=val_scores, threshold=threshold
    )

    # Test metrics
    test_scores = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= threshold).astype(int)
    test_metrics = compute_metrics(
        y_test, test_pred, y_score=test_scores, threshold=threshold
    )

    _print_split_results("Random Forest", threshold, val_metrics, test_metrics)

    return best_model, search, threshold, val_metrics, test_metrics


def train_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train SVM (RBF kernel) with RandomizedSearchCV."""
    print("\n" + "=" * 60)
    print("SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 60)
    print(
        "Teaching: SVM finds the maximum-margin hyperplane that separates"
        " classes in a (possibly transformed) feature space. With an RBF"
        " kernel it can capture complex non-linear boundaries. We use"
        " decision_function (not predict_proba) for threshold tuning."
    )

    # kernel='rbf' and class_weight='balanced' are FIXED (not in search grid)
    classifier = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    pipeline = _make_full_pipeline("svm", classifier)

    param_grid = {
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", 0.01, 0.1, 1.0],
    }

    f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)
    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=20,
        cv=5,
        scoring=f2_scorer,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("  Running RandomizedSearchCV (20 iterations, 5-fold CV)...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"  Best params: {search.best_params_}")

    # SVM CRITICAL: use decision_function, NOT predict_proba
    val_scores = best_model.decision_function(X_val)
    threshold = select_threshold(y_val, val_scores, beta=2.0)
    print(f"  Selected threshold (F2-optimal on validation): {threshold:.4f}")

    # Validation metrics
    val_pred = (val_scores >= threshold).astype(int)
    val_metrics = compute_metrics(
        y_val, val_pred, y_score=val_scores, threshold=threshold
    )

    # Test metrics
    test_scores = best_model.decision_function(X_test)
    test_pred = (test_scores >= threshold).astype(int)
    test_metrics = compute_metrics(
        y_test, test_pred, y_score=test_scores, threshold=threshold
    )

    _print_split_results("SVM (RBF)", threshold, val_metrics, test_metrics)

    return best_model, search, threshold, val_metrics, test_metrics


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost with RandomizedSearchCV."""
    print("\n" + "=" * 60)
    print("XGBOOST")
    print("=" * 60)
    print(
        "Teaching: XGBoost builds trees sequentially, where each new tree"
        " corrects the errors of the previous ensemble. It uses gradient"
        " boosting with regularisation, making it one of the strongest"
        " tabular-data classifiers available."
    )

    # scale_pos_weight from TRAIN split only
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"  scale_pos_weight = {n_neg}/{n_pos} = {scale_pos_weight:.2f}")

    classifier = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    pipeline = _make_full_pipeline("xgboost", classifier)

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__subsample": [0.7, 0.85, 1.0],
        "classifier__colsample_bytree": [0.7, 0.85, 1.0],
        "classifier__min_child_weight": [1, 3, 5],
    }

    f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)
    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=20,
        cv=5,
        scoring=f2_scorer,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("  Running RandomizedSearchCV (20 iterations, 5-fold CV)...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"  Best params: {search.best_params_}")

    # Threshold selection on validation set
    val_scores = best_model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, val_scores, beta=2.0)
    print(f"  Selected threshold (F2-optimal on validation): {threshold:.4f}")

    # Validation metrics
    val_pred = (val_scores >= threshold).astype(int)
    val_metrics = compute_metrics(
        y_val, val_pred, y_score=val_scores, threshold=threshold
    )

    # Test metrics
    test_scores = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= threshold).astype(int)
    test_metrics = compute_metrics(
        y_test, test_pred, y_score=test_scores, threshold=threshold
    )

    _print_split_results("XGBoost", threshold, val_metrics, test_metrics)

    return best_model, search, threshold, val_metrics, test_metrics


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_split_results(name, threshold, val_metrics, test_metrics):
    """Print validation and test results for a single model."""
    print(f"\n--- {name} VALIDATION SET ---")
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
    print(f"\n--- {name} TEST SET ---")
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


def _save_confusion_matrix(
    model,
    X_test,
    y_test,
    threshold,
    model_label,
    filename,
    use_decision_function=False,
):
    """Save a confusion matrix PNG for the given model."""
    if use_decision_function:
        scores = model.decision_function(X_test)
    else:
        scores = model.predict_proba(X_test)[:, 1]
    y_pred = (scores >= threshold).astype(int)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Non-hazardous", "Hazardous"]
    )
    disp.ax_.set_title(f"{model_label} -- Test Set\n(threshold={threshold:.3f})")
    disp.figure_.savefig(FIGURES_DIR / filename, dpi=100, bbox_inches="tight")
    plt.close("all")


def _print_comparison_table():
    """Load both CSVs and print a comparison table sorted by validation F2."""
    val_df = pd.read_csv(ARTIFACTS_DIR / "validation_metrics.csv")
    test_df = pd.read_csv(ARTIFACTS_DIR / "model_metrics.csv")

    # Merge to get validation ranking with test metrics
    val_order = val_df.sort_values("selection_rank")[["model", "selection_rank"]].copy()
    merged = val_order.merge(test_df, on="model", how="inner")
    merged = merged.sort_values("selection_rank")

    print("\n" + "=" * 100)
    print("=== 4-MODEL COMPARISON (TEST METRICS) ===")
    print("=== Ranked by validation hazardous_f2 (descending) ===")
    print("=" * 100)

    # Header
    header = (
        f"{'Rank':>4}  {'Model':<15} {'Variant':<12} {'Thresh':>6}"
        f"  {'Prec':>6} {'Recall':>6} {'F1':>6} {'F2':>6}"
        f"  {'AUC':>6} {'Acc':>6}"
    )
    print(header)
    print("-" * len(header))

    for _, row in merged.iterrows():
        print(
            f"{int(row['selection_rank']):>4}  {row['model']:<15} {row['variant']:<12}"
            f" {row['threshold']:>6.3f}"
            f"  {row['hazardous_precision']:>6.3f} {row['hazardous_recall']:>6.3f}"
            f" {row['hazardous_f1']:>6.3f} {row['hazardous_f2']:>6.3f}"
            f"  {row['roc_auc']:>6.3f} {row['accuracy']:>6.3f}"
        )

    print()
    print(
        "Teaching: Fair comparison = same data, same split, same metrics."
        " All four models were trained on the same training fold, threshold-tuned"
        " on the same validation fold, and evaluated once on the same test fold."
        " This ensures differences reflect genuine model capability, not data leakage."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading data splits...")
    splits = make_splits(random_state=RANDOM_STATE)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]

    print(f"  Train:      {X_train.shape}, hazardous rate = {y_train.mean():.4f}")
    print(f"  Validation: {X_val.shape}, hazardous rate = {y_val.mean():.4f}")
    print(f"  Test:       {X_test.shape}, hazardous rate = {y_test.mean():.4f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Train 3 new models (logreg already done -- we do NOT retrain it)
    # -----------------------------------------------------------------------

    # --- Random Forest ---
    rf_model, rf_search, rf_thresh, rf_val, rf_test = train_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # --- SVM ---
    svm_model, svm_search, svm_thresh, svm_val, svm_test = train_svm(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # --- XGBoost ---
    xgb_model, xgb_search, xgb_thresh, xgb_val, xgb_test = train_xgboost(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # -----------------------------------------------------------------------
    # Save artifacts (idempotent: load -> remove model rows -> append -> save)
    # -----------------------------------------------------------------------
    models_data = [
        {
            "name": "random_forest",
            "search": rf_search,
            "threshold": rf_thresh,
            "val_metrics": rf_val,
            "test_metrics": rf_test,
            "model": rf_model,
            "use_decision_function": False,
            "figure_filename": "rf_confusion_matrix.png",
            "figure_label": "Random Forest",
        },
        {
            "name": "svm",
            "search": svm_search,
            "threshold": svm_thresh,
            "val_metrics": svm_val,
            "test_metrics": svm_test,
            "model": svm_model,
            "use_decision_function": True,
            "figure_filename": "svm_confusion_matrix.png",
            "figure_label": "SVM (RBF)",
        },
        {
            "name": "xgboost",
            "search": xgb_search,
            "threshold": xgb_thresh,
            "val_metrics": xgb_val,
            "test_metrics": xgb_test,
            "model": xgb_model,
            "use_decision_function": False,
            "figure_filename": "xgboost_confusion_matrix.png",
            "figure_label": "XGBoost",
        },
    ]

    print("\n--- Saving artifacts ---")

    for md in models_data:
        hyperparams = json.dumps(md["search"].best_params_)

        # Test metrics -> model_metrics.csv
        test_row = {
            "model": md["name"],
            "variant": "tuned",
            "split": "test",
            "seed": RANDOM_STATE,
            "threshold": md["threshold"],
            "hazardous_precision": md["test_metrics"]["hazardous_precision"],
            "hazardous_recall": md["test_metrics"]["hazardous_recall"],
            "hazardous_f1": md["test_metrics"]["hazardous_f1"],
            "hazardous_f2": md["test_metrics"]["hazardous_f2"],
            "roc_auc": md["test_metrics"]["roc_auc"],
            "accuracy": md["test_metrics"]["accuracy"],
            "tn": md["test_metrics"]["tn"],
            "fp": md["test_metrics"]["fp"],
            "fn": md["test_metrics"]["fn"],
            "tp": md["test_metrics"]["tp"],
            "hyperparams_json": hyperparams,
        }
        _save_metrics_csv(ARTIFACTS_DIR / "model_metrics.csv", test_row, md["name"])

        # Validation metrics -> validation_metrics.csv (rank assigned later)
        val_row = {
            "model": md["name"],
            "variant": "tuned",
            "split": "validation",
            "seed": RANDOM_STATE,
            "threshold": md["threshold"],
            "hazardous_precision": md["val_metrics"]["hazardous_precision"],
            "hazardous_recall": md["val_metrics"]["hazardous_recall"],
            "hazardous_f1": md["val_metrics"]["hazardous_f1"],
            "hazardous_f2": md["val_metrics"]["hazardous_f2"],
            "roc_auc": md["val_metrics"]["roc_auc"],
            "accuracy": md["val_metrics"]["accuracy"],
            "tn": md["val_metrics"]["tn"],
            "fp": md["val_metrics"]["fp"],
            "fn": md["val_metrics"]["fn"],
            "tp": md["val_metrics"]["tp"],
            "hyperparams_json": hyperparams,
            "selection_rank": 0,  # placeholder -- re-ranked below
        }
        _save_metrics_csv(ARTIFACTS_DIR / "validation_metrics.csv", val_row, md["name"])

        # Confusion matrix figure
        _save_confusion_matrix(
            md["model"],
            X_test,
            y_test,
            md["threshold"],
            md["figure_label"],
            md["figure_filename"],
            use_decision_function=md["use_decision_function"],
        )
        print(f"  Saved: {md['figure_filename']}")

    # Re-rank ALL models in validation_metrics.csv (including existing logreg)
    _rank_validation_models(ARTIFACTS_DIR / "validation_metrics.csv")
    print("  Re-ranked validation_metrics.csv")

    # -----------------------------------------------------------------------
    # Print final comparison table
    # -----------------------------------------------------------------------
    _print_comparison_table()


if __name__ == "__main__":
    main()
