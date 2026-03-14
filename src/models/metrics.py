"""Canonical metric computation for the Seismic Bumps classifier."""

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    fbeta_score,
)
from typing import Dict, Any


def compute_metrics(
    y_true,
    y_pred,
    y_score=None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute the canonical metric suite for the hazardous-class evaluation.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1) — already thresholded
        y_score: Continuous score for ROC-AUC (predict_proba[:,1] or decision_function)
                 Pass None to skip ROC-AUC.
        threshold: The threshold used to binarize y_score into y_pred (for record-keeping)

    Returns:
        Dict with keys:
            hazardous_precision, hazardous_recall, hazardous_f1, hazardous_f2,
            roc_auc, accuracy, tn, fp, fn, tp, threshold
    """
    # pos_label=1 is always explicit — hazardous is the positive class
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )
    f2 = fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc = roc_auc_score(y_true, y_score) if y_score is not None else None

    return {
        "hazardous_precision": float(prec),
        "hazardous_recall": float(rec),
        "hazardous_f1": float(f1),
        "hazardous_f2": float(f2),
        "roc_auc": float(roc) if roc is not None else None,
        "accuracy": float(acc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def select_threshold(y_val_true, y_val_score, beta: float = 2.0) -> float:
    """
    Select the threshold on the validation set that maximizes hazardous-class F-beta score.

    Tie-breaking rules:
    1. Among thresholds within 0.01 of best F-beta, prefer the one with higher recall.
    2. If still tied, prefer the threshold closest to 0.5.

    Args:
        y_val_true: Ground truth labels for validation split
        y_val_score: Continuous scores (predict_proba[:,1] or decision_function)
        beta: F-beta beta value (default 2.0 = F2 = recall-weighted)

    Returns:
        Best threshold (float)
    """
    y_val_true = np.asarray(y_val_true)
    y_val_score = np.asarray(y_val_score)

    thresholds = np.linspace(0.01, 0.99, 99)
    best_fbeta = -1.0
    best_recall = -1.0
    best_thresh = 0.5

    for t in thresholds:
        y_pred = (y_val_score >= t).astype(int)
        fb = fbeta_score(y_val_true, y_pred, beta=beta, pos_label=1, zero_division=0)
        rec = float(np.sum((y_pred == 1) & (y_val_true == 1))) / max(
            np.sum(y_val_true == 1), 1
        )
        if fb > best_fbeta + 0.01:
            best_fbeta = fb
            best_recall = rec
            best_thresh = t
        elif abs(fb - best_fbeta) <= 0.01:
            if rec > best_recall:
                best_recall = rec
                best_thresh = t
            elif rec == best_recall and abs(t - 0.5) < abs(best_thresh - 0.5):
                best_thresh = t

    return float(best_thresh)
