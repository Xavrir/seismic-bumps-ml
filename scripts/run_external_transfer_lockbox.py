"""Run lockbox-safe external transfer evaluation across model families."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from xgboost import XGBClassifier

from load_arff import load_seismic_bumps
from external_transfer import (
    add_external_prior_features,
    build_external_alignment_features,
    fit_external_reference,
)
from preprocess import BINARY_COLS, ORDINAL_COLS, build_pipeline
from metrics import compute_metrics, select_threshold

RANDOM_STATE = 42
ARTIFACTS_DIR = Path("artifacts")
EXTERNAL_WINDOWS_CSV = (
    ARTIFACTS_DIR / "external" / "usgs_mining_events_aggregated_8h.csv"
)
VALIDATION_METRICS_CSV = ARTIFACTS_DIR / "validation_metrics.csv"

LOCKBOX_CV_CSV = ARTIFACTS_DIR / "external_transfer_lockbox_cv.csv"
LOCKBOX_SUMMARY_CSV = ARTIFACTS_DIR / "external_transfer_lockbox_summary.csv"
LOCKBOX_SELECTED_CSV = ARTIFACTS_DIR / "external_transfer_lockbox_selected.csv"
LOCKBOX_RESULTS_CSV = ARTIFACTS_DIR / "external_transfer_lockbox_results.csv"

MODEL_SPECS = [
    {"name": "logreg", "model_type": "logreg"},
    {"name": "random_forest", "model_type": "rf"},
    {"name": "xgboost", "model_type": "xgboost"},
]

BASELINE_VARIANT = "baseline_lockbox"
EXTERNAL_VARIANT = "external_prior_lockbox"


def _numeric_cols(X: pd.DataFrame) -> list[str]:
    excluded = set(ORDINAL_COLS + BINARY_COLS + ["class"])
    return [c for c in X.columns if c not in excluded]


def _strip_classifier_prefix(params: dict) -> dict:
    return {k.replace("classifier__", ""): v for k, v in params.items()}


def _load_base_params() -> dict:
    val_df = pd.read_csv(VALIDATION_METRICS_CSV)
    out = {}
    for spec in MODEL_SPECS:
        row = val_df[val_df["model"] == spec["name"]].iloc[0]
        out[spec["name"]] = _strip_classifier_prefix(
            json.loads(row["hyperparams_json"])
        )
    return out


def _build_classifier(model_name: str, params: dict, y_train: pd.Series):
    if model_name == "logreg":
        merged = {"max_iter": 1000, **params}
        return LogisticRegression(random_state=RANDOM_STATE, **merged)

    if model_name == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, **params)

    if model_name == "xgboost":
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw = n_neg / max(n_pos, 1)
        merged = {**params, "scale_pos_weight": spw}
        return XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", **merged)

    raise ValueError(f"Unsupported model: {model_name}")


def _fit_and_predict_scores(
    model_name: str, model_type: str, params: dict, X_train, y_train, X_eval
):
    pipeline = build_pipeline(model_type, numeric_cols=_numeric_cols(X_train))
    X_train_proc = pipeline.fit_transform(X_train)
    model = _build_classifier(model_name, params, y_train)
    model.fit(X_train_proc, y_train)
    X_eval_proc = pipeline.transform(X_eval)
    scores = model.predict_proba(X_eval_proc)[:, 1]
    return scores, pipeline, model


def _prepare_external_reference(freeze_end: str) -> dict:
    windows = pd.read_csv(EXTERNAL_WINDOWS_CSV)
    ts = pd.to_datetime(windows["window_start_utc"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(freeze_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    frozen = windows[ts <= cutoff].copy()
    if frozen.empty:
        raise ValueError("External frozen windows are empty")
    ext_features = build_external_alignment_features(frozen)
    return fit_external_reference(ext_features)


def _augment(X: pd.DataFrame, reference: dict, use_external: bool) -> pd.DataFrame:
    return add_external_prior_features(X, reference) if use_external else X


def _evaluate_variant_cv(
    model_name: str,
    model_type: str,
    params: dict,
    reference: dict,
    use_external: bool,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    n_splits: int,
    n_repeats: int,
) -> pd.DataFrame:
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )
    rows = []

    for fold_idx, (outer_tr, outer_te) in enumerate(rskf.split(X_dev, y_dev), start=1):
        X_outer_train = X_dev.iloc[outer_tr]
        y_outer_train = y_dev.iloc[outer_tr]
        X_outer_test = X_dev.iloc[outer_te]
        y_outer_test = y_dev.iloc[outer_te]

        X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
            X_outer_train,
            y_outer_train,
            test_size=0.2,
            stratify=y_outer_train,
            random_state=RANDOM_STATE + fold_idx,
        )

        X_inner_train = _augment(X_inner_train, reference, use_external)
        X_inner_val = _augment(X_inner_val, reference, use_external)
        X_outer_test = _augment(X_outer_test, reference, use_external)

        val_scores, _, _ = _fit_and_predict_scores(
            model_name,
            model_type,
            params,
            X_inner_train,
            y_inner_train,
            X_inner_val,
        )
        threshold = select_threshold(y_inner_val, val_scores, beta=2.0)

        test_scores, _, _ = _fit_and_predict_scores(
            model_name,
            model_type,
            params,
            X_inner_train,
            y_inner_train,
            X_outer_test,
        )
        test_pred = (test_scores >= threshold).astype(int)
        m = compute_metrics(
            y_outer_test, test_pred, y_score=test_scores, threshold=threshold
        )

        rows.append(
            {
                "model": model_name,
                "variant": EXTERNAL_VARIANT if use_external else BASELINE_VARIANT,
                "fold": fold_idx,
                "threshold": threshold,
                "hazardous_precision": m["hazardous_precision"],
                "hazardous_recall": m["hazardous_recall"],
                "hazardous_f1": m["hazardous_f1"],
                "hazardous_f2": m["hazardous_f2"],
                "roc_auc": m["roc_auc"],
                "accuracy": m["accuracy"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tp": m["tp"],
            }
        )

    return pd.DataFrame(rows)


def _select_variant_per_model(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.sort_values(
        ["model", "f2_mean", "recall_mean", "auc_mean"],
        ascending=[True, False, False, False],
    )
    return ranked.groupby("model", as_index=False).head(1).reset_index(drop=True)


def _evaluate_lockbox_once(
    selected: pd.DataFrame,
    params_by_model: dict,
    reference: dict,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_lockbox: pd.DataFrame,
    y_lockbox: pd.Series,
) -> pd.DataFrame:
    X_train_dev, X_val_dev, y_train_dev, y_val_dev = train_test_split(
        X_dev,
        y_dev,
        test_size=0.2,
        stratify=y_dev,
        random_state=RANDOM_STATE,
    )

    rows = []
    for _, sel in selected.iterrows():
        model_name = sel["model"]
        model_type = next(
            spec["model_type"] for spec in MODEL_SPECS if spec["name"] == model_name
        )
        use_external = sel["variant"] == EXTERNAL_VARIANT

        X_train_use = _augment(X_train_dev, reference, use_external)
        X_val_use = _augment(X_val_dev, reference, use_external)
        X_lock_use = _augment(X_lockbox, reference, use_external)

        val_scores, pipeline, model = _fit_and_predict_scores(
            model_name,
            model_type,
            params_by_model[model_name],
            X_train_use,
            y_train_dev,
            X_val_use,
        )
        threshold = select_threshold(y_val_dev, val_scores, beta=2.0)

        X_lock_proc = pipeline.transform(X_lock_use)
        lock_scores = model.predict_proba(X_lock_proc)[:, 1]
        lock_pred = (lock_scores >= threshold).astype(int)
        m = compute_metrics(
            y_lockbox, lock_pred, y_score=lock_scores, threshold=threshold
        )

        rows.append(
            {
                "model": model_name,
                "variant": sel["variant"],
                "split": "lockbox",
                "seed": RANDOM_STATE,
                "threshold": threshold,
                "hazardous_precision": m["hazardous_precision"],
                "hazardous_recall": m["hazardous_recall"],
                "hazardous_f1": m["hazardous_f1"],
                "hazardous_f2": m["hazardous_f2"],
                "roc_auc": m["roc_auc"],
                "accuracy": m["accuracy"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tp": m["tp"],
                "cv_f2_mean": sel["f2_mean"],
                "cv_recall_mean": sel["recall_mean"],
                "cv_auc_mean": sel["auc_mean"],
                "hyperparams_json": json.dumps(
                    {
                        **params_by_model[model_name],
                        "external_prior": use_external,
                    }
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    if not EXTERNAL_WINDOWS_CSV.exists():
        raise FileNotFoundError(
            f"Missing {EXTERNAL_WINDOWS_CSV}. Run scripts/fetch_external_usgs.py first."
        )

    n_splits = 5
    n_repeats = 2
    freeze_end = "2023-12-31"

    reference = _prepare_external_reference(freeze_end=freeze_end)
    params_by_model = _load_base_params()

    df = load_seismic_bumps()
    X = df.drop(columns=["class"])
    y = df["class"]

    X_dev, X_lockbox, y_dev, y_lockbox = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv_parts = []
    for spec in MODEL_SPECS:
        model_name = spec["name"]
        model_type = spec["model_type"]

        cv_parts.append(
            _evaluate_variant_cv(
                model_name,
                model_type,
                params_by_model[model_name],
                reference,
                use_external=False,
                X_dev=X_dev,
                y_dev=y_dev,
                n_splits=n_splits,
                n_repeats=n_repeats,
            )
        )
        cv_parts.append(
            _evaluate_variant_cv(
                model_name,
                model_type,
                params_by_model[model_name],
                reference,
                use_external=True,
                X_dev=X_dev,
                y_dev=y_dev,
                n_splits=n_splits,
                n_repeats=n_repeats,
            )
        )

    cv_df = pd.concat(cv_parts, ignore_index=True)
    cv_df.to_csv(LOCKBOX_CV_CSV, index=False)

    summary = (
        cv_df.groupby(["model", "variant"], as_index=False)
        .agg(
            folds=("fold", "count"),
            threshold_mean=("threshold", "mean"),
            threshold_std=("threshold", "std"),
            recall_mean=("hazardous_recall", "mean"),
            recall_std=("hazardous_recall", "std"),
            f2_mean=("hazardous_f2", "mean"),
            f2_std=("hazardous_f2", "std"),
            auc_mean=("roc_auc", "mean"),
            auc_std=("roc_auc", "std"),
        )
        .sort_values(
            ["model", "f2_mean", "recall_mean", "auc_mean"],
            ascending=[True, False, False, False],
        )
    )
    summary.to_csv(LOCKBOX_SUMMARY_CSV, index=False)

    selected = _select_variant_per_model(summary)
    selected["freeze_end_utc"] = freeze_end
    selected["cv_splits"] = n_splits
    selected["cv_repeats"] = n_repeats
    selected.to_csv(LOCKBOX_SELECTED_CSV, index=False)

    lockbox_results = _evaluate_lockbox_once(
        selected,
        params_by_model,
        reference,
        X_dev,
        y_dev,
        X_lockbox,
        y_lockbox,
    )
    lockbox_results["freeze_end_utc"] = freeze_end
    lockbox_results.to_csv(LOCKBOX_RESULTS_CSV, index=False)

    print("=== EXTERNAL TRANSFER LOCKBOX EVALUATION ===")
    print(summary.to_string(index=False))
    print("\nSelected variants:")
    print(
        selected[["model", "variant", "f2_mean", "recall_mean", "auc_mean"]].to_string(
            index=False
        )
    )
    print("\nLockbox results:")
    print(
        lockbox_results[
            [
                "model",
                "variant",
                "threshold",
                "hazardous_recall",
                "hazardous_precision",
                "hazardous_f2",
                "roc_auc",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {LOCKBOX_CV_CSV}")
    print(f"Saved: {LOCKBOX_SUMMARY_CSV}")
    print(f"Saved: {LOCKBOX_SELECTED_CSV}")
    print(f"Saved: {LOCKBOX_RESULTS_CSV}")


if __name__ == "__main__":
    main()
