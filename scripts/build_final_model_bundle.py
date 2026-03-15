"""Train and save the frozen final model bundle for inference."""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.load_arff import load_seismic_bumps
from src.features.preprocess import BINARY_COLS, ORDINAL_COLS, build_pipeline
from src.models.metrics import compute_metrics
from src.models.risk_levels import to_danger_flag, to_risk_level, to_risk_score

RANDOM_STATE = 42

ARTIFACTS_DIR = Path("artifacts")
FINAL_POLICY_DIR = ARTIFACTS_DIR / "final_policy"
POLICY_JSON = FINAL_POLICY_DIR / "final_policy.json"
BUNDLE_PKL = FINAL_POLICY_DIR / "model_bundle.pkl"
LOCKBOX_EVAL_CSV = FINAL_POLICY_DIR / "final_lockbox_eval.csv"
LOCKBOX_RESULTS_CSV = ARTIFACTS_DIR / "external_transfer_lockbox_results.csv"

MODEL_TO_TYPE = {
    "logreg": "logreg",
    "random_forest": "rf",
    "xgboost": "xgboost",
}


def _numeric_cols(features: pd.DataFrame) -> list[str]:
    excluded = set(ORDINAL_COLS + BINARY_COLS + ["class"])
    return [column for column in features.columns if column not in excluded]


def _load_policy() -> dict:
    if not POLICY_JSON.exists():
        raise FileNotFoundError(
            f"Missing {POLICY_JSON}. Run scripts/finalize_lockbox_policy.py first."
        )
    return json.loads(POLICY_JSON.read_text(encoding="utf-8"))


def _read_policy_fields(policy: dict) -> tuple[str, str, float, float]:
    model_name = policy["model"]
    variant = policy["variant"]
    threshold = float(policy["operating_threshold"])
    watch_floor = float(policy.get("watch_floor", 0.30))
    return model_name, variant, threshold, watch_floor


def _validate_policy_thresholds(threshold: float, watch_floor: float) -> None:
    if not (0.0 <= watch_floor < threshold <= 1.0):
        raise ValueError(
            f"Invalid thresholds: watch_floor={watch_floor}, threshold={threshold}. "
            "Require 0 <= watch_floor < threshold <= 1."
        )


def _load_model_params(model_name: str, variant: str, policy: dict) -> dict:
    if "hyperparams" in policy:
        params = dict(policy["hyperparams"])
        params.pop("external_prior", None)
        return params

    if LOCKBOX_RESULTS_CSV.exists():
        lockbox = pd.read_csv(LOCKBOX_RESULTS_CSV)
        match = lockbox[
            (lockbox["model"] == model_name) & (lockbox["variant"] == variant)
        ]
        if not match.empty:
            params = json.loads(match.iloc[0]["hyperparams_json"])
            params.pop("external_prior", None)
            return params

    raise ValueError(
        "Cannot resolve variant-specific hyperparameters. "
        f"Expected either policy.hyperparams or a matching row in {LOCKBOX_RESULTS_CSV} "
        f"for model={model_name}, variant={variant}."
    )


def _build_classifier(model_name: str, params: dict, labels: pd.Series):
    if model_name == "logreg":
        return LogisticRegression(
            random_state=RANDOM_STATE, **{"max_iter": 1000, **params}
        )

    if model_name == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, **params)

    if model_name == "xgboost":
        n_neg = int((labels == 0).sum())
        n_pos = int((labels == 1).sum())
        return XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            **{**params, "scale_pos_weight": n_neg / max(n_pos, 1)},
        )

    raise ValueError(f"Unsupported model: {model_name}")


def _split_dev_and_lockbox() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = load_seismic_bumps()
    features = dataset.drop(columns=["class"])
    labels = dataset["class"]

    X_dev, X_lockbox, y_dev, y_lockbox = train_test_split(
        features,
        labels,
        test_size=0.15,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    return (
        pd.DataFrame(X_dev).copy(),
        pd.DataFrame(X_lockbox).copy(),
        pd.Series(y_dev).copy(),
        pd.Series(y_lockbox).copy(),
    )


def _evaluate_lockbox(
    model,
    pipeline,
    X_lockbox: pd.DataFrame,
    y_lockbox: pd.Series,
    threshold: float,
    watch_floor: float,
) -> tuple[pd.DataFrame, dict]:
    X_lockbox_proc = pipeline.transform(X_lockbox)
    lockbox_proba = np.asarray(model.predict_proba(X_lockbox_proc), dtype=float)
    lockbox_scores = lockbox_proba[:, 1]
    lockbox_pred = (lockbox_scores >= threshold).astype(int)

    metrics = compute_metrics(
        y_lockbox,
        lockbox_pred,
        y_score=lockbox_scores,
        threshold=threshold,
    )

    evaluation = pd.DataFrame(
        {
            "predicted_probability": lockbox_scores,
            "predicted_label": lockbox_pred,
            "risk_score": [to_risk_score(score) for score in lockbox_scores],
            "risk_level": [
                to_risk_level(score, threshold=threshold, watch_floor=watch_floor)
                for score in lockbox_scores
            ],
            "dangerous_flag": [
                to_danger_flag(score, threshold=threshold) for score in lockbox_scores
            ],
            "actual_label": y_lockbox.to_numpy(),
        }
    )
    return evaluation, metrics


def main() -> None:
    policy = _load_policy()
    model_name, variant, threshold, watch_floor = _read_policy_fields(policy)
    _validate_policy_thresholds(threshold, watch_floor)

    if "external_prior" in variant:
        raise ValueError(
            "Final bundle currently supports frozen baseline variants only. "
            "If external prior is selected, build a dedicated frozen reference bundle first."
        )

    params = _load_model_params(model_name, variant, policy)
    X_dev, X_lockbox, y_dev, y_lockbox = _split_dev_and_lockbox()

    model_type = MODEL_TO_TYPE.get(model_name)
    if model_type is None:
        raise ValueError(f"Unsupported model type mapping for model={model_name}")

    pipeline = build_pipeline(model_type, numeric_cols=_numeric_cols(X_dev))
    X_dev_proc = pipeline.fit_transform(X_dev)
    model = _build_classifier(model_name, params, y_dev)
    model.fit(X_dev_proc, y_dev)

    lockbox_eval, lockbox_metrics = _evaluate_lockbox(
        model,
        pipeline,
        X_lockbox,
        y_lockbox,
        threshold,
        watch_floor,
    )
    lockbox_eval.to_csv(LOCKBOX_EVAL_CSV, index=False)

    bundle = {
        "model_name": model_name,
        "variant": variant,
        "threshold": threshold,
        "watch_floor": watch_floor,
        "use_external_prior": False,
        "pipeline": pipeline,
        "model": model,
    }
    with BUNDLE_PKL.open("wb") as handle:
        pickle.dump(bundle, handle)

    print("=== FINAL MODEL BUNDLE BUILT ===")
    print(f"Model={model_name} Variant={variant}")
    print(f"Threshold={threshold:.3f}, WatchFloor={watch_floor:.2f}")
    print(
        "Lockbox metrics: recall={:.3f} precision={:.3f} F2={:.3f} AUC={:.3f}".format(
            lockbox_metrics["hazardous_recall"],
            lockbox_metrics["hazardous_precision"],
            lockbox_metrics["hazardous_f2"],
            lockbox_metrics["roc_auc"],
        )
    )
    print(f"Saved: {BUNDLE_PKL}")
    print(f"Saved: {LOCKBOX_EVAL_CSV}")


if __name__ == "__main__":
    main()
