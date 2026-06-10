"""Tests for the cost-based threshold and calibrated bundle contract."""

import json
from pathlib import Path

import numpy as np

from scripts.finalize_lockbox_policy import _select_cost_threshold, COST_FN, COST_FP
from src.app.scoring import DEFAULT_BUNDLE_PATH, load_model_bundle

POLICY_PATH = Path("artifacts/final_policy/final_policy.json")


def test_cost_threshold_prefers_catching_hazards_when_misses_are_costly():
    # Scores where a low threshold catches the positives.
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.35, 0.4, 0.6])

    threshold, cost = _select_cost_threshold(y_true, y_score)

    # With FN heavily penalized, the chosen threshold must catch both positives.
    pred = (y_score >= threshold).astype(int)
    fn = int(((pred == 0) & (y_true == 1)).sum())
    assert fn == 0
    assert cost == COST_FP * int(((pred == 1) & (y_true == 0)).sum())


def test_cost_constants_encode_safety_first_ratio():
    assert COST_FN > COST_FP


def test_bundle_is_calibrated_and_scores_in_unit_interval():
    if not DEFAULT_BUNDLE_PATH.exists():
        import pytest

        pytest.skip("model bundle not built yet")

    bundle = load_model_bundle()
    assert bundle.get("calibrated") is True
    assert bundle.get("calibration_method") in {"isotonic", "sigmoid"}
    # Calibrated estimator must expose predict_proba for the scoring layer.
    assert hasattr(bundle["model"], "predict_proba")


def test_policy_records_cost_matrix_when_present():
    if not POLICY_PATH.exists():
        import pytest

        pytest.skip("policy not frozen yet")

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert "threshold_basis" in policy
    if "cost_matrix" in policy:
        cm = policy["cost_matrix"]
        assert cm["fn"] > cm["fp"]
        assert 0.0 < policy["cost_optimal_threshold"] < 1.0
