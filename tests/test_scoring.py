"""Tests for Seismic Risk Console scoring helpers."""

import pytest

from src.app.scoring import (
    PREDICTION_COLUMNS,
    ScoringInputError,
    load_model_bundle,
    load_required_columns,
    load_sample_input,
    read_csv_input,
    score_features,
    validate_features,
)
from src.models.risk_levels import to_risk_level


def test_score_sample_input_returns_prediction_columns():
    bundle = load_model_bundle()
    predictions = score_features(load_sample_input().head(2), bundle)

    for column in PREDICTION_COLUMNS:
        assert column in predictions.columns
    assert predictions["predicted_probability"].between(0, 1).all()
    assert predictions["risk_score"].between(0, 100).all()
    assert set(predictions["risk_level"]).issubset({"low", "watch", "dangerous"})


def test_validate_features_reports_missing_required_columns():
    required = load_required_columns()
    incomplete = load_sample_input().drop(columns=[required[0]])

    with pytest.raises(ScoringInputError, match=f"Missing required column: {required[0]}"):
        validate_features(incomplete, required_columns=required)


def test_read_csv_input_reports_invalid_csv():
    with pytest.raises(ScoringInputError, match="Could not read CSV input"):
        read_csv_input(b'"unterminated')


def test_validate_features_rejects_non_numeric_values():
    features = load_sample_input().head(1).astype({"energy": "object"})
    features.loc[features.index[0], "energy"] = "not-a-number"

    with pytest.raises(ScoringInputError, match="Numeric columns contain"):
        validate_features(features)


def test_validate_features_rejects_blank_categorical_values():
    features = load_sample_input().head(1)
    features.loc[features.index[0], "seismic"] = None

    with pytest.raises(
        ScoringInputError,
        match="Categorical column contains blank values: seismic",
    ):
        validate_features(features)


def test_validate_features_rejects_header_only_csv():
    features = load_sample_input().head(0)

    with pytest.raises(ScoringInputError, match="Input CSV has no rows to score"):
        validate_features(features)


def test_risk_level_boundaries_match_policy_contract():
    threshold = 0.512
    watch_floor = 0.30

    assert to_risk_level(0.299, threshold, watch_floor) == "low"
    assert to_risk_level(0.300, threshold, watch_floor) == "watch"
    assert to_risk_level(0.511, threshold, watch_floor) == "watch"
    assert to_risk_level(0.512, threshold, watch_floor) == "dangerous"


def test_extra_columns_are_ignored_for_scoring_schema():
    features = load_sample_input().head(1)
    features["operator_note"] = "demo"

    validated = validate_features(features)

    assert "operator_note" not in validated.columns
    assert list(validated.columns) == load_required_columns()
