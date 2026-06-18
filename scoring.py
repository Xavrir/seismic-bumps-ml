"""Reusable scoring helpers for the Seismic Risk Console."""

from __future__ import annotations

import pickle
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from risk_levels import to_danger_flag, to_risk_level, to_risk_score

PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_POLICY_DIR = PROJECT_ROOT / "artifacts" / "final_policy"
DEFAULT_BUNDLE_PATH = FINAL_POLICY_DIR / "model_bundle.pkl"
DEFAULT_SAMPLE_INPUT_PATH = FINAL_POLICY_DIR / "sample_input.csv"

CATEGORICAL_OPTIONS = {
    "seismic": ("a", "b", "c", "d"),
    "seismoacoustic": ("a", "b", "c", "d"),
    "ghazard": ("a", "b", "c", "d"),
    "shift": ("N", "W"),
}

NUMERIC_COLUMNS = (
    "genergy",
    "gpuls",
    "gdenergy",
    "gdpuls",
    "nbumps",
    "nbumps2",
    "nbumps3",
    "nbumps4",
    "nbumps5",
    "nbumps6",
    "nbumps7",
    "nbumps89",
    "energy",
    "maxenergy",
)

PREDICTION_COLUMNS = (
    "predicted_probability",
    "risk_score",
    "risk_level",
    "dangerous_flag",
)

class ScoringInputError(ValueError):
    """Raised when uploaded or entered scoring data cannot be scored."""

def load_required_columns(sample_path: Path = DEFAULT_SAMPLE_INPUT_PATH) -> list[str]:
    """Read the canonical feature order from the bundled sample CSV."""
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample input file: {sample_path}")
    return list(pd.read_csv(sample_path, nrows=0).columns)

def load_sample_input(sample_path: Path = DEFAULT_SAMPLE_INPUT_PATH) -> pd.DataFrame:
    """Load the bundled sample rows used by the UI and tests."""
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample input file: {sample_path}")
    return pd.read_csv(sample_path)

def load_model_bundle(bundle_path: Path = DEFAULT_BUNDLE_PATH) -> dict:
    """Load the frozen sklearn bundle."""
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing model bundle file: {bundle_path}")
    with bundle_path.open("rb") as handle:
        return pickle.load(handle)

def read_csv_input(source: bytes | BinaryIO) -> pd.DataFrame:
    """Parse uploaded CSV bytes into a DataFrame with a user-facing error."""
    try:
        if isinstance(source, bytes):
            return pd.read_csv(BytesIO(source))
        return pd.read_csv(source)
    except Exception as exc:
        raise ScoringInputError(f"Could not read CSV input: {exc}") from exc

def validate_features(
    features: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Validate and normalize feature rows for model scoring."""
    if required_columns is None:
        required_columns = load_required_columns()

    if features.empty:
        raise ScoringInputError("Input CSV has no rows to score.")

    missing = [column for column in required_columns if column not in features.columns]
    if missing:
        raise ScoringInputError(
            "Missing required column"
            + ("s" if len(missing) > 1 else "")
            + f": {', '.join(missing)}"
        )

    normalized = features.loc[:, required_columns].copy()
    for column, allowed_values in CATEGORICAL_OPTIONS.items():
        if normalized[column].isna().any():
            raise ScoringInputError(f"Categorical column contains blank values: {column}")

        values = normalized[column].astype(str)
        invalid = sorted(set(values) - set(allowed_values))
        if invalid:
            allowed = ", ".join(allowed_values)
            found = ", ".join(invalid[:5])
            raise ScoringInputError(
                f"Invalid value for {column}: {found}. Expected one of: {allowed}."
            )
        normalized[column] = normalized[column].astype(str)

    for column in NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    numeric_with_nulls = [
        column for column in NUMERIC_COLUMNS if normalized[column].isna().any()
    ]
    if numeric_with_nulls:
        raise ScoringInputError(
            "Numeric columns contain blank or non-numeric values: "
            + ", ".join(numeric_with_nulls)
        )

    return normalized

def score_features(features: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Score shift rows and append probability, score, level, and flag columns."""
    threshold = float(bundle["threshold"])
    watch_floor = float(bundle.get("watch_floor", 0.30))
    if not (0.0 <= watch_floor < threshold <= 1.0):
        raise ValueError(
            f"Invalid bundle thresholds: watch_floor={watch_floor}, threshold={threshold}"
        )

    normalized = validate_features(features)
    pipeline = bundle["pipeline"]
    model = bundle["model"]
    processed_features = pipeline.transform(normalized)
    probabilities = model.predict_proba(processed_features)[:, 1]

    output = normalized.copy()
    output["predicted_probability"] = probabilities
    output["risk_score"] = [to_risk_score(probability) for probability in probabilities]
    output["risk_level"] = [
        to_risk_level(probability, threshold=threshold, watch_floor=watch_floor)
        for probability in probabilities
    ]
    output["dangerous_flag"] = [
        to_danger_flag(probability, threshold=threshold)
        for probability in probabilities
    ]
    return output

def prediction_summary(predictions: pd.DataFrame) -> dict[str, int]:
    """Return counts by risk level for concise UI summaries."""
    counts = predictions["risk_level"].value_counts()
    return {
        "low": int(counts.get("low", 0)),
        "watch": int(counts.get("watch", 0)),
        "dangerous": int(counts.get("dangerous", 0)),
    }
