"""Predict shift-level mine danger scores using the frozen final model bundle."""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.models.risk_levels import to_danger_flag, to_risk_level, to_risk_score


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict hazardous risk for shift rows"
    )
    parser.add_argument("--input-csv", required=True, help="Input feature CSV path")
    parser.add_argument(
        "--bundle",
        default="artifacts/final_policy/model_bundle.pkl",
        help="Path to frozen model bundle",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/final_policy/predictions.csv",
        help="Where predictions should be written",
    )
    return parser.parse_args()


def _load_bundle(bundle_path: Path) -> dict:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing bundle file: {bundle_path}")
    with bundle_path.open("rb") as handle:
        return pickle.load(handle)


def _validate_thresholds(threshold: float, watch_floor: float) -> None:
    if not (0.0 <= watch_floor < threshold <= 1.0):
        raise ValueError(
            f"Invalid thresholds in bundle: watch_floor={watch_floor}, threshold={threshold}. "
            "Require 0 <= watch_floor < threshold <= 1."
        )


def _build_prediction_frame(
    features: pd.DataFrame,
    probabilities,
    threshold: float,
    watch_floor: float,
) -> pd.DataFrame:
    output = features.copy()
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


def main() -> None:
    args = _parse_args()
    bundle = _load_bundle(Path(args.bundle))

    threshold = float(bundle["threshold"])
    watch_floor = float(bundle.get("watch_floor", 0.30))
    _validate_thresholds(threshold, watch_floor)

    pipeline = bundle["pipeline"]
    model = bundle["model"]

    features = pd.read_csv(args.input_csv)
    features_processed = pipeline.transform(features)
    probabilities = model.predict_proba(features_processed)[:, 1]

    prediction_frame = _build_prediction_frame(
        features,
        probabilities,
        threshold,
        watch_floor,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_path, index=False)

    print("=== PREDICTIONS COMPLETE ===")
    print(f"Rows scored: {len(prediction_frame)}")
    print(f"Threshold: {threshold:.3f}, WatchFloor: {watch_floor:.2f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
