"""Build a curated demo CSV for the Seismic Risk Console walkthrough video.

Reads the real seismic-bumps dataset, scores every row through the FROZEN policy
bundle, and picks a small mix of shifts that span low / watch / dangerous so the
Upload-CSV demo shows colourful tiles and the red "dangerous" alert banner.

Nothing here retrains or mutates the model: it only *reads* the frozen bundle and
*reads* the source data. Output is the 18 feature columns only (no label, no scores),
with a header identical to the bundled sample so the app accepts the upload.

Usage:
    python scripts/make_demo_csv.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scoring import (
    load_model_bundle,
    load_required_columns,
    score_features,
)
from load_arff import load_seismic_bumps

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "demo" / "seismic_demo_shifts.csv"

# How many rows to show for each risk level in the demo file.
WANT = {"low": 5, "watch": 2, "dangerous": 2}


def _pick(scored: pd.DataFrame, level: str, n: int) -> pd.DataFrame:
    """Pick n rows at a given risk level, preferring genuinely hazardous shifts.

    For 'dangerous' we sort hazardous (class == 1) rows first so the demo is
    authentic rather than a model false-alarm; ties broken by highest probability.
    """
    pool = scored[scored["risk_level"] == level]
    if pool.empty:
        return pool
    pool = pool.sort_values(
        by=["class", "predicted_probability"], ascending=[False, False]
    )
    return pool.head(n)


def main() -> None:
    required = load_required_columns()  # canonical 18-feature header order
    bundle = load_model_bundle()

    df = load_seismic_bumps()
    features = df.loc[:, required].copy()

    scored = score_features(features, bundle)
    scored["class"] = df["class"].to_numpy()  # keep label only to pick honest rows

    chosen = pd.concat([_pick(scored, level, n) for level, n in WANT.items()])

    # Report what we actually got, then verify it spans all three levels.
    counts = chosen["risk_level"].value_counts().to_dict()
    print("Selected demo rows by risk level:", counts)
    for level in WANT:
        if counts.get(level, 0) == 0:
            raise SystemExit(
                f"No rows available at risk level '{level}'. Widen the candidate pool."
            )

    # Write feature columns only — drop label + scored columns so the app re-scores.
    out = chosen.loc[:, required].reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out)} rows -> {OUTPUT_PATH}")
    print("Header matches sample:", list(out.columns) == required)


if __name__ == "__main__":
    main()
