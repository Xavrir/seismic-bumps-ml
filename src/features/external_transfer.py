"""Feature alignment and cautious transfer utilities for external seismic priors."""

from __future__ import annotations

import numpy as np
import pandas as pd


UCI_BUMP_COLS = [
    "nbumps",
    "nbumps2",
    "nbumps3",
    "nbumps4",
    "nbumps5",
    "nbumps6",
    "nbumps7",
    "nbumps89",
]


def build_uci_alignment_features(X: pd.DataFrame) -> pd.DataFrame:
    """Project UCI rows into a small feature space comparable to external events."""
    out = pd.DataFrame(index=X.index)
    out["activity_total"] = X[UCI_BUMP_COLS].sum(axis=1)
    out["activity_ge_2"] = X[
        ["nbumps2", "nbumps3", "nbumps4", "nbumps5", "nbumps6", "nbumps7", "nbumps89"]
    ].sum(axis=1)
    out["activity_ge_3"] = X[
        ["nbumps3", "nbumps4", "nbumps5", "nbumps6", "nbumps7", "nbumps89"]
    ].sum(axis=1)
    out["activity_ge_4"] = X[
        ["nbumps4", "nbumps5", "nbumps6", "nbumps7", "nbumps89"]
    ].sum(axis=1)
    out["energy_total_log1p"] = np.log1p(
        pd.to_numeric(X["energy"], errors="coerce").clip(lower=0)
    )
    out["energy_peak_log1p"] = np.log1p(
        pd.to_numeric(X["maxenergy"], errors="coerce").clip(lower=0)
    )
    out["energy_mean_per_event_log1p"] = np.log1p(
        pd.to_numeric(X["energy"], errors="coerce").clip(lower=0)
        / (out["activity_total"] + 1.0)
    )
    return out.fillna(0.0)


def build_external_alignment_features(external_windows: pd.DataFrame) -> pd.DataFrame:
    """Project aggregated external windows into the same aligned feature space."""
    out = pd.DataFrame(index=external_windows.index)
    out["activity_total"] = pd.to_numeric(
        external_windows["event_count"], errors="coerce"
    )
    out["activity_ge_2"] = pd.to_numeric(
        external_windows["count_mag_ge_2"], errors="coerce"
    )
    out["activity_ge_3"] = pd.to_numeric(
        external_windows["count_mag_ge_3"], errors="coerce"
    )
    out["activity_ge_4"] = pd.to_numeric(
        external_windows["count_mag_ge_4"], errors="coerce"
    )
    out["energy_total_log1p"] = np.log1p(
        pd.to_numeric(external_windows["energy_proxy_sum"], errors="coerce").clip(
            lower=0
        )
    )
    out["energy_peak_log1p"] = np.log1p(
        pd.to_numeric(external_windows["energy_proxy_max"], errors="coerce").clip(
            lower=0
        )
    )
    out["energy_mean_per_event_log1p"] = np.log1p(
        pd.to_numeric(external_windows["energy_proxy_mean"], errors="coerce").clip(
            lower=0
        )
    )
    return out.fillna(0.0)


def fit_external_reference(
    external_features: pd.DataFrame, ridge: float = 1e-6
) -> dict:
    """Fit a simple Gaussian reference model on external aligned features."""
    values = external_features.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    cov = np.cov(values, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = cov + np.eye(cov.shape[0]) * ridge
    inv_cov = np.linalg.pinv(cov)

    sorted_columns = {
        col: np.sort(external_features[col].to_numpy(dtype=float))
        for col in external_features.columns
    }

    return {
        "columns": list(external_features.columns),
        "mean": mean,
        "inv_cov": inv_cov,
        "sorted_columns": sorted_columns,
    }


def _percentile_rank(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_values, values, side="right")
    return positions / max(len(sorted_values), 1)


def add_external_prior_features(X: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Append external-reference prior features to UCI feature rows."""
    aligned = build_uci_alignment_features(X)
    cols = reference["columns"]
    aligned_values = aligned[cols].to_numpy(dtype=float)
    centered = aligned_values - reference["mean"]
    d2 = np.einsum("ij,jk,ik->i", centered, reference["inv_cov"], centered)
    distance = np.sqrt(np.maximum(d2, 0.0))

    out = X.copy()
    out["external_prior_mahalanobis"] = distance
    out["external_prior_similarity"] = 1.0 / (1.0 + distance)

    for source_col, feature_name in [
        ("activity_total", "external_prior_activity_total_pct"),
        ("energy_total_log1p", "external_prior_energy_total_pct"),
        ("energy_peak_log1p", "external_prior_energy_peak_pct"),
    ]:
        sorted_values = reference["sorted_columns"][source_col]
        out[feature_name] = _percentile_rank(
            sorted_values,
            aligned[source_col].to_numpy(dtype=float),
        )

    return out
