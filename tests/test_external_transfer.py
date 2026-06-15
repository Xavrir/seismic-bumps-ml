import pandas as pd

from external_transfer import (
    add_external_prior_features,
    build_external_alignment_features,
    build_uci_alignment_features,
    fit_external_reference,
)


def test_build_uci_alignment_features_expected_columns():
    X = pd.DataFrame(
        {
            "energy": [10.0, 20.0],
            "maxenergy": [5.0, 15.0],
            "nbumps": [1, 2],
            "nbumps2": [0, 1],
            "nbumps3": [0, 1],
            "nbumps4": [0, 0],
            "nbumps5": [0, 0],
            "nbumps6": [0, 0],
            "nbumps7": [0, 0],
            "nbumps89": [0, 0],
        }
    )
    feat = build_uci_alignment_features(X)
    assert set(feat.columns) == {
        "activity_total",
        "activity_ge_2",
        "activity_ge_3",
        "activity_ge_4",
        "energy_total_log1p",
        "energy_peak_log1p",
        "energy_mean_per_event_log1p",
    }


def test_external_reference_and_prior_features_round_trip():
    external = pd.DataFrame(
        {
            "event_count": [1, 2, 4],
            "count_mag_ge_2": [0, 1, 3],
            "count_mag_ge_3": [0, 1, 2],
            "count_mag_ge_4": [0, 0, 1],
            "energy_proxy_sum": [100.0, 300.0, 900.0],
            "energy_proxy_max": [80.0, 200.0, 500.0],
            "energy_proxy_mean": [100.0, 150.0, 225.0],
        }
    )
    ext_feat = build_external_alignment_features(external)
    ref = fit_external_reference(ext_feat)

    uci = pd.DataFrame(
        {
            "energy": [100.0, 500.0],
            "maxenergy": [50.0, 200.0],
            "nbumps": [1, 2],
            "nbumps2": [0, 1],
            "nbumps3": [0, 0],
            "nbumps4": [0, 1],
            "nbumps5": [0, 0],
            "nbumps6": [0, 0],
            "nbumps7": [0, 0],
            "nbumps89": [0, 0],
            "shift": ["N", "W"],
            "seismic": ["a", "b"],
            "seismoacoustic": ["a", "b"],
            "ghazard": ["a", "b"],
        }
    )

    aug = add_external_prior_features(uci, ref)
    assert "external_prior_mahalanobis" in aug.columns
    assert "external_prior_similarity" in aug.columns
    assert "external_prior_activity_total_pct" in aug.columns
    assert (
        (aug["external_prior_similarity"] > 0) & (aug["external_prior_similarity"] <= 1)
    ).all()
