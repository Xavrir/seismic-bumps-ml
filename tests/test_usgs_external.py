import pandas as pd

from usgs_external import (
    aggregate_events_to_shift_windows,
    build_usgs_query_url,
    magnitude_to_energy_proxy,
)


def test_build_usgs_query_url_contains_required_params():
    url = build_usgs_query_url(
        starttime="2020-01-01",
        endtime="2020-12-31",
        eventtype="mining explosion",
        minmagnitude=1.5,
    )
    assert "format=csv" in url
    assert "starttime=2020-01-01" in url
    assert "endtime=2020-12-31" in url
    assert "eventtype=mining+explosion" in url
    assert "minmagnitude=1.5" in url


def test_magnitude_to_energy_proxy_monotonic_in_magnitude():
    mags = pd.Series([1.0, 2.0, 3.0])
    energy = magnitude_to_energy_proxy(mags)
    assert energy.iloc[0] < energy.iloc[1] < energy.iloc[2]


def test_aggregate_events_to_shift_windows_basic_shape_and_counts():
    events = pd.DataFrame(
        {
            "time": [
                "2020-01-01T00:10:00Z",
                "2020-01-01T01:15:00Z",
                "2020-01-01T08:20:00Z",
            ],
            "mag": [1.1, 1.3, 2.0],
            "depth": [5.0, 7.0, 4.0],
            "latitude": [10.0, 10.2, 10.5],
            "longitude": [20.0, 20.2, 20.5],
        }
    )

    agg = aggregate_events_to_shift_windows(events, window_hours=8)
    assert len(agg) == 2
    assert list(agg["event_count"]) == [2, 1]
    assert list(agg["count_mag_ge_2"]) == [0, 1]
    assert list(agg["count_mag_ge_3"]) == [0, 0]
    assert list(agg["count_mag_ge_4"]) == [0, 0]
    assert "energy_proxy_sum" in agg.columns
    assert (agg["mag_std"] >= 0).all()
