"""Utilities for fetching and aggregating USGS mining-event catalogs."""

from __future__ import annotations

from io import StringIO
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

USGS_FDSN_EVENT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def build_usgs_query_url(
    starttime: str,
    endtime: str,
    eventtype: str = "mining explosion",
    minmagnitude: float = 0.0,
) -> str:
    """Build a deterministic USGS ComCat CSV query URL."""
    params = {
        "format": "csv",
        "starttime": starttime,
        "endtime": endtime,
        "eventtype": eventtype,
        "minmagnitude": minmagnitude,
        "orderby": "time-asc",
    }
    return f"{USGS_FDSN_EVENT_URL}?{urlencode(params)}"


def fetch_usgs_events_csv(url: str, timeout: int = 60) -> pd.DataFrame:
    """Fetch USGS CSV endpoint and return a DataFrame."""
    with urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    df = pd.read_csv(StringIO(body))
    if "time" not in df.columns:
        raise ValueError("USGS response missing required 'time' column")
    return df


def magnitude_to_energy_proxy(magnitude: pd.Series) -> pd.Series:
    """Convert magnitude to energy proxy in joules via Gutenberg-Richter formula."""
    mag = pd.to_numeric(magnitude, errors="coerce")
    return 10 ** (1.5 * mag + 4.8)


def aggregate_events_to_shift_windows(
    events: pd.DataFrame, window_hours: int = 8
) -> pd.DataFrame:
    """Aggregate event-level USGS rows into fixed-hour windows.

    The output is designed to resemble shift-level aggregation, similar in spirit
    to the Seismic Bumps problem framing.
    """
    if "time" not in events.columns:
        raise ValueError("events must include 'time' column")

    df = events.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).copy()

    df["mag"] = pd.to_numeric(df.get("mag"), errors="coerce")
    df["depth"] = pd.to_numeric(df.get("depth"), errors="coerce")
    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["energy_proxy"] = magnitude_to_energy_proxy(df["mag"])
    df["count_mag_ge_2"] = (df["mag"] >= 2.0).astype(int)
    df["count_mag_ge_3"] = (df["mag"] >= 3.0).astype(int)
    df["count_mag_ge_4"] = (df["mag"] >= 4.0).astype(int)

    freq = f"{window_hours}h"
    df["window_start_utc"] = df["time"].dt.floor(freq)

    grouped = (
        df.groupby("window_start_utc", as_index=False)
        .agg(
            event_count=("time", "size"),
            count_mag_ge_2=("count_mag_ge_2", "sum"),
            count_mag_ge_3=("count_mag_ge_3", "sum"),
            count_mag_ge_4=("count_mag_ge_4", "sum"),
            mag_mean=("mag", "mean"),
            mag_max=("mag", "max"),
            mag_std=("mag", "std"),
            depth_mean=("depth", "mean"),
            depth_max=("depth", "max"),
            depth_std=("depth", "std"),
            energy_proxy_sum=("energy_proxy", "sum"),
            energy_proxy_max=("energy_proxy", "max"),
            energy_proxy_mean=("energy_proxy", "mean"),
            latitude_mean=("latitude", "mean"),
            longitude_mean=("longitude", "mean"),
        )
        .sort_values("window_start_utc")
    )

    for col in ["mag_std", "depth_std"]:
        grouped[col] = grouped[col].fillna(0.0)

    grouped["window_end_utc"] = grouped["window_start_utc"] + pd.to_timedelta(
        window_hours, unit="h"
    )

    ordered_cols = [
        "window_start_utc",
        "window_end_utc",
        "event_count",
        "count_mag_ge_2",
        "count_mag_ge_3",
        "count_mag_ge_4",
        "mag_mean",
        "mag_max",
        "mag_std",
        "depth_mean",
        "depth_max",
        "depth_std",
        "energy_proxy_sum",
        "energy_proxy_max",
        "energy_proxy_mean",
        "latitude_mean",
        "longitude_mean",
    ]

    return grouped[ordered_cols].reset_index(drop=True)


def summarize_external_catalog(events: pd.DataFrame) -> dict:
    """Compute concise QA summary stats for fetched external event data."""
    n_rows = int(len(events))
    min_time = pd.to_datetime(events["time"], utc=True, errors="coerce").min()
    max_time = pd.to_datetime(events["time"], utc=True, errors="coerce").max()
    mag = pd.to_numeric(events.get("mag"), errors="coerce")

    return {
        "rows": n_rows,
        "min_time": str(min_time) if pd.notna(min_time) else None,
        "max_time": str(max_time) if pd.notna(max_time) else None,
        "mag_mean": float(np.nanmean(mag)) if np.isfinite(np.nanmean(mag)) else None,
        "mag_max": float(np.nanmax(mag)) if np.isfinite(np.nanmax(mag)) else None,
    }
