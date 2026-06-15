"""Fetch and aggregate external USGS mining-event data for next-phase modeling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from usgs_external import (
    aggregate_events_to_shift_windows,
    build_usgs_query_url,
    fetch_usgs_events_csv,
    summarize_external_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch USGS mining explosion events and aggregate to 8-hour windows"
    )
    parser.add_argument("--start", default="2010-01-01", help="start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="end date (YYYY-MM-DD)")
    parser.add_argument(
        "--eventtype",
        default="mining explosion",
        help="USGS event type filter (e.g., 'mining explosion', 'quarry blast')",
    )
    parser.add_argument("--minmag", type=float, default=0.0, help="minimum magnitude")
    parser.add_argument(
        "--window-hours", type=int, default=8, help="aggregation window"
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/external",
        help="output directory for generated CSV files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    query_url = build_usgs_query_url(
        starttime=args.start,
        endtime=args.end,
        eventtype=args.eventtype,
        minmagnitude=args.minmag,
    )

    print("Fetching USGS catalog...")
    print(f"URL: {query_url}")
    events = fetch_usgs_events_csv(query_url)
    summary = summarize_external_catalog(events)

    raw_path = out_dir / "usgs_mining_events_raw.csv"
    events.to_csv(raw_path, index=False)

    print(
        "Fetched rows={rows}, time range=({start} -> {end}), mag_mean={mag_mean}, mag_max={mag_max}".format(
            rows=summary["rows"],
            start=summary["min_time"],
            end=summary["max_time"],
            mag_mean=summary["mag_mean"],
            mag_max=summary["mag_max"],
        )
    )

    print(f"Aggregating to {args.window_hours}-hour windows...")
    agg = aggregate_events_to_shift_windows(events, window_hours=args.window_hours)
    agg_path = out_dir / "usgs_mining_events_aggregated_8h.csv"
    agg.to_csv(agg_path, index=False)

    print(f"Saved raw catalog: {raw_path}")
    print(f"Saved aggregated catalog: {agg_path}")
    print(f"Aggregated windows: {len(agg)}")


if __name__ == "__main__":
    main()
