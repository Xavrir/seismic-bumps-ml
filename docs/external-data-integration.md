# External Data Integration (Phase 1)

This project now includes a concrete ingestion path for external seismic events from USGS ComCat.

## What is added

- `scripts/fetch_external_usgs.py`
  - Fetches event-level USGS catalog rows by date range and event type.
  - Writes raw CSV to `artifacts/external/usgs_mining_events_raw.csv`.
  - Aggregates events into 8-hour windows and writes:
    `artifacts/external/usgs_mining_events_aggregated_8h.csv`.

- `src/data/usgs_external.py`
  - Query URL builder
  - CSV fetcher
  - Magnitude-to-energy proxy conversion
  - Shift-window aggregation helpers

## How to run

```bash
python3 scripts/fetch_external_usgs.py \
  --start 2010-01-01 \
  --end 2025-12-31 \
  --eventtype "mining explosion" \
  --minmag 0.0
```

You can also try:

```bash
python3 scripts/fetch_external_usgs.py --eventtype "quarry blast"
```

## Why this matters

UCI Seismic Bumps has limited positive samples (170 hazardous rows), so recall is hard to improve reliably. USGS event catalogs provide additional mining-related seismic behavior that can be transformed into shift-like statistics.

This does **not** directly produce the same hazard label as UCI. It gives extra seismic context that can be used in later stages for:

1. feature pretraining,
2. anomaly signatures,
3. region- or period-level priors,
4. stress-testing threshold policies.

## Next integration step

To go beyond ingestion and actually improve recall, we should add a training script that uses the aggregated external windows as auxiliary data in a staged workflow:

1. pretrain a representation model on external event windows,
2. fine-tune on UCI labeled data,
3. re-optimize threshold by hazardous F2 on validation,
4. compare against the current baseline in `artifacts/week1_experiments.csv`.
