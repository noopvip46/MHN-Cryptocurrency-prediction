# STanHop Book Depth Pipeline

This repo collects Binance book depth data, pivots it into fixed-width features, and computes MHN-ready features like imbalance ratios, deltas, and rolling z-scores.

## Quick Start

1. Activate your Python environment:
```powershell
& ".\.venv\Scripts\Activate.ps1"
```
2. Run the data pipeline for live updates:
```powershell
python data_pipeline.py
```
3. Enter trading pairs (e.g. `ETHUSDT,BTCUSDT`) and `period` (e.g. `1d` for live daily update).

## Live vs Historical

- **Live use**: call `run_pipeline([...], "1d")` regularly (e.g. daily cron/job) to fetch new data and refresh features.
- **Historical analysis**: use `run_pipeline([...], "7d")` or date strings (e.g. `2026-03-08`) for backtest batch runs.

## Core files

- `data_pipeline.py`: production pipeline with functions:
  - `fetch_and_merge(...)`
  - `pivot_merged(...)`
  - `engineer_features(...)`
  - `run_pipeline(...)`
- `dataCleanUpAndExtraction.ipynb`: exploratory notebook and manual feature experiments.
- `book_depth_utils.py`: helper functions for downloading Binance book-depth data.

## Feature-engineering design choices (STanHop style)

- Use immediate `T` vs `T-1` deltas for “impact” features.
- Use rolling windows (e.g. 60 and 240 minutes) for regime normalization.

## Example Python usage

```python
from data_pipeline import run_pipeline
pairs = ["ETHUSDT", "BTCUSDT"]
df = run_pipeline(pairs, period="1d")
print(df.head())
```

## Notes

- Functions and classes for ongoing retraining and live production reuse.
- The notebook is good for experimentation; the pipeline is for repeatable live/historical runs.
