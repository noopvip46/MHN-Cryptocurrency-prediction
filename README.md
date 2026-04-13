# Flash Crash Predictor

A research project for predicting flash crashes in cryptocurrency markets using order book depth, trade data, and on-chain signals. The core architecture combines MHN, STanHop, and HopCPT — a novel combination for this problem.

STanHop handles multivariate time series across both temporal and variate axes. MHN acts as associative memory that retrieves similar historical crash precursor patterns. HopCPT wraps everything with conformal prediction to give statistically valid uncertainty bounds on each prediction.

Built as part of UoB Masters module ITML608.

---

## Project structure

```
Project/
├── config.py                          central config — all paths, constants, and live settings
├── run.py                             entry point for the full offline pipeline
├── env.example                        environment variable template (copy to .env)
│
├── data_collection/
│   ├── book_depth_utils.py            downloads Binance book depth snapshots (batch)
│   ├── trades_utils.py                downloads Binance historical trades (batch)
│   └── onchain_utils.py               Alchemy on-chain data — our own JSON-RPC client
│                                        (no official Python SDK exists)
│
├── feature_extraction/
│   └── data_pipeline.py               fetch → merge → pivot → engineer → on-chain merge
│
├── models/
│   ├── base.py                        abstract base class all models implement
│   ├── data_adapter.py                loads labeled CSV, builds sliding window sequences
│   ├── hopfield/
│   │   ├── mhn.py                     Modern Hopfield Network
│   │   ├── stanhop.py                 Sparse Tandem Hopfield Network
│   │   └── hopcpt.py                  conformal prediction wrapper (any model)
│   ├── deep_learning/
│   │   ├── lstm.py                    bidirectional LSTM with attention pooling
│   │   └── transformer.py             transformer encoder with CLS token pooling
│   └── baselines/
│       └── ml_models.py               XGBoost, Random Forest, Logistic Regression
│
├── live/
│   ├── binance_stream.py              Binance futures WebSocket → 30 s snapshot aggregator
│   ├── snapshot_buffer.py             thread-safe rolling buffer (capped at window × 2880 rows)
│   ├── feature_computer.py            raw snapshots → engineered features + live labeling
│   ├── predictor.py                   inference loop with atomic model-swap support
│   ├── trainer.py                     rolling retrain loop, swaps predictor model atomically
│   └── run_live.py                    entry point for live production mode
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_modelling.ipynb
│
├── bookDepth_data/                    book depth CSVs (not committed)
├── trades_data/                       trades CSVs (not committed)
└── onchain_data/                      Alchemy on-chain CSVs (not committed)
```

---

## Setup

```bash
pip install pandas numpy torch scikit-learn xgboost requests websockets
```

Copy the environment template and fill in your Alchemy API key:

```bash
cp env.example .env
# edit .env and set ALCHEMY_API_KEY
```

Alchemy key is optional — the pipeline and live system both run without it, but on-chain features will be zero. Get a free key at [dashboard.alchemy.com](https://dashboard.alchemy.com). Enable **Ethereum Mainnet** in your dashboard.

---

## Offline pipeline

Run the full pipeline from scratch:

```bash
python run.py --pairs BTCUSDT,ETHUSDT --period 6m --model stanhop
```

Skip steps if data is already downloaded:

```bash
# skip download, just train
python run.py --skip-download --skip-extract --model stanhop

# train without conformal wrapper
python run.py --skip-download --skip-extract --model lstm --no-conformal

# run an XGBoost baseline
python run.py --skip-download --skip-extract --model xgboost
```

The pipeline runs four steps in order:

| Step | What it does |
|---|---|
| **1 — Download** | Fetches Binance book depth + trades + Alchemy on-chain data for the given period |
| **2 — Extract** | Pivots raw CSVs, engineers features, merges on-chain columns |
| **3 — Label** | Applies volatility-adjusted flash crash labels |
| **4 — Train** | Trains the chosen model, optionally wraps with HopCPT |

---

## Live production mode

Runs the full system in real time: ingests data, predicts every 30 seconds, and retrains on a rolling window.

```bash
python live/run_live.py

# with warm restart — buffer survives crashes/restarts
python live/run_live.py --buffer-file live_buffer.parquet

# custom window and retrain interval
python live/run_live.py --window 14 --retrain-hours 12
```

### How it works

```
BinanceStream  ──┐
 @aggTrade        │                             ┌──► Predictor  (every 30 s)
 @depth20@500ms  ├──► FeatureComputer ──► Buffer │
 @forceOrder     │                             └──► Trainer    (every 24 h)
OnchainStream  ──┘
```

**Rolling window** — `SnapshotBuffer` is capped at `--window` days × 2880 snapshots. As new rows arrive the oldest fall off. The Trainer always retrains on whatever is in the buffer — no manual pruning needed.

**Live labeling** — Labels cannot be known at the moment of a snapshot (they depend on what happens next). `FeatureComputer.generate_label()` fires every tick, looks back `CRASH_HORIZON = 20` snapshots (10 minutes), checks whether a crash occurred in that window, and writes the label retroactively into the buffer. The Trainer only uses labeled rows.

**Atomic model swap** — Retraining runs in a background thread. When a new model is ready it is swapped into the Predictor atomically so inference never blocks or drops a cycle.

### Live options

| Flag | Default | Description |
|---|---|---|
| `--pairs` | `BTCUSDT,ETHUSDT` | Trading pairs to stream |
| `--model` | `stanhop` | Architecture: `mhn`, `stanhop`, `lstm`, `transformer` |
| `--window` | `30` | Rolling training window in days |
| `--retrain-hours` | `24` | How often to retrain |
| `--no-conformal` | off | Disable HopCPT wrapper |
| `--alpha` | `0.1` | HopCPT miscoverage level |
| `--buffer-file` | none | Parquet path for warm restart |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |

---

## Data

### Exchange data (Binance)
Order book depth snapshots at ~30 second intervals and historical trades, both downloaded from the Binance public data archive. No API key required.

Features per pair: order book imbalance ratios, notional imbalance, trade flow imbalance (buy/sell pressure), VWAP returns, liquidation count, and rolling z-scores at 60 and 240 minute windows.

### On-chain data (Alchemy)
ETH mainnet features fetched via our own JSON-RPC client (Alchemy has no official Python SDK). Features are **global market-state columns** — not per-pair — so BTCUSDT and ETHUSDT both carry the same on-chain values at each timestamp. Requires `ALCHEMY_API_KEY` in `.env`.

| Feature | Source | Signal |
|---|---|---|
| `base_fee_gwei_mean/max` | `eth_feeHistory` | Network congestion / panic activity |
| `gas_used_ratio_mean` | `eth_feeHistory` | Block fullness |
| `large_transfer_count/eth` | `alchemy_getAssetTransfers` (>50 ETH) | Whale moves |
| `exchange_inflow_count/eth` | Same, filtered to known CEX wallets | Selling pressure |

### Flash crash label
A crash is flagged at timestamp T if the cumulative VWAP return over the next 10 minutes (20 snapshots) falls more than 3 standard deviations below the current rolling volatility. The label pair is `BTCUSDT` by default.

---

## Models

All models implement the same interface and can be swapped freely.

```python
from models import SequenceDataset, STanHopModel, HopCPT

ds = SequenceDataset("bookDepth_data/all_pairs_labeled.csv", seq_len=120)
(X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = ds.get_splits()

model = STanHopModel(seq_len=120, n_features=ds.n_features)
cpt   = HopCPT(model, alpha=0.1)
cpt.fit(X_tr, y_tr, X_val=X_cal, y_val=y_cal)
cpt.calibrate(X_cal, y_cal)

print(cpt.evaluate(X_te, y_te))
```

Use `ds.get_flat_splits()` for sklearn-style models (XGBoost, Random Forest, Logistic Regression) since they expect 2D input.

The data split is always chronological — 70% train, 15% calibration (for HopCPT), 15% test. No shuffling.

---

## Configuration

All constants live in `config.py`. Key settings:

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_PAIRS` | `BTCUSDT, ETHUSDT` | Trading pairs |
| `DEFAULT_PERIOD` | `6m` | Historical download window |
| `SNAPSHOT_INTERVAL_S` | `30` | Seconds between snapshots |
| `SEQ_LEN` | `120` | Input sequence length (= 60 min) |
| `CRASH_HORIZON` | `20` | Snapshots forward for label (= 10 min) |
| `CRASH_SIGMA` | `3.0` | Volatility threshold for crash label |
| `LABEL_PAIR` | `BTCUSDT` | Pair used to generate the label |
| `LIVE_WINDOW_DAYS` | `30` | Rolling training window in live mode |
| `RETRAIN_INTERVAL_HOURS` | `24` | Live retrain frequency |
