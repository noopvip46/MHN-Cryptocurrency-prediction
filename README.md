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
│   ├── book_depth_utils.py            downloads + pivots Binance book depth snapshots
│   ├── trades_utils.py                downloads + aggregates Binance historical trades
│   └── onchain_utils.py               Alchemy on-chain data — our own JSON-RPC client
│                                        (no official Python SDK exists)
│
├── feature_extraction/
│   └── data_pipeline.py               pivot → engineer → on-chain merge → trade merge → time features
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
├── docs/
│   ├── DATA_COLLECTION.md             all raw data sources — fields, formats, disk layout
│   └── FEATURE_EXTRACTION.md         full pipeline from raw data to model tensors
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_modelling.ipynb
│
├── bookDepth_data/                    book depth CSVs (not committed)
├── trades_data/                       trades CSVs (not committed)
├── onchain_data/                      Alchemy on-chain CSVs (not committed)
└── checkpoints/                       model checkpoints saved during training (not committed)
```

---

## Setup

**conda (recommended)**

PyTorch's package name on conda is `pytorch`, not `torch`. Install from the official pytorch channel:

```bash
# CPU only (testing / no GPU)
conda install pytorch cpuonly -c pytorch
conda install -c conda-forge pandas numpy scikit-learn xgboost requests websockets

# GPU — replace 12.1 with your installed CUDA version
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge pandas numpy scikit-learn xgboost requests websockets
```

**pip (alternative)**

```bash
# CPU only
pip install pandas numpy torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn xgboost requests websockets

# GPU — pip bundles CUDA inside the wheel (~2 GB)
pip install pandas numpy torch scikit-learn xgboost requests websockets
```

Copy the environment template and fill in your Alchemy API key:

```bash
cp env.example .env
# edit .env and set ALCHEMY_API_KEY
```

Alchemy key is optional — the pipeline runs without it using `--no-onchain`. Get a free key at [dashboard.alchemy.com](https://dashboard.alchemy.com). Enable **Ethereum Mainnet** in your dashboard.

---

## Offline pipeline

### Common run commands

```bash
# Full pipeline from scratch
python run.py --pairs BTCUSDT,ETHUSDT --period 6m --model stanhop

# Skip on-chain — no Alchemy key needed, much faster
python run.py --pairs BTCUSDT,ETHUSDT --period 6m --model stanhop --no-onchain

# Data already downloaded — skip straight to extract + label + train
python run.py --skip-download --model stanhop --no-onchain

# Data already extracted — skip straight to train
python run.py --skip-download --skip-extract --skip-label --model stanhop

# Different model, no conformal wrapper
python run.py --skip-download --skip-extract --skip-label --model lstm --no-conformal

# ML baselines (XGBoost, Random Forest, Logistic Regression)
python run.py --skip-download --skip-extract --skip-label --model xgboost
python run.py --skip-download --skip-extract --skip-label --model random_forest

# Resume a training run interrupted by Ctrl+C
python run.py --skip-download --skip-extract --skip-label --model stanhop \
    --resume checkpoints/stanhop_checkpoint.pt

# Memory-only pipeline — no intermediate CSVs, only all_pairs_labeled.csv written
python run.py --no-onchain --no-save --model stanhop

# Custom hyperparameters
python run.py --skip-download --skip-extract --skip-label --model stanhop \
    --hidden-dim 256 --n-heads 8 --top-k 20 --dropout 0.2 --lr 5e-4 --batch-size 512

python run.py --skip-download --skip-extract --skip-label --model lstm \
    --hidden-dim 256 --n-layers 3 --dropout 0.3 --lr 5e-4

python run.py --skip-download --skip-extract --skip-label --model xgboost --device cuda \
    --xgb-max-depth 4 --xgb-lr 0.02 --xgb-n-estimators 1000 --xgb-early-stopping 50

# Run on a other datasets with different pair names
python run.py --skip-download --skip-extract --skip-label \
    --data-file /path/to/their_labeled.csv --label-pair SOLUSDT --model stanhop --no-conformal
```

### All flags

**Pipeline control**

| Flag | Default | Description |
|---|---|---|
| `--pairs` | `BTCUSDT,ETHUSDT` | Comma-separated trading pairs |
| `--period` | `6m` | Download period: `7d`, `2w`, `6m`, `1y`, or `YYYY-MM-DD/YYYY-MM-DD` |
| `--model` | `stanhop` | Architecture: `mhn`, `stanhop`, `lstm`, `transformer`, `xgboost`, `random_forest`, `logistic` |
| `--seq-len` | `120` | Sequence length in snapshots (120 = 60 min) |
| `--epochs` | `50` | Training epochs for DL models |
| `--alpha` | `0.1` | HopCPT miscoverage level — target coverage = 1 − alpha |
| `--skip-download` | off | Skip download step (use existing data) |
| `--skip-extract` | off | Skip feature extraction step |
| `--skip-label` | off | Skip label generation step |
| `--no-conformal` | off | Train without the HopCPT wrapper |
| `--no-onchain` | off | Skip Alchemy on-chain data entirely |
| `--no-save` | off | Memory-only pipeline — no intermediate CSVs |
| `--checkpoint-dir` | `checkpoints/` | Directory for per-epoch DL training checkpoints |
| `--resume` | none | Path to a `.pt` checkpoint — resumes training from that epoch |
| `--device` | `auto` | Compute device: `cpu`, `cuda`, or `auto` |
| `--data-file` | none | Path to a pre-built labeled CSV — skips download/extract/label entirely |
| `--label-pair` | `BTCUSDT` | Pair name whose `_flash_crash_label` column is the prediction target |

**Hyperparameters — shared DL** (apply to MHN, STanHop, LSTM, Transformer)

| Flag | Default | Description |
|---|---|---|
| `--hidden-dim` | `128` | Hidden / model dimension |
| `--n-heads` | `4` | Number of attention heads |
| `--n-layers` | `2` (LSTM) / `3` (Transformer) | Number of stacked layers |
| `--dropout` | `0.2` (LSTM) / `0.1` (others) | Dropout rate |
| `--lr` | `1e-3` | AdamW learning rate |
| `--batch-size` | `256` | Mini-batch size |

**Hyperparameters — model-specific**

| Flag | Default | Applies to | Description |
|---|---|---|---|
| `--top-k` | `10` | STanHop | Sparse attention — keep top-k scores per query |
| `--n-patterns` | `64` | MHN | Number of learnable memory patterns |
| `--dim-feedforward` | `256` | Transformer | Feedforward dimension inside each encoder layer |

**Hyperparameters — XGBoost**

| Flag | Default | Description |
|---|---|---|
| `--xgb-n-estimators` | `500` | Max trees — early stopping usually cuts this short |
| `--xgb-max-depth` | `6` | Maximum tree depth |
| `--xgb-lr` | `0.05` | Learning rate / eta |
| `--xgb-early-stopping` | `30` | Stop after N rounds without val improvement |

### Pipeline steps

| Step | Function | What it does |
|---|---|---|
| **1 — Download** | `step_download` | Fetches Binance book depth ZIPs, pivots on the fly, aggregates trades to 30 s bins. Raw data is deleted immediately after processing — peak disk ~600 MB. |
| **2 — Extract** | `run_pipeline` | Aligns pairs via merge_asof spine, engineers stationary features, merges on-chain columns, merges pre-aggregated trade features, adds time encoding. |
| **3 — Label** | `step_label` | Applies volatility-adjusted forward-looking flash crash labels. Drops the last 20 rows (incomplete forward window). |
| **4 — Train** | `step_train` | Trains the chosen model; optionally wraps with HopCPT conformal prediction. |

### Disk layout after a full run

```
bookDepth_data/
  BTCUSDT/BTCUSDT_merged.csv          ~50 MB  (6 months, wide format)
  ETHUSDT/ETHUSDT_merged.csv          ~50 MB
  all_pairs_labeled.csv               ~15 MB  (final training dataset)

trades_data/
  BTCUSDT/BTCUSDT_trades_agg.csv      ~20 MB  (30 s bins)
  ETHUSDT/ETHUSDT_trades_agg.csv      ~20 MB

checkpoints/
  stanhop_checkpoint.pt               last completed epoch weights + optimizer state
```

Total: ~155 MB for 6 months of two pairs. Peak during download: ~650 MB (one day's raw trades).

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

## Documentation

| Doc | Description |
|---|---|
| [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md) | Raw data sources — fields, formats, download behaviour, disk usage |
| [docs/FEATURE_EXTRACTION.md](docs/FEATURE_EXTRACTION.md) | Full pipeline from raw data to model tensors — every column, formula, and transformation |

---

## Data

### Exchange data (Binance)

Order book depth snapshots at 30-second intervals and historical trades, both downloaded from the Binance public data archive. No API key required.

**Book depth:** raw daily ZIPs are downloaded, pivoted from long format (one row per price level) to wide format (one row per snapshot) immediately, and the raw CSV deleted. Final output is one `{PAIR}_merged.csv` per pair (~50 MB for 6 months).

**Trades:** raw daily ZIPs (~500 MB/day uncompressed) are downloaded, aggregated to 30-second bins, and the raw CSV deleted. Final output is one `{PAIR}_trades_agg.csv` per pair (~20 MB for 6 months). Peak disk use at any moment during download is ~650 MB (one day's data only).

### Features

All features fed to the model are stationary — raw per-level depth and absolute notional aggregates are dropped during feature engineering. With two pairs and no on-chain data the model sees **30 features** per snapshot (13 per pair + 4 time features).

| Feature group | Features per pair | What it captures |
|---|---|---|
| Order-book ratios | `depth_imbalance_ratio`, `notional_imbalance_ratio` | Signed quantity and value skew ∈ (−1, 1) |
| Deltas | `depth_imbalance_ratio_delta`, `notional_imbalance_ratio_delta`, `total_notional_delta` | Tick-to-tick change in book state |
| Z-scores | `notional_z` (60 min), `notional_regime_z` (240 min) | Deviation from recent and regime-level norms |
| Trade flow | `vwap_return`, `buy_ratio`, `trade_flow_imbalance` | Price change, aggressor direction |
| Trade activity | `trade_intensity_z`, `trade_intensity_regime_z`, `trade_notional_z` | Activity surge vs short and long-term baseline |
| Time (global) | `hour_sin/cos`, `dow_sin/cos` | Cyclically-encoded time of day and day of week |

With on-chain enabled: +7 global ETH mainnet features = **37 features** total.

### On-chain data (Alchemy)

ETH mainnet features fetched via our own JSON-RPC client (Alchemy has no official Python SDK). Features are **global market-state columns** shared across all pairs. Requires `ALCHEMY_API_KEY` in `.env`.

| Feature | Source | Signal |
|---|---|---|
| `base_fee_gwei_mean/max` | `eth_feeHistory` | Network congestion / panic activity |
| `gas_used_ratio_mean` | `eth_feeHistory` | Block fullness |
| `large_transfer_count/eth` | `alchemy_getAssetTransfers` (>50 ETH) | Whale moves |
| `exchange_inflow_count/eth` | Same, filtered to known CEX wallets | Selling pressure |

### Flash crash label

A crash is flagged at timestamp `t` if the **cumulative VWAP return over the next 10 minutes is a `CRASH_SIGMA`-sigma event on the 10-minute return timescale**:

```
label[t] = 1  if  Σ vwap_return[t+1 … t+20]  <  −CRASH_SIGMA × vwap_volatility[t] × sqrt(20)
```

The `sqrt(20)` scaling is critical — the forward return is a sum of 20 periods so its standard deviation is `vwap_volatility × sqrt(20)`, not `vwap_volatility` alone. The threshold is also dynamic, adapting to the current volatility regime. With `CRASH_SIGMA=2.0` the expected crash rate is ~2.3%. The label pair is `BTCUSDT` by default.

---

## Models

All models implement the same interface (`fit`, `predict_proba`, `evaluate`, `save`, `load`) and can be swapped freely.

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

Use `ds.get_flat_splits()` for sklearn-style models (XGBoost, Random Forest, Logistic Regression) — they expect 2D `(N, seq_len × n_features)` input.

The data split is always chronological — 70% train, 15% calibration (for HopCPT), 15% test. No shuffling.

### Class imbalance

Flash crashes occur 0.5–2% of the time — a model predicting "no crash" every time scores ~99% accuracy and is useless. All models derive class weights from the training split only (never the full dataset, to avoid leakage):

- **DL models** (MHN, STanHop, LSTM, Transformer) — `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`
- **XGBoost** — `scale_pos_weight = n_neg / n_pos`
- **Random Forest, Logistic Regression** — `class_weight='balanced'`

STanHop's sparse top-k attention also helps at the retrieval level — it can surface rare historical crash precursor patterns rather than diluting them with the dominant normal-market signal, complementing loss-level weighting.

### Evaluation metrics

Accuracy is not reported — it is uninformative on imbalanced data. All models report:

| Metric | What it measures |
|---|---|
| **PR-AUC** (`avg_prec`) | Area under Precision-Recall curve — primary metric |
| **ROC-AUC** (`roc_auc`) | Threshold-independent discrimination |
| **F1** | Harmonic mean of precision and recall at threshold 0.5 |
| **Precision** | Of all crash alerts raised, what fraction were real |
| **Recall** | Of all actual crashes, what fraction were caught |

HopCPT additionally reports:

| Metric | What it measures |
|---|---|
| `conformal_coverage` | Fraction where true label fell inside the prediction set — should be ≥ 1 − α |
| `uncertain_rate` | Fraction where the model abstained (both classes in set) |
| `empty_set_rate` | Fraction where neither class was included |

### Checkpointing

DL models save a checkpoint to `checkpoints/{model}_checkpoint.pt` after every completed epoch. The file contains the network weights, optimizer state, and scheduler state so training can resume exactly where it left off. On Ctrl+C mid-epoch, the last completed epoch's checkpoint is preserved.

```bash
# Resume from a checkpoint
python run.py --skip-download --skip-extract --skip-label --model stanhop \
    --resume checkpoints/stanhop_checkpoint.pt
```

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
| `CRASH_SIGMA` | `3.0` | Volatility threshold multiplier for crash label |
| `LABEL_PAIR` | `BTCUSDT` | Pair used as the training target |
| `LIVE_WINDOW_DAYS` | `30` | Rolling training window in live mode |
| `RETRAIN_INTERVAL_HOURS` | `24` | Live retrain frequency |
