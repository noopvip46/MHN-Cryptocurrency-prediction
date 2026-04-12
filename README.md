# Flash Crash Predictor

A research project for predicting flash crashes in cryptocurrency markets using order book depth and trade data from Binance. The core architecture combines MHN, STanHop, and HopCPT, which we believe is a novel combination for this problem.

The idea is that STanHop handles the multivariate time series, MHN acts as an associative memory that retrieves similar historical crash precursor patterns, and HopCPT wraps the whole thing with conformal prediction to give statistically valid uncertainty bounds on predictions.

This project was built as part of UoB Masters module ITML608.

## Project structure

```
Project/
├── config.py                        central config for all paths and constants
├── run.py                           single entry point to run the full pipeline
│
├── notebooks/
│   ├── 01_data_collection.ipynb     download book depth and trades data from Binance
│   ├── 02_feature_extraction.ipynb  feature engineering, cleaning, and labelling
│   └── 03_modelling.ipynb           demo of training and evaluating all models
│
├── data_collection/
│   ├── book_depth_utils.py          downloads Binance book depth snapshots
│   └── trades_utils.py              downloads Binance historical trades
│
├── feature_extraction/
│   └── data_pipeline.py             full fetch, merge, pivot, feature engineering pipeline
│
└── models/
    ├── base.py                      abstract base class all models implement
    ├── data_adapter.py              loads labeled CSV and builds sliding window sequences
    ├── hopfield/
    │   ├── mhn.py                   Modern Hopfield Network
    │   ├── stanhop.py               Sparse Tandem Hopfield Network
    │   └── hopcpt.py                conformal prediction wrapper for any model
    ├── deep_learning/
    │   ├── lstm.py                  bidirectional LSTM with attention pooling
    │   └── transformer.py           transformer encoder with CLS token pooling
    └── baselines/
        └── ml_models.py             XGBoost, Random Forest, Logistic Regression
```

## Quick start

Run the full pipeline from scratch:

```bash
python run.py --pairs BTCUSDT,ETHUSDT --period 6m --model stanhop
```

Skip steps if you already have data:

```bash
# already downloaded data, just train
python run.py --skip-download --skip-extract --model stanhop

# train without conformal prediction wrapper
python run.py --skip-download --skip-extract --model lstm --no-conformal

# run an XGBoost baseline
python run.py --skip-download --skip-extract --model xgboost
```

## Data

Data comes from the Binance public data API. No API key is needed. Book depth snapshots are collected at roughly 30 second intervals. Trades data is aggregated into those same intervals.

Features include order book imbalance ratios, notional imbalance, trade flow imbalance, VWAP returns, buy/sell pressure, and rolling z-scores at 60 and 240 minute windows.

The flash crash label is generated using a volatility-adjusted threshold: a crash is flagged at timestamp T if the cumulative return over the next 10 minutes falls more than 3 standard deviations below the current rolling volatility.

Data files are not committed to git. Run the pipeline to fetch your own copy.

## Models

All models implement the same interface so you can swap them freely.

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

For sklearn-style models use `ds.get_flat_splits()` instead since they expect 2D input.

The data split is always chronological: 70% train, 15% calibration for HopCPT, 15% test. No shuffling since this is time series data.

## Requirements

```bash
pip install pandas numpy torch scikit-learn xgboost requests
```
