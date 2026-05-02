# Feature Extraction

This document traces every transformation from downloaded data to the final tensor fed into the model. The pipeline runs inside `feature_extraction/data_pipeline.py`; labeling runs in `run.py`.

---

## Pipeline overview

```
bookDepth_data/{PAIR}_merged.csv  ──┐
trades_data/{PAIR}_trades_agg.csv ──┼──► 1: Align pairs  ──► 2: Engineer OB features
onchain_data/ETHUSDT-onchain-*.csv ─┘         (merge_asof)        (stationary only)
                                                                         │
                                          5: Time features ◄── 4: Trade feature merge
                                                   │                (merge_asof)
                                                   ▼
                                            6: Label  ──► all_pairs_labeled.csv  ──► Model
```

Intermediate files are written only when `--no-save` is not passed (default saves `all_pairs_cleaned.csv` and `all_pairs_with_trades.csv` as debugging checkpoints). The entire pipeline can run in memory with `--no-save`.

---

## Step 1 — Align pairs (`pivot_merged`)

**Input:** `{PAIR}_merged.csv` per pair — already wide-format (one row per snapshot, all price levels as columns), produced by `book_depth_utils.download_book_depth_range`  
**Output:** single DataFrame, one row per spine timestamp

The `{PAIR}_merged.csv` files are loaded and aligned using `merge_asof` with a 30-second tolerance, using the first pair's timestamps as the spine. This ensures:

- Exactly one row per spine snapshot — no row inflation
- ETHUSDT columns filled from the nearest matching timestamp (within ±30 s)
- Any gap larger than 30 s produces NaN for that pair's columns (extremely rare)

An outer merge is explicitly avoided — it would double the row count when two pairs' snapshot timestamps differ by even a fraction of a second, producing NaN-filled gaps that corrupt rolling window computations and inflate the apparent crash rate.

**Column layout after merge:**

```
timestamp
BTCUSDT_depth_{pct}       quantity at percentage level (bid: negative, ask: positive)
BTCUSDT_notional_{pct}    notional value at level
ETHUSDT_depth_{pct}
ETHUSDT_notional_{pct}
…
```

---

## Step 2 — Feature engineering (`engineer_features`)

**Input:** aligned wide DataFrame  
**Output:** stationary features only — all raw per-level and absolute aggregate columns are dropped

Applied independently per trading pair. All column names are prefixed with `{PAIR}_`.

Only stationary, normalised features survive. Raw depth and notional columns are non-stationary (absolute market size grows over time) and provide no information beyond what the ratios and z-scores already capture.

### Order-book ratio features

| Column | Formula | What it captures |
|---|---|---|
| `{P}_depth_imbalance_ratio` | `(Σask_depth − Σbid_depth) / (Σask + Σbid)` | Normalised quantity skew ∈ (−1, 1); positive = more asks |
| `{P}_notional_imbalance_ratio` | `(Σask_notional − Σbid_notional) / (Σask + Σbid)` | Normalised value skew ∈ (−1, 1) |

> Notional is more informative than raw depth because it accounts for the price at each level — a deep ask wall further from mid carries less weight than a thin one close to mid.

### Delta features

First differences over the 30-second snapshot cadence (T − T−1).

| Column | What it captures |
|---|---|
| `{P}_depth_imbalance_ratio_delta` | Rate of change in quantity skew |
| `{P}_notional_imbalance_ratio_delta` | Rate of change in value skew |
| `{P}_total_notional_delta` | Change in total book value — sudden thinning of the book |

### Rolling z-score features

Applied to `log1p(total_notional)` to stabilise variance across market regimes.

| Column | Window | What it captures |
|---|---|---|
| `{P}_notional_z` | 120 snapshots (~60 min) | Short-term deviation from the recent norm |
| `{P}_notional_regime_z` | 480 snapshots (~240 min) | Deviation from the current market regime |

`min_periods=1` so no rows are dropped at the start of the series. Zero standard deviation is replaced with `NaN` (handled downstream by ffill/bfill in `SequenceDataset`).

### Dropped after this step

- `{PAIR}_depth_{pct}` and `{PAIR}_notional_{pct}` — all raw per-level columns (absolute, non-stationary)

---

## Step 3 — On-chain merge (`merge_onchain`)

**Input:** engineered DataFrame + `onchain_data/ETHUSDT/ETHUSDT-onchain-*.csv`  
**Output:** same DataFrame + 7 global columns  
**Skipped when:** `--no-onchain` flag is passed

On-chain features are joined on `timestamp` (left join — book depth rows are the spine). Gaps between ETH block timestamps (~12 s) and snapshot timestamps (~30 s) are closed with forward-fill then back-fill. No zero-filling is used.

| Column | Type | Description |
|---|---|---|
| `base_fee_gwei_mean` | float | Mean EIP-1559 base fee in the 30 s window (gwei) |
| `base_fee_gwei_max` | float | Peak base fee — sharp spikes signal network panic |
| `gas_used_ratio_mean` | float | Mean block fullness (0–1) — congestion proxy |
| `large_transfer_count` | int | ETH transfers > 50 ETH — whale activity |
| `large_transfer_eth` | float | Total ETH moved by whale transfers |
| `exchange_inflow_count` | int | Whale transfers **to** known exchange wallets |
| `exchange_inflow_eth` | float | Total ETH flowing to exchanges — sell pressure signal |

These columns are present in every row regardless of trading pair — they are global market-state signals, not per-pair.

---

## Step 4 — Trade feature merge (`merge_trades`)

**Input:** engineered DataFrame + `trades_data/{PAIR}/{PAIR}_trades_agg.csv` (pre-aggregated 30 s bins, ~20 MB per pair)  
**Output:** same DataFrame with stationary trade features added per pair

Trade bins are joined to the depth spine using `merge_asof` with a 20-second tolerance. Derived features are computed on the full concatenated series so rolling windows span day boundaries correctly.

### Derived trade features

Raw trade aggregates (`trade_count`, `trade_volume`, `trade_notional`, `buy_volume`, `sell_volume`) are used only as intermediates and are dropped after derivation.

| Column | Formula | What it captures |
|---|---|---|
| `{P}_vwap_return` | `pct_change(trade_notional / trade_volume)` | Tick-to-tick % VWAP change — stationary price series |
| `{P}_vwap_volatility` | `rolling(120, min_periods=1).std(vwap_return)` | Rolling price instability; window = 120 snapshots (~60 min). **Label threshold only — excluded from model X.** |
| `{P}_buy_ratio` | `buy_volume / trade_volume` | Aggressive buy pressure ∈ (0, 1) |
| `{P}_trade_flow_imbalance` | `(buy_volume − sell_volume) / trade_volume` | Net directional pressure ∈ (−1, 1) |
| `{P}_trade_intensity_z` | rolling z-score of `log1p(trade_count)`, window 120 | Short-term activity surge vs recent norm |
| `{P}_trade_intensity_regime_z` | rolling z-score of `log1p(trade_count)`, window 480 | Activity vs longer-term regime |
| `{P}_trade_notional_z` | rolling z-score of `log1p(trade_notional)`, window 120 | Value-weighted activity surge |

> VWAP itself is an intermediate only — computed as `trade_notional / trade_volume`, used to derive `vwap_return`, then discarded.

---

## Step 5 — Time features (`add_time_features`)

**Input:** merged DataFrame  
**Output:** same DataFrame + 4 global columns

Flash crashes have pronounced time-of-day and day-of-week patterns — Asian-session thin liquidity windows, US and EU open volatility spikes. Sine/cosine encoding preserves the cyclic topology so 23:59 is adjacent to 00:00.

| Column | Formula | Cycle |
|---|---|---|
| `hour_sin` | `sin(2π × fractional_hour / 24)` | 24-hour |
| `hour_cos` | `cos(2π × fractional_hour / 24)` | 24-hour |
| `dow_sin` | `sin(2π × day_of_week / 7)` | 7-day (Mon = 0) |
| `dow_cos` | `cos(2π × day_of_week / 7)` | 7-day |

---

## Step 6 — Label generation (`step_label`)

**Input:** in-memory DataFrame from `run_pipeline` (or `all_pairs_with_trades.csv` if `--skip-extract`)  
**Output:** `bookDepth_data/all_pairs_labeled.csv`

The flash crash label is a **forward-looking binary indicator** — it tells us at time `t` whether a crash *will* happen in the next 10 minutes. Computed per pair; only `BTCUSDT_flash_crash_label` is used as the training target by default (`LABEL_PAIR` in `config.py`).

### Building blocks

**① VWAP return** (computed in Step 4)

```
vwap[t]        = trade_notional[t] / trade_volume[t]
vwap_return[t] = (vwap[t] − vwap[t−1]) / vwap[t−1]        ← pct_change(), stationary
```

**② VWAP volatility** (computed in Step 4)

```
vwap_volatility[t] = rolling_std(vwap_return, window=120, min_periods=1)[t]
```

Dynamic threshold denominator — adapts to the current regime. A 3-sigma drop in a calm market is far more significant than in a volatile one.

**③ Forward cumulative return**

```
fwd_cum_return[t] = Σ vwap_return[t+k]   for k = 1 … 20
                  = vwap_return.rolling(20).sum().shift(−20)
```

**④ Crash threshold**

```
threshold[t] = −CRASH_SIGMA × vwap_volatility[t] × sqrt(CRASH_HORIZON)
             = −2.0 × vwap_volatility[t] × sqrt(20)
```

The `sqrt(CRASH_HORIZON)` factor is essential. `fwd_cum_return` is a sum of `CRASH_HORIZON` iid returns, so its standard deviation is `vwap_volatility × sqrt(CRASH_HORIZON)` — not `vwap_volatility` alone. Without this scaling the threshold sits at `-CRASH_SIGMA / sqrt(20) ≈ -0.67` standard deviations of the forward distribution, labelling ~25% of rows as crashes regardless of `CRASH_SIGMA`.

### Label rule

```
flash_crash_label[t] = 1  if  fwd_cum_return[t] < −CRASH_SIGMA × vwap_volatility[t] × sqrt(CRASH_HORIZON)
                        0  otherwise
```

A crash is flagged at `t` if the **cumulative VWAP return over the next 10 minutes is a `CRASH_SIGMA`-sigma event on the 10-minute return timescale**.

### Constants (from `config.py`)

| Constant | Value | Meaning |
|---|---|---|
| `CRASH_HORIZON` | `20` | Snapshots in the forward window (20 × 30 s = 10 min) |
| `CRASH_SIGMA` | `2.0` | Sigma threshold on the 10-minute return distribution |
| `LABEL_PAIR` | `BTCUSDT` | Pair whose label is used as the training target |

**Tuning `CRASH_SIGMA`:**

| CRASH_SIGMA | Expected crash rate | Training events (6 months) |
|---|---|---|
| 2.0 | ~2.3% | ~10,000 |
| 2.5 | ~0.6% | ~3,000 |
| 3.0 | ~0.13% | ~650 — too few for deep learning |

### Tail trimming

The last `CRASH_HORIZON = 20` rows are dropped. Their forward window extends beyond available data so their labels would be NaN — keeping them would introduce incomplete samples.

### Expected class distribution

Flash crashes are rare. Expect a crash rate of **0.5–2%** depending on the period and market conditions, giving a class ratio of roughly 1:50 to 1:200. Class imbalance is handled by the model — see the *Into the model* section.

---

## Final dataset — `all_pairs_labeled.csv`

One row per 30-second snapshot. Only stationary features appear — all raw per-level and absolute aggregate columns are dropped during Steps 2 and 4.

```
timestamp

# Per pair (repeated for each pair in DEFAULT_PAIRS)
{PAIR}_depth_imbalance_ratio          ∈ (−1, 1)  normalised quantity skew
{PAIR}_notional_imbalance_ratio       ∈ (−1, 1)  normalised value skew
{PAIR}_depth_imbalance_ratio_delta    T vs T-1 change in depth skew
{PAIR}_notional_imbalance_ratio_delta T vs T-1 change in notional skew
{PAIR}_total_notional_delta           T vs T-1 change in total book value
{PAIR}_notional_z                     rolling z-score log(notional), window 120
{PAIR}_notional_regime_z              rolling z-score log(notional), window 480
{PAIR}_vwap_return                    tick-to-tick % VWAP change
{PAIR}_buy_ratio                      aggressive buy fraction ∈ (0, 1)
{PAIR}_trade_flow_imbalance           (buy − sell) / total ∈ (−1, 1)
{PAIR}_trade_intensity_z              rolling z-score log(trade_count), window 120
{PAIR}_trade_intensity_regime_z       rolling z-score log(trade_count), window 480
{PAIR}_trade_notional_z               rolling z-score log(trade_notional), window 120
# {PAIR}_vwap_volatility              present in CSV, excluded from model X (label threshold only)

# Time context, cyclically encoded (global — not per-pair)
hour_sin                              sin(2π × hour / 24)
hour_cos                              cos(2π × hour / 24)
dow_sin                               sin(2π × day_of_week / 7)
dow_cos                               cos(2π × day_of_week / 7)

# Global on-chain (shared, not per-pair) — present only when --no-onchain is not passed
base_fee_gwei_mean
base_fee_gwei_max
gas_used_ratio_mean
large_transfer_count
large_transfer_eth
exchange_inflow_count
exchange_inflow_eth

# Labels (one per pair, only BTCUSDT used as training target)
{PAIR}_flash_crash_label
```

**Feature count:**

| Configuration | Features |
|---|---|
| 2 pairs, no on-chain | 30 (13 per pair + 4 time) |
| 2 pairs, with on-chain | 37 (13 per pair + 4 time + 7 global) |

---

## Into the model — `SequenceDataset`

`models/data_adapter.py` loads `all_pairs_labeled.csv` and produces sliding window tensors.

### Feature matrix vs target vector

- **X** — every numeric column except `timestamp`, all `_flash_crash_label` columns, and `_vwap_volatility` (used only to compute the label, not a predictive feature)
- **y** — the label column for `LABEL_PAIR` extracted separately

The labels are not deleted — they live in `y` and are used throughout: `y_tr` drives the loss, `y_cal` drives HopCPT calibration, and `y_te` is ground truth at evaluation. The model never sees `y` as an input.

### Sliding window construction

Each sample is a contiguous block of `SEQ_LEN = 120` consecutive rows (60 minutes). The label for the window is taken from the **last row** of that window.

```
X shape: (N, 120, n_features)   dtype: float32
y shape: (N,)                   dtype: int8     values: {0, 1}
```

NaNs remaining after ffill/bfill (at the very start of the series) are zero-filled.

### Class imbalance

Flash crashes occur 0.5–2% of the time. Without correction, a model predicting "no crash" every time achieves ~99% accuracy while being completely useless.

All models derive class weights from the training split only (never the full dataset, to avoid data leakage):

- **DL models** (MHN, STanHop, LSTM, Transformer) — `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`
- **XGBoost** — `scale_pos_weight = n_neg / n_pos`
- **Random Forest, Logistic Regression** — `class_weight='balanced'`

STanHop's sparse top-k attention provides an architectural advantage on top of this: it can selectively retrieve rare historical crash precursor patterns from its associative memory rather than averaging them out with the dominant normal-market signal. This is retrieval-level sparsity, complementary to loss-level class weighting.

### Evaluation metrics

Accuracy is not reported — it is uninformative on imbalanced data. All models are evaluated with:

| Metric | What it measures |
|---|---|
| **PR-AUC** (`avg_prec`) | Area under the Precision-Recall curve — primary metric; directly reflects performance on the rare positive class |
| **ROC-AUC** (`roc_auc`) | Threshold-independent discrimination; useful for architecture comparison |
| **F1** | Harmonic mean of precision and recall at threshold 0.5 |
| **Precision** | Of all crash alerts raised, what fraction were real |
| **Recall** | Of all actual crashes, what fraction were caught |

HopCPT adds three additional metrics after calibration:

| Metric | What it measures |
|---|---|
| `conformal_coverage` | Fraction of test samples where the true label fell inside the prediction set — should be ≥ 1 − α |
| `uncertain_rate` | Fraction of samples where the model abstained (both classes in the set) |
| `empty_set_rate` | Fraction of samples where neither class was included |

### Chronological split

No shuffling. Windows are assigned in time order:

| Split | Fraction | Use |
|---|---|---|
| Train | 70% | Model parameter optimisation; class weights computed here |
| Calibration | 15% | HopCPT nonconformity score calibration; never seen during training |
| Test | 15% | Final held-out evaluation only |

For sklearn-style models (XGBoost, etc.) `get_flat_splits()` returns `X` reshaped to `(N, 120 × n_features)`.
