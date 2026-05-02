# Data Collection

All raw data is fetched before the feature engineering pipeline runs. Three sources feed the system: Binance book depth, Binance trades, and Alchemy on-chain data. No API key is needed for Binance. Alchemy requires `ALCHEMY_API_KEY` in `.env`.

---

## 1. Binance Book Depth

**Source:** `https://data.binance.vision/data/futures/um/daily/bookDepth`  
**Module:** `data_collection/book_depth_utils.py`  
**Final output:** `bookDepth_data/{SYMBOL}/{SYMBOL}_merged.csv`

### Download behaviour

Each day's data is a ZIP file containing a single long-format CSV (~57,600 rows). The downloader processes each day in sequence:

1. Download ZIP (~5 MB)
2. Extract → raw CSV (`{SYMBOL}-bookDepth-{date}.csv`, ~10 MB)
3. Pivot long → wide: one row per snapshot, all price levels as columns
4. Save pivoted CSV (`{SYMBOL}-bookDepth-pivoted-{date}.csv`, ~200 KB)
5. Delete ZIP + raw CSV immediately

Peak disk use at any moment: ~15 MB (one day's ZIP + raw CSV). After all days are downloaded, daily pivoted files are concatenated into a single `{SYMBOL}_merged.csv` (~50 MB for 6 months) and the daily files deleted.

### Raw format (long, one row per price level)

| Field | Type | Description |
|---|---|---|
| `timestamp` | datetime | Snapshot timestamp (UTC) |
| `percentage` | float | Distance from mid price as a percentage; negative = bid side, positive = ask side |
| `depth` | float | Quantity at this price level (base asset units) |
| `notional` | float | Value at this level (quote asset = depth × price) |

**Snapshot cadence:** ~30 seconds  
**Percentage levels:** typically `±0.1`, `±0.2`, … up to `±5.0` from mid. Percentage values are normalised to one decimal place during pivoting to ensure consistent column names across days (Binance occasionally formats `-5` as an integer instead of `-5.0`).

### Pivoted format (wide, one row per snapshot)

```
timestamp
{SYMBOL}_depth_{pct}       quantity at percentage level pct
{SYMBOL}_notional_{pct}    notional value at percentage level pct
…                          repeated for each level on both sides
```

---

## 2. Binance Trades

**Source:** `https://data.binance.vision/data/futures/um/daily/trades`  
**Module:** `data_collection/trades_utils.py`  
**Final output:** `trades_data/{SYMBOL}/{SYMBOL}_trades_agg.csv`

### Download behaviour

Each day's raw trades ZIP is large (~80 MB compressed, ~500 MB uncompressed). The downloader processes each day in sequence:

1. Download ZIP (~80 MB)
2. Extract → raw CSV (`{SYMBOL}-trades-{date}.csv`, ~500 MB)
3. Aggregate to 30-second bins (~2,880 rows)
4. Save daily aggregated CSV (`{SYMBOL}-trades-agg-{date}.csv`, ~100 KB)
5. Delete ZIP + raw CSV immediately

Peak disk use at any moment: ~580 MB (one day's ZIP + raw CSV). After all days are processed, daily aggregated files are concatenated into a single `{SYMBOL}_trades_agg.csv` (~20 MB for 6 months) and the daily files deleted.

### Raw format (tick-by-tick)

| Field | Type | Description |
|---|---|---|
| `id` | int64 | Trade ID |
| `price` | float64 | Execution price |
| `qty` | float64 | Quantity traded (base asset) |
| `quote_qty` | float64 | Notional value (quote asset = price × qty) |
| `time` | int64 | Trade timestamp (Unix milliseconds) |
| `is_buyer_maker` | bool | `True` = seller is aggressor (taker sell); `False` = buyer is aggressor (taker buy) |

### Aggregated format (30-second bins)

Trades are floored to 30-second boundaries (`time // 30_000 × 30_000`) and grouped. The resulting file has ~2,880 rows per day.

| Column | Formula | Description |
|---|---|---|
| `timestamp` | `ts_30s` converted to datetime | Bin start time (UTC) |
| `trade_count` | `count(id)` | Number of individual trades in the bin |
| `trade_volume` | `Σqty` | Total quantity traded |
| `trade_notional` | `Σquote_qty` | Total value traded |
| `buy_volume` | `Σqty` where `is_buyer_maker=False` | Aggressive buy quantity |
| `sell_volume` | `Σqty` where `is_buyer_maker=True` | Aggressive sell quantity |

These raw aggregates are used only as inputs to `_derive_trade_features` during feature engineering — they do not appear in the final model input.

---

## 3. Alchemy On-Chain (ETH Mainnet)

**Source:** Alchemy JSON-RPC — `https://eth-mainnet.g.alchemy.com/v2/{key}`  
**Module:** `data_collection/onchain_utils.py`  
**Output:** `onchain_data/ETHUSDT/ETHUSDT-onchain-{YYYY-MM-DD}.csv`

> Alchemy has no official Python SDK. `onchain_utils.py` is our own thin JSON-RPC client.

One file per day. Each row is a 30-second window aligned to the book depth snapshot cadence.

| Field | Type | Source method | Description |
|---|---|---|---|
| `timestamp_ms` | int64 | derived | Window start, Unix milliseconds |
| `block_number_start` | int64 | `eth_feeHistory` | First ETH block in this 30 s window |
| `block_number_end` | int64 | `eth_feeHistory` | Last ETH block in this 30 s window |
| `base_fee_gwei_mean` | float | `eth_feeHistory` | Mean EIP-1559 base fee across blocks in window (gwei) |
| `base_fee_gwei_max` | float | `eth_feeHistory` | Peak base fee in window — spike detector |
| `gas_used_ratio_mean` | float | `eth_feeHistory` | Mean gas used / gas limit — block fullness / congestion |
| `large_transfer_count` | int | `alchemy_getAssetTransfers` | Native ETH transfers > 50 ETH in window |
| `large_transfer_eth` | float | `alchemy_getAssetTransfers` | Total ETH moved by large transfers |
| `exchange_inflow_count` | int | `alchemy_getAssetTransfers` | Large transfers **to** known CEX wallets (Binance, Coinbase, Kraken, OKX) |
| `exchange_inflow_eth` | float | `alchemy_getAssetTransfers` | Total ETH flowing into exchanges |

**On-chain features are global** — they are not per-pair. Every row in the merged dataset carries the same ETH chain values for that timestamp. Bitcoin has no EVM chain so these are treated as shared market-state signals.

**Block timing:** ETH post-Merge produces one block every ~12 seconds (~2–3 blocks per 30 s window). Block numbers are estimated from the post-Merge reference point and fine-tuned with a small number of RPC calls. Fee history is fetched in batches of 1024 blocks.

**Exchange wallet addresses** are defined in `NETWORK_CONFIG` inside `onchain_utils.py`.

On-chain download can be skipped entirely with `--no-onchain` if no Alchemy API key is available.

---

## Period format

All three downloaders accept the same period string format:

| Format | Example | Meaning |
|---|---|---|
| `Nd` | `7d` | Last N days |
| `Nw` | `2w` | Last N weeks |
| `Nm` | `6m` | Last N months |
| `Ny` | `1y` | Last N years |
| `YYYY-MM-DD` | `2026-03-17` | Single day |
| `YYYY-MM-DD/YYYY-MM-DD` | `2026-03-01/2026-03-20` | Explicit date range |

All periods are clamped to yesterday — today's data is not yet available on Binance's archive.

---

## Disk usage summary

| Phase | Peak | After completion |
|---|---|---|
| Book depth download (per day) | ~15 MB (ZIP + raw CSV) | ~200 KB (pivoted CSV) |
| Book depth total (6 months, 2 pairs) | ~15 MB | ~100 MB (2 × merged CSV) |
| Trades download (per day) | ~580 MB (ZIP + raw CSV) | ~100 KB (agg CSV) |
| Trades total (6 months, 2 pairs) | ~580 MB | ~40 MB (2 × agg CSV) |
| On-chain (6 months) | negligible | ~5 MB |
| **Total after full run** | **~580 MB peak** | **~155 MB** |

## Output summary

| Directory | Final file | Size (6 months) | Committed |
|---|---|---|---|
| `bookDepth_data/{PAIR}/` | `{PAIR}_merged.csv` | ~50 MB/pair | No |
| `trades_data/{PAIR}/` | `{PAIR}_trades_agg.csv` | ~20 MB/pair | No |
| `onchain_data/ETHUSDT/` | `ETHUSDT-onchain-*.csv` (daily) | ~5 MB total | No |
| `bookDepth_data/` | `all_pairs_labeled.csv` | ~15 MB | No |
