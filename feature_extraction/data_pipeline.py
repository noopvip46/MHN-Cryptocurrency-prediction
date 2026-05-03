import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from the project root (data_collection, config)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_collection.onchain_utils import ONCHAIN_FEATURE_COLUMNS    # noqa: E402


# ── Step 1: Pivot ──────────────────────────────────────────────────────────────

def pivot_merged(out_base: str = "bookDepth_data", save: bool = False) -> pd.DataFrame:
    """Load per-pair {PAIR}_merged.csv files (already wide) and align on timestamp.

    {PAIR}_merged.csv is produced by book_depth_utils.download_book_depth_range —
    one row per snapshot, columns {PAIR}_depth_{pct} / {PAIR}_notional_{pct}.

    Pairs are aligned with merge_asof (tolerance 30 s) using the first pair's
    timestamps as the spine.  This avoids the 2× row explosion that outer-merge
    produces when BTCUSDT and ETHUSDT snapshots land at slightly different times.

    save=True  : writes all_pairs_cleaned.csv (checkpoint for debugging).
    save=False : returns the merged DataFrame without touching disk.
    """
    merged_files = sorted(glob.glob(f"{out_base}/*/*_merged.csv"))

    if not merged_files:
        raise FileNotFoundError(f"No *_merged.csv files found under {out_base}/")

    pair_dfs = []
    for merged_file in merged_files:
        pair = Path(merged_file).parts[-2]
        df   = pd.read_csv(merged_file, parse_dates=["timestamp"])
        if df.empty:
            print(f"  [{pair}] empty merged file, skipping")
            continue

        # ── Coalesce integer/float duplicate columns ──────────────────────
        # Some daily CSVs name levels as integers (e.g. _depth_-1), others
        # as floats (_depth_-1.0).  When concatenated, each variant covers
        # ~40–60% of rows — neither alone reaches full coverage.  Here we
        # merge them: fill the integer column from the float column, then
        # drop the float variant.  After this, every level has 100% non-null.
        float_cols = [c for c in df.columns
                      if re.match(r'.+_(depth|notional)_-?\d+\.0$', c)]
        n_coalesced = 0
        for fcol in float_cols:
            int_col = re.sub(r'\.0$', '', fcol)
            if int_col in df.columns:
                df[int_col] = df[int_col].fillna(df[fcol])
                df = df.drop(columns=[fcol])
                n_coalesced += 1
            else:
                # Float-only column — rename to integer convention
                df = df.rename(columns={fcol: int_col})
                n_coalesced += 1
        if n_coalesced:
            print(f"  [{pair}] coalesced {n_coalesced} duplicate int/float columns")

        pair_dfs.append(df.sort_values("timestamp").reset_index(drop=True))
        print(f"  [{pair}] loaded: rows={len(df):,}  cols={len(df.columns)}")

    if not pair_dfs:
        raise ValueError("All merged files were empty.")

    # Use the first pair as the timestamp spine; join all others via merge_asof.
    # This keeps exactly one row per spine snapshot and never inflates row count.
    combined = pair_dfs[0]
    for other in pair_dfs[1:]:
        combined = pd.merge_asof(
            combined,
            other,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("30s"),
        )
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    if save:
        out_file = Path(out_base) / "all_pairs_cleaned.csv"
        combined.to_csv(out_file, index=False)
        print(f"  saved → {out_file}  rows={len(combined):,}  cols={len(combined.columns)}")

    return combined


# ── Step 2: Feature engineering ───────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    trading_pairs: list,
    roll_window: int = 120,
    regime_window: int = 480,
) -> pd.DataFrame:
    """Compute stationary order-book features; drop all raw non-stationary columns.

    Per pair computes:
      Core book features:
        - depth_imbalance_ratio       ∈ (−1, 1)  normalised quantity skew
        - inner_book_ratio_z          top-of-book liquidity thinness (z-scored)
        - book_slope_z                shape of the book: near vs far liquidity

      Spike-correction features:
        - depth_change_z              rate of change in total depth (book recovery signal)
        - imbalance_momentum          change in depth imbalance (shifting pressure)

    Raw per-level columns ({PAIR}_depth_{pct}, {PAIR}_notional_{pct}) are dropped
    after the ratios are computed — they are non-stationary.

    The input DataFrame is not modified in place.
    """
    df = df.copy()

    for pair in trading_pairs:
        pair = pair.strip().upper()

        # ── Depth ──────────────────────────────────────────────────────────────
        depth_neg = [c for c in df.columns if c.startswith(f"{pair}_depth_-")]
        depth_pos = [c for c in df.columns if c.startswith(f"{pair}_depth_")
                     and not c.startswith(f"{pair}_depth_-")]

        total_depth_neg = df[depth_neg].sum(axis=1)
        total_depth_pos = df[depth_pos].sum(axis=1)
        total_depth     = total_depth_neg + total_depth_pos

        depth_imbalance = total_depth_pos - total_depth_neg
        df[f"{pair}_depth_imbalance_ratio"] = (
            depth_imbalance / total_depth.replace(0, np.nan)
        )

        # ── Spike-correction: depth change rate ──────────────────────────────
        # Z-scored rate of change in total depth. After a spike, rapid depth
        # recovery (positive change) signals limit orders flooding back in →
        # mean-reversion. Continued depth drain (negative) → continuation.
        depth_pct_change = total_depth.pct_change().fillna(0)
        dc_mean = depth_pct_change.rolling(roll_window, min_periods=1).mean()
        dc_std  = depth_pct_change.rolling(roll_window, min_periods=1).std().replace(0, np.nan)
        df[f"{pair}_depth_change_z"] = ((depth_pct_change - dc_mean) / dc_std).clip(-5, 5)

        # ── Spike-correction: imbalance momentum ─────────────────────────────
        # Change in depth imbalance ratio over last 5 snapshots (~2.5 min).
        # Shifting imbalance = directional pressure building or unwinding.
        imb = df[f"{pair}_depth_imbalance_ratio"]
        df[f"{pair}_imbalance_momentum"] = imb.diff(5).fillna(0)

        # ── Notional (aggregates only — used for book slope, not kept as features)
        notional_neg = [c for c in df.columns if c.startswith(f"{pair}_notional_-")]
        notional_pos = [c for c in df.columns if c.startswith(f"{pair}_notional_")
                        and not c.startswith(f"{pair}_notional_-")]

        total_notional = df[notional_neg].sum(axis=1) + df[notional_pos].sum(axis=1)

        # ── Top-of-book thinness (book slope) ─────────────────────────────────
        inner_bid_col = f"{pair}_notional_-1"
        inner_ask_col = f"{pair}_notional_1"

        if inner_bid_col in df.columns and inner_ask_col in df.columns:
            close_notional = df[inner_bid_col] + df[inner_ask_col]
            print(f"  [{pair}] book slope using ±1%  ({inner_bid_col} / {inner_ask_col})")

            inner_frac = close_notional / total_notional.replace(0, np.nan)
            inner_mean = inner_frac.rolling(roll_window, min_periods=1).mean()
            inner_std  = inner_frac.rolling(roll_window, min_periods=1).std().replace(0, np.nan)
            df[f"{pair}_inner_book_ratio_z"] = (inner_frac - inner_mean) / inner_std

            outer_cols = [c for c in notional_neg + notional_pos
                          if re.search(r"_notional_-?[45]$", c)]
            if outer_cols:
                outer_notional = df[outer_cols].sum(axis=1)
                book_slope     = close_notional / outer_notional.replace(0, np.nan)
                slope_mean     = book_slope.rolling(roll_window, min_periods=1).mean()
                slope_std      = book_slope.rolling(roll_window, min_periods=1).std().replace(0, np.nan)
                df[f"{pair}_book_slope_z"] = (book_slope - slope_mean) / slope_std
        else:
            print(f"  [{pair}] book slope skipped — ±1% notional columns not found")

        # ── Drop raw non-stationary columns ────────────────────────────────────
        raw_cols = depth_neg + depth_pos + notional_neg + notional_pos
        df = df.drop(columns=[c for c in raw_cols if c in df.columns])

    return df


# ── Step 3: On-chain merge ────────────────────────────────────────────────────

def merge_onchain(
    df: pd.DataFrame,
    onchain_base: str = "onchain_data",
    symbol: str = "ETHUSDT",
) -> pd.DataFrame:
    """Join on-chain features into the feature DataFrame on timestamp.

    On-chain features are global market-state columns shared across all pairs.
    Gaps between 30 s snapshots and ~12 s block times are closed with
    forward-fill then back-fill. Returns df unchanged if no on-chain data found.
    """
    onchain_dir  = Path(onchain_base) / symbol
    daily_files  = sorted(glob.glob(str(onchain_dir / f"{symbol}-onchain-*.csv")))

    if not daily_files:
        print(f"  [onchain] no data found under {onchain_dir} — skipping merge")
        return df

    onchain = pd.concat([pd.read_csv(f) for f in daily_files], ignore_index=True)
    onchain["timestamp"] = pd.to_datetime(onchain["timestamp_ms"], unit="ms")
    onchain = (
        onchain[["timestamp"] + ONCHAIN_FEATURE_COLUMNS]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    df = df.copy().merge(onchain, on="timestamp", how="left")
    df[ONCHAIN_FEATURE_COLUMNS] = df[ONCHAIN_FEATURE_COLUMNS].ffill().bfill()

    print(
        f"  [onchain] merged {len(daily_files)} daily files — "
        f"{len(ONCHAIN_FEATURE_COLUMNS)} global features added"
    )
    return df


# ── Step 4: Trade feature merge ───────────────────────────────────────────────

def _derive_trade_features(agg: pd.DataFrame, roll_window: int, regime_window: int) -> pd.DataFrame:
    """Compute stationary derived features from pre-aggregated trade data; drop raw columns.

    Input columns : timestamp, trade_count, trade_volume, trade_notional,
                    buy_volume, sell_volume

    Output keeps stationary derived columns relevant for spike-correction prediction:

    Core price features:
      vwap_return            — single-step return
      vwap_volatility        — rolling vol (used for barrier sizing, excluded from model X)
      vwap_return_5/10/20    — rolling cumulative returns at 2.5/5/10-min horizons

    Spike-correction features:
      return_zscore           — how abnormal is the current return (mean-reversion signal)
      price_acceleration      — 2nd derivative of price: decelerating = reversal signal
      volume_return_corr      — rolling correlation between volume and returns
                                (high volume confirming move → continuation)
      volume_surge            — current volume vs rolling average (spike on thin volume → revert)
      return_skew             — rolling skewness of returns (asymmetry → directional pressure)

    Volume & flow features:
      vwap_vol_ratio          — short/long-term vol ratio (volatility acceleration)
      buy_ratio               — fraction of volume that is taker-buy
      trade_intensity_z       — z-scored trade count
      trade_intensity_regime_z — long-window variant

    Raw aggregate columns are dropped — they are non-stationary level quantities.
    """
    agg = agg.copy().sort_values("timestamp").reset_index(drop=True)

    vwap = agg["trade_notional"] / agg["trade_volume"].replace(0, np.nan)
    agg["vwap_return"]          = vwap.pct_change()
    agg["vwap_volatility"]      = agg["vwap_return"].rolling(roll_window, min_periods=1).std()

    # Multi-horizon cumulative returns — momentum at multiple scales.
    for h in [5, 10, 20]:
        agg[f"vwap_return_{h}"] = agg["vwap_return"].rolling(h, min_periods=1).sum()

    # ── Spike-correction features ─────────────────────────────────────────

    # Return z-score: how many σ is the current return from its rolling mean?
    # Extreme values (|z| > 2-3) indicate abnormal moves likely to revert.
    ret_mean = agg["vwap_return"].rolling(roll_window, min_periods=1).mean()
    ret_std  = agg["vwap_return"].rolling(roll_window, min_periods=1).std().replace(0, np.nan)
    agg["return_zscore"] = ((agg["vwap_return"] - ret_mean) / ret_std).clip(-5, 5)

    # Price acceleration: change in return (2nd derivative of log-price).
    # Positive when price is accelerating in current direction (continuation),
    # negative when decelerating (reversal signal).
    agg["price_acceleration"] = agg["vwap_return"].diff()

    # Volume-price confirmation: rolling correlation between volume and abs returns.
    # High correlation = volume confirms the move (continuation).
    # Low/negative = divergence (potential reversal).
    log_vol = np.log1p(agg["trade_volume"])
    abs_ret = agg["vwap_return"].abs()
    agg["volume_return_corr"] = log_vol.rolling(20, min_periods=5).corr(abs_ret).fillna(0)

    # Volume surge: ratio of current trade volume to its rolling mean.
    # Spikes on thin volume are more likely to revert than those on heavy volume.
    vol_mean = log_vol.rolling(roll_window, min_periods=1).mean()
    vol_std  = log_vol.rolling(roll_window, min_periods=1).std().replace(0, np.nan)
    agg["volume_surge"] = ((log_vol - vol_mean) / vol_std).clip(-5, 5)

    # Return skewness: rolling 3rd moment of returns.
    # Positive skew = right tail (upward pressure), negative = downward pressure.
    # Persistent skew in one direction predicts continuation; sudden reversal of
    # skew predicts mean-reversion.
    agg["return_skew"] = agg["vwap_return"].rolling(30, min_periods=10).skew().fillna(0).clip(-3, 3)

    # ── Volatility & flow features ────────────────────────────────────────

    vol_short = agg["vwap_return"].rolling(10,  min_periods=1).std()
    vol_long  = agg["vwap_return"].rolling(60,  min_periods=1).std().replace(0, np.nan)
    agg["vwap_vol_ratio"] = (vol_short / vol_long).clip(0, 10)

    agg["buy_ratio"] = agg["buy_volume"] / agg["trade_volume"].replace(0, np.nan)

    log_count = np.log1p(agg["trade_count"])

    agg["trade_intensity_z"] = (
        (log_count - log_count.rolling(roll_window,   min_periods=1).mean()) /
        log_count.rolling(roll_window,   min_periods=1).std().replace(0, np.nan)
    )
    agg["trade_intensity_regime_z"] = (
        (log_count - log_count.rolling(regime_window, min_periods=1).mean()) /
        log_count.rolling(regime_window, min_periods=1).std().replace(0, np.nan)
    )

    # Drop raw non-stationary columns — only derived stationary features survive
    raw_trade_cols = ["trade_count", "trade_volume", "trade_notional", "buy_volume", "sell_volume"]
    agg = agg.drop(columns=[c for c in raw_trade_cols if c in agg.columns])

    return agg


def merge_trades(
    df: pd.DataFrame,
    trading_pairs: list,
    trades_base: str = "trades_data",
    roll_window: int = 120,
    regime_window: int = 480,
    out_path: str = None,
) -> pd.DataFrame:
    """Join pre-aggregated trade features onto the book-depth feature DataFrame.

    Reads {PAIR}_trades_agg.csv produced by trades_utils.download_trades_range —
    already aggregated to 30-second bins (~2 880 rows/day, not millions).
    Joined to depth timestamps via merge_asof with a 20-second tolerance.
    Derived features (vwap_return, vwap_volatility, etc.) are computed on the
    full concatenated series so rolling windows span day boundaries correctly.

    vwap_volatility is kept in the output CSV for the label step but is
    excluded from model input features.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    for pair in trading_pairs:
        pair     = pair.strip().upper()
        agg_path = Path(trades_base) / pair / f"{pair}_trades_agg.csv"

        if not agg_path.exists():
            print(f"  [{pair}] {agg_path.name} not found — skipping trade features")
            continue

        agg = pd.read_csv(agg_path, parse_dates=["timestamp"])
        agg = _derive_trade_features(agg, roll_window, regime_window)
        agg = agg.rename(columns={c: f"{pair}_{c}" for c in agg.columns if c != "timestamp"})

        df = pd.merge_asof(
            df.sort_values("timestamp"),
            agg.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("20s"),
        )
        print(f"  [{pair}] trade features joined  bins={len(agg)}")

    if out_path:
        df.to_csv(out_path, index=False)
        print(f"  saved → {out_path}  rows={len(df)}, cols={len(df.columns)}")

    return df


# ── Step 5: Time features ─────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclically-encoded time context columns from the timestamp.

    Flash crashes have pronounced time-of-day and day-of-week patterns
    (thin Asian-session liquidity, US/EU open volatility spikes).
    Sine/cosine encoding preserves the cyclic topology — 23:59 is adjacent
    to 00:00, and the model never sees an artificial discontinuity.

    Adds four columns (no pair prefix — they are global):
      hour_sin, hour_cos   — 24-hour cycle
      dow_sin,  dow_cos    — 7-day cycle (Monday=0)
    """
    ts = pd.to_datetime(df["timestamp"])
    df = df.copy()

    hour = ts.dt.hour + ts.dt.minute / 60.0         # fractional hour ∈ [0, 24)
    dow  = ts.dt.dayofweek.astype(float)             # ∈ [0, 7)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"]  = np.sin(2 * np.pi * dow  /  7.0)
    df["dow_cos"]  = np.cos(2 * np.pi * dow  /  7.0)

    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def infer_pairs(df: pd.DataFrame) -> list:
    """Infer trading pair names from DataFrame column names."""
    seen, pairs = set(), []
    for col in df.columns:
        if col == "timestamp":
            continue
        pair = col.split("_")[0]
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


# ── Top-level pipeline ─────────────────────────────────────────────────────────

def run_pipeline(
    trading_pairs: list,
    out_base: str        = "bookDepth_data",
    trades_base: str     = "trades_data",
    roll_window: int     = 120,
    regime_window: int   = 480,
    onchain_base: str    = "onchain_data",
    onchain_symbol: str  = "ETHUSDT",
    skip_onchain: bool   = False,
    save_intermediates: bool = False,
) -> pd.DataFrame:
    """Pivot → engineer → (on-chain merge) → trade merge → return DataFrame.

    Download is handled by step_download() in run.py — not here.

    save_intermediates=False (default): nothing written to disk except the
      final all_pairs_labeled.csv produced by step_label().  Keeps the entire
      pipeline in memory — no large intermediate files.

    save_intermediates=True: saves all_pairs_cleaned.csv and
      all_pairs_with_trades.csv as checkpoints (useful for debugging or
      resuming with --skip-extract / --skip-label).

    skip_onchain=True: skip the on-chain merge step entirely.
    """
    from config import ALL_PAIRS_TRADES

    trading_pairs = [p.strip().upper() for p in trading_pairs if p.strip()]

    print("[pipeline] loading book depth...")
    cleaned = pivot_merged(out_base=out_base, save=save_intermediates)

    print("[pipeline] engineering order-book features...")
    features = engineer_features(cleaned, trading_pairs, roll_window, regime_window)

    if not skip_onchain:
        print("[pipeline] merging on-chain features...")
        features = merge_onchain(features, onchain_base=onchain_base, symbol=onchain_symbol)
    else:
        print("[pipeline] on-chain merge skipped (--no-onchain)")

    print("[pipeline] merging trade features...")
    features = merge_trades(
        features,
        trading_pairs,
        trades_base   = trades_base,
        roll_window   = roll_window,
        regime_window = regime_window,
        out_path      = str(ALL_PAIRS_TRADES) if save_intermediates else None,
    )

    print("[pipeline] adding time features...")
    features = add_time_features(features)

    # Cross-pair divergence features — only meaningful when both BTC and ETH are present.
    # BTC and ETH are highly correlated; divergence from that correlation is informative:
    #   return spread:    BTC selling off while ETH holds → pair-specific pressure
    #   imbalance spread: one book being drained while the other stays balanced
    btc_ret = "BTCUSDT_vwap_return"
    eth_ret = "ETHUSDT_vwap_return"
    btc_imb = "BTCUSDT_depth_imbalance_ratio"
    eth_imb = "ETHUSDT_depth_imbalance_ratio"
    if btc_ret in features.columns and eth_ret in features.columns:
        features["btceth_return_spread"] = features[btc_ret] - features[eth_ret]
        print("[pipeline] added btceth_return_spread")
    if btc_imb in features.columns and eth_imb in features.columns:
        features["btceth_imbalance_spread"] = features[btc_imb] - features[eth_imb]
        print("[pipeline] added btceth_imbalance_spread")

    # Drop cold-start rows where rolling windows are under-filled.
    # The longest window is regime_window (480 rows = ~4 hours at 30s cadence).
    # Before that point notional_z == notional_regime_z, multi-horizon returns
    # are all identical, etc. — pure noise that would corrupt training.
    features = features.iloc[regime_window:].reset_index(drop=True)
    print(f"[pipeline] dropped first {regime_window} cold-start rows")

    # Defragment: columns were added one-by-one across the pipeline steps;
    # a single copy consolidates memory and eliminates pandas PerformanceWarning.
    features = features.copy()

    print(f"[pipeline] done  rows={len(features)}  cols={len(features.columns)}")
    return features


# ── Standalone download + merge helper (used by run.py step_download) ─────────

def fetch_and_merge(
    trading_pairs: list,
    period: str,
    out_base: str = "bookDepth_data",
) -> dict:
    """Download book depth data for given pairs/period and merge daily CSVs per pair.

    Kept as a standalone helper so step_download in run.py can call it without
    triggering the rest of the pipeline.  Not called by run_pipeline().
    """
    from data_collection import book_depth_utils

    summary = book_depth_utils.download_book_depth_range(
        trading_pairs, period, out_base=out_base, pause_seconds=0.15
    )

    import glob as _glob
    for pair in trading_pairs:
        path = Path(out_base) / pair.strip().upper()
        if not path.is_dir():
            continue

        daily_files = sorted(
            f for f in _glob.glob(str(path / "*.csv"))
            if "_merged" not in Path(f).name
        )
        if not daily_files:
            print(f"  [{pair}] no daily CSVs found, skipping merge")
            continue

        merged_df = pd.concat([pd.read_csv(f) for f in daily_files], ignore_index=True)
        out_path  = path / f"{pair.upper()}_merged.csv"
        merged_df.to_csv(out_path, index=False)
        print(f"  [{pair}] merged {len(daily_files)} files → {out_path.name}")

    return summary


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw   = input("Trading pairs (comma-separated) or enter for default [BTCUSDT,ETHUSDT]: ").strip().upper()
    pairs = ["BTCUSDT", "ETHUSDT"] if raw == "" else raw.split(",")
    df    = run_pipeline(pairs)
    print(df.head())
