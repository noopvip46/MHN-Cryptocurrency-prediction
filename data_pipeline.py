import glob
from pathlib import Path

import numpy as np
import pandas as pd

import book_depth_utils


# Fetching and merging, this file is the main data processing pipeline. Produces the cleaned/feature-engineered CSV ready for modelling.
"""
This file meant for ongoing use, not just a one-off historical run. For one-off use the notebook is more convenient for iterating on cleaning/feature engineering steps, 
but the pipeline can be re-run as needed to pull new data and update the merged files.
"""
def fetch_and_merge(trading_pairs: list, period: str, out_base: str = "bookDepth_data") -> dict:
    """Download book depth data for given pairs/period and merge daily CSVs per pair.

    For live use, call with period="1d" (or a specific date) each day to pull the
    latest data and append it to the merged file.

    Returns the download summary dict from book_depth_utils.
    """
    summary = book_depth_utils.download_book_depth_range(
        trading_pairs, period, out_base=out_base, pause_seconds=0.15
    )

    for pair in trading_pairs:
        path = Path(out_base) / pair.strip().upper()
        if not path.is_dir():
            continue

        # Exclude the merged file itself so re-runs don't double-count rows
        daily_files = sorted(
            f for f in glob.glob(str(path / "*.csv"))
            if "_merged" not in Path(f).name
        )
        if not daily_files:
            print(f"  [{pair}] no daily CSVs found, skipping merge")
            continue

        merged_df = pd.concat([pd.read_csv(f) for f in daily_files], ignore_index=True)
        out_path = path / f"{pair.upper()}_merged.csv"
        merged_df.to_csv(out_path, index=False)
        print(f"  [{pair}] merged {len(daily_files)} files → {out_path.name}")

    return summary


# ── Cleaning & Pivoting ────────────────────────────────────────────────────────

def pivot_merged(out_base: str = "bookDepth_data") -> pd.DataFrame:
    """Load all *_merged.csv files, pivot (timestamp × percentage), combine pairs.

    Saves the result to {out_base}/all_pairs_cleaned.csv and returns it as a
    DataFrame ready for feature engineering.
    """
    merged_files = sorted(glob.glob(f"{out_base}/*/*_merged*.csv"))
    if not merged_files:
        raise FileNotFoundError(f"No merged CSV files found under {out_base}/")

    all_cleaned = []
    for merged_file in merged_files:
        pair = Path(merged_file).parts[-2]
        df = pd.read_csv(merged_file, parse_dates=["timestamp"])
        if df.empty:
            print(f"  [{pair}] empty merged file, skipping")
            continue

        pivoted = df.pivot_table(
            index="timestamp",
            columns="percentage",
            values=["depth", "notional"],
            aggfunc="first",
        )
        pivoted.columns = [f"{pair}_{v}_{k}" for v, k in pivoted.columns]
        pivoted = pivoted.reset_index()
        all_cleaned.append(pivoted)
        print(f"  [{pair}] pivoted: rows={len(pivoted)}, cols={len(pivoted.columns)}")

    if not all_cleaned:
        raise ValueError("All merged files were empty.")

    combined = pd.concat(all_cleaned, ignore_index=True, sort=False)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    out_file = Path(out_base) / "all_pairs_cleaned.csv"
    combined.to_csv(out_file, index=False)
    print(f"  saved → {out_file}  rows={len(combined)}, cols={len(combined.columns)}")

    return combined


# ── Feature Engineering ────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    trading_pairs: list,
    roll_window: int = 60,
    regime_window: int = 240,
) -> pd.DataFrame:
    """Add engineered features to the cleaned pivot DataFrame.

    Per pair this adds:
      - depth imbalance (pos − neg) and imbalance ratio
      - notional imbalance and imbalance ratio  (neg/pos columns are kept)
      - T vs T-1 deltas: depth_imbalance_ratio, notional_imbalance_ratio, total_notional
      - rolling z-score at roll_window (short-term) and regime_window (long-term)

    Raw aggregates (total_depth_*, total_notional, log_notional) are dropped at the end.
    The input DataFrame is not modified in place.
    """
    df = df.copy()

    for pair in trading_pairs:
        pair = pair.strip().upper()

        # ── Depth ──────────────────────────────────────────────────────────────
        depth_neg = [c for c in df.columns if c.startswith(f"{pair}_depth_-")]
        depth_pos = [c for c in df.columns if c.startswith(f"{pair}_depth_") and not c.startswith(f"{pair}_depth_-")]

        total_depth_neg = df[depth_neg].sum(axis=1)
        total_depth_pos = df[depth_pos].sum(axis=1)
        total_depth = total_depth_neg + total_depth_pos

        df[f"{pair}_depth_imbalance"] = total_depth_pos - total_depth_neg
        df[f"{pair}_depth_imbalance_ratio"] = df[f"{pair}_depth_imbalance"] / total_depth.replace(0, np.nan)

        # ── Notional ───────────────────────────────────────────────────────────
        notional_neg = [c for c in df.columns if c.startswith(f"{pair}_notional_-")]
        notional_pos = [c for c in df.columns if c.startswith(f"{pair}_notional_") and not c.startswith(f"{pair}_notional_-")]

        df[f"{pair}_total_notional_neg"] = df[notional_neg].sum(axis=1)
        df[f"{pair}_total_notional_pos"] = df[notional_pos].sum(axis=1)
        total_notional = df[f"{pair}_total_notional_neg"] + df[f"{pair}_total_notional_pos"]
        df[f"{pair}_total_notional"] = total_notional  # temp; dropped below after delta

        df[f"{pair}_notional_imbalance"] = df[f"{pair}_total_notional_pos"] - df[f"{pair}_total_notional_neg"]
        df[f"{pair}_notional_imbalance_ratio"] = df[f"{pair}_notional_imbalance"] / total_notional.replace(0, np.nan)

        # ── T vs T-1 deltas ────────────────────────────────────────────────────
        df[f"{pair}_depth_imbalance_ratio_delta"] = df[f"{pair}_depth_imbalance_ratio"].diff()
        df[f"{pair}_notional_imbalance_ratio_delta"] = df[f"{pair}_notional_imbalance_ratio"].diff()
        df[f"{pair}_total_notional_delta"] = df[f"{pair}_total_notional"].diff()

        # ── Rolling z-score normalization ──────────────────────────────────────
        log_notional = np.log1p(total_notional)

        roll_mean = log_notional.rolling(roll_window, min_periods=1).mean()
        roll_std = log_notional.rolling(roll_window, min_periods=1).std().replace(0, np.nan)
        df[f"{pair}_notional_z"] = (log_notional - roll_mean) / roll_std

        regime_avg = log_notional.rolling(regime_window, min_periods=1).mean()
        regime_std = log_notional.rolling(regime_window, min_periods=1).std().replace(0, np.nan)
        df[f"{pair}_notional_regime_z"] = (log_notional - regime_avg) / regime_std

    # Drop raw aggregates that were only needed as intermediates
    drop_cols = []
    for pair in trading_pairs:
        pair = pair.strip().upper()
        drop_cols += [
            f"{pair}_total_notional",   # kept delta, not the raw total
        ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def infer_pairs(df: pd.DataFrame) -> list:
    """Infer trading pair names from DataFrame column names.

    Assumes columns follow the pattern <PAIR>_<feature>_<level>.
    """
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
    period: str,
    out_base: str = "bookDepth_data",
    roll_window: int = 60,
    regime_window: int = 240,
) -> pd.DataFrame:
    """Full pipeline: fetch → merge → pivot → feature engineering.

    For historical runs use e.g. period="7d".
    For live daily updates call with period="1d" (or yesterday's date "YYYY-MM-DD").

    Returns the feature-engineered DataFrame ready for modelling.
    """
    trading_pairs = [p.strip().upper() for p in trading_pairs if p.strip()]

    print(f"[pipeline] fetching {trading_pairs}  period={period!r}")
    summary = fetch_and_merge(trading_pairs, period, out_base=out_base)
    print(
        f"[pipeline] downloaded={summary['downloaded']}  "
        f"missing={summary['skipped_404']}  errors={summary['errors']}"
    )

    print("[pipeline] pivoting...")
    cleaned = pivot_merged(out_base=out_base)

    print("[pipeline] engineering features...")
    features = engineer_features(cleaned, trading_pairs, roll_window, regime_window)
    print(f"[pipeline] done  rows={len(features)}  cols={len(features.columns)}")

    return features


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw = input("Trading pairs (comma-separated) or 0 for default [ETHUSDT]: ").strip().upper()
    pairs = ["ETHUSDT"] if raw in ("0", "") else raw.split(",")

    period = input("Period (e.g. 7d, 1d, 2026-03-17): ").strip() or "7d"

    df = run_pipeline(pairs, period)
    print(df.head())
