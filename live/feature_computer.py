"""
live/feature_computer.py

Converts raw BinanceStream snapshots + on-chain block data into an engineered
feature row whose column names exactly match the offline pipeline output
(all_pairs_labeled.csv).

Rolling statistics (z-scores, deltas) are computed from a context window
passed in by the caller (last ROLL_WINDOW_REGIME rows from SnapshotBuffer).

Live labeling
-------------
Labels cannot be assigned at the moment of the snapshot — they depend on
whether a crash happens CRASH_HORIZON ticks later.  generate_label() is
called by the live system after CRASH_HORIZON new snapshots have arrived
and applies the label retroactively via SnapshotBuffer.update_label().
"""

import numpy as np
import pandas as pd
import threading
from typing import Dict, List, Optional


class FeatureComputer:
    """Stateful per-tick feature transformer.

    Maintains:
        _prev       last tick's computed values (for delta features)
        _onchain    latest on-chain block features (updated by OnchainStream)
        _vwap_buf   rolling VWAP deque for live label generation
    """

    def __init__(
        self,
        pairs:         List[str],
        roll_window:   int = 120,
        regime_window: int = 480,
        crash_horizon: int = 20,
        crash_sigma:   float = 3.0,
        label_pair:    str = "BTCUSDT",
    ):
        self.pairs         = [p.upper() for p in pairs]
        self.roll_window   = roll_window
        self.regime_window = regime_window
        self.crash_horizon = crash_horizon
        self.crash_sigma   = crash_sigma
        self.label_pair    = label_pair.upper()

        self._prev: Dict[str, dict] = {p: {} for p in self.pairs}
        self._onchain: dict         = {}
        self._onchain_lock          = threading.Lock()

        # Rolling VWAP buffer for label generation (length = regime_window + crash_horizon)
        from collections import deque
        self._vwap_buf: deque = deque(maxlen=regime_window + crash_horizon + 10)

    # ── on-chain update (called from OnchainStream callback thread) ────────────

    def update_onchain(self, features: dict):
        with self._onchain_lock:
            self._onchain = dict(features)

    # ── main compute ───────────────────────────────────────────────────────────

    def compute(
        self,
        snapshots: Dict[str, dict],
        context:   pd.DataFrame,
    ) -> dict:
        """Build one feature row from the current tick's snapshots.

        Args:
            snapshots: {symbol: snapshot_dict} for every pair this tick.
            context:   Last regime_window rows already in SnapshotBuffer
                       (used for rolling z-score computation).
        Returns:
            Feature dict ready to append to SnapshotBuffer.
        """
        ts_ms = next(iter(snapshots.values()))["timestamp_ms"]
        row: dict = {"timestamp_ms": ts_ms}

        for pair in self.pairs:
            snap = snapshots.get(pair, {})
            self._add_depth_features(row, pair, snap)
            self._add_trade_features(row, pair, snap)
            self._add_rolling_features(row, pair, context)
            self._cache_prev(pair, row)

        self._add_onchain_features(row)

        # Track VWAP for label generation
        if self.label_pair in self.pairs:
            self._vwap_buf.append(snapshots.get(self.label_pair, {}).get("vwap", 0.0))

        return row

    # ── label generation (called CRASH_HORIZON ticks after the target row) ─────

    def generate_label(self, buffer) -> Optional[float]:
        """Compute and write the flash-crash label for the row CRASH_HORIZON ticks ago.

        Returns the label value (0 or 1), or None if not enough history.
        Call this every tick; it writes via buffer.update_label().
        """
        if len(self._vwap_buf) < self.crash_horizon + 1:
            return None

        vwap_list   = list(self._vwap_buf)
        target_idx  = len(vwap_list) - 1 - self.crash_horizon   # the row being labeled
        target_vwap = vwap_list[target_idx]
        if target_vwap <= 0:
            return None

        # Forward returns from target row to now
        fwd_returns = [
            (vwap_list[target_idx + k] - target_vwap) / target_vwap
            for k in range(1, self.crash_horizon + 1)
        ]
        cumulative_return = sum(fwd_returns)

        # Volatility estimate from context around the target row
        recent_returns = [
            (vwap_list[i] - vwap_list[i - 1]) / vwap_list[i - 1]
            for i in range(max(1, target_idx - self.roll_window), target_idx + 1)
            if vwap_list[i - 1] > 0
        ]
        vol = float(np.std(recent_returns)) if len(recent_returns) > 1 else 1e-6

        label = 1 if cumulative_return < -self.crash_sigma * vol else 0

        label_col = f"{self.label_pair}_flash_crash_label"
        buffer.update_label(self.crash_horizon, label_col, label)
        return label

    # ── private helpers ────────────────────────────────────────────────────────

    def _add_depth_features(self, row: dict, pair: str, snap: dict):
        bids = snap.get("bids", [])
        asks = snap.get("asks", [])

        bid_depth    = sum(q for _, q in bids)
        ask_depth    = sum(q for _, q in asks)
        bid_notional = sum(p * q for p, q in bids)
        ask_notional = sum(p * q for p, q in asks)

        total_depth    = bid_depth    + ask_depth
        total_notional = bid_notional + ask_notional

        depth_imb    = bid_depth    - ask_depth
        notional_imb = bid_notional - ask_notional

        row[f"{pair}_depth_imbalance"]         = depth_imb
        row[f"{pair}_depth_imbalance_ratio"]   = depth_imb    / total_depth    if total_depth    > 0 else 0.0
        row[f"{pair}_total_notional_pos"]       = bid_notional
        row[f"{pair}_total_notional_neg"]       = ask_notional
        row[f"{pair}_notional_imbalance"]       = notional_imb
        row[f"{pair}_notional_imbalance_ratio"] = notional_imb / total_notional if total_notional > 0 else 0.0

        # Store for delta and z-score computation
        row[f"__{pair}_total_notional"] = total_notional   # temp, stripped before save

    def _add_trade_features(self, row: dict, pair: str, snap: dict):
        vwap      = snap.get("vwap",      0.0)
        total_qty = snap.get("total_qty", 0.0)
        buy_qty   = snap.get("buy_qty",   0.0)

        prev_vwap   = self._prev.get(pair, {}).get("_vwap", vwap)
        vwap_return = (vwap - prev_vwap) / prev_vwap if prev_vwap > 0 else 0.0

        row[f"{pair}_vwap"]        = vwap
        row[f"{pair}_vwap_return"] = vwap_return
        row[f"{pair}_buy_ratio"]   = buy_qty / total_qty if total_qty > 0 else 0.5
        row[f"{pair}_liq_count"]   = snap.get("liq_count", 0)
        row[f"{pair}_liq_qty"]     = snap.get("liq_qty",   0.0)

    def _add_rolling_features(self, row: dict, pair: str, context: pd.DataFrame):
        total_notional = row.get(f"__{pair}_total_notional", 0.0)
        log_notional   = np.log1p(total_notional)

        # Deltas vs previous tick
        prev = self._prev.get(pair, {})
        row[f"{pair}_depth_imbalance_ratio_delta"]    = (
            row.get(f"{pair}_depth_imbalance_ratio",   0.0)
            - prev.get(f"{pair}_depth_imbalance_ratio",   0.0)
        )
        row[f"{pair}_notional_imbalance_ratio_delta"] = (
            row.get(f"{pair}_notional_imbalance_ratio", 0.0)
            - prev.get(f"{pair}_notional_imbalance_ratio", 0.0)
        )
        row[f"{pair}_total_notional_delta"] = (
            total_notional - prev.get("_total_notional", total_notional)
        )

        # Rolling z-scores from context window
        pos_col = f"{pair}_total_notional_pos"
        neg_col = f"{pair}_total_notional_neg"
        if not context.empty and pos_col in context.columns and neg_col in context.columns:
            ctx_log = np.log1p(context[pos_col] + context[neg_col])

            roll   = ctx_log.tail(self.roll_window)
            regime = ctx_log.tail(self.regime_window)

            rs, rv = roll.mean(),   roll.std()
            gs, gv = regime.mean(), regime.std()

            row[f"{pair}_notional_z"]        = (log_notional - rs) / rv if rv > 0 else 0.0
            row[f"{pair}_notional_regime_z"] = (log_notional - gs) / gv if gv > 0 else 0.0
        else:
            row[f"{pair}_notional_z"]        = 0.0
            row[f"{pair}_notional_regime_z"] = 0.0

    def _cache_prev(self, pair: str, row: dict):
        self._prev[pair] = {
            f"{pair}_depth_imbalance_ratio":    row.get(f"{pair}_depth_imbalance_ratio",   0.0),
            f"{pair}_notional_imbalance_ratio": row.get(f"{pair}_notional_imbalance_ratio", 0.0),
            "_total_notional": row.get(f"__{pair}_total_notional", 0.0),
            "_vwap":           row.get(f"{pair}_vwap", 0.0),
        }
        # Remove temp keys before the row is saved to buffer
        row.pop(f"__{pair}_total_notional", None)

    def _add_onchain_features(self, row: dict):
        from data_collection.onchain_utils import ONCHAIN_FEATURE_COLUMNS
        with self._onchain_lock:
            onchain = dict(self._onchain)
        for col in ONCHAIN_FEATURE_COLUMNS:
            row[col] = onchain.get(col, 0.0)
