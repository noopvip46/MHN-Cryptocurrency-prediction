# Loads all_pairs_labeled.csv, builds sliding window sequences, and returns
# strictly chronological train / calibration / test splits.
#
# We use a three-way split because HopCPT needs a calibration set that the
# base model never sees during training. The order is always train → cal → test
# with no shuffling since this is time series data and future cannot leak into past.
#
# Event-based indexing: after building sliding windows over ALL rows, we keep
# only windows whose final row is a spike event (CUSUM-detected).  This means
# models train on the order book state leading up to each spike, not on every
# 30-second snapshot.
#
# Label column convention: {PAIR}_spike_label  (1=reversal, 0=continuation)

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data


class WindowDataset(torch.utils.data.Dataset):
    """Lazy sliding-window PyTorch Dataset.

    Holds only a reference to the base feature view and labels array.
    Each window is made contiguous and converted to a tensor on demand in
    __getitem__ — only one window (seq_len × n_features) is in memory per
    worker at a time, not the entire training split.

    RAM cost: O(T × F) for the base array, not O(N_windows × seq_len × F).
    """

    def __init__(self, X_view: np.ndarray, y: np.ndarray):
        # X_view : (N, seq_len, F)  — may be a non-contiguous stride-tricks view
        # y      : (N,)             — int8 labels
        self._X = X_view
        self._y = y

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        # .copy() makes one window contiguous and writable so torch can wrap it.
        # stride_tricks views are read-only; ascontiguousarray alone doesn't fix that.
        x = torch.from_numpy(self._X[idx].copy())   # (seq_len, F) float32
        y = torch.tensor(float(self._y[idx]))
        return x, y


def make_balanced_sampler(y: np.ndarray, pos_fraction: float = 0.10):
    """Create a WeightedRandomSampler that oversamples positives.

    During training with extreme imbalance (e.g. 1:290), random batches may
    contain zero positives for hundreds of iterations — the model never sees
    what a crash looks like and learns to predict all-zeros.

    This sampler assigns higher sampling weight to positive rows so that each
    batch contains ~pos_fraction (default 10%) crash examples.  The model
    trains on a rebalanced distribution but is evaluated on the honest
    imbalanced test set, so metrics reflect real-world performance.

    Returns a WeightedRandomSampler that can be passed as `sampler=` to
    a DataLoader (replaces shuffle=True).
    """
    y = np.asarray(y, dtype=int)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    if n_pos == 0 or n_neg == 0:
        return None   # can't balance a single-class dataset

    # Weight each sample so that the expected fraction of positives per batch
    # equals pos_fraction.  w_pos/w_neg = (pos_fraction / (1-pos_fraction)) * (n_neg / n_pos).
    w_pos = pos_fraction / max(n_pos, 1)
    w_neg = (1.0 - pos_fraction) / max(n_neg, 1)

    weights = np.where(y == 1, w_pos, w_neg).astype(np.float64)

    return torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(y),         # one full "epoch" = N draws
        replacement=True,           # required for oversampling minority class
    )


class SequenceDataset:

    def __init__(self, csv_path, seq_len=120, label_pair="BTCUSDT", train_ratio=0.70, cal_ratio=0.15):
        if train_ratio + cal_ratio >= 1.0:
            raise ValueError(f"train_ratio ({train_ratio}) + cal_ratio ({cal_ratio}) must be < 1.0")

        self.csv_path    = Path(csv_path)
        self.seq_len     = seq_len
        self.label_pair  = label_pair.strip().upper()
        self.train_ratio = train_ratio
        self.cal_ratio   = cal_ratio

        # Populated by load()
        self._X            = None
        self._y            = None
        self._feature_names = []
        self._n_train      = 0
        self._n_cal        = 0

    def load(self):
        """Read CSV, build sliding windows, filter to spike events, split chronologically."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Labeled CSV not found: {self.csv_path}\n"
                "Run the pipeline first to generate it."
            )

        print(f"[SequenceDataset] loading {self.csv_path} ...")
        df = pd.read_csv(self.csv_path)

        # Sort chronologically — mandatory for valid time series splits
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        # ── Detect label columns: spike-based (new) or crash-based (legacy) ──
        event_col = f"{self.label_pair}_spike_event"
        label_col = f"{self.label_pair}_spike_label"

        if event_col not in df.columns or label_col not in df.columns:
            # Fall back to legacy crash label for backward compat
            legacy_col = f"{self.label_pair}_flash_crash_label"
            if legacy_col in df.columns:
                print(f"[SequenceDataset] using legacy label column: {legacy_col}")
                event_col = None   # no event filtering — train on all rows
                label_col = legacy_col
            else:
                available = [c for c in df.columns if "label" in c.lower()]
                raise KeyError(f"Label columns not found for {self.label_pair}. "
                               f"Available: {available}")

        # ── Separate features from metadata/labels ────────────────────────────
        # Drop timestamp, all label/event/direction columns, vwap_volatility
        # (used to compute barriers, would leak label info to the model)
        spike_cols    = [c for c in df.columns if any(
            c.endswith(s) for s in ("_spike_event", "_spike_label", "_spike_direction")
        )]
        legacy_cols   = [c for c in df.columns if c.endswith("_flash_crash_label")]
        vwap_vol_cols = [c for c in df.columns if c.endswith("_vwap_volatility")]

        drop_cols = ["timestamp"] + spike_cols + legacy_cols + vwap_vol_cols
        feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Forward-fill then back-fill NaNs — book depth snapshots can have small gaps
        feature_df = feature_df.ffill().bfill()

        # ── Propagate spike-characterizing features ──────────────────────────
        # spike_magnitude, spike_dir_signed, spike_vol_normalized are only non-zero
        # at event rows.  In a window, only the last row has values — the other 19
        # rows are 0.  DL models struggle with such sparse signals.
        # Solution: forward-fill these columns so the spike info "persists" across
        # multiple rows.  This way every row in an event window carries the spike
        # context (what kind of spike are we classifying?).
        spike_feature_cols = [c for c in feature_df.columns if any(
            c.endswith(s) for s in ("_spike_magnitude", "_spike_dir_signed", "_spike_vol_normalized")
        )]
        for col in spike_feature_cols:
            # Replace 0s with NaN so ffill works, then fill forward up to seq_len rows
            series = feature_df[col].replace(0.0, np.nan)
            feature_df[col] = series.ffill(limit=self.seq_len).fillna(0.0)

        self._feature_names = feature_df.columns.tolist()
        features_arr = feature_df.values.astype(np.float32)
        labels_arr   = df[label_col].fillna(0).astype(int).values

        T, F = features_arr.shape
        print(f"[SequenceDataset] raw data: {T} rows x {F} features")

        n_windows = T - self.seq_len + 1
        if n_windows <= 0:
            raise ValueError(f"seq_len={self.seq_len} exceeds data length {T}. "
                             "Reduce seq_len or add more data.")

        # Build sliding window view: (n_windows, seq_len, F)
        X_all = np.lib.stride_tricks.sliding_window_view(
            features_arr, (self.seq_len, F)
        )[:, 0]
        y_all = labels_arr[self.seq_len - 1:]

        # ── Event-based filtering ─────────────────────────────────────────────
        # Keep only windows whose final row is a spike event.
        # This is the core difference from crash prediction: we train on
        # events, not every snapshot.
        #
        # UNION strategy: use spike events from ALL pairs, not just the
        # primary label pair.  For BTC events, use BTC's spike_label; for
        # ETH-only events, use ETH's spike_label.  This roughly doubles
        # the training set and lets the model learn cross-asset reversal
        # patterns.
        if event_col is not None:
            # Collect all pairs' event and label columns
            all_event_cols = [c for c in df.columns if c.endswith("_spike_event")]
            all_label_cols = {c.replace("_spike_event", ""): c.replace("_spike_event", "_spike_label")
                              for c in all_event_cols
                              if c.replace("_spike_event", "_spike_label") in df.columns}

            # Build union event mask: a row is an event if ANY pair triggers
            union_event = np.zeros(T, dtype=np.int8)
            for ecol in all_event_cols:
                union_event = np.maximum(union_event, df[ecol].fillna(0).astype(np.int8).values)

            # Build union label: prefer primary pair's label, fall back to any available
            union_label = np.full(T, -1, dtype=np.int8)   # -1 = no label
            # First fill from non-primary pairs
            for pair_prefix, lcol in all_label_cols.items():
                pair_event_col = f"{pair_prefix}_spike_event"
                mask = df[pair_event_col].fillna(0).astype(int).values == 1
                union_label[mask] = df[lcol].fillna(0).astype(int).values[mask]
            # Then overwrite with primary pair (takes priority)
            primary_mask = df[event_col].fillna(0).astype(int).values == 1
            union_label[primary_mask] = df[label_col].fillna(0).astype(int).values[primary_mask]

            event_flags = union_event[self.seq_len - 1:]   # align with windows
            label_flags = union_label[self.seq_len - 1:]
            event_mask  = (event_flags == 1) & (label_flags >= 0)

            n_events = int(event_mask.sum())
            n_primary = int(primary_mask[self.seq_len - 1:][event_mask].sum()) if n_events > 0 else 0
            n_other   = n_events - n_primary
            print(f"[SequenceDataset] {n_events} spike events (union) out of "
                  f"{n_windows} windows ({100*n_events/n_windows:.2f}%)")
            print(f"  primary ({self.label_pair}): {n_primary}  "
                  f"other pairs: {n_other}")

            if n_events == 0:
                raise ValueError("No spike events found. Check CUSUM_H threshold "
                                 "or re-run label generation.")

            # Copy event windows — stride-tricks views can't be boolean-indexed safely
            event_indices = np.where(event_mask)[0]
            X = np.array([X_all[i].copy() for i in event_indices], dtype=np.float32)
            y = label_flags[event_mask].astype(np.int8)
        else:
            # Legacy mode: use all windows
            X = X_all
            y = y_all.astype(np.int8)
            event_indices = np.arange(len(y))

        # ── Chronological split based on event positions ──────────────────────
        n_events_total = len(y)
        self._n_train = math.floor(n_events_total * self.train_ratio)
        self._n_cal   = math.floor(n_events_total * self.cal_ratio)

        self._X = X
        self._y = y

        self._print_diagnostics()
        return self

    def get_splits(self):
        """Returns (X_train, y_train), (X_cal, y_cal), (X_test, y_test).

        X shapes: (N, seq_len, n_features), strictly chronological, no shuffle.
        Only spike event windows are included.
        """
        self._check_loaded()
        a, b = self._n_train, self._n_train + self._n_cal
        return (
            (self._X[:a],  self._y[:a]),
            (self._X[a:b], self._y[a:b]),
            (self._X[b:],  self._y[b:]),
        )

    def get_flat_splits(self):
        """Same as get_splits but X is flattened to (N, seq_len * n_features).

        Use this for sklearn-style models like XGBoost that expect 2D input.
        """
        (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = self.get_splits()
        flatten = lambda x: x.reshape(x.shape[0], -1)
        return (
            (flatten(X_tr),  y_tr),
            (flatten(X_cal), y_cal),
            (flatten(X_te),  y_te),
        )

    @property
    def n_features(self):
        self._check_loaded()
        return self._X.shape[2]

    @property
    def n_timesteps(self):
        return self.seq_len

    @property
    def feature_names(self):
        self._check_loaded()
        return list(self._feature_names)

    def _check_loaded(self):
        if self._X is None:
            raise RuntimeError("Call .load() before accessing data.")

    def _print_diagnostics(self):
        N = len(self._y)
        a = self._n_train
        b = a + self._n_cal
        splits = {"train": self._y[:a], "cal": self._y[a:b], "test": self._y[b:]}

        print(f"[SequenceDataset] events={N}  seq_len={self.seq_len}  n_features={self._X.shape[2]}")
        for split_name, split_y in splits.items():
            n     = len(split_y)
            n_pos = int(split_y.sum())
            n_neg = n - n_pos
            pct   = 100.0 * n_pos / max(n, 1)
            ratio_str = f"{n_pos}:{n_neg}" if n_pos > 0 else "no reversals"
            print(f"  {split_name:<6}: {n:>6} events | reversal={n_pos} ({pct:.1f}%) "
                  f"continuation={n_neg} ({100-pct:.1f}%) | ratio {ratio_str}")
