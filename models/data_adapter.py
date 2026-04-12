# Loads all_pairs_labeled.csv, builds sliding window sequences, and returns
# strictly chronological train / calibration / test splits.
#
# We use a three-way split because HopCPT needs a calibration set that the
# base model never sees during training. The order is always train → cal → test
# with no shuffling since this is time series data and future cannot leak into past.
#
# Label column convention produced by the extraction notebook: {PAIR}_flash_crash_label

import math
from pathlib import Path

import numpy as np
import pandas as pd


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
        # Read the CSV, sort chronologically, build sliding windows, print diagnostics.
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Labeled CSV not found: {self.csv_path}\n"
                "Run the feature extraction notebook first to generate it."
            )

        print(f"[SequenceDataset] loading {self.csv_path} ...")
        df = pd.read_csv(self.csv_path)

        # Sort chronologically — mandatory for valid time series splits
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        label_col = f"{self.label_pair}_flash_crash_label"
        if label_col not in df.columns:
            available = [c for c in df.columns if c.endswith("_flash_crash_label")]
            raise KeyError(f"Label column '{label_col}' not found. Available label columns: {available}")

        # Separate features from labels and timestamp
        label_cols_all = [c for c in df.columns if c.endswith("_flash_crash_label")]
        drop_cols = ["timestamp"] + label_cols_all
        feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Forward-fill then back-fill NaNs — book depth snapshots can have small gaps
        feature_df = feature_df.ffill().bfill()

        self._feature_names = feature_df.columns.tolist()
        features_arr = feature_df.values.astype(np.float32)
        labels_arr   = df[label_col].fillna(0).astype(int).values

        T, F = features_arr.shape
        print(f"[SequenceDataset] raw data: {T} rows x {F} features")

        # Build sliding windows: window i covers rows [i, i+seq_len), label is at row i+seq_len-1
        n_windows = T - self.seq_len + 1
        if n_windows <= 0:
            raise ValueError(f"seq_len={self.seq_len} exceeds data length {T}. Reduce seq_len or add more data.")

        X = np.empty((n_windows, self.seq_len, F), dtype=np.float32)
        y = np.empty(n_windows, dtype=np.int8)
        for i in range(n_windows):
            X[i] = features_arr[i : i + self.seq_len]
            y[i] = labels_arr[i + self.seq_len - 1]

        self._X       = X
        self._y       = y
        self._n_train = math.floor(n_windows * self.train_ratio)
        self._n_cal   = math.floor(n_windows * self.cal_ratio)

        self._print_diagnostics()
        return self

    def get_splits(self):
        # Returns (X_train, y_train), (X_cal, y_cal), (X_test, y_test)
        # X shapes: (N, seq_len, n_features), strictly chronological, no shuffle.
        self._check_loaded()
        a, b = self._n_train, self._n_train + self._n_cal
        return (
            (self._X[:a],  self._y[:a]),
            (self._X[a:b], self._y[a:b]),
            (self._X[b:],  self._y[b:]),
        )

    def get_flat_splits(self):
        # Same as get_splits but X is flattened to (N, seq_len * n_features).
        # Use this for sklearn-style models like XGBoost that expect 2D input.
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

        print(f"[SequenceDataset] windows={N}  seq_len={self.seq_len}  n_features={self._X.shape[2]}")
        for split_name, split_y in splits.items():
            n     = len(split_y)
            n_pos = int(split_y.sum())
            n_neg = n - n_pos
            pct   = 100.0 * n_pos / max(n, 1)
            ratio = f"1:{int(n_neg / max(n_pos, 1))}" if n_pos > 0 else "no positives"
            print(f"  {split_name:<6}: {n:>6} windows | crash={n_pos} ({pct:.2f}%) | class ratio {ratio}")
