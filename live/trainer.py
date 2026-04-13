"""
live/trainer.py

Rolling retrain loop.

Every RETRAIN_INTERVAL_HOURS the Trainer:
  1. Pulls the full rolling window from SnapshotBuffer
  2. Filters to rows that have been labeled (CRASH_HORIZON delay applied)
  3. Splits 70 / 15 / 15 chronologically — same as offline pipeline
  4. Retrains the model and recalibrates HopCPT
  5. Swaps the Predictor's model atomically

The first training run fires immediately at startup so the Predictor
has a model before the first retrain interval elapses.

Rolling window constraint
-------------------------
The buffer is capped at LIVE_WINDOW_DAYS * 2880 rows by SnapshotBuffer.
The Trainer always trains on whatever is in the buffer — window management
is handled entirely by the buffer's maxlen, not here.
"""

import threading
import time
from typing import Callable, List, Optional

import numpy as np


class Trainer:
    """Background retraining loop with atomic model swap.

    Usage:
        trainer = Trainer(buffer, predictor, model_factory, seq_len=120)
        threading.Thread(target=trainer.run, daemon=True).start()
    """

    def __init__(
        self,
        buffer,
        predictor,
        model_factory:       Callable,   # callable() → new untrained model instance
        seq_len:             int,
        train_ratio:         float = 0.70,
        cal_ratio:           float = 0.15,
        label_col:           str   = "BTCUSDT_flash_crash_label",
        use_conformal:       bool  = True,
        alpha:               float = 0.1,
        retrain_interval_s:  int   = 86400,
        min_rows:            int   = 300,
    ):
        self._buffer              = buffer
        self._predictor           = predictor
        self._model_factory       = model_factory
        self._seq_len             = seq_len
        self._train_ratio         = train_ratio
        self._cal_ratio           = cal_ratio
        self._label_col           = label_col
        self._use_conformal       = use_conformal
        self._alpha               = alpha
        self._retrain_interval_s  = retrain_interval_s
        self._min_rows            = min_rows
        self._stop                = threading.Event()

    # ── public API ─────────────────────────────────────────────────────────────

    def stop(self):
        self._stop.set()

    # ── core retrain ───────────────────────────────────────────────────────────

    def retrain_once(self) -> bool:
        """Pull buffer → split → train → calibrate → swap. Returns True on success."""
        df = self._buffer.as_dataframe()

        if df.empty:
            print("  [Trainer] buffer empty — skipping")
            return False

        # Only use rows that have been labeled (label col present and not NaN)
        if self._label_col not in df.columns:
            print(f"  [Trainer] label column '{self._label_col}' not yet available — skipping")
            return False

        labeled = df[df[self._label_col].notna()].reset_index(drop=True)

        if len(labeled) < self._min_rows:
            print(
                f"  [Trainer] only {len(labeled)} labeled rows "
                f"(need {self._min_rows}) — skipping"
            )
            return False

        feature_cols = self._infer_feature_cols(labeled)
        if not feature_cols:
            print("  [Trainer] no feature columns found — skipping")
            return False

        # ── build sliding-window sequences ────────────────────────────────────
        arr    = labeled[feature_cols].values.astype("float32")
        labels = labeled[self._label_col].values.astype("float32")

        X_all, y_all = [], []
        for i in range(self._seq_len, len(arr)):
            X_all.append(arr[i - self._seq_len : i])
            y_all.append(labels[i])

        if not X_all:
            print("  [Trainer] not enough rows to form sequences — skipping")
            return False

        X_all = np.array(X_all)   # (N, seq_len, n_features)
        y_all = np.array(y_all)

        n       = len(X_all)
        n_train = int(n * self._train_ratio)
        n_cal   = int(n * self._cal_ratio)

        X_tr,  y_tr  = X_all[:n_train],                y_all[:n_train]
        X_cal, y_cal = X_all[n_train:n_train + n_cal], y_all[n_train:n_train + n_cal]
        X_te,  y_te  = X_all[n_train + n_cal:],        y_all[n_train + n_cal:]

        crash_rate = float(y_tr.mean()) if len(y_tr) > 0 else 0.0
        print(
            f"  [Trainer] retraining  "
            f"train={len(X_tr)}  cal={len(X_cal)}  test={len(X_te)}  "
            f"crash_rate={crash_rate:.3f}  features={len(feature_cols)}"
        )

        # ── train ─────────────────────────────────────────────────────────────
        try:
            model = self._model_factory(len(feature_cols))
            model.fit(X_tr, y_tr)

            if self._use_conformal:
                from models import HopCPT
                cpt = HopCPT(model, alpha=self._alpha)
                cpt.fit(X_tr, y_tr, X_val=X_cal, y_val=y_cal)
                cpt.calibrate(X_cal, y_cal)
                live_model = cpt
            else:
                live_model = model

            metrics = live_model.evaluate(X_te, y_te)
            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  [Trainer] done  {metric_str}")

            # ── atomic model swap ──────────────────────────────────────────────
            self._predictor.set_model(live_model, feature_cols)
            return True

        except Exception as e:
            import traceback
            print(f"  [Trainer] retrain failed: {e}")
            traceback.print_exc()
            return False

    # ── blocking loop ──────────────────────────────────────────────────────────

    def run(self):
        """Blocking retrain loop. Run in a daemon thread.

        Fires immediately on startup, then every retrain_interval_s.
        """
        print("  [Trainer] initial training run...")
        self.retrain_once()

        while not self._stop.wait(timeout=self._retrain_interval_s):
            print("  [Trainer] scheduled retrain starting...")
            self.retrain_once()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _infer_feature_cols(self, df) -> List[str]:
        exclude = {"timestamp_ms"}
        return [
            c for c in df.columns
            if c not in exclude and not c.endswith("_label")
            and df[c].dtype in (float, int, "float32", "float64", "int32", "int64")
        ]
