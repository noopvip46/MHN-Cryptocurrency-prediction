"""
live/predictor.py

Prediction loop — runs every SNAPSHOT_INTERVAL_S seconds.

Reads the last SEQ_LEN rows from SnapshotBuffer, runs model inference,
and fires the on_prediction callback with the result.

The model can be swapped atomically by the Trainer at any time without
blocking or dropping a prediction cycle.

Warmup: no predictions are emitted until the buffer holds at least
LIVE_WARMUP_SNAPSHOTS rows (default = SEQ_LEN = 120).
"""

import threading
import time
from typing import Callable, List, Optional

import numpy as np


class Predictor:
    """Periodic inference loop with atomic model-swap support.

    Usage:
        predictor = Predictor(buffer, seq_len=120, warmup=120)
        predictor.on_prediction(my_callback)

        # swap model whenever retraining completes:
        predictor.set_model(new_model, feature_cols)

        # run in a background thread:
        threading.Thread(target=predictor.run,
                         kwargs={"interval_s": 30}, daemon=True).start()
    """

    def __init__(
        self,
        buffer,
        seq_len: int,
        warmup:  int,
    ):
        self._buffer   = buffer
        self._seq_len  = seq_len
        self._warmup   = warmup
        self._stop     = threading.Event()

        self._model        = None
        self._feature_cols: Optional[List[str]] = None
        self._model_lock   = threading.RLock()

        self._callback: Optional[Callable[[dict], None]] = None

    # ── public API ─────────────────────────────────────────────────────────────

    def on_prediction(self, callback: Callable[[dict], None]) -> "Predictor":
        self._callback = callback
        return self

    def set_model(self, model, feature_cols: List[str]):
        """Atomically swap the model and its expected feature column list."""
        with self._model_lock:
            self._model        = model
            self._feature_cols = list(feature_cols)
        print(f"  [Predictor] model swapped  features={len(feature_cols)}")

    def stop(self):
        self._stop.set()

    # ── single-shot predict ────────────────────────────────────────────────────

    def predict_once(self) -> Optional[dict]:
        """Run one inference step. Returns result dict or None if not ready."""
        with self._model_lock:
            model = self._model
            cols  = self._feature_cols

        if model is None or cols is None:
            return None
        if len(self._buffer) < self._warmup:
            return None

        df = self._buffer.tail(self._seq_len)
        if len(df) < self._seq_len:
            return None

        # Guard: drop any columns that aren't in the feature list
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"  [Predictor] missing cols in buffer: {missing[:5]}...")
            return None

        X = df[cols].values.astype("float32")
        X = X[np.newaxis, ...]   # shape: (1, seq_len, n_features)

        try:
            prediction = model.predict(X)
            return {
                "timestamp_ms": int(df["timestamp_ms"].iloc[-1]),
                "prediction":   prediction,
            }
        except Exception as e:
            print(f"  [Predictor] inference error: {e}")
            return None

    # ── blocking loop ──────────────────────────────────────────────────────────

    def run(self, interval_s: int = 30):
        """Blocking prediction loop. Run in a daemon thread."""
        while not self._stop.wait(timeout=interval_s):
            buf_len = len(self._buffer)

            if buf_len < self._warmup:
                remaining = self._warmup - buf_len
                print(f"  [Predictor] warming up — {remaining} snapshots remaining")
                continue

            if self._model is None:
                print("  [Predictor] waiting for initial model from Trainer...")
                continue

            result = self.predict_once()
            if result is None:
                continue

            if self._callback:
                try:
                    self._callback(result)
                except Exception as e:
                    print(f"  [Predictor] callback error: {e}")
            else:
                pred = result["prediction"]
                ts   = result["timestamp_ms"]
                print(f"  [Predictor] ts={ts}  flash_crash={pred}")
