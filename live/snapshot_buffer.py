"""
live/snapshot_buffer.py

Thread-safe rolling feature buffer backed by a bounded deque.

maxlen = LIVE_WINDOW_DAYS * (86400 / SNAPSHOT_INTERVAL_S)
       = 30 * 2880 = 86,400 rows by default

As new rows are appended the oldest fall off automatically.
The Predictor reads tail(SEQ_LEN); the Trainer reads the full buffer.

Supports save/load (parquet) for warm restarts — the buffer is restored
to its previous state without re-downloading historical data.
"""

import threading
from collections import deque
from pathlib import Path
from typing import Optional

import pandas as pd


class SnapshotBuffer:
    """Bounded, thread-safe deque of feature-row dicts.

    Usage:
        buf = SnapshotBuffer(maxlen=86_400)
        buf.append({"timestamp_ms": ..., "ETHUSDT_depth_imbalance": ..., ...})
        df  = buf.tail(120)   # last 120 rows as DataFrame
    """

    def __init__(self, maxlen: int):
        self._maxlen = maxlen
        self._deque: deque = deque(maxlen=maxlen)
        self._lock   = threading.RLock()

    # ── write ──────────────────────────────────────────────────────────────────

    def append(self, row: dict):
        with self._lock:
            self._deque.append(row)

    def update_label(self, offset_from_end: int, label_col: str, value):
        """Set a label on a past row (used by live labeler after CRASH_HORIZON ticks)."""
        with self._lock:
            if offset_from_end >= len(self._deque):
                return
            idx  = len(self._deque) - 1 - offset_from_end
            row  = dict(self._deque[idx])
            row[label_col] = value
            # deque doesn't support item assignment — rebuild via list splice
            lst       = list(self._deque)
            lst[idx]  = row
            self._deque.clear()
            self._deque.extend(lst)

    # ── read ───────────────────────────────────────────────────────────────────

    def tail(self, n: int) -> pd.DataFrame:
        with self._lock:
            rows = list(self._deque)[-n:]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def as_dataframe(self) -> pd.DataFrame:
        with self._lock:
            rows = list(self._deque)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)

    # ── persistence ────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Persist buffer to parquet for warm restart."""
        df = self.as_dataframe()
        if not df.empty:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)

    def load(self, path: str):
        """Restore buffer from a previously saved parquet file."""
        p = Path(path)
        if not p.exists():
            return
        df = pd.read_parquet(p)
        # Only load up to maxlen rows (most recent)
        df = df.tail(self._maxlen)
        with self._lock:
            self._deque.clear()
            for _, row in df.iterrows():
                self._deque.append(row.to_dict())
        print(f"  [SnapshotBuffer] loaded {len(self._deque)} rows from {path}")
