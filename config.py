from pathlib import Path

ROOT_DIR        = Path(__file__).resolve().parent
BOOK_DEPTH_DIR  = ROOT_DIR / "bookDepth_data"
TRADES_DIR      = ROOT_DIR / "trades_data"
ONCHAIN_DIR     = ROOT_DIR / "onchain_data"
PROCESSED_DIR   = ROOT_DIR / "bookDepth_data"  # processed CSVs live alongside raw data

ALL_PAIRS_CLEANED = PROCESSED_DIR / "all_pairs_cleaned.csv"
ALL_PAIRS_TRADES  = PROCESSED_DIR / "all_pairs_with_trades.csv"
ALL_PAIRS_LABELED = PROCESSED_DIR / "all_pairs_labeled.csv"

DEFAULT_PAIRS       = ["BTCUSDT", "ETHUSDT"]
DEFAULT_PERIOD      = "6m"
SNAPSHOT_INTERVAL_S = 30   # ~30 seconds between book depth snapshots

ROLL_WINDOW_SHORT  = 120   # snapshots (~60 min)
ROLL_WINDOW_REGIME = 480   # snapshots (~240 min)

# ── CUSUM event filter (López de Prado AFML Ch. 2) ────────────────────────────
# Maintains running cumulative sums of upward/downward deviations in returns.
# Triggers a spike event when either cumsum exceeds threshold h; resets after.
CUSUM_H        = 0.015     # threshold for CUSUM trigger (1.5% cumulative deviation)
CUSUM_EXPECTED = 0.0       # expected return (drift) subtracted before accumulation

# ── Triple Barrier labeling (López de Prado AFML Ch. 3) ──────────────────────
# For each spike event detected by CUSUM, we place three barriers:
#   - take-profit (continuation): price moves further in spike direction by pt × σ
#   - stop-loss   (reversal):     price reverses against spike direction by sl × σ
#   - timeout:                    max_hold snapshots elapse without hitting either
# Label = which barrier is hit first.
BARRIER_PT       = None    # not used — simple forward-return labeling
BARRIER_SL       = None    # not used — simple forward-return labeling
BARRIER_MAX_HOLD = 10      # forward window: 10 snapshots (~5 min) to observe reversal
BARRIER_VOL_SPAN = 120     # lookback window for rolling volatility estimate (snapshots)

# Label encoding:
#   binary mode  → 1 = reversal (mean-reversion), 0 = continuation/timeout
#   ternary mode → 0 = reversal, 1 = timeout, 2 = continuation
LABEL_MODE       = "binary"

LABEL_PAIR    = "ETHUSDT"  # primary pair — paper targets Ethereum spike-corrections

SEQ_LEN     = 20    # input sequence length in snapshots (~10 min context for spike-correction)
TRAIN_RATIO = 0.70
CAL_RATIO   = 0.15  # calibration set for HopCPT, test set is whatever remains

# On-chain feature settings (Alchemy)
# Network follows the pattern: https://{network}.g.alchemy.com/v2/{key}
# User must enable the target network in their Alchemy dashboard first.
#
# On-chain features are global market-state columns, not per-pair.
# They are joined on timestamp so every row (regardless of pair) shares
# the same ETH chain values at that moment in time.
ONCHAIN_SYMBOL = "ETHUSDT"   # subdirectory name / filename prefix in onchain_data/

# Canonical column list — single source of truth shared with data_pipeline.py.
from data_collection.onchain_utils import ONCHAIN_FEATURE_COLUMNS  # noqa: E402

# ── Live production ────────────────────────────────────────────────────────────
LIVE_WINDOW_DAYS       = 30    # rolling training window kept in SnapshotBuffer
RETRAIN_INTERVAL_HOURS = 24    # how often the Trainer retrains on the rolling window
LIVE_WARMUP_SNAPSHOTS  = SEQ_LEN  # min rows before first prediction (= 120)

BINANCE_WS_BASE = "wss://fstream.binance.com/stream"  # futures combined stream
