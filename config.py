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

CRASH_HORIZON = 20         # how many snapshots forward we look to define the crash label (~10 min)
CRASH_SIGMA   = 3.0        # how many standard deviations below current volatility counts as a crash
LABEL_PAIR    = "BTCUSDT"  # primary pair we generate the label for

SEQ_LEN     = 120   # input sequence length in snapshots
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
