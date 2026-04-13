"""
live/run_live.py

Entry point for the live flash-crash prediction system.

Usage:
    python live/run_live.py
    python live/run_live.py --pairs ETHUSDT,BTCUSDT --model stanhop
    python live/run_live.py --window 14 --retrain-hours 12 --buffer-file live_buf.parquet

System flow:
    BinanceStream  ──┐
    OnchainStream  ──┴──► FeatureComputer ──► SnapshotBuffer ──► Predictor
                                                     │
                                                  Trainer (rolling retrain)

Rolling window:
    SnapshotBuffer is capped at --window days × 2880 snapshots.
    As new rows arrive the oldest fall off automatically.
    Retraining always uses whatever is currently in the buffer.

Live labeling:
    Labels are assigned CRASH_HORIZON ticks (10 min) after each snapshot
    by FeatureComputer.generate_label(), which writes retroactively into
    SnapshotBuffer.  The Trainer filters to labeled rows only.

Warm restart:
    Pass --buffer-file to persist and restore the buffer across restarts.
    The system resumes from the last saved state without re-downloading data.

Requirements:
    pip install websockets
    ALCHEMY_API_KEY set in .env  (on-chain features; optional)
"""

import sys
import signal
import argparse
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    DEFAULT_PAIRS,
    SEQ_LEN, TRAIN_RATIO, CAL_RATIO,
    ROLL_WINDOW_SHORT, ROLL_WINDOW_REGIME,
    CRASH_HORIZON, CRASH_SIGMA, LABEL_PAIR,
    SNAPSHOT_INTERVAL_S,
    LIVE_WINDOW_DAYS, RETRAIN_INTERVAL_HOURS, LIVE_WARMUP_SNAPSHOTS,
    ONCHAIN_SYMBOL,
)
from live.binance_stream  import BinanceStream
from live.snapshot_buffer import SnapshotBuffer
from live.feature_computer import FeatureComputer
from live.predictor       import Predictor
from live.trainer         import Trainer


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Flash Crash Predictor — live mode")
    p.add_argument("--pairs",
                   default=",".join(DEFAULT_PAIRS),
                   help="Comma-separated trading pairs (default: BTCUSDT,ETHUSDT)")
    p.add_argument("--model",
                   default="stanhop",
                   choices=["mhn", "stanhop", "lstm", "transformer"],
                   help="Model architecture (default: stanhop)")
    p.add_argument("--window",
                   type=int, default=LIVE_WINDOW_DAYS,
                   help=f"Rolling training window in days (default: {LIVE_WINDOW_DAYS})")
    p.add_argument("--retrain-hours",
                   type=int, default=RETRAIN_INTERVAL_HOURS,
                   help=f"Retrain interval in hours (default: {RETRAIN_INTERVAL_HOURS})")
    p.add_argument("--no-conformal",
                   action="store_true",
                   help="Disable HopCPT conformal wrapper")
    p.add_argument("--alpha",
                   type=float, default=0.1,
                   help="HopCPT miscoverage level (default: 0.1)")
    p.add_argument("--device",
                   default="auto", choices=["cpu", "cuda", "auto"],
                   help="Compute device (default: auto)")
    p.add_argument("--buffer-file",
                   default=None,
                   help="Parquet path for buffer persistence (warm restart)")
    return p.parse_args()


# ── helpers ────────────────────────────────────────────────────────────────────

def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


def make_model_factory(model_name: str, seq_len: int, device: str):
    """Returns a factory callable(n_features) → untrained model."""
    from models import (
        MHNFlashCrashModel, STanHopModel,
        LSTMFlashCrashModel, TransformerFlashCrashModel,
    )
    MODEL_MAP = {
        "mhn":         MHNFlashCrashModel,
        "stanhop":     STanHopModel,
        "lstm":        LSTMFlashCrashModel,
        "transformer": TransformerFlashCrashModel,
    }
    cls = MODEL_MAP[model_name]
    return lambda n_features: cls(seq_len=seq_len, n_features=n_features, device=device)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    pairs  = [p.strip().upper() for p in args.pairs.split(",")]
    device = resolve_device(args.device)

    ticks_per_day = 86400 // SNAPSHOT_INTERVAL_S   # 2880
    max_buf       = args.window * ticks_per_day

    print(f"\nFlash Crash Predictor — LIVE")
    print(f"  Pairs          : {pairs}")
    print(f"  Model          : {args.model}  device={device}")
    print(f"  Rolling window : {args.window} days  ({max_buf:,} max snapshots)")
    print(f"  Retrain every  : {args.retrain_hours} hours")
    print(f"  Conformal      : {not args.no_conformal}  alpha={args.alpha}")
    print(f"  Buffer file    : {args.buffer_file or 'none (in-memory only)'}")

    # ── SnapshotBuffer ────────────────────────────────────────────────────────
    buffer = SnapshotBuffer(maxlen=max_buf)
    if args.buffer_file:
        buffer.load(args.buffer_file)

    # ── FeatureComputer ───────────────────────────────────────────────────────
    computer = FeatureComputer(
        pairs         = pairs,
        roll_window   = ROLL_WINDOW_SHORT,
        regime_window = ROLL_WINDOW_REGIME,
        crash_horizon = CRASH_HORIZON,
        crash_sigma   = CRASH_SIGMA,
        label_pair    = LABEL_PAIR,
    )

    # ── Predictor ─────────────────────────────────────────────────────────────
    def on_prediction(result: dict):
        pred = result["prediction"]
        ts   = result["timestamp_ms"]
        tag  = "⚠  FLASH CRASH" if (hasattr(pred, '__iter__') and 1 in pred) or pred == 1 else "   normal"
        print(f"  [LIVE] {ts}  {tag}  raw={pred}")

    predictor = Predictor(
        buffer  = buffer,
        seq_len = SEQ_LEN,
        warmup  = LIVE_WARMUP_SNAPSHOTS,
    ).on_prediction(on_prediction)

    # ── Trainer ───────────────────────────────────────────────────────────────
    model_factory = make_model_factory(args.model, SEQ_LEN, device)

    trainer = Trainer(
        buffer             = buffer,
        predictor          = predictor,
        model_factory      = model_factory,
        seq_len            = SEQ_LEN,
        train_ratio        = TRAIN_RATIO,
        cal_ratio          = CAL_RATIO,
        label_col          = f"{LABEL_PAIR}_flash_crash_label",
        use_conformal      = not args.no_conformal,
        alpha              = args.alpha,
        retrain_interval_s = args.retrain_hours * 3600,
        min_rows           = LIVE_WARMUP_SNAPSHOTS * 4,
    )

    # ── Binance stream callback ───────────────────────────────────────────────
    # Accumulate snapshots until all pairs have reported, then compute features.
    _pending: dict = {}
    _pending_lock  = threading.Lock()

    def on_binance_snapshot(snapshot: dict):
        symbol = snapshot["symbol"]
        with _pending_lock:
            _pending[symbol] = snapshot
            if set(_pending.keys()) != set(pairs):
                return   # still waiting for the other pair(s) this tick
            tick_snaps = dict(_pending)
            _pending.clear()

        # Compute features (outside the lock to minimise hold time)
        context = buffer.tail(ROLL_WINDOW_REGIME)
        row     = computer.compute(tick_snaps, context)
        buffer.append(row)

        # Generate label for the snapshot CRASH_HORIZON ticks ago
        computer.generate_label(buffer)

        # Persist every 100 ticks if a file was specified
        if args.buffer_file and len(buffer) % 100 == 0:
            buffer.save(args.buffer_file)

    binance_stream = BinanceStream(
        symbols             = pairs,
        snapshot_interval_s = SNAPSHOT_INTERVAL_S,
    ).on_snapshot(on_binance_snapshot)

    # ── Alchemy on-chain stream (optional) ────────────────────────────────────
    onchain_stream = None
    try:
        from data_collection.onchain_utils import OnchainStream
        onchain_stream = OnchainStream.from_env()
        print("  [OnchainStream] Alchemy key found — on-chain features enabled")
    except EnvironmentError:
        print("  [WARN] ALCHEMY_API_KEY not set — on-chain features will be zero")

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    def shutdown(sig, _frame):
        print(f"\n  Received signal {sig} — shutting down...")
        binance_stream.stop()
        predictor.stop()
        trainer.stop()
        if onchain_stream:
            onchain_stream.stop()
        if args.buffer_file:
            buffer.save(args.buffer_file)
            print(f"  Buffer saved → {args.buffer_file} ({len(buffer)} rows)")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Launch threads ────────────────────────────────────────────────────────
    threads = [
        threading.Thread(target=binance_stream.start,  name="BinanceStream", daemon=True),
        threading.Thread(target=trainer.run,            name="Trainer",       daemon=True),
        threading.Thread(
            target=predictor.run,
            kwargs={"interval_s": SNAPSHOT_INTERVAL_S},
            name="Predictor",
            daemon=True,
        ),
    ]
    if onchain_stream:
        threads.append(threading.Thread(
            target=onchain_stream.start,
            args=(computer.update_onchain,),
            name="OnchainStream",
            daemon=True,
        ))

    print("\n  Starting threads...")
    for t in threads:
        t.start()
        print(f"  [{t.name}] started")

    print(f"\n  System live — press Ctrl+C to stop.\n")
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
