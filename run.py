# Main script to run the full flash crash prediction pipeline end to end.
# Use --help to see all options.

import argparse
import sys
import traceback

import pandas as pd
import numpy as np

from config import (
    DEFAULT_PAIRS, DEFAULT_PERIOD, SEQ_LEN,
    ALL_PAIRS_TRADES, ALL_PAIRS_LABELED,
    CRASH_HORIZON, CRASH_SIGMA,
    BOOK_DEPTH_DIR, TRADES_DIR,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flash Crash Predictor — end-to-end pipeline"
    )
    parser.add_argument(
        "--pairs",
        default=",".join(DEFAULT_PAIRS),
        help="Comma-separated trading pairs (default: BTCUSDT,ETHUSDT)",
    )
    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help="Data period e.g. 6m, 1y (default: 6m)",
    )
    parser.add_argument(
        "--model",
        default="stanhop",
        choices=["mhn", "stanhop", "lstm", "transformer", "xgboost", "random_forest", "logistic"],
        help="Model to train (default: stanhop)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help=f"Input sequence length in snapshots (default: {SEQ_LEN})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for deep learning models (default: 50)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="HopCPT miscoverage level (default: 0.1)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download step",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip feature extraction step",
    )
    parser.add_argument(
        "--skip-label",
        action="store_true",
        help="Skip label generation step",
    )
    parser.add_argument(
        "--no-conformal",
        action="store_true",
        help="Do not wrap model with HopCPT conformal predictor",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Compute device (default: auto)",
    )
    return parser.parse_args()


def step_download(pairs, period):
    print("\n\u2550\u2550\u2550 Step 1/3: Downloading data \u2550\u2550\u2550")
    from data_collection import book_depth_utils, trades_utils
    import glob
    from pathlib import Path

    # Book depth
    print(f"  Downloading book depth for {pairs} over {period} ...")
    bd_summary = book_depth_utils.download_book_depth_range(
        pairs, period, out_base=str(BOOK_DEPTH_DIR), pause_seconds=0.15
    )
    print(f"  Book depth — downloaded: {bd_summary['downloaded']}, "
          f"missing: {bd_summary['skipped_404']}, errors: {bd_summary['errors']}")

    # Merge book depth daily CSVs
    for pair in pairs:
        path = BOOK_DEPTH_DIR / pair
        if not path.is_dir():
            print(f"  [WARN] No bookDepth_data directory for {pair}")
            continue
        all_files = sorted(
            f for f in glob.glob(str(path / "*.csv"))
            if "_merged" not in Path(f).name
        )
        if not all_files:
            print(f"  [WARN] No daily CSV files found for {pair}")
            continue
        out_path = path / f"{pair}_merged.csv"
        first = True
        for f in all_files:
            chunk = pd.read_csv(f)
            chunk.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
            first = False
        print(f"  [{pair}] merged {len(all_files)} daily files -> {out_path.name}")

    # Trades
    print(f"  Downloading trades for {pairs} over {period} ...")
    tr_summary = trades_utils.download_trades_range(
        pairs, period, out_base=str(TRADES_DIR), pause_seconds=0.15
    )
    print(f"  Trades — downloaded: {tr_summary['downloaded']}, "
          f"missing: {tr_summary['skipped_404']}, errors: {tr_summary['errors']}")

    TRADE_DTYPES = {
        "id": "int64",
        "price": "float64",
        "qty": "float64",
        "quote_qty": "float64",
        "time": "int64",
        "is_buyer_maker": "bool",
    }

    # Merge trade daily CSVs
    for pair in pairs:
        path = TRADES_DIR / pair
        if not path.is_dir():
            print(f"  [WARN] No trades_data directory for {pair}")
            continue
        all_files = sorted(glob.glob(str(path / f"{pair}-trades-*.csv")))
        if not all_files:
            print(f"  [WARN] No daily trade CSVs found for {pair}")
            continue
        out_path = path / f"{pair}_trades_merged.csv"
        first = True
        for f in all_files:
            chunk = pd.read_csv(f, dtype=TRADE_DTYPES)
            chunk.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
            first = False
            Path(f).unlink()
        print(f"  [{pair}] merged {len(all_files)} daily trade files -> {out_path.name}")

    print("  Download step complete.")


def step_extract():
    print("\n\u2550\u2550\u2550 Step 2/3: Feature extraction \u2550\u2550\u2550")
    from feature_extraction.data_pipeline import run_pipeline
    print("  Running feature extraction pipeline ...")
    run_pipeline()
    print("  Feature extraction complete.")


def step_label(pairs):
    print("\n\u2550\u2550\u2550 Step 2b: Label generation \u2550\u2550\u2550")
    df = pd.read_csv(ALL_PAIRS_TRADES, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    for pair in pairs:
        vwap_ret = df[f"{pair}_vwap_return"]
        vwap_vol = df[f"{pair}_vwap_volatility"]
        fwd = vwap_ret.rolling(CRASH_HORIZON).sum().shift(-CRASH_HORIZON)
        df[f"{pair}_flash_crash_label"] = (fwd < -CRASH_SIGMA * vwap_vol).astype("Int8")

        n_crashes = int(df[f"{pair}_flash_crash_label"].sum())
        pct = 100 * n_crashes / len(df)
        print(f"  {pair}: {n_crashes} crash events flagged ({pct:.2f}%)")

    df = df.iloc[:-CRASH_HORIZON].reset_index(drop=True)
    df.to_csv(ALL_PAIRS_LABELED, index=False)
    print(f"  Labeled dataset saved: {ALL_PAIRS_LABELED}")
    print(f"  Rows: {len(df)}, cols: {len(df.columns)}")


def resolve_device(device_arg):
    if device_arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


def step_train(model_name, seq_len, epochs, alpha, use_conformal, device):
    print(f"\n\u2550\u2550\u2550 Step 3/3: Training ({model_name}) \u2550\u2550\u2550")
    from models import (
        SequenceDataset, HopCPT,
        MHNFlashCrashModel, STanHopModel,
        LSTMFlashCrashModel, TransformerFlashCrashModel,
        MLBaselinesModel,
    )
    from config import LABEL_PAIR

    print(f"  Loading dataset from {ALL_PAIRS_LABELED} ...")
    ds = SequenceDataset(str(ALL_PAIRS_LABELED), seq_len=seq_len, label_pair=LABEL_PAIR)
    ds.load()
    print(f"  Dataset loaded: n_features={ds.n_features}")

    ML_MODELS = {"xgboost", "random_forest", "logistic"}
    is_ml = model_name in ML_MODELS

    MODEL_MAP = {
        "mhn":           lambda: MHNFlashCrashModel(seq_len=seq_len, n_features=ds.n_features, epochs=epochs, device=device),
        "stanhop":       lambda: STanHopModel(seq_len=seq_len, n_features=ds.n_features, epochs=epochs, device=device),
        "lstm":          lambda: LSTMFlashCrashModel(seq_len=seq_len, n_features=ds.n_features, epochs=epochs, device=device),
        "transformer":   lambda: TransformerFlashCrashModel(seq_len=seq_len, n_features=ds.n_features, epochs=epochs, device=device),
        "xgboost":       lambda: MLBaselinesModel("xgboost"),
        "random_forest": lambda: MLBaselinesModel("random_forest"),
        "logistic":      lambda: MLBaselinesModel("logistic"),
    }

    model = MODEL_MAP[model_name]()

    if is_ml:
        (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = ds.get_flat_splits()
    else:
        (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = ds.get_splits()

    if use_conformal and not is_ml:
        print(f"  Wrapping {model_name} with HopCPT (alpha={alpha}) ...")
        cpt = HopCPT(model, alpha=alpha)
        cpt.fit(X_tr, y_tr, X_val=X_cal, y_val=y_cal)
        cpt.calibrate(X_cal, y_cal)
        metrics = cpt.evaluate(X_te, y_te)
        sets = cpt.predict_set(X_te)
        print(f"\n  Conformal prediction set breakdown:")
        print(f"    Crash only    (1): {(sets == 1).sum()}")
        print(f"    No crash only (0): {(sets == 0).sum()}")
        print(f"    Uncertain     (2): {(sets == 2).sum()}")
        print(f"    Empty set    (-1): {(sets == -1).sum()}")
    else:
        if use_conformal and is_ml:
            print(f"  [NOTE] HopCPT skipped for ML baseline model '{model_name}'")
        print(f"  Training {model_name} ...")
        model.fit(X_tr, y_tr)
        metrics = model.evaluate(X_te, y_te)

    return metrics


def print_summary(metrics, model_name):
    print("\n" + "\u2550" * 60)
    print(f"  Results — {model_name}")
    print("\u2550" * 60)
    col_w = max(len(k) for k in metrics) + 2
    for k, v in metrics.items():
        print(f"  {k:<{col_w}}: {v:.4f}")
    print("\u2550" * 60)


def main():
    args = parse_args()
    pairs = [p.strip().upper() for p in args.pairs.split(",")]
    period = args.period
    seq_len = args.seq_len
    epochs = args.epochs
    alpha = args.alpha
    device = resolve_device(args.device)
    use_conformal = not args.no_conformal

    print(f"\nFlash Crash Predictor")
    print(f"  Pairs:    {pairs}")
    print(f"  Period:   {period}")
    print(f"  Model:    {args.model}")
    print(f"  Device:   {device}")
    print(f"  Conformal:{use_conformal}")

    try:
        if not args.skip_download:
            step_download(pairs, period)
        else:
            print("\n[SKIP] Data download")

        if not args.skip_extract:
            step_extract()
        else:
            print("\n[SKIP] Feature extraction")

        if not args.skip_label:
            step_label(pairs)
        else:
            print("\n[SKIP] Label generation")

        metrics = step_train(
            model_name=args.model,
            seq_len=seq_len,
            epochs=epochs,
            alpha=alpha,
            use_conformal=use_conformal,
            device=device,
        )

        print_summary(metrics, args.model)

    except Exception:
        print("\n[ERROR] Pipeline failed:")
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
