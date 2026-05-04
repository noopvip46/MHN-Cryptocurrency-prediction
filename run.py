# Spike-Correction Predictor — end-to-end offline pipeline
# Detects price spikes (up & down) using CUSUM filtering and predicts
# whether they will revert (mean-reversion) or continue (momentum)
# using Triple Barrier labeling (López de Prado, AFML).
# Run with --help for all options.

import argparse
import sys
import traceback
import time

import pandas as pd
import numpy as np

from config import (
    DEFAULT_PAIRS, DEFAULT_PERIOD, SEQ_LEN,
    ALL_PAIRS_TRADES, ALL_PAIRS_LABELED,
    CUSUM_H, CUSUM_EXPECTED,
    BARRIER_MAX_HOLD, BARRIER_VOL_SPAN,
    BOOK_DEPTH_DIR, TRADES_DIR, ONCHAIN_DIR, ONCHAIN_SYMBOL,
    ROLL_WINDOW_SHORT, ROLL_WINDOW_REGIME,
)

SEP = "=" * 55


def parse_args():
    p = argparse.ArgumentParser(description="Spike-Correction Predictor — offline pipeline")

    # Data
    p.add_argument("--pairs",    default=",".join(DEFAULT_PAIRS),
                   help="Comma-separated trading pairs  (default: BTCUSDT,ETHUSDT)")
    p.add_argument("--period",   default=DEFAULT_PERIOD,
                   help="Download period e.g. 6m, 1y, 7d  (default: 6m)")

    # Model
    p.add_argument("--model",    default="stanhop",
                   choices=["mhn", "stanhop", "lstm", "transformer",
                            "xgboost", "random_forest", "logistic"],
                   help="Model architecture  (default: stanhop)")
    p.add_argument("--seq-len",  type=int, default=SEQ_LEN,
                   help=f"Sequence length in snapshots  (default: {SEQ_LEN})")
    p.add_argument("--epochs",   type=int, default=50,
                   help="Training epochs for DL models  (default: 50)")
    p.add_argument("--alpha",    type=float, default=0.1,
                   help="HopCPT miscoverage level  (default: 0.1)")

    # Skip flags
    p.add_argument("--skip-download",  action="store_true", help="Skip download step")
    p.add_argument("--skip-extract",   action="store_true", help="Skip feature extraction step")
    p.add_argument("--skip-label",     action="store_true", help="Skip label generation step")
    p.add_argument("--no-conformal",   action="store_true", help="Disable HopCPT wrapper")

    # Data flags
    p.add_argument("--no-onchain",  action="store_true",
                   help="Skip on-chain (Alchemy) data entirely — faster, no API key needed")
    p.add_argument("--no-save",     action="store_true",
                   help="Do not save intermediate CSVs — only all_pairs_labeled.csv is written. "
                        "Incompatible with --skip-extract (no checkpoint to resume from).")

    # Device
    p.add_argument("--device",   default="auto", choices=["cpu", "cuda", "auto"],
                   help="Compute device  (default: auto)")

    # Checkpointing
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   help="Directory to save per-epoch checkpoints during DL training  (default: checkpoints/)")
    p.add_argument("--resume",  default=None,
                   help="Path to a .pt checkpoint file — resumes DL training from that epoch")

    # ── Hyperparameters — shared DL ───────────────────────────────────────────
    p.add_argument("--hidden-dim",      type=int,   default=128,
                   help="Hidden/model dimension for all DL models  (default: 128)")
    p.add_argument("--n-heads",         type=int,   default=4,
                   help="Number of attention heads  (default: 4)")
    p.add_argument("--n-layers",        type=int,   default=None,
                   help="Number of stacked layers — LSTM default 2, Transformer default 3")
    p.add_argument("--dropout",         type=float, default=None,
                   help="Dropout rate — LSTM default 0.2, others default 0.1")
    p.add_argument("--lr",              type=float, default=1e-3,
                   help="Learning rate for DL optimiser  (default: 1e-3)")
    p.add_argument("--batch-size",      type=int,   default=256,
                   help="Mini-batch size for DL training  (default: 256)")

    # ── Hyperparameters — model-specific ──────────────────────────────────────
    p.add_argument("--top-k",           type=int,   default=10,
                   help="[STanHop] sparse attention top-k per query  (default: 10)")
    p.add_argument("--n-patterns",      type=int,   default=64,
                   help="[MHN] number of learnable memory patterns  (default: 64)")
    p.add_argument("--dim-feedforward", type=int,   default=256,
                   help="[Transformer] feedforward dim inside each encoder layer  (default: 256)")

    # ── Hyperparameters — XGBoost ─────────────────────────────────────────────
    p.add_argument("--xgb-n-estimators",    type=int,   default=1000,
                   help="[XGBoost] max trees (early stopping usually cuts this short)  (default: 1000)")
    p.add_argument("--xgb-max-depth",       type=int,   default=4,
                   help="[XGBoost] max tree depth — shallower reduces overfit on rare positives  (default: 4)")
    p.add_argument("--xgb-min-child-weight", type=int,  default=20,
                   help="[XGBoost] min samples per leaf — prevents splits on tiny positive subsets  (default: 20)")
    p.add_argument("--xgb-lr",              type=float, default=0.05,
                   help="[XGBoost] learning rate / eta  (default: 0.05)")
    p.add_argument("--xgb-subsample",       type=float, default=0.8,
                   help="[XGBoost] row subsampling ratio per tree  (default: 0.8)")
    p.add_argument("--xgb-colsample",       type=float, default=0.7,
                   help="[XGBoost] feature subsampling ratio per tree  (default: 0.7)")
    p.add_argument("--xgb-early-stopping",  type=int,   default=30,
                   help="[XGBoost] stop after N rounds without val improvement  (default: 30)")

    # External dataset support
    p.add_argument("--data-file",   default=None,
                   help="Path to a pre-built labeled CSV (skips download/extract/label). "
                        "Must contain timestamp, feature columns, and {label-pair}_spike_event/label.")
    p.add_argument("--label-pair",  default=None,
                   help="Pair name whose spike_label column is used as the target "
                        "(default: LABEL_PAIR from config.py, currently BTCUSDT). "
                        "Use this when running on a colleague's dataset with different pair names.")

    return p.parse_args()


# ── Step 1: Download ──────────────────────────────────────────────────────────

def step_download(pairs, period, no_onchain: bool = False):
    print(f"\n{SEP}")
    print(f"  Step 1/4 — Download")
    print(f"  Pairs: {pairs}   Period: {period}")
    print(SEP)

    from data_collection import book_depth_utils, trades_utils

    # ── Book depth ────────────────────────────────────────────────────────────
    # Downloads ZIP per day → pivots long→wide immediately → deletes raw CSV.
    # Peak disk: one day's ZIP + raw CSV (~10 MB). Final: {PAIR}_merged.csv (~50 MB/pair).
    print(f"\n  [1/3] Book depth")
    t0 = time.time()
    bd = book_depth_utils.download_book_depth_range(
        pairs, period, out_base=str(BOOK_DEPTH_DIR), pause_seconds=0.1
    )
    print(f"  Book depth done in {time.time()-t0:.0f}s — "
          f"ok: {bd['downloaded']}  missing: {bd['skipped_404']}  errors: {bd['errors']}")

    # ── Trades ────────────────────────────────────────────────────────────────
    # Downloads ZIP per day → aggregates ~500 MB raw CSV to 30 s bins (~100 KB) → deletes raw.
    # Peak disk: one day's ZIP + raw CSV (~600 MB). Final: {PAIR}_trades_agg.csv (~20 MB/pair).
    print(f"\n  [2/3] Trades")
    t0 = time.time()
    tr = trades_utils.download_trades_range(
        pairs, period, out_base=str(TRADES_DIR), pause_seconds=0.15
    )
    print(f"  Trades done in {time.time()-t0:.0f}s — "
          f"ok: {tr['downloaded']}  missing: {tr['skipped_404']}  errors: {tr['errors']}")

    # ── On-chain (optional) ───────────────────────────────────────────────────
    if no_onchain:
        print("\n  [3/3] On-chain skipped (--no-onchain)")
    else:
        print(f"\n  [3/3] On-chain (Alchemy)")
        try:
            from data_collection.onchain_utils import AlchemyClient, download_onchain_range
            client = AlchemyClient.from_env()
            t0 = time.time()
            oc = download_onchain_range(
                period, out_base=str(ONCHAIN_DIR), symbol=ONCHAIN_SYMBOL, client=client
            )
            print(f"  On-chain done in {time.time()-t0:.0f}s — "
                  f"ok: {oc['downloaded']}  skipped: {oc['skipped']}  errors: {oc['errors']}")
        except EnvironmentError as e:
            print(f"  [SKIP] On-chain — {e}")
            print("  Tip: set ALCHEMY_API_KEY in .env or use --no-onchain to suppress this warning.")

    print(f"\n  Download complete.")


# ── Step 2: Feature extraction ────────────────────────────────────────────────

def step_extract(pairs, no_onchain: bool = False, save_intermediates: bool = False):
    print(f"\n{SEP}")
    print(f"  Step 2/4 — Feature extraction")
    print(SEP)

    if save_intermediates:
        print("  [INFO] --save-intermediates: will write all_pairs_cleaned.csv "
              "and all_pairs_with_trades.csv")

    from feature_extraction.data_pipeline import run_pipeline
    t0 = time.time()
    features_df = run_pipeline(
        trading_pairs      = pairs,
        out_base           = str(BOOK_DEPTH_DIR),
        trades_base        = str(TRADES_DIR),
        roll_window        = ROLL_WINDOW_SHORT,
        regime_window      = ROLL_WINDOW_REGIME,
        onchain_base       = str(ONCHAIN_DIR),
        onchain_symbol     = ONCHAIN_SYMBOL,
        skip_onchain       = no_onchain,
        save_intermediates = save_intermediates,
    )
    print(f"  Feature extraction complete in {time.time()-t0:.0f}s.")
    return features_df


# ── Step 3: Label generation ──────────────────────────────────────────────────

def _cusum_filter(returns, h, expected=0.0):
    """CUSUM event filter (López de Prado AFML Ch. 2).

    Maintains two running cumulative sums tracking upward and downward
    deviations from the expected return.  When either sum exceeds the
    threshold h, a spike event is recorded and both sums are reset.

    Returns
    -------
    events : list of (index, direction, magnitude)
        index     — row index where the spike was detected
        direction — +1 for upward spike, -1 for downward spike
        magnitude — absolute value of the cumulative sum at trigger (≥ h)
    """
    s_pos, s_neg = 0.0, 0.0
    events = []
    for i, r in enumerate(returns):
        diff = r - expected
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)
        if s_pos > h:
            events.append((i, +1, s_pos))
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg < -h:
            events.append((i, -1, abs(s_neg)))
            s_pos = 0.0
            s_neg = 0.0
    return events


def _volatility_label(returns, event_idx, max_hold):
    """Post-spike volatility labeling.

    After a spike event, measure the realized volatility over the next max_hold
    snapshots.  Label = 1 if the post-spike period is highly volatile (above
    median for the dataset), label = 0 if the spike is quickly absorbed and
    the market calms down.

    This reframes the prediction task from "direction" (unpredictable) to
    "activity regime" (predictable from order book state):
    - Thin books + volume surge → continued high volatility (label=1)
    - Deep books + low intensity → spike absorbed (label=0)

    Parameters
    ----------
    returns   : array-like, per-snapshot returns
    event_idx : int, row where the spike was detected
    max_hold  : int, forward window (snapshots) over which to measure volatility

    Returns
    -------
    realized_vol : float
        Standard deviation of returns in the forward window
    """
    T = len(returns)
    end = min(event_idx + max_hold + 1, T)
    fwd_returns = returns[event_idx + 1 : end]

    if len(fwd_returns) < 3:
        return 0.0

    return float(np.std(fwd_returns))


def step_label(pairs, features_df=None):
    """Generate spike-correction labels using CUSUM filter + Triple Barrier.

    Pipeline (López de Prado, Advances in Financial Machine Learning):
    1. CUSUM filter detects spike events (both up and down) in vwap_return.
    2. For each spike event, Triple Barrier determines whether the price
       reverts (mean-reversion → label=1) or continues/times-out (label=0).
    3. Events from all pairs are combined; the label from the primary pair
       (LABEL_PAIR) is used as the training target.

    Spike events are a small subset of all rows — models train only on these
    event rows, not every 30-second snapshot.

    If features_df is provided (in-memory from step_extract), labels are applied
    directly.  Otherwise all_pairs_with_trades.csv is read from disk.
    """
    print(f"\n{SEP}")
    print(f"  Step 3/4 — Label generation (CUSUM h={CUSUM_H}, "
          f"forward-return reversal, window={BARRIER_MAX_HOLD} snapshots)")
    print(SEP)

    if features_df is not None:
        df = features_df.sort_values("timestamp").reset_index(drop=True)
    else:
        if not ALL_PAIRS_TRADES.exists():
            raise FileNotFoundError(
                f"{ALL_PAIRS_TRADES} not found.\n"
                "Re-run without --skip-extract, or use --save-intermediates on the extract step."
            )
        df = pd.read_csv(ALL_PAIRS_TRADES, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    from config import LABEL_PAIR

    for pair in pairs:
        ret_col = f"{pair}_vwap_return"
        vol_col = f"{pair}_vwap_volatility"

        if ret_col not in df.columns:
            print(f"  [WARN] {ret_col} not found, skipping {pair}")
            continue

        returns = df[ret_col].fillna(0.0).values
        vol     = df[vol_col].fillna(0.01).values if vol_col in df.columns else \
                  pd.Series(returns).rolling(BARRIER_VOL_SPAN, min_periods=30).std().fillna(0.01).values

        # 1) CUSUM filter — detect spike events
        events = _cusum_filter(returns, h=CUSUM_H, expected=CUSUM_EXPECTED)
        print(f"  {pair}: CUSUM detected {len(events)} spike events "
              f"(up={sum(1 for _,d,_ in events if d>0)}, "
              f"down={sum(1 for _,d,_ in events if d<0)})")

        # 2) Post-spike volatility measurement
        spike_event = np.zeros(len(df), dtype=np.int8)
        spike_dir   = np.zeros(len(df), dtype=np.int8)
        spike_mag   = np.zeros(len(df), dtype=np.float32)
        spike_fwd_vol = np.zeros(len(df), dtype=np.float32)

        # Skip events too close to the end (need max_hold snapshots forward)
        valid_events = [(idx, d, mag) for idx, d, mag in events
                        if idx + BARRIER_MAX_HOLD < len(df)]

        for idx, direction, magnitude in valid_events:
            fwd_vol = _volatility_label(returns, idx, max_hold=BARRIER_MAX_HOLD)
            spike_event[idx] = 1
            spike_dir[idx]   = direction
            spike_mag[idx]   = magnitude
            spike_fwd_vol[idx] = fwd_vol

        # Label by median split: high post-spike volatility (1) vs low (0)
        event_vols = spike_fwd_vol[spike_event == 1]
        vol_median = np.median(event_vols) if len(event_vols) > 0 else 0.0
        spike_label = np.where(
            (spike_event == 1) & (spike_fwd_vol > vol_median), 1, 0
        ).astype(np.int8)

        df[f"{pair}_spike_event"]     = pd.array(spike_event, dtype="Int8")
        df[f"{pair}_spike_label"]     = pd.array(spike_label, dtype="Int8")
        df[f"{pair}_spike_direction"] = pd.array(spike_dir, dtype="Int8")

        # ── Spike-characterizing MODEL features (not dropped by data_adapter) ─
        df[f"{pair}_spike_magnitude"] = spike_mag
        df[f"{pair}_spike_dir_signed"] = spike_dir.astype(np.float32)
        local_vol_at_event = np.where(spike_event == 1, vol, np.nan)
        vol_norm = np.where(spike_event == 1,
                            spike_mag / np.maximum(local_vol_at_event, 1e-6), 0.0)
        df[f"{pair}_spike_vol_normalized"] = vol_norm.astype(np.float32)

        n_events  = int(spike_event.sum())
        n_high    = int(spike_label.sum())
        n_low     = n_events - n_high
        print(f"  {pair}: {n_events} valid events → "
              f"high_vol={n_high} (50%)  low_vol={n_low} (50%)"
              f"  median_vol={vol_median:.6f}"
              f"  (fwd window={BARRIER_MAX_HOLD} snapshots = {BARRIER_MAX_HOLD*30//60} min)")

    # ── For the primary label pair, copy its spike columns as the training target
    primary_event_col = f"{LABEL_PAIR}_spike_event"
    primary_label_col = f"{LABEL_PAIR}_spike_label"

    if primary_event_col not in df.columns:
        raise KeyError(f"Primary label pair {LABEL_PAIR} has no spike events. "
                       f"Check that {LABEL_PAIR} is in --pairs.")

    # Trim rows at the end that can't have valid forward barriers
    df = df.iloc[:-BARRIER_MAX_HOLD].reset_index(drop=True)

    df.to_csv(ALL_PAIRS_LABELED, index=False)
    n_events_total = int(df[primary_event_col].sum())
    n_high_total = int(df.loc[df[primary_event_col] == 1, primary_label_col].sum())
    print(f"\n  Saved: {ALL_PAIRS_LABELED}  rows={len(df):,}  cols={len(df.columns)}")
    print(f"  Training target ({LABEL_PAIR}): {n_events_total} events, "
          f"high_vol={n_high_total}, low_vol={n_events_total - n_high_total}")
    return df


# ── Step 4: Train ─────────────────────────────────────────────────────────────

def resolve_device(arg):
    if arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return arg


def step_train(model_name, seq_len, epochs, alpha, use_conformal, device,
               checkpoint_dir=None, resume_from=None, data_file=None, label_pair=None,
               hparams=None):
    print(f"\n{SEP}")
    print(f"  Step 4/4 — Train ({model_name})  device={device}")
    print(SEP)

    from models import (
        SequenceDataset, HopCPT,
        MHNFlashCrashModel, STanHopModel,
        LSTMFlashCrashModel, TransformerFlashCrashModel,
        MLBaselinesModel,
    )
    from config import LABEL_PAIR as _DEFAULT_LABEL_PAIR

    csv_path   = data_file  if data_file   else str(ALL_PAIRS_LABELED)
    label_pair = label_pair if label_pair  else _DEFAULT_LABEL_PAIR

    print(f"  Loading {csv_path}  (label_pair={label_pair}) ...")
    ds = SequenceDataset(csv_path, seq_len=seq_len, label_pair=label_pair)
    ds.load()
    print(f"  n_features={ds.n_features}  seq_len={seq_len}")

    ML_MODELS = {"xgboost", "random_forest", "logistic"}
    is_ml     = model_name in ML_MODELS

    hp = hparams or {}   # dict of overrides; absent keys fall back to model defaults

    MODEL_MAP = {
        # hp.get(key) or default  — the "or" handles None (flag not passed) correctly.
        # dict.get(key, default) only falls back when the key is ABSENT; since we
        # always insert every key (even when the CLI flag was not supplied, value=None),
        # we must use "or" so None values also fall back to the model's own default.
        "mhn": lambda: MHNFlashCrashModel(
            seq_len=seq_len, n_features=ds.n_features, epochs=epochs, device=device,
            hidden_dim        = hp.get("hidden_dim")  or 128,
            n_heads           = hp.get("n_heads")     or 4,
            n_memory_patterns = hp.get("n_patterns")  or 64,
            dropout           = hp.get("dropout")     or 0.1,
            lr                = hp.get("lr")          or 1e-3,
            batch_size        = hp.get("batch_size")  or 256,
        ),
        "stanhop": lambda: STanHopModel(
            seq_len=seq_len, n_features=ds.n_features, epochs=epochs, device=device,
            hidden_dim = hp.get("hidden_dim") or 128,
            n_heads    = hp.get("n_heads")    or 4,
            top_k      = hp.get("top_k")      or 10,
            dropout    = hp.get("dropout")    or 0.1,
            lr         = hp.get("lr")         or 1e-3,
            batch_size = hp.get("batch_size") or 256,
        ),
        "lstm": lambda: LSTMFlashCrashModel(
            seq_len=seq_len, n_features=ds.n_features, epochs=epochs, device=device,
            hidden_dim = hp.get("hidden_dim") or 128,
            n_layers   = hp.get("n_layers")   or 2,
            dropout    = hp.get("dropout")    or 0.2,
            lr         = hp.get("lr")         or 1e-3,
            batch_size = hp.get("batch_size") or 256,
        ),
        "transformer": lambda: TransformerFlashCrashModel(
            seq_len=seq_len, n_features=ds.n_features, epochs=epochs, device=device,
            d_model         = hp.get("hidden_dim")      or 128,
            n_heads         = hp.get("n_heads")         or 4,
            n_layers        = hp.get("n_layers")        or 3,
            dim_feedforward = hp.get("dim_feedforward") or 256,
            dropout         = hp.get("dropout")         or 0.1,
            lr              = hp.get("lr")              or 1e-3,
            batch_size      = hp.get("batch_size")      or 256,
        ),
        "xgboost": lambda: MLBaselinesModel(
            "xgboost", device=device,
            n_estimators          = hp.get("xgb_n_estimators")    or 1000,
            max_depth             = hp.get("xgb_max_depth")        or 4,
            min_child_weight      = hp.get("xgb_min_child_weight") or 20,
            learning_rate         = hp.get("xgb_lr")               or 0.05,
            subsample             = hp.get("xgb_subsample")        or 0.8,
            colsample_bytree      = hp.get("xgb_colsample")        or 0.7,
            early_stopping_rounds = hp.get("xgb_early_stopping")   or 30,
        ),
        "random_forest": lambda: MLBaselinesModel("random_forest", device=device),
        "logistic":      lambda: MLBaselinesModel("logistic",      device=device),
    }

    model = MODEL_MAP[model_name]()

    if is_ml:
        (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = ds.get_flat_splits()
    else:
        (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = ds.get_splits()

    # DL training kwargs — passed into fit() for checkpoint saving / resuming
    dl_kwargs = {}
    if not is_ml:
        dl_kwargs["checkpoint_dir"] = checkpoint_dir
        dl_kwargs["resume_from"]    = resume_from
        if checkpoint_dir:
            print(f"  Checkpoints → {checkpoint_dir}/  (overwritten each epoch)")
        if resume_from:
            print(f"  Resuming from: {resume_from}")

    try:
        if use_conformal and not is_ml:
            print(f"  Wrapping with HopCPT (alpha={alpha}) ...")
            cpt = HopCPT(model, alpha=alpha)
            cpt.fit(X_tr, y_tr, X_val=X_cal, y_val=y_cal, **dl_kwargs)
            cpt.calibrate(X_cal, y_cal)
            metrics = cpt.evaluate(X_te, y_te)
            sets    = cpt.predict_set(X_te)
            print(f"  Conformal set breakdown:")
            print(f"    Crash only    (1): {(sets == 1).sum()}")
            print(f"    No crash only (0): {(sets == 0).sum()}")
            print(f"    Uncertain     (2): {(sets == 2).sum()}")
            print(f"    Empty set    (-1): {(sets == -1).sum()}")
        else:
            if use_conformal and is_ml:
                print(f"  [NOTE] HopCPT skipped for ML baseline '{model_name}'")
            print(f"  Training ...")
            t0 = time.time()
            if is_ml:
                # Always pass val split so XGBoost can report eval metrics during training
                model.fit(X_tr, y_tr, X_val=X_cal, y_val=y_cal)
            else:
                model.fit(X_tr, y_tr, X_val=X_cal, y_val=y_cal, **dl_kwargs)
            print(f"  Training done in {time.time()-t0:.0f}s")
            metrics = model.evaluate(X_te, y_te)

    except KeyboardInterrupt:
        print("\n  Training stopped by user.  Checkpoint saved (see above).")
        sys.exit(0)

    return metrics


def print_summary(metrics, model_name):
    print(f"\n{SEP}")
    print(f"  Results — {model_name}")
    print(SEP)
    col_w = max(len(k) for k in metrics) + 2
    for k, v in metrics.items():
        print(f"  {k:<{col_w}}: {v:.4f}")
    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args          = parse_args()
    pairs         = [p.strip().upper() for p in args.pairs.split(",")]
    device        = resolve_device(args.device)
    use_conformal = not args.no_conformal
    save_inter    = not args.no_save

    # Warn about incompatible flag combo
    if args.no_save and args.skip_extract:
        print("[WARN] --no-save + --skip-extract: no intermediate file to resume from.")
        print("       Remove --skip-extract or add --save-intermediates on the previous run.")

    print(f"\nSpike-Correction Predictor")
    print(f"  Pairs      : {pairs}")
    print(f"  Period     : {args.period}")
    print(f"  Model      : {args.model}  device={device}  conformal={use_conformal}")
    print(f"  Labeling   : CUSUM (h={CUSUM_H}) + forward-return reversal (window={BARRIER_MAX_HOLD} snapshots = {BARRIER_MAX_HOLD*30//60} min)")
    print(f"  On-chain   : {'disabled' if args.no_onchain else 'enabled'}")
    print(f"  Intermediates saved: {save_inter}")
    if args.resume:
        print(f"  Resume from: {args.resume}")

    try:
        t_total = time.time()
        features_df = None   # passed in-memory when --no-save

        if not args.skip_download:
            step_download(pairs, args.period, no_onchain=args.no_onchain)
        else:
            print("\n[SKIP] Download")

        if not args.skip_extract:
            features_df = step_extract(
                pairs,
                no_onchain=args.no_onchain,
                save_intermediates=not args.no_save,
            )
        else:
            print("\n[SKIP] Feature extraction")

        if not args.skip_label:
            step_label(pairs, features_df=features_df)
        else:
            print("\n[SKIP] Label generation")

        metrics = step_train(
            model_name      = args.model,
            seq_len         = args.seq_len,
            epochs          = args.epochs,
            alpha           = args.alpha,
            use_conformal   = use_conformal,
            device          = device,
            checkpoint_dir  = args.checkpoint_dir,
            resume_from     = args.resume,
            data_file       = args.data_file,
            label_pair      = args.label_pair,
            hparams         = {
                "hidden_dim":        args.hidden_dim,
                "n_heads":           args.n_heads,
                "n_layers":          args.n_layers,
                "dropout":           args.dropout,
                "lr":                args.lr,
                "batch_size":        args.batch_size,
                "top_k":             args.top_k,
                "n_patterns":        args.n_patterns,
                "dim_feedforward":   args.dim_feedforward,
                "xgb_n_estimators":    args.xgb_n_estimators,
                "xgb_max_depth":       args.xgb_max_depth,
                "xgb_min_child_weight": args.xgb_min_child_weight,
                "xgb_lr":              args.xgb_lr,
                "xgb_subsample":       args.xgb_subsample,
                "xgb_colsample":       args.xgb_colsample,
                "xgb_early_stopping":  args.xgb_early_stopping,
            },
        )
        print_summary(metrics, args.model)
        print(f"\n  Total time: {time.time()-t_total:.0f}s")

    except Exception:
        print("\n[ERROR] Pipeline failed:")
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
