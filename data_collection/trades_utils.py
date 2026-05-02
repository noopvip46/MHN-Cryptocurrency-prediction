"""
data_collection/trades_utils.py

Downloads Binance futures trade data and immediately aggregates each day to
30-second bins.  The raw tick-by-tick CSV (~500 MB/day) is deleted the moment
aggregation is done — only the ~2 880-row daily summary survives on disk.

Peak disk use at any moment: one day's ZIP (~80 MB) + one day's raw CSV
(~500 MB) = under 600 MB total, regardless of the period length.

Final output per pair: {PAIR}_trades_agg.csv  (~20 MB for 6 months)
Columns: timestamp, trade_count, trade_volume, trade_notional, buy_volume, sell_volume
"""

import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BINANCE_DATA_BASE = "https://data.binance.vision/data/futures/um/daily/trades"

_TRADE_DTYPES = {
    "id":             "int64",
    "price":          "float64",
    "qty":            "float64",
    "quote_qty":      "float64",
    "time":           "int64",
    "is_buyer_maker": "bool",
}

AGG_COLUMNS = [
    "timestamp",
    "trade_count",
    "trade_volume",
    "trade_notional",
    "buy_volume",
    "sell_volume",
]


# ── Period helpers ────────────────────────────────────────────────────────────

def _parse_period(period: str):
    period = period.strip().lower()
    today  = datetime.utcnow().date()
    if "/" in period:
        s, e  = period.split("/", 1)
        start = datetime.strptime(s.strip(), "%Y-%m-%d").date()
        end   = datetime.strptime(e.strip(), "%Y-%m-%d").date()
        if end < start:
            raise ValueError("End date must be after start date.")
        return start, end
    if period.endswith("d"):
        end = today - timedelta(days=1)
        return end - timedelta(days=int(period[:-1]) - 1), end
    if period.endswith("w"):
        end = today - timedelta(days=1)
        return end - timedelta(days=int(period[:-1]) * 7 - 1), end
    if period.endswith("m"):
        import calendar
        months = int(period[:-1])
        end    = today - timedelta(days=1)
        y, m   = end.year, end.month
        for _ in range(months):
            m -= 1
            if m == 0:
                m = 12; y -= 1
        _, last = calendar.monthrange(y, m)
        return datetime(y, m, min(end.day, last)).date(), end
    if period.endswith("y"):
        end = today - timedelta(days=1)
        return datetime(end.year - int(period[:-1]), end.month, end.day).date(), end
    d = datetime.strptime(period, "%Y-%m-%d").date()
    return d, d


def _date_list(start, end):
    cur = start
    while cur <= end:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


# ── Core aggregation ──────────────────────────────────────────────────────────

def _aggregate_to_30s(raw_csv: Path) -> pd.DataFrame:
    """Read one day's raw trades CSV and collapse to 30-second bins.

    Input : millions of tick rows (~500 MB)
    Output: ~2 880 rows, 6 columns (~100 KB)

    The raw file is NOT deleted here — caller handles cleanup.
    """
    df = pd.read_csv(raw_csv, dtype=_TRADE_DTYPES)

    df["buy_qty"]  = df["qty"].where(~df["is_buyer_maker"].astype(bool), 0.0)
    df["sell_qty"] = df["qty"].where( df["is_buyer_maker"].astype(bool), 0.0)
    df["ts_30s"]   = df["time"] // 30_000 * 30_000   # floor to 30 s in Unix ms

    agg = (
        df.groupby("ts_30s", sort=True)
        .agg(
            trade_count    = ("id",        "count"),
            trade_volume   = ("qty",       "sum"),
            trade_notional = ("quote_qty", "sum"),
            buy_volume     = ("buy_qty",   "sum"),
            sell_volume    = ("sell_qty",  "sum"),
        )
        .reset_index()
    )
    agg["timestamp"] = pd.to_datetime(agg["ts_30s"], unit="ms")
    return agg[AGG_COLUMNS]


# ── Single-day downloader ─────────────────────────────────────────────────────

def download_trades(
    symbol: str,
    date_str: str,
    out_base: str = "trades_data",
    delete_zip: bool = True,
    session=None,
) -> Path | None:
    """Download one day of trades, aggregate to 30 s bins, delete raw CSV.

    Returns path to the daily aggregated CSV, or None if day not available.
    At peak: ZIP + raw CSV on disk simultaneously (~600 MB), both gone after.
    Output: {out_base}/{SYMBOL}/{SYMBOL}-trades-agg-{YYYY-MM-DD}.csv
    """
    symbol  = symbol.strip().upper()
    out_dir = Path(out_base) / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_csv = out_dir / f"{symbol}-trades-agg-{date_str}.csv"
    if agg_csv.exists():
        return agg_csv

    zip_name  = f"{symbol}-trades-{date_str}.zip"
    local_zip = out_dir / zip_name
    url       = f"{BINANCE_DATA_BASE}/{symbol}/{zip_name}"
    s         = session or requests.Session()

    if not local_zip.exists():
        resp = s.get(url, stream=True, timeout=60)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        with open(local_zip, "wb") as fh:
            for chunk in resp.iter_content(65_536):
                if chunk:
                    fh.write(chunk)

    with zipfile.ZipFile(local_zip, "r") as z:
        z.extractall(out_dir)
    if delete_zip:
        local_zip.unlink(missing_ok=True)

    raw_csv = out_dir / f"{symbol}-trades-{date_str}.csv"
    agg_df  = _aggregate_to_30s(raw_csv)       # 500 MB → ~100 KB in memory
    agg_df.to_csv(agg_csv, index=False)
    raw_csv.unlink(missing_ok=True)             # raw gone immediately

    return agg_csv


# ── Range downloader ──────────────────────────────────────────────────────────

def download_trades_range(
    symbols,
    period: str,
    out_base: str = "trades_data",
    pause_seconds: float = 0.15,
) -> dict:
    """Download and aggregate trades for all symbols over a date range.

    Shows a live progress counter.  At no point does more than one day's
    raw data exist on disk (~600 MB peak, not GB or TB).

    Daily aggregated CSVs are merged into {SYMBOL}_trades_agg.csv at the end
    and the daily files are deleted — final footprint ~20 MB for 6 months.
    """
    start_date, end_date = _parse_period(period)
    yesterday = datetime.utcnow().date() - timedelta(days=1)
    end_date  = min(end_date, yesterday)
    if start_date > end_date:
        raise ValueError("Period resolves to zero days after clamping to yesterday.")

    symbols = [s.strip().upper() for s in symbols if s.strip()]
    dates   = list(_date_list(start_date, end_date))
    total   = len(dates)

    result = {
        "requested_symbols": symbols,
        "period":            period,
        "start_date":        start_date.isoformat(),
        "end_date":          end_date.isoformat(),
        "days_requested":    total,
        "downloaded":        0,
        "skipped_404":       0,
        "errors":            0,
        "details":           [],
    }

    session = requests.Session()

    for symbol in symbols:
        daily_paths = []
        ok = skipped = errors = 0
        t0 = time.time()

        print(f"  [{symbol}] trades: 0/{total}", end="", flush=True)

        for i, date_str in enumerate(dates, 1):
            time.sleep(pause_seconds)
            try:
                path = download_trades(
                    symbol, date_str,
                    out_base=out_base, delete_zip=True, session=session,
                )
                if path is None:
                    skipped += 1
                    result["details"].append(f"{symbol} {date_str}: missing")
                else:
                    ok += 1
                    daily_paths.append(path)
                    result["details"].append(f"{symbol} {date_str}: ok")
            except Exception as ex:
                errors += 1
                result["details"].append(f"{symbol} {date_str}: error {ex}")

            elapsed = time.time() - t0
            rate    = i / elapsed if elapsed > 0 else 0
            eta     = (total - i) / rate if rate > 0 else 0
            print(
                f"\r  [{symbol}] trades: {i}/{total}  "
                f"ok={ok}  missing={skipped}  err={errors}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s   ",
                end="", flush=True,
            )

        print()   # newline after progress line

        result["downloaded"]  += ok
        result["skipped_404"] += skipped
        result["errors"]      += errors

        # Merge daily agg files into one file per symbol, delete dailies
        if daily_paths:
            merged = pd.concat(
                [pd.read_csv(p, parse_dates=["timestamp"]) for p in daily_paths],
                ignore_index=True,
            ).sort_values("timestamp").reset_index(drop=True)

            out_path = Path(out_base) / symbol / f"{symbol}_trades_agg.csv"
            merged.to_csv(out_path, index=False)

            for p in daily_paths:
                p.unlink(missing_ok=True)

            size_mb = out_path.stat().st_size / 1_048_576
            print(
                f"  [{symbol}] trades_agg.csv saved  "
                f"rows={len(merged):,}  size={size_mb:.1f} MB"
            )

    return result
