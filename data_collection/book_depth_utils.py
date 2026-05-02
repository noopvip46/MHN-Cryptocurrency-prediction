"""
data_collection/book_depth_utils.py

Downloads Binance futures book-depth snapshots and immediately pivots each
daily CSV from long format (one row per level) to wide format (one row per
snapshot).  The raw long-format CSV is deleted as soon as the pivot is done.

Raw format  : ~57 600 rows/day  (2 880 snapshots × ~20 price levels)
Pivoted fmt : ~2 880 rows/day   (one row per 30 s snapshot, all levels as cols)

Peak disk use at any moment: one day's ZIP + one day's raw CSV (~10 MB total).
Final output per pair: {PAIR}_merged.csv  (~50 MB for 6 months)
"""

import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BINANCE_DATA_BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"


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


# ── Core pivot ────────────────────────────────────────────────────────────────

def _pivot_daily(raw_csv: Path, symbol: str) -> pd.DataFrame:
    """Pivot one day's long-format book-depth CSV to wide format.

    Input : ~57 600 rows (timestamp, percentage, depth, notional)
    Output: ~2 880 rows (timestamp, {SYMBOL}_depth_{pct}, {SYMBOL}_notional_{pct}, …)

    Percentage values are normalised to 1-decimal floats before pivoting so that
    Binance's inconsistent formatting (e.g. -5 vs -5.0 across days) never
    produces duplicate columns.

    Caller is responsible for deleting raw_csv afterwards.
    """
    df = pd.read_csv(raw_csv, parse_dates=["timestamp"])
    # Normalise: -5 (int-like) and -5.0 (float) → -5.0; keep sub-integer levels
    df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce").round(1)
    df = df.dropna(subset=["percentage"])
    pivoted = df.pivot_table(
        index="timestamp",
        columns="percentage",
        values=["depth", "notional"],
        aggfunc="first",
    )
    pivoted.columns = [f"{symbol}_{v}_{k}" for v, k in pivoted.columns]
    return pivoted.reset_index()


# ── Single-day downloader ─────────────────────────────────────────────────────

def download_book_depth(
    symbol: str,
    date_str: str,
    out_base: str = "bookDepth_data",
    delete_zip: bool = True,
    session=None,
) -> Path | None:
    """Download one day of book depth, pivot to wide format, delete raw CSV.

    Returns path to the pivoted CSV, or None if day not available.
    Peak disk: ZIP + raw CSV (~10 MB) — both gone after this call returns.
    Output: {out_base}/{SYMBOL}/{SYMBOL}-bookDepth-pivoted-{YYYY-MM-DD}.csv
    """
    symbol  = symbol.strip().upper()
    out_dir = Path(out_base) / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    pivoted_csv = out_dir / f"{symbol}-bookDepth-pivoted-{date_str}.csv"
    if pivoted_csv.exists():
        return pivoted_csv

    zip_name  = f"{symbol}-bookDepth-{date_str}.zip"
    local_zip = out_dir / zip_name
    url       = f"{BINANCE_DATA_BASE}/{symbol}/{zip_name}"
    s         = session or requests.Session()

    if not local_zip.exists():
        resp = s.get(url, stream=True, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        with open(local_zip, "wb") as fh:
            for chunk in resp.iter_content(8_192):
                if chunk:
                    fh.write(chunk)

    with zipfile.ZipFile(local_zip, "r") as z:
        z.extractall(out_dir)
    if delete_zip:
        local_zip.unlink(missing_ok=True)

    raw_csv = out_dir / f"{symbol}-bookDepth-{date_str}.csv"
    pivoted = _pivot_daily(raw_csv, symbol)
    pivoted.to_csv(pivoted_csv, index=False)
    raw_csv.unlink(missing_ok=True)             # raw gone immediately

    return pivoted_csv


# ── Range downloader ──────────────────────────────────────────────────────────

def download_book_depth_range(
    symbols,
    period: str,
    out_base: str = "bookDepth_data",
    pause_seconds: float = 0.1,
) -> dict:
    """Download and pivot book depth for all symbols over a date range.

    Shows a live progress counter.  At no point does more than one day's
    raw data exist on disk (~10 MB peak).

    Daily pivoted CSVs are merged into {SYMBOL}_merged.csv at the end and
    the daily files are deleted — final footprint ~50 MB for 6 months.
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

        print(f"  [{symbol}] book depth: 0/{total}", end="", flush=True)

        for i, date_str in enumerate(dates, 1):
            time.sleep(pause_seconds)
            try:
                path = download_book_depth(
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
                f"\r  [{symbol}] book depth: {i}/{total}  "
                f"ok={ok}  missing={skipped}  err={errors}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s   ",
                end="", flush=True,
            )

        print()   # newline after progress line

        result["downloaded"]  += ok
        result["skipped_404"] += skipped
        result["errors"]      += errors

        # Merge daily pivoted CSVs into one file per symbol, delete dailies
        if daily_paths:
            merged = pd.concat(
                [pd.read_csv(p, parse_dates=["timestamp"]) for p in daily_paths],
                ignore_index=True,
            ).sort_values("timestamp").reset_index(drop=True)

            out_path = Path(out_base) / symbol / f"{symbol}_merged.csv"
            merged.to_csv(out_path, index=False)

            for p in daily_paths:
                p.unlink(missing_ok=True)

            size_mb = out_path.stat().st_size / 1_048_576
            print(
                f"  [{symbol}] merged.csv saved  "
                f"rows={len(merged):,}  cols={len(merged.columns)}  size={size_mb:.1f} MB"
            )

    return result
