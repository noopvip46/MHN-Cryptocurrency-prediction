from datetime import datetime, timedelta
from pathlib import Path
import zipfile
import requests
import time

BINANCE_DATA_BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"


def _parse_period(period: str):
    """Parse a user period string into a (start_date, end_date) pair.
    Accepts:
      - '7d', '1w', '2w', '1m', '3m', '1y', etc.
      - 'YYYY-MM-DD' (single day)
      - 'YYYY-MM-DD/YYYY-MM-DD' range.
    """
    period = period.strip().lower()
    today = datetime.utcnow().date()
    if "/" in period:
        start_text, end_text = [p.strip() for p in period.split("/", 1)]
        start = datetime.strptime(start_text, "%Y-%m-%d").date()
        end = datetime.strptime(end_text, "%Y-%m-%d").date()
        if end < start:
            raise ValueError("End date must be after start date.")
        return start, end

    if period.endswith("d"):
        days = int(period[:-1])
        end = today - timedelta(days=1)
        start = end - timedelta(days=days - 1)
        return start, end
    if period.endswith("w"):
        weeks = int(period[:-1])
        end = today - timedelta(days=1)
        start = end - timedelta(days=weeks * 7 - 1)
        return start, end
    if period.endswith("m"):
        months = int(period[:-1])
        end = today - timedelta(days=1)
        year = end.year
        month = end.month
        while months > 0:
            month -= 1
            if month == 0:
                month = 12
                year -= 1
            months -= 1
        # clamp day to month length
        import calendar
        _, last_day = calendar.monthrange(year, month)
        day = min(end.day, last_day)
        start = datetime(year, month, day).date()
        return start, end
    if period.endswith("y"):
        years = int(period[:-1])
        end = today - timedelta(days=1)
        start = datetime(end.year - years, end.month, end.day).date()
        return start, end

    # Try parse single date
    start = datetime.strptime(period, "%Y-%m-%d").date()
    return start, start


def _date_list(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


def download_book_depth(symbol: str, date_str: str, out_base: str = "bookDepth_data", delete_zip=True, session=None):
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("Symbol is required")

    out_dir = Path(out_base) / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol}-bookDepth-{date_str}.zip"
    local_zip = out_dir / filename
    url = f"{BINANCE_DATA_BASE}/{symbol}/{filename}"
    s = session or requests.Session()

    if not local_zip.exists():
        resp = s.get(url, stream=True, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        with open(local_zip, "wb") as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)

    with zipfile.ZipFile(local_zip, "r") as z:
        z.extractall(out_dir)

    if delete_zip and local_zip.exists():
        local_zip.unlink()

    return out_dir


def download_book_depth_range(symbols, period: str, out_base: str = "bookDepth_data", pause_seconds: float = 0.1):
    start_date, end_date = _parse_period(period)
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    if end_date >= today:
        end_date = yesterday
    if start_date > end_date:
        raise ValueError("Invalid period: after adjusting for latest available date, no past days remain. Please pick an earlier date or shorter period.")

    result = {
        "requested_symbols": [s.strip().upper() for s in symbols if s.strip()],
        "period": period,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "days_requested": (end_date - start_date).days + 1,
        "downloaded": 0,
        "skipped_404": 0,
        "errors": 0,
        "details": [],
    }
    session = requests.Session()

    for symbol in result["requested_symbols"]:
        for date_str in _date_list(start_date, end_date):
            time.sleep(pause_seconds)
            try:
                extracted = download_book_depth(symbol, date_str, out_base=out_base, delete_zip=True, session=session)
                if extracted is None:
                    result["skipped_404"] += 1
                    result["details"].append(f"{symbol} {date_str}: missing (not available)")
                else:
                    result["downloaded"] += 1
                    result["details"].append(f"{symbol} {date_str}: downloaded")
            except Exception as ex:
                result["errors"] += 1
                result["details"].append(f"{symbol} {date_str}: error {ex}")

    return result

