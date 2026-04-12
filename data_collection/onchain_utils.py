"""
data_collection/onchain_utils.py

Alchemy on-chain data collector for EVM chains.
Alchemy has no official Python SDK — this module is our own thin client.

Network URL pattern:   https://{network}.g.alchemy.com/v2/{api_key}
WebSocket URL pattern: wss://{network}.g.alchemy.com/v2/{api_key}

The user must enable the target network in their Alchemy dashboard first.
Set ALCHEMY_API_KEY (and optionally ALCHEMY_NETWORK) in your .env file.

Feature design
--------------
On-chain features are GLOBAL market columns, not per-pair.
Bitcoin has no EVM chain so BTCUSDT cannot have its own on-chain data.
To keep a uniform feature vector across all pairs, on-chain columns are shared:
    both ETHUSDT and BTCUSDT rows carry the same ETH chain values.
Pairs not covered by any network entry receive zero-filled on-chain columns.
The canonical feature column list is ONCHAIN_FEATURE_COLUMNS — any code that
merges on-chain data must use exactly these columns in this order.

Historical (batch) mode
-----------------------
Per-day CSVs written to: <out_base>/<symbol>/<symbol>-onchain-<YYYY-MM-DD>.csv
Each row is a 30-second window aligned to SNAPSHOT_INTERVAL_S.

Columns:
    timestamp_ms           Unix ms (window start)
    block_number_start     first block in the window
    block_number_end       last block in the window
    base_fee_gwei_mean     mean EIP-1559 base fee (gwei)   — congestion / panic
    base_fee_gwei_max      peak base fee in window          — spike detector
    gas_used_ratio_mean    mean gas used / gas limit        — block fullness
    large_transfer_count   native ETH transfers > threshold — whale moves
    large_transfer_eth     total ETH in large transfers
    exchange_inflow_count  large ETH transfers TO known CEX wallets — sell pressure
    exchange_inflow_eth    total ETH flowing into exchanges

Real-time stream mode
---------------------
OnchainStream subscribes to newHeads via WebSocket and fires a callback on
every new block.  Requires: pip install websockets
"""

import os
import time
import json
import asyncio
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Canonical feature column list
# Any downstream code that reads on-chain CSVs must reference this list.
# Pairs without chain support are zero-filled to exactly these columns.
# ─────────────────────────────────────────────────────────────────────────────

ONCHAIN_FEATURE_COLUMNS: List[str] = [
    "base_fee_gwei_mean",
    "base_fee_gwei_max",
    "gas_used_ratio_mean",
    "large_transfer_count",
    "large_transfer_eth",
    "exchange_inflow_count",
    "exchange_inflow_eth",
]

# ─────────────────────────────────────────────────────────────────────────────
# Network configuration registry
# Add a new entry here to support additional EVM chains.
# ─────────────────────────────────────────────────────────────────────────────

NETWORK_CONFIG: Dict[str, dict] = {
    "eth-mainnet": {
        # Trading pairs whose on-chain signals come from this network.
        # BTC has no EVM chain — not listed here.
        "pairs": ["ETHUSDT"],

        # Which feature groups are meaningful on this network.
        # L2 gas fees are near-zero and not predictive — set fee_history False there.
        "fee_history":     True,
        "asset_transfers": True,
        "exchange_inflows": True,

        # Whale transfer threshold in native ETH
        "large_tx_eth": 50.0,

        # Known centralised exchange hot/cold wallets.
        # Transfers TO these addresses = coins moving to exchange = selling pressure.
        # Sources: Etherscan labels, Dune Analytics exchange wallet lists.
        "exchange_wallets": {
            "Binance": [
                "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",  # Binance hot wallet 1
                "0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8",  # Binance cold wallet
                "0xF977814e90dA44bFA03b6295A0616a897441aceC",  # Binance 8
                "0x28C6c06298d514Db089934071355E5743bf21d60",  # Binance 14
            ],
            "Coinbase": [
                "0x71660c4005BA85c37ccec55d0C4493E66Fe775d3",  # Coinbase hot wallet
                "0xA9D1e08C7793af67e9d92fe308d5697FB81d3E43",  # Coinbase 2
            ],
            "Kraken": [
                "0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2",
                "0x0A869d79a7052C7f1b55a8EbAbbEa3420F0D1E13",
            ],
            "OKX": [
                "0x6cC5F688a315f3dC28A7781717a9A798a59fDA7b",
            ],
        },
    },
    # ── L2 examples (fee_history disabled — gas is noise on L2s) ──────────────
    # Uncomment and enable in Alchemy dashboard if you want L2 transfer data.
    #
    # "base-mainnet": {
    #     "pairs": [],           # no distinct pair yet
    #     "fee_history":     False,
    #     "asset_transfers": True,
    #     "exchange_inflows": True,
    #     "large_tx_eth":    500.0,
    #     "exchange_wallets": {},
    # },
    # "arb-mainnet": {
    #     "pairs": [],
    #     "fee_history":     False,
    #     "asset_transfers": True,
    #     "exchange_inflows": True,
    #     "large_tx_eth":    500.0,
    #     "exchange_wallets": {},
    # },
}

# Flat set of all exchange wallet addresses across all networks (lowercase) for fast lookup
_ALL_EXCHANGE_WALLETS: Dict[str, set] = {
    network: {
        addr.lower()
        for wallets in cfg["exchange_wallets"].values()
        for addr in wallets
    }
    for network, cfg in NETWORK_CONFIG.items()
    if cfg.get("exchange_inflows")
}

# ─────────────────────────────────────────────────────────────────────────────
# Internal constants
# ─────────────────────────────────────────────────────────────────────────────

ALCHEMY_DEFAULT_NETWORK = "eth-mainnet"
FEE_HISTORY_BATCH       = 1024   # hard Alchemy limit per eth_feeHistory call
ETH_BLOCK_TIME_S        = 12     # post-Merge ETH slot time (constant)
SNAPSHOT_INTERVAL_S     = 30     # must match config.SNAPSHOT_INTERVAL_S

# Post-Merge reference for fast block-number estimation
_MERGE_BLOCK = 15_537_394
_MERGE_TS    = 1_663_224_162     # Unix seconds, Sep 15 2022


# ─────────────────────────────────────────────────────────────────────────────
# Period helpers  (same contract as trades_utils / book_depth_utils)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_period(period: str):
    """Parse a period string into (start_date, end_date).
    Accepts: '7d', '1w', '2w', '1m', '3m', '1y', 'YYYY-MM-DD', 'YYYY-MM-DD/YYYY-MM-DD'.
    """
    import calendar
    period = period.strip().lower()
    today  = datetime.utcnow().date()

    if "/" in period:
        s, e  = [p.strip() for p in period.split("/", 1)]
        start = datetime.strptime(s, "%Y-%m-%d").date()
        end   = datetime.strptime(e, "%Y-%m-%d").date()
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
        months = int(period[:-1])
        end    = today - timedelta(days=1)
        yr, mo = end.year, end.month
        for _ in range(months):
            mo -= 1
            if mo == 0:
                mo = 12; yr -= 1
        _, last = calendar.monthrange(yr, mo)
        return datetime(yr, mo, min(end.day, last)).date(), end
    if period.endswith("y"):
        end = today - timedelta(days=1)
        return datetime(end.year - int(period[:-1]), end.month, end.day).date(), end

    start = datetime.strptime(period, "%Y-%m-%d").date()
    return start, start


def _date_list(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


# ─────────────────────────────────────────────────────────────────────────────
# Environment loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_alchemy_env():
    """Read ALCHEMY_API_KEY and ALCHEMY_NETWORK from .env or process environment."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    api_key = os.getenv("ALCHEMY_API_KEY", "")
    network = os.getenv("ALCHEMY_NETWORK", ALCHEMY_DEFAULT_NETWORK)
    if not api_key:
        raise EnvironmentError(
            "ALCHEMY_API_KEY not set. Add it to your .env file:\n"
            "  ALCHEMY_API_KEY=your_key_here\n"
            "  ALCHEMY_NETWORK=eth-mainnet   # optional, defaults to eth-mainnet"
        )
    return api_key, network


# ─────────────────────────────────────────────────────────────────────────────
# Alchemy JSON-RPC client
# ─────────────────────────────────────────────────────────────────────────────

class AlchemyClient:
    """Synchronous JSON-RPC wrapper for Alchemy.
    No official Python SDK exists — we build our own.

    Usage:
        client = AlchemyClient.from_env()
        print(client.get_block_number())
    """

    def __init__(self, api_key: str, network: str = ALCHEMY_DEFAULT_NETWORK):
        self.network  = network
        self.http_url = f"https://{network}.g.alchemy.com/v2/{api_key}"
        self.ws_url   = f"wss://{network}.g.alchemy.com/v2/{api_key}"
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._req_id  = 0

    @classmethod
    def from_env(cls) -> "AlchemyClient":
        api_key, network = _load_alchemy_env()
        return cls(api_key, network)

    # ── low-level ──────────────────────────────────────────────────────────────

    def _rpc(self, method: str, params: list):
        self._req_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id":      self._req_id,
            "method":  method,
            "params":  params,
        }
        resp = self._session.post(self.http_url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Alchemy RPC [{method}]: {data['error']}")
        return data["result"]

    # ── standard eth methods ───────────────────────────────────────────────────

    def get_block_number(self) -> int:
        return int(self._rpc("eth_blockNumber", []), 16)

    def get_block(self, block_number: int) -> dict:
        """Block header + tx hashes (not full tx objects)."""
        return self._rpc("eth_getBlockByNumber", [hex(block_number), False])

    def get_fee_history(self, block_count: int, newest_block: int) -> dict:
        """eth_feeHistory — up to 1024 blocks per call."""
        return self._rpc("eth_feeHistory", [hex(block_count), hex(newest_block), []])

    # ── alchemy extension ──────────────────────────────────────────────────────

    def get_asset_transfers(
        self,
        from_block:    int,
        to_block:      int,
        min_value_eth: float = 0.0,
        page_key:      Optional[str] = None,
    ) -> dict:
        """alchemy_getAssetTransfers — native ETH (external) transfers.
        Returns {"transfers": [...], "pageKey": str|None}.
        """
        params: dict = {
            "fromBlock":        hex(from_block),
            "toBlock":          hex(to_block),
            "category":         ["external"],
            "withMetadata":     True,
            "excludeZeroValue": True,
            "maxCount":         hex(1000),
        }
        if page_key:
            params["pageKey"] = page_key
        result    = self._rpc("alchemy_getAssetTransfers", [params])
        threshold = min_value_eth
        transfers = [
            t for t in result.get("transfers", [])
            if float(t.get("value") or 0) >= threshold
        ]
        return {"transfers": transfers, "pageKey": result.get("pageKey")}

    def get_all_asset_transfers(
        self,
        from_block:    int,
        to_block:      int,
        min_value_eth: float = 0.0,
    ) -> List[dict]:
        """Paginate through all transfers in a block range above the value threshold."""
        all_transfers: List[dict] = []
        page_key = None
        while True:
            result   = self.get_asset_transfers(from_block, to_block, min_value_eth, page_key)
            all_transfers.extend(result["transfers"])
            page_key = result.get("pageKey")
            if not page_key:
                break
            time.sleep(0.1)
        return all_transfers

    # ── block discovery ────────────────────────────────────────────────────────

    def estimate_block_at_timestamp(self, target_ts: int) -> int:
        """Fast block estimate using post-Merge 12 s constant slot time."""
        return _MERGE_BLOCK + round((target_ts - _MERGE_TS) / ETH_BLOCK_TIME_S)

    def find_block_at_timestamp(self, target_ts: int) -> int:
        """Block number whose timestamp is closest to target_ts.
        Uses post-Merge estimation then walks ±1 block to fine-tune.
        Typically requires only 1–3 RPC calls.
        """
        estimated = max(0, self.estimate_block_at_timestamp(target_ts))
        block     = self.get_block(estimated)
        block_ts  = int(block["timestamp"], 16)

        if block_ts < target_ts:
            while block_ts < target_ts:
                estimated += 1
                block_ts   = int(self.get_block(estimated)["timestamp"], 16)
        else:
            while block_ts > target_ts and estimated > 0:
                estimated -= 1
                block_ts   = int(self.get_block(estimated)["timestamp"], 16)

        return estimated


# ─────────────────────────────────────────────────────────────────────────────
# Historical (batch) download
# ─────────────────────────────────────────────────────────────────────────────

def download_onchain(
    client:    AlchemyClient,
    date_str:  str,
    out_base:  str  = "onchain_data",
    symbol:    str  = "ETHUSDT",
    overwrite: bool = False,
) -> Optional[Path]:
    """Download one day of on-chain features and save as a CSV.

    Features produced are exactly ONCHAIN_FEATURE_COLUMNS.
    Returns the output directory path, or None if no blocks were found.
    Skips silently if the file already exists (unless overwrite=True).
    """
    out_dir  = Path(out_base) / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{symbol}-onchain-{date_str}.csv"

    if out_file.exists() and not overwrite:
        return out_dir

    cfg      = NETWORK_CONFIG.get(client.network, {})
    exc_set  = _ALL_EXCHANGE_WALLETS.get(client.network, set())
    large_th = cfg.get("large_tx_eth", 50.0)

    day      = datetime.strptime(date_str, "%Y-%m-%d")
    start_ts = int(day.timestamp())
    end_ts   = start_ts + 86400 - 1

    # ── find start block; estimate end block from 12 s slot time ──────────────
    start_block = client.find_block_at_timestamp(start_ts)
    end_block   = start_block + int(86400 / ETH_BLOCK_TIME_S)   # ~7200 blocks/day

    # ── fee history ────────────────────────────────────────────────────────────
    block_rows = []
    if cfg.get("fee_history", True):
        cursor = end_block
        while cursor >= start_block:
            count   = min(FEE_HISTORY_BATCH, cursor - start_block + 1)
            history = client.get_fee_history(count, cursor)

            base_fees  = history.get("baseFeePerGas", [])   # length = count + 1
            gas_ratios = history.get("gasUsedRatio",  [])   # length = count
            oldest     = int(history.get("oldestBlock", "0x0"), 16)

            for i, (bf, gr) in enumerate(zip(base_fees, gas_ratios)):
                b_num = oldest + i
                b_ts  = start_ts + (b_num - start_block) * ETH_BLOCK_TIME_S
                block_rows.append({
                    "block_number":  b_num,
                    "timestamp_s":   b_ts,
                    "base_fee_gwei": int(bf, 16) / 1e9,
                    "gas_used_ratio": float(gr),
                })

            cursor -= count
            time.sleep(0.08)

    if not block_rows:
        return None

    blocks_df = (
        pd.DataFrame(block_rows)
        .query("@start_ts <= timestamp_s <= @end_ts")
        .sort_values("block_number")
        .reset_index(drop=True)
    )

    # ── asset transfers — fetch once, tag by destination ──────────────────────
    transfer_rows: List[dict] = []
    if cfg.get("asset_transfers", True):
        raw_transfers = client.get_all_asset_transfers(
            start_block, end_block, min_value_eth=large_th
        )
        for t in raw_transfers:
            raw_block = t.get("blockNum", "0x0")
            b_num     = int(raw_block, 16) if isinstance(raw_block, str) else int(raw_block)
            value     = float(t.get("value") or 0)
            to_addr   = (t.get("to") or "").lower()
            transfer_rows.append({
                "block_number":   b_num,
                "eth_value":      value,
                # Tag inflows to known exchange wallets in the same pass — no extra calls
                "to_exchange":    to_addr in exc_set,
            })

    transfers_df = (
        pd.DataFrame(transfer_rows)
        if transfer_rows
        else pd.DataFrame(columns=["block_number", "eth_value", "to_exchange"])
    )

    # ── aggregate to SNAPSHOT_INTERVAL_S (30 s) windows ──────────────────────
    snap = SNAPSHOT_INTERVAL_S
    blocks_df["window_ts_ms"] = (
        (blocks_df["timestamp_s"] // snap * snap * 1000).astype("int64")
    )

    fee_agg = (
        blocks_df.groupby("window_ts_ms")
        .agg(
            block_number_start  =("block_number",  "min"),
            block_number_end    =("block_number",  "max"),
            base_fee_gwei_mean  =("base_fee_gwei", "mean"),
            base_fee_gwei_max   =("base_fee_gwei", "max"),
            gas_used_ratio_mean =("gas_used_ratio","mean"),
        )
        .reset_index()
    )

    if not transfers_df.empty:
        block_to_win = blocks_df.set_index("block_number")["window_ts_ms"].to_dict()
        transfers_df["window_ts_ms"] = (
            transfers_df["block_number"].map(block_to_win).dropna().astype("int64")
        )
        transfers_df = transfers_df.dropna(subset=["window_ts_ms"])
        transfers_df["window_ts_ms"] = transfers_df["window_ts_ms"].astype("int64")

        # All large transfers (whale signal)
        all_tx_agg = (
            transfers_df.groupby("window_ts_ms")
            .agg(
                large_transfer_count=("eth_value", "count"),
                large_transfer_eth  =("eth_value", "sum"),
            )
            .reset_index()
        )
        # Exchange-bound inflows only (sell pressure signal)
        exc_agg = (
            transfers_df[transfers_df["to_exchange"]]
            .groupby("window_ts_ms")
            .agg(
                exchange_inflow_count=("eth_value", "count"),
                exchange_inflow_eth  =("eth_value", "sum"),
            )
            .reset_index()
        )
        result = fee_agg.merge(all_tx_agg, on="window_ts_ms", how="left")
        result = result.merge(exc_agg,     on="window_ts_ms", how="left")
    else:
        result = fee_agg.copy()

    # ── fill missing and enforce canonical column order ───────────────────────
    for col in ["large_transfer_count", "exchange_inflow_count"]:
        result[col] = result.get(col, 0).fillna(0).astype(int)
    for col in ["large_transfer_eth", "exchange_inflow_eth"]:
        result[col] = result.get(col, 0.0).fillna(0.0)

    result = result.rename(columns={"window_ts_ms": "timestamp_ms"})
    result = result.sort_values("timestamp_ms").reset_index(drop=True)

    # Preserve block metadata + canonical feature columns
    meta_cols    = ["timestamp_ms", "block_number_start", "block_number_end"]
    ordered_cols = meta_cols + ONCHAIN_FEATURE_COLUMNS
    result       = result.reindex(columns=ordered_cols, fill_value=0)

    result.to_csv(out_file, index=False)
    return out_dir


def download_onchain_range(
    period:        str,
    out_base:      str   = "onchain_data",
    symbol:        str   = "ETHUSDT",
    overwrite:     bool  = False,
    pause_seconds: float = 0.2,
    client: Optional[AlchemyClient] = None,
) -> dict:
    """Download on-chain features for a date range, day by day.

    Mirrors the interface of download_trades_range / download_book_depth_range.
    """
    start_date, end_date = _parse_period(period)
    today     = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    if end_date >= today:
        end_date = yesterday
    if start_date > end_date:
        raise ValueError("Invalid period: no past days remain after adjustment.")

    if client is None:
        client = AlchemyClient.from_env()

    result = {
        "symbol":         symbol,
        "period":         period,
        "start_date":     start_date.isoformat(),
        "end_date":       end_date.isoformat(),
        "days_requested": (end_date - start_date).days + 1,
        "downloaded":     0,
        "skipped":        0,
        "errors":         0,
        "details":        [],
    }

    for date_str in _date_list(start_date, end_date):
        time.sleep(pause_seconds)
        try:
            out = download_onchain(
                client, date_str,
                out_base=out_base, symbol=symbol, overwrite=overwrite,
            )
            if out is None:
                result["skipped"] += 1
                result["details"].append(f"{date_str}: no blocks found (skipped)")
            else:
                result["downloaded"] += 1
                result["details"].append(f"{date_str}: downloaded")
        except Exception as ex:
            result["errors"] += 1
            result["details"].append(f"{date_str}: error — {ex}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Real-time stream
# ─────────────────────────────────────────────────────────────────────────────

class OnchainStream:
    """Subscribe to Alchemy newHeads via WebSocket and emit per-block features.

    Requires: pip install websockets

    The callback receives a dict with keys matching ONCHAIN_FEATURE_COLUMNS
    (transfer counts are 0 in real-time mode — only gas features are available
    per-block without a separate transfer query).

    Usage:
        stream = OnchainStream.from_env()

        def on_block(features: dict):
            print(features)

        stream.start(on_block)     # blocks; call stream.stop() from another thread
    """

    def __init__(self, api_key: str, network: str = ALCHEMY_DEFAULT_NETWORK):
        self.ws_url = f"wss://{network}.g.alchemy.com/v2/{api_key}"
        self._stop  = False

    @classmethod
    def from_env(cls) -> "OnchainStream":
        api_key, network = _load_alchemy_env()
        return cls(api_key, network)

    def stop(self):
        self._stop = True

    def start(self, callback: Callable[[dict], None]):
        """Start the WebSocket loop (blocking)."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets package required for real-time streaming.\n"
                "Install with: pip install websockets"
            )
        asyncio.run(self._stream(callback))

    async def _stream(self, callback: Callable[[dict], None]):
        import websockets

        subscribe_msg = json.dumps({
            "jsonrpc": "2.0",
            "id":      1,
            "method":  "eth_subscribe",
            "params":  ["newHeads"],
        })

        async with websockets.connect(self.ws_url) as ws:
            await ws.send(subscribe_msg)
            await ws.recv()   # consume subscription confirmation

            while not self._stop:
                try:
                    raw  = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    data = json.loads(raw)
                    head = data.get("params", {}).get("result", {})
                    if not head:
                        continue

                    base_fee_hex = head.get("baseFeePerGas", "0x0")
                    features = {
                        # metadata
                        "block_number":  int(head.get("number",    "0x0"), 16),
                        "timestamp_ms":  int(head.get("timestamp", "0x0"), 16) * 1000,
                        # ONCHAIN_FEATURE_COLUMNS
                        "base_fee_gwei_mean":    int(base_fee_hex, 16) / 1e9,
                        "base_fee_gwei_max":     int(base_fee_hex, 16) / 1e9,
                        "gas_used_ratio_mean": (
                            int(head.get("gasUsed",  "0x0"), 16) /
                            max(int(head.get("gasLimit", "0x1"), 16), 1)
                        ),
                        # Transfer counts not available per-block without extra queries
                        "large_transfer_count":  0,
                        "large_transfer_eth":    0.0,
                        "exchange_inflow_count": 0,
                        "exchange_inflow_eth":   0.0,
                    }
                    callback(features)

                except asyncio.TimeoutError:
                    continue   # no block in 30 s — normal during quiet periods
                except Exception:
                    if not self._stop:
                        raise
