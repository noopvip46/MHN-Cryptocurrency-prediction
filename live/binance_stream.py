"""
live/binance_stream.py

Binance futures WebSocket client.
Subscribes to aggTrade, depth20@500ms, and forceOrder for all configured pairs
via a single combined stream connection.

Every SNAPSHOT_INTERVAL_S seconds the accumulated window is packaged into a
snapshot dict and fired to the registered on_snapshot callback.

Snapshot dict keys:
    timestamp_ms    window-end Unix milliseconds
    symbol          e.g. "ETHUSDT"
    bids            list of [price, qty] — latest top-20 bids at window close
    asks            list of [price, qty] — latest top-20 asks at window close
    vwap            volume-weighted average price over the window
    total_qty       total traded quantity (buys + sells)
    buy_qty         aggressive-buy quantity (taker buys)
    sell_qty        aggressive-sell quantity (taker sells)
    trade_count     number of individual trade events
    liq_count       liquidation order count in window
    liq_qty         total quantity liquidated in window
"""

import asyncio
import json
import time
from typing import Callable, Dict, List, Optional


BINANCE_WS_BASE = "wss://fstream.binance.com/stream"


class BinanceStream:
    """Async WebSocket stream → 30-second snapshot aggregator.

    Usage (in a thread):
        stream = BinanceStream(["ETHUSDT", "BTCUSDT"])
        stream.on_snapshot(my_callback)
        threading.Thread(target=stream.start, daemon=True).start()
    """

    def __init__(self, symbols: List[str], snapshot_interval_s: int = 30):
        self.symbols              = [s.strip().upper() for s in symbols]
        self.snapshot_interval_s  = snapshot_interval_s
        self._stop                = False
        self._callback: Optional[Callable[[dict], None]] = None

        # Per-symbol state
        self._depth: Dict[str, dict] = {
            s: {"bids": [], "asks": []} for s in self.symbols
        }
        self._trades: Dict[str, dict] = {
            s: self._empty_trade_acc() for s in self.symbols
        }
        self._liqs: Dict[str, dict] = {
            s: {"count": 0, "qty": 0.0} for s in self.symbols
        }

    # ── public API ─────────────────────────────────────────────────────────────

    def on_snapshot(self, callback: Callable[[dict], None]) -> "BinanceStream":
        """Register the callback fired every snapshot_interval_s."""
        self._callback = callback
        return self

    def stop(self):
        self._stop = True

    def start(self):
        """Blocking entry point — run in a daemon thread."""
        asyncio.run(self._run())

    # ── internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_trade_acc() -> dict:
        return {"qty": 0.0, "notional": 0.0, "buy_qty": 0.0, "count": 0}

    async def _run(self):
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets package required.\nInstall with: pip install websockets"
            )

        streams = []
        for s in self.symbols:
            sym = s.lower()
            streams += [
                f"{sym}@aggTrade",
                f"{sym}@depth20@500ms",
                f"{sym}@forceOrder",
            ]
        url = f"{BINANCE_WS_BASE}?streams=" + "/".join(streams)

        async with websockets.connect(url, ping_interval=20) as ws:
            tick_task = asyncio.create_task(self._tick_loop())
            try:
                while not self._stop:
                    try:
                        raw  = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        self._dispatch(json.loads(raw))
                    except asyncio.TimeoutError:
                        continue
            finally:
                tick_task.cancel()

    def _dispatch(self, msg: dict):
        stream = msg.get("stream", "")
        data   = msg.get("data", {})
        if "@aggTrade"  in stream: self._on_trade(data)
        elif "@depth20" in stream: self._on_depth(data)
        elif "@forceOrder" in stream: self._on_liq(data)

    def _sym(self, data: dict) -> Optional[str]:
        s = data.get("s", "").upper()
        return s if s in self.symbols else None

    def _on_trade(self, data: dict):
        sym = self._sym(data)
        if not sym:
            return
        qty   = float(data.get("q", 0))
        price = float(data.get("p", 0))
        acc   = self._trades[sym]
        acc["qty"]      += qty
        acc["notional"] += price * qty
        acc["count"]    += 1
        # m=True means buyer is maker → seller is aggressor (taker sell)
        if not data.get("m", False):
            acc["buy_qty"] += qty

    def _on_depth(self, data: dict):
        sym = self._sym(data)
        if not sym:
            return
        self._depth[sym] = {
            "bids": [[float(p), float(q)] for p, q in data.get("b", [])],
            "asks": [[float(p), float(q)] for p, q in data.get("a", [])],
        }

    def _on_liq(self, data: dict):
        order = data.get("o", {})
        sym   = order.get("s", "").upper()
        if sym not in self.symbols:
            return
        self._liqs[sym]["count"] += 1
        self._liqs[sym]["qty"]   += float(order.get("q", 0))

    async def _tick_loop(self):
        """Fire one snapshot per symbol every snapshot_interval_s, wall-aligned."""
        now   = time.time()
        sleep = self.snapshot_interval_s - (now % self.snapshot_interval_s)
        await asyncio.sleep(sleep)

        while not self._stop:
            ts_ms = int(time.time() * 1000)
            for sym in self.symbols:
                snap = self._build_snapshot(sym, ts_ms)
                self._trades[sym] = self._empty_trade_acc()
                self._liqs[sym]   = {"count": 0, "qty": 0.0}
                if self._callback:
                    try:
                        self._callback(snap)
                    except Exception as e:
                        print(f"  [BinanceStream] callback error ({sym}): {e}")
            await asyncio.sleep(self.snapshot_interval_s)

    def _build_snapshot(self, symbol: str, ts_ms: int) -> dict:
        depth = self._depth[symbol]
        acc   = self._trades[symbol]
        liq   = self._liqs[symbol]
        total = acc["qty"]
        vwap  = acc["notional"] / total if total > 0 else 0.0
        return {
            "timestamp_ms": ts_ms,
            "symbol":       symbol,
            "bids":         depth["bids"],
            "asks":         depth["asks"],
            "vwap":         vwap,
            "total_qty":    total,
            "buy_qty":      acc["buy_qty"],
            "sell_qty":     total - acc["buy_qty"],
            "trade_count":  acc["count"],
            "liq_count":    liq["count"],
            "liq_qty":      liq["qty"],
        }
