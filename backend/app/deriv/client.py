"""Async Deriv WebSocket client — READ-ONLY.

* Every outgoing message passes `safety.check_outgoing_message` (allowlist of market-data calls).
* No `authorize`: the app never holds a token.
* Auto-reconnect with exponential backoff + jitter, keep-alive pings, resubscription, and
  reconnect hooks (used by the collector to backfill gaps).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

import websockets

from .. import safety

log = logging.getLogger("deriv")

TickCallback = Callable[[dict], Awaitable[None] | None]


class DerivAPIError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class RateLimiter:
    def __init__(self, rps: float):
        self.interval = 1.0 / max(rps, 0.01)
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self.interval


class DerivClient:
    def __init__(self, url: str, app_id: str, max_rps: float = 2.0, ping_interval: float = 25.0,
                 request_timeout: float = 30.0, max_backoff: float = 60.0, proxy: str | bool | None = True):
        sep = "&" if "?" in url else "?"
        self.url = f"{url}{sep}app_id={app_id}" if "app_id=" not in url else url
        self.ping_interval = ping_interval
        self.request_timeout = request_timeout
        self.max_backoff = max_backoff
        self.proxy = proxy  # True = honour HTTPS_PROXY env (websockets >= 15); None = direct
        self.limiter = RateLimiter(max_rps)
        self._ws: Any = None
        self._req_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._tick_subs: dict[str, TickCallback] = {}
        self._reconnect_hooks: list[Callable[[], Awaitable[None]]] = []
        self._disconnect_hooks: list[Callable[[], None]] = []
        self._connected = asyncio.Event()
        self._stop = asyncio.Event()
        self._runner: asyncio.Task | None = None
        # health
        self.connects = 0
        self.disconnects = 0
        self.last_message_at: float | None = None
        self.last_error: str | None = None
        self.connected_since: float | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> asyncio.Task:
        if self._runner is None or self._runner.done():
            self._stop.clear()
            self._runner = asyncio.create_task(self._run_forever(), name="deriv-client")
        return self._runner

    async def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._runner:
            self._runner.cancel()
            try:
                await self._runner
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def on_reconnect(self, hook: Callable[[], Awaitable[None]]) -> None:
        self._reconnect_hooks.append(hook)

    def on_disconnect(self, hook: Callable[[], None]) -> None:
        self._disconnect_hooks.append(hook)

    def health(self) -> dict:
        return {
            "connected": self.is_connected, "url": self.url.split("?")[0], "connects": self.connects,
            "disconnects": self.disconnects, "last_message_at": self.last_message_at,
            "last_error": self.last_error, "connected_since": self.connected_since,
            "subscriptions": sorted(self._tick_subs.keys()),
        }

    async def wait_connected(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _run_forever(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url, open_timeout=20, ping_interval=None, close_timeout=5,
                                              max_size=2 ** 24, proxy=self.proxy) as ws:
                    self._ws = ws
                    self.connects += 1
                    self.connected_since = time.time()
                    self.last_error = None
                    attempt = 0
                    self._connected.set()
                    log.info("connected to Deriv")
                    reader = asyncio.create_task(self._reader(ws))
                    pinger = asyncio.create_task(self._pinger())
                    resub = asyncio.create_task(self._after_connect())
                    done, pending = await asyncio.wait({reader, pinger}, return_when=asyncio.FIRST_COMPLETED)
                    for t in pending | {resub}:
                        t.cancel()
                    for t in done:
                        if t.exception():
                            raise t.exception()  # type: ignore[misc]
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.last_error = f"{type(e).__name__}: {e}"
                log.warning("deriv connection error: %s", self.last_error)
            finally:
                if self._connected.is_set():
                    self.disconnects += 1
                    for h in self._disconnect_hooks:
                        try:
                            h()
                        except Exception:  # noqa: BLE001
                            pass
                self._connected.clear()
                self._ws = None
                self.connected_since = None
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(ConnectionError("connection lost"))
                self._pending.clear()
            if self._stop.is_set():
                break
            attempt += 1
            delay = min(self.max_backoff, 0.5 * 2 ** min(attempt, 8)) * (0.7 + 0.6 * random.random())
            await asyncio.sleep(delay)

    async def _after_connect(self) -> None:
        # resubscribe live ticks
        for sym in list(self._tick_subs):
            try:
                await self._send({"ticks": sym, "subscribe": 1}, expect_response=True)
            except Exception as e:  # noqa: BLE001
                log.warning("resubscribe %s failed: %s", sym, e)
        if self.connects > 1:
            for hook in self._reconnect_hooks:
                try:
                    await hook()
                except Exception as e:  # noqa: BLE001
                    log.warning("reconnect hook failed: %s", e)

    async def _pinger(self) -> None:
        while True:
            await asyncio.sleep(self.ping_interval)
            await self.request({"ping": 1}, timeout=15)

    async def _reader(self, ws) -> None:  # noqa: ANN001
        async for raw in ws:
            self.last_message_at = time.time()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            req_id = msg.get("req_id")
            mtype = msg.get("msg_type")
            if mtype == "tick" and "tick" in msg:
                tick = msg["tick"]
                cb = self._tick_subs.get(tick.get("symbol"))
                if cb is not None:
                    r = cb(tick)
                    if asyncio.iscoroutine(r):
                        await r
                # first tick response of a subscription may also carry req_id
            if req_id is not None and req_id in self._pending:
                fut = self._pending.pop(req_id)
                if not fut.done():
                    fut.set_result(msg)

    # ------------------------------------------------------------------ requests
    async def _send(self, msg: dict, expect_response: bool = True, timeout: float | None = None) -> dict | None:
        safety.check_outgoing_message(msg)  # HARD GATE — raises for any trading/auth call
        if not self._connected.is_set():
            ok = await self.wait_connected(timeout or self.request_timeout)
            if not ok:
                raise ConnectionError("not connected to Deriv")
        await self.limiter.wait()
        self._req_id += 1
        req_id = self._req_id
        msg = {**msg, "req_id": req_id}
        fut: asyncio.Future | None = None
        if expect_response:
            fut = asyncio.get_running_loop().create_future()
            self._pending[req_id] = fut
        ws = self._ws
        if ws is None:
            raise ConnectionError("not connected to Deriv")
        await ws.send(json.dumps(msg))
        if fut is None:
            return None
        resp = await asyncio.wait_for(fut, timeout or self.request_timeout)
        if "error" in resp:
            err = resp["error"]
            raise DerivAPIError(err.get("code", "Error"), err.get("message", ""))
        return resp

    async def request(self, msg: dict, timeout: float | None = None) -> dict:
        resp = await self._send(msg, expect_response=True, timeout=timeout)
        assert resp is not None
        return resp

    async def subscribe_ticks(self, symbol: str, callback: TickCallback) -> None:
        self._tick_subs[symbol] = callback
        if self.is_connected:
            await self._send({"ticks": symbol, "subscribe": 1}, expect_response=True)

    # ------------------------------------------------------------------ convenience
    async def active_symbols(self) -> list[dict]:
        r = await self.request({"active_symbols": "brief", "product_type": "basic"})
        return r.get("active_symbols", [])

    async def ticks_history(self, symbol: str, end: int | str = "latest", count: int = 5000,
                            start: int | None = None) -> tuple[list[int], list[float]]:
        msg: dict[str, Any] = {"ticks_history": symbol, "end": end, "count": count, "style": "ticks",
                               "adjust_start_time": 1}
        if start is not None:
            msg["start"] = start
        r = await self.request(msg)
        h = r.get("history", {})
        return [int(t) for t in h.get("times", [])], [float(p) for p in h.get("prices", [])]

    async def candles(self, symbol: str, granularity: int, end: int | str = "latest", count: int = 5000,
                      start: int | None = None) -> list[dict]:
        msg: dict[str, Any] = {"ticks_history": symbol, "end": end, "count": count, "style": "candles",
                               "granularity": granularity, "adjust_start_time": 1}
        if start is not None:
            msg["start"] = start
        r = await self.request(msg)
        return r.get("candles", [])

    # Explicitly unsupported — kept so any accidental call fails loudly.
    async def buy(self, *_, **__):  # pragma: no cover - always raises
        safety.assert_no_trading("buy")

    async def sell(self, *_, **__):  # pragma: no cover - always raises
        safety.assert_no_trading("sell")
