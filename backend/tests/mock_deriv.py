"""TEST-ONLY local server speaking the Deriv WebSocket protocol subset the app uses.

It records every request it receives so tests can prove that no trading message was ever sent, and it can
drop all connections on demand to test reconnect + gap backfill.
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
import websockets

ACTIVE = [
    {"symbol": "TBOOM1000", "display_name": "Boom 1000 Index", "market": "synthetic_index", "submarket": "crash_index",
     "pip": 0.001, "exchange_is_open": 1},
    {"symbol": "TCRASH500", "display_name": "Crash 500 Index", "market": "synthetic_index", "submarket": "crash_index",
     "pip": 0.001, "exchange_is_open": 1},
    {"symbol": "TBOOM150", "display_name": "Boom 150 Index", "market": "synthetic_index", "submarket": "crash_index",
     "pip": 0.001, "exchange_is_open": 1},
    {"symbol": "TVOL10", "display_name": "Volatility 10 Index", "market": "synthetic_index", "submarket": "random_index",
     "pip": 0.001, "exchange_is_open": 1},
]


class MockDeriv:
    def __init__(self, series: dict[str, tuple[np.ndarray, np.ndarray]], start_index: int, tick_interval: float = 0.02):
        self.series = series
        self.pos = {s: start_index for s in series}
        self.tick_interval = tick_interval
        self.received: list[dict] = []
        self.conns: set = set()
        self.subs: dict = {}
        self.paused = False
        self._server = None
        self._task = None
        self.port = None

    async def start(self) -> int:
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        self._task = asyncio.create_task(self._ticker())
        return self.port

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        self._server.close()
        await self._server.wait_closed()

    async def drop_all(self) -> None:
        for ws in list(self.conns):
            await ws.close()

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(self.tick_interval)
            for sym, (ep, q) in self.series.items():
                if self.pos[sym] >= len(ep) - 1:
                    continue
                self.pos[sym] += 1  # market moves on even while nobody is connected (=> gaps to backfill)
                if self.paused:
                    continue
                i = self.pos[sym]
                for ws, syms in list(self.subs.items()):
                    if sym in syms:
                        try:
                            await ws.send(json.dumps({"msg_type": "tick", "tick": {"symbol": sym, "epoch": int(ep[i]),
                                                                                   "quote": float(q[i])}}))
                        except Exception:  # noqa: BLE001
                            pass

    async def _handle(self, ws) -> None:  # noqa: ANN001
        self.conns.add(ws)
        self.subs[ws] = set()
        try:
            async for raw in ws:
                m = json.loads(raw)
                self.received.append(m)
                rid = m.get("req_id")
                if "active_symbols" in m:
                    await ws.send(json.dumps({"msg_type": "active_symbols", "active_symbols": ACTIVE, "req_id": rid}))
                elif "ping" in m:
                    await ws.send(json.dumps({"msg_type": "ping", "ping": "pong", "req_id": rid}))
                elif "ticks_history" in m:
                    await ws.send(json.dumps(self._history(m)))
                elif "ticks" in m:
                    sym = m["ticks"]
                    self.subs[ws].add(sym)
                    ep, q = self.series[sym]
                    i = self.pos[sym]
                    await ws.send(json.dumps({"msg_type": "tick", "req_id": rid,
                                              "tick": {"symbol": sym, "epoch": int(ep[i]), "quote": float(q[i])}}))
                else:
                    await ws.send(json.dumps({"msg_type": "error", "req_id": rid,
                                              "error": {"code": "Forbidden", "message": "not supported in mock"}}))
        except websockets.ConnectionClosed:
            pass
        finally:
            self.conns.discard(ws)
            self.subs.pop(ws, None)

    def _history(self, m: dict) -> dict:
        sym = m["ticks_history"]
        ep, q = self.series[sym]
        hi = self.pos[sym]
        end = m.get("end", "latest")
        if end != "latest":
            hi = min(hi, int(np.searchsorted(ep, int(end), side="right")) - 1)
        count = int(m.get("count", 5000))
        if m.get("style") == "candles":
            g = int(m["granularity"])
            e, qq = ep[:hi + 1], q[:hi + 1]
            b = (e // g) * g
            out = []
            for k in np.unique(b)[-count:]:
                sel = qq[b == k]
                out.append({"epoch": int(k), "open": float(sel[0]), "high": float(sel.max()), "low": float(sel.min()),
                            "close": float(sel[-1])})
            return {"msg_type": "candles", "candles": out, "req_id": m.get("req_id")}
        lo = max(0, hi - count + 1)
        if hi < 0:
            return {"msg_type": "history", "history": {"times": [], "prices": []}, "req_id": m.get("req_id")}
        return {"msg_type": "history", "req_id": m.get("req_id"),
                "history": {"times": ep[lo:hi + 1].tolist(), "prices": q[lo:hi + 1].tolist()}}
