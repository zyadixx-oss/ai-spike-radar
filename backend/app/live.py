"""In-process live hub: recent ticks per symbol + pub/sub for WebSocket broadcast."""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Any


class LiveHub:
    def __init__(self, maxlen: int = 6000):
        self.maxlen = maxlen
        self.ticks: dict[str, deque] = {}
        self.listeners: set[asyncio.Queue] = set()
        self.radar: dict[str, Any] = {}
        self.last_tick_at: dict[str, float] = {}
        self.count: dict[str, int] = {}   # monotonic number of distinct ticks seen per symbol

    def seed(self, symbol: str, epochs: list[int], quotes: list[float]) -> None:
        dq = self.ticks.setdefault(symbol, deque(maxlen=self.maxlen))
        last = dq[-1][0] if dq else -1
        for e, q in zip(epochs, quotes):
            if e > last:
                dq.append((int(e), float(q)))
                last = e
                self.count[symbol] = self.count.get(symbol, 0) + 1

    def reset(self, symbol: str, epochs: list[int], quotes: list[float]) -> None:
        """Replace the buffer (sorted, de-duplicated) — used after a gap fill."""
        old = list(self.ticks.get(symbol, ()))
        merged = dict(old)
        before = len(merged)
        merged.update({int(e): float(q) for e, q in zip(epochs, quotes)})
        self.count[symbol] = self.count.get(symbol, 0) + (len(merged) - before)
        self.ticks[symbol] = deque(sorted(merged.items())[-self.maxlen:], maxlen=self.maxlen)

    def add_tick(self, symbol: str, epoch: int, quote: float) -> bool:
        dq = self.ticks.setdefault(symbol, deque(maxlen=self.maxlen))
        if dq and epoch <= dq[-1][0]:
            return False
        dq.append((int(epoch), float(quote)))
        self.count[symbol] = self.count.get(symbol, 0) + 1
        import time
        self.last_tick_at[symbol] = time.time()
        self.publish({"type": "tick", "symbol": symbol, "epoch": int(epoch), "quote": float(quote)})
        return True

    def recent(self, symbol: str) -> list[tuple[int, float]]:
        return list(self.ticks.get(symbol, ()))

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=5000)
        self.listeners.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.listeners.discard(q)

    def publish(self, msg: dict) -> None:
        for q in list(self.listeners):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                # slow consumer: drop oldest
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except Exception:  # noqa: BLE001
                    pass


hub = LiveHub(maxlen=20000)
