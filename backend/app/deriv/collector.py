"""Data collector: discovery, historical backfill (resumable), live ticks, gap detection & filling."""
from __future__ import annotations

import asyncio
import logging
import time

from sqlalchemy import func, select

from .. import db
from ..config import DERIV_NATIVE_GRANULARITIES, settings
from ..live import LiveHub, hub as default_hub
from .client import DerivAPIError, DerivClient
from .discovery import match_boom_crash

log = logging.getLogger("collector")


class Collector:
    def __init__(self, client: DerivClient, engine=None, live_hub: LiveHub | None = None):  # noqa: ANN001
        self.client = client
        self.engine = engine or db.get_engine()
        self.hub = live_hub or default_hub
        self.symbols: list[dict] = []
        self.missing: list[str] = []
        self._buffer: list[dict] = []
        self._flush_lock = asyncio.Lock()
        self._tasks: list[asyncio.Task] = []
        self.status: dict[str, dict] = {}
        self.last_live_epoch: dict[str, int] = {}
        self.gap_from: dict[str, int] = {}
        client.on_disconnect(self._snapshot_on_disconnect)
        client.on_reconnect(self.fill_all_gaps)

    def _snapshot_on_disconnect(self) -> None:
        """Remember where each live stream stopped — the reconnect backfill starts there, not at the newest
        stored tick (fresh post-reconnect ticks may already be stored by then)."""
        for sym, e in self.last_live_epoch.items():
            self.gap_from.setdefault(sym, e)

    # ------------------------------------------------------------------ discovery
    async def discover(self) -> list[dict]:
        active = await self.client.active_symbols()
        found, missing = match_boom_crash(active)
        for row in found:
            db.upsert(self.engine, db.symbols, row, ["symbol"])
        # mark previously-known symbols that disappeared as unavailable
        found_syms = {r["symbol"] for r in found}
        with self.engine.begin() as conn:
            for r in conn.execute(select(db.symbols.c.symbol)).all():
                if r.symbol not in found_syms:
                    conn.execute(db.symbols.update().where(db.symbols.c.symbol == r.symbol).values(available=False))
        self.symbols, self.missing = found, missing
        if missing:
            db.log_event("warn", "discovery", f"not offered by Deriv right now: {', '.join(missing)}", self.engine)
        db.log_event("info", "discovery", f"discovered {len(found)} Boom/Crash symbols: "
                     + ", ".join(f"{r['display_name']}={r['symbol']}" for r in found), self.engine)
        return found

    # ------------------------------------------------------------------ helpers
    def _state(self, symbol: str) -> dict:
        with self.engine.connect() as conn:
            r = conn.execute(select(db.backfill_state).where(db.backfill_state.c.symbol == symbol)).mappings().first()
        return dict(r) if r else {"symbol": symbol, "oldest_epoch": None, "newest_epoch": None,
                                  "ticks_complete": False, "candles_done": "{}"}

    def _save_state(self, st: dict) -> None:
        st = {**st, "updated_at": time.time()}
        db.upsert(self.engine, db.backfill_state, st, ["symbol"])

    def _store_ticks(self, symbol: str, times: list[int], prices: list[float]) -> int:
        return db.insert_ignore(self.engine, db.ticks,
                                ({"symbol": symbol, "epoch": int(t), "quote": float(p)} for t, p in zip(times, prices)))

    def _db_range(self, symbol: str) -> tuple[int | None, int | None]:
        with self.engine.connect() as conn:
            r = conn.execute(select(func.min(db.ticks.c.epoch), func.max(db.ticks.c.epoch))
                             .where(db.ticks.c.symbol == symbol)).first()
        return (r[0], r[1]) if r else (None, None)

    # ------------------------------------------------------------------ backfill
    async def backfill_ticks(self, symbol: str, max_pages: int | None = None) -> int:
        """Page backwards through tick history. Resumable via backfill_state."""
        st = self._state(symbol)
        if st.get("ticks_complete"):
            return 0
        lo, hi = self._db_range(symbol)
        oldest = st.get("oldest_epoch") or lo
        limit_epoch = int(time.time()) - settings.max_backfill_days * 86400
        end: int | str = (oldest - 1) if oldest else "latest"
        total, pages = 0, 0
        while True:
            try:
                times, prices = await self.client.ticks_history(symbol, end=end, count=5000)
            except DerivAPIError as e:
                db.log_event("warn", "backfill", f"{symbol}: {e}", self.engine)
                break
            if not times:
                st["ticks_complete"] = True
                break
            total += self._store_ticks(symbol, times, prices)
            new_oldest = min(times)
            if oldest is not None and new_oldest >= oldest:
                st["ticks_complete"] = True  # API returned nothing older
                break
            oldest = new_oldest
            st["oldest_epoch"] = oldest
            st["newest_epoch"] = max(st.get("newest_epoch") or 0, max(times))
            self._save_state(st)
            pages += 1
            self.status.setdefault(symbol, {})["backfill_oldest"] = oldest
            if oldest <= limit_epoch:
                st["ticks_complete"] = True
                break
            if max_pages is not None and pages >= max_pages:
                break
            end = oldest - 1
        self._save_state(st)
        db.log_event("info", "backfill", f"{symbol}: +{total} ticks (oldest={st.get('oldest_epoch')}, "
                     f"complete={st.get('ticks_complete')})", self.engine)
        return total

    async def backfill_candles(self, symbol: str, max_pages: int = 6) -> dict[int, int]:
        """Official Deriv candles for longer timeframes (used where tick coverage is short)."""
        st = self._state(symbol)
        done = db.jloads(st.get("candles_done"), {}) or {}
        out: dict[int, int] = {}
        for g in sorted(DERIV_NATIVE_GRANULARITIES):
            if done.get(str(g)):
                continue
            end: int | str = "latest"
            n = 0
            for _ in range(max_pages):
                try:
                    rows = await self.client.candles(symbol, g, end=end, count=5000)
                except DerivAPIError as e:
                    db.log_event("warn", "backfill", f"{symbol} candles {g}: {e}", self.engine)
                    break
                if not rows:
                    break
                n += db.insert_ignore(self.engine, db.candles_native, (
                    {"symbol": symbol, "granularity": g, "epoch": int(c["epoch"]), "open": float(c["open"]),
                     "high": float(c["high"]), "low": float(c["low"]), "close": float(c["close"])} for c in rows))
                oldest = min(int(c["epoch"]) for c in rows)
                if len(rows) < 5000:
                    break
                end = oldest - 1
            done[str(g)] = True
            out[g] = n
        st["candles_done"] = db.jdump(done)
        self._save_state(st)
        return out

    # ------------------------------------------------------------------ gaps
    async def fill_gap(self, symbol: str, since_epoch: int) -> int:
        """Fetch everything after `since_epoch` up to now by paging back from latest."""
        end: int | str = "latest"
        total = 0
        for _ in range(200):
            times, prices = await self.client.ticks_history(symbol, end=end, count=5000)
            if not times:
                break
            keep = [(t, p) for t, p in zip(times, prices) if t > since_epoch]
            if keep:
                total += self._store_ticks(symbol, [k[0] for k in keep], [k[1] for k in keep])
            if min(times) <= since_epoch:
                break
            end = min(times) - 1
        return total

    async def fill_all_gaps(self) -> None:
        await self.flush()
        from ..research.data import load_recent_ticks
        for s in self.symbols:
            sym = s["symbol"]
            _, hi = self._db_range(sym)
            since = self.gap_from.pop(sym, hi)
            if since is None:
                continue
            try:
                n = await self.fill_gap(sym, since)
                db.log_event("info", "gapfill", f"{sym}: reconnect backfill +{n} ticks after {since}", self.engine)
                await self.flush()
                ep, q = load_recent_ticks(sym, self.hub.maxlen, engine=self.engine)
                self.hub.reset(sym, ep.tolist(), q.tolist())
            except Exception as e:  # noqa: BLE001
                db.log_event("error", "gapfill", f"{sym}: {e}", self.engine)
        self.detect_gaps(lookback_seconds=86400)
        await self.repair_recorded_gaps()

    async def maintenance(self) -> dict:
        """Periodic: flush, record holes in the last day (e.g. a feed that went silent while connected),
        then try to fill them from history."""
        await self.flush()
        holes = await self.repair_holes()
        new = await asyncio.to_thread(self.detect_gaps, 86400)
        fixed = await self.repair_recorded_gaps() + holes
        if fixed:
            from ..research.data import load_recent_ticks
            for s in self.symbols:
                ep, q = load_recent_ticks(s["symbol"], self.hub.maxlen, engine=self.engine)
                self.hub.reset(s["symbol"], ep.tolist(), q.tolist())
        return {"new_gaps": new, "repaired": fixed}

    async def repair_holes(self, lookback_seconds: int = 86400, max_per_symbol: int = 50) -> int:
        """Fill short holes (> 3x median tick interval) that are too small to count as data gaps."""
        import numpy as np
        from ..research.data import load_ticks
        filled = 0
        for s in self.symbols:
            sym = s["symbol"]
            ep, _ = await asyncio.to_thread(load_ticks, sym, int(time.time()) - lookback_seconds, None, self.engine)
            if len(ep) < 10:
                continue
            d = np.diff(ep)
            thr = max(2.0, 3 * float(np.median(d)))
            for i in np.where(d > thr)[0][-max_per_symbol:]:
                a, b = int(ep[i]), int(ep[i + 1])
                try:
                    times, prices = await self.client.ticks_history(sym, end=b - 1, count=min(5000, int(b - a) + 5))
                except Exception:  # noqa: BLE001
                    continue
                keep = [(t, p) for t, p in zip(times, prices) if a < t < b]
                if keep:
                    filled += self._store_ticks(sym, [k[0] for k in keep], [k[1] for k in keep]) > 0
        return int(filled)

    async def repair_recorded_gaps(self, limit: int = 50) -> int:
        """Try to fill recorded gaps from history; mark them filled when the hole is closed."""
        import numpy as np
        rows = db.exec_sql("SELECT * FROM data_gaps WHERE filled = :f ORDER BY id DESC LIMIT :n",
                           {"f": False, "n": limit}, self.engine)
        fixed = 0
        for g in rows:
            try:
                times, prices = await self.client.ticks_history(g["symbol"], end=int(g["end_epoch"]) - 1, count=5000)
            except Exception:  # noqa: BLE001
                continue
            keep = [(t, p) for t, p in zip(times, prices) if g["start_epoch"] < t < g["end_epoch"]]
            if keep:
                self._store_ticks(g["symbol"], [k[0] for k in keep], [k[1] for k in keep])
            ep = np.array([r["epoch"] for r in db.exec_sql(
                "SELECT epoch FROM ticks WHERE symbol=:s AND epoch>=:a AND epoch<=:b ORDER BY epoch",
                {"s": g["symbol"], "a": g["start_epoch"], "b": g["end_epoch"]}, self.engine)])
            from ..research.data import gap_threshold
            if len(ep) >= 2 and np.max(np.diff(ep)) <= gap_threshold(ep if len(ep) > 3 else np.array([0, 1, 2])):
                with self.engine.begin() as conn:
                    conn.execute(db.data_gaps.update().where(db.data_gaps.c.id == g["id"]).values(filled=True))
                fixed += 1
        return fixed

    def detect_gaps(self, lookback_seconds: int | None = None) -> int:
        """Record holes in stored tick data that history could not fill."""
        import numpy as np
        from ..research.data import load_ticks
        found = 0
        for s in self.symbols or [dict(r) for r in db.exec_sql("SELECT symbol FROM symbols", engine=self.engine)]:
            sym = s["symbol"]
            start = int(time.time()) - lookback_seconds if lookback_seconds else None
            ep, _ = load_ticks(sym, start=start, engine=self.engine)
            if len(ep) < 3:
                continue
            d = np.diff(ep)
            med = float(np.median(d))
            thr = max(settings.gap_min_seconds, settings.gap_factor * med)
            idx = np.where(d > thr)[0]
            rows = [{"symbol": sym, "start_epoch": int(ep[i]), "end_epoch": int(ep[i + 1]),
                     "seconds": float(d[i]), "detected_at": time.time(), "filled": False, "source": "scan"} for i in idx]
            found += db.insert_ignore(self.engine, db.data_gaps, rows)
        return found

    # ------------------------------------------------------------------ live
    async def _on_tick(self, tick: dict) -> None:
        sym, ep, q = tick.get("symbol"), tick.get("epoch"), tick.get("quote")
        if sym is None or ep is None or q is None:
            return
        if self.hub.add_tick(sym, int(ep), float(q)):
            self._buffer.append({"symbol": sym, "epoch": int(ep), "quote": float(q)})
            self.last_live_epoch[sym] = int(ep)

    async def flush_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            await self.flush()

    async def flush(self) -> int:
        async with self._flush_lock:
            buf, self._buffer = self._buffer, []
        if not buf:
            return 0
        return await asyncio.to_thread(db.insert_ignore, self.engine, db.ticks, buf)

    async def start_live(self) -> None:
        for s in self.symbols:
            sym = s["symbol"]
            # seed the live buffer from DB so features work immediately
            from ..research.data import load_recent_ticks
            ep, q = load_recent_ticks(sym, self.hub.maxlen, engine=self.engine)
            self.hub.seed(sym, ep.tolist(), q.tolist())
            await self.client.subscribe_ticks(sym, self._on_tick)
        self._tasks.append(asyncio.create_task(self.flush_loop(), name="tick-flush"))

    async def run(self) -> None:
        """Full collector lifecycle — runs forever."""
        self.client.start()
        while True:
            if await self.client.wait_connected(60):
                break
            db.log_event("warn", "collector", f"waiting for Deriv connection: {self.client.last_error}", self.engine)
        while True:
            try:
                await self.discover()
                break
            except Exception as e:  # noqa: BLE001
                db.log_event("error", "discovery", str(e), self.engine)
                await asyncio.sleep(30)
        # Close any hole between the last stored tick and now, then go live, then deep backfill.
        for s in self.symbols:
            _, hi = self._db_range(s["symbol"])
            if hi is not None:
                try:
                    await self.fill_gap(s["symbol"], hi)
                except Exception as e:  # noqa: BLE001
                    db.log_event("error", "gapfill", f"{s['symbol']}: {e}", self.engine)
        await self.start_live()
        from ..research.data import load_recent_ticks
        for s in self.symbols:
            try:
                await self.backfill_ticks(s["symbol"])
                await self.flush()
                ep, q = load_recent_ticks(s["symbol"], self.hub.maxlen, engine=self.engine)
                self.hub.reset(s["symbol"], ep.tolist(), q.tolist())  # live features usable immediately
                await self.backfill_candles(s["symbol"])
            except Exception as e:  # noqa: BLE001
                db.log_event("error", "backfill", f"{s['symbol']}: {e}", self.engine)
        self.detect_gaps()
        # periodic maintenance (gap repair every minute) and rediscovery (hourly)
        k = 0
        while True:
            await asyncio.sleep(60)
            k += 1
            try:
                if self.client.is_connected:
                    await self.maintenance()
                if k % 60 == 0:
                    await self.discover()
            except Exception as e:  # noqa: BLE001
                db.log_event("error", "collector", f"maintenance: {e}", self.engine)
