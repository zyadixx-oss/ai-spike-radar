"""Deriv client & collector against a local protocol mock: discovery, backfill, live ticks,
disconnect -> reconnect -> gap backfill, and proof that no trading message is ever sent."""
import asyncio

import numpy as np
import pytest

from app import db, safety
from app.deriv.client import DerivClient
from app.deriv.collector import Collector
from app.live import LiveHub
from tests.mock_deriv import MockDeriv
from tests.sim import simulate


async def _wait(cond, timeout=15.0):  # noqa: ANN001
    t = 0.0
    while t < timeout:
        if cond():
            return True
        await asyncio.sleep(0.05)
        t += 0.05
    return False


@pytest.fixture()
def series():
    import time
    start = int(time.time()) - 30000
    ep1, q1, _ = simulate(n=30000, mean_gap=300, seed=11, start_epoch=start)
    ep2, q2, _ = simulate(n=30000, mean_gap=300, direction=-1, seed=12, start_epoch=start)
    return {"TBOOM1000": (ep1, q1), "TCRASH500": (ep2, q2)}


async def test_full_collector_cycle(engine, series):
    mock = MockDeriv(series, start_index=12000, tick_interval=0.01)
    port = await mock.start()
    client = DerivClient(f"ws://127.0.0.1:{port}", "1", max_rps=200, ping_interval=0.5, max_backoff=0.3, proxy=None)
    hub = LiveHub(maxlen=5000)
    col = Collector(client, engine, hub)
    client.start()
    try:
        assert await client.wait_connected(10)
        found = await col.discover()
        assert [s["symbol"] for s in found] == ["TBOOM1000", "TCRASH500"]
        assert len(col.missing) == 8
        # backfill history (resumable)
        n = await col.backfill_ticks("TBOOM1000", max_pages=2)
        assert n == 10000
        st = col._state("TBOOM1000")
        assert st["oldest_epoch"] is not None and not st["ticks_complete"]
        n2 = await col.backfill_ticks("TBOOM1000")  # resumes from saved oldest epoch, runs to the start
        assert n2 > 0 and col._state("TBOOM1000")["ticks_complete"]
        await col.backfill_candles("TBOOM1000", max_pages=1)
        assert db.exec_sql("SELECT COUNT(*) AS n FROM candles_native", engine=engine)[0]["n"] > 0
        # live ticks
        await col.start_live()
        before = hub.count.get("TBOOM1000", 0)
        assert await _wait(lambda: hub.count.get("TBOOM1000", 0) > before + 50)
        # drop connection while the market keeps moving, then pause the stream briefly so ticks are missed
        mock.paused = True
        await mock.drop_all()
        await asyncio.sleep(0.6)
        mock.paused = False
        assert await _wait(lambda: client.connects >= 2, 10), client.last_error
        await col.flush()
        # after reconnect: gap filled from history => stored ticks are contiguous
        await asyncio.sleep(1.0)
        m = await col.maintenance()  # repairs any hole left while the feed was silent but connected
        await col.flush()
        ep = np.array([r["epoch"] for r in db.exec_sql("SELECT epoch FROM ticks WHERE symbol='TBOOM1000' ORDER BY epoch",
                                                       engine=engine)])
        assert len(ep) > 12000
        assert np.max(np.diff(ep)) == 1, "gap left after reconnect"
        assert client.disconnects >= 1
    finally:
        await client.stop()
        await mock.stop()
    sent_types = {k for m in mock.received for k in m if k not in ("req_id", "subscribe", "passthrough")}
    assert sent_types <= set(safety.ALLOWED_DERIV_CALLS) | {"end", "count", "style", "adjust_start_time", "start",
                                                            "granularity", "product_type"}
    assert not sent_types & set(safety.FORBIDDEN_DERIV_CALLS)
