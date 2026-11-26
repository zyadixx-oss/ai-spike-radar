"""Command line: `python -m app.cli <command>`

  probe                 connect to the REAL Deriv API, discover Boom/Crash symbols, fetch a few ticks (read-only)
  serve [--port 8000]   run API + collector + radar + workers
  job TYPE [SYMBOL]     run one job synchronously (scan_spikes | research | train | arena | evaluate | gaps)
  safety                print the safety flags
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from . import db, safety
from .config import settings


async def probe() -> int:
    from .deriv.client import DerivClient
    from .deriv.discovery import match_boom_crash
    proxy = {"auto": True, "none": None}.get(settings.deriv_ws_proxy, settings.deriv_ws_proxy)
    c = DerivClient(settings.deriv_ws_url, settings.deriv_app_id, settings.deriv_max_rps, proxy=proxy)
    c.start()
    ok = await c.wait_connected(25)
    if not ok:
        print(f"FAILED to connect to {settings.deriv_ws_url}: {c.last_error}")
        await c.stop()
        return 2
    print(f"connected to {settings.deriv_ws_url} (app_id={settings.deriv_app_id})")
    active = await c.active_symbols()
    found, missing = match_boom_crash(active)
    print(f"active symbols: {len(active)} | Boom/Crash targets found: {len(found)} | missing: {missing}")
    for s in found:
        times, prices = await c.ticks_history(s["symbol"], count=20)
        dt = (sum(b - a for a, b in zip(times, times[1:])) / max(len(times) - 1, 1)) if times else None
        print(f"  {s['display_name']:<16} code={s['symbol']:<12} ticks={len(times)} last={prices[-1] if prices else None} "
              f"mean_dt={dt:.2f}s" if dt else f"  {s['display_name']}: no ticks")
    try:
        await c.buy()  # always raises before any network I/O
    except safety.TradingForbiddenError as e:
        print(f"safety check OK — trading refused locally before sending: {e}")
    await c.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="app.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    sub.add_parser("safety")
    sp = sub.add_parser("serve")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    jp = sub.add_parser("job")
    jp.add_argument("type")
    jp.add_argument("symbol", nargs="?")
    a = ap.parse_args(argv)
    safety.startup_check()
    if a.cmd == "safety":
        print(json.dumps(safety.status(), indent=2))
        return 0
    if a.cmd == "probe":
        return asyncio.run(probe())
    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("app.main:app", host=a.host, port=a.port, log_level="info")
        return 0
    if a.cmd == "job":
        from .engine import tasks  # noqa: F401
        from .engine.jobs import HANDLERS
        t0 = time.time()
        out = HANDLERS[a.type]({"symbol": a.symbol} if a.symbol else {})
        print(json.dumps(db.clean_json(out), indent=2, default=str)[:20000])
        print(f"done in {time.time() - t0:.1f}s")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
