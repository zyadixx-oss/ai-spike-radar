"""FastAPI application: REST + WebSocket, plus background services (collector, radar, job workers)."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import db, safety
from .api import routes
from .config import settings
from .deriv.client import DerivClient
from .deriv.collector import Collector
from .engine import tasks  # noqa: F401 — registers job handlers
from .engine.jobs import worker_loop
from .engine.radar import Radar, radar_summary
from .engine.tasks import DynamicScheduler
from .live import hub

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

FAST_JOBS = ["evaluate", "scan_spikes", "gaps"]
SLOW_JOBS = ["research", "train", "arena"]


class Services:
    client: DerivClient | None = None
    collector: Collector | None = None
    radar: Radar | None = None
    tasks: list[asyncio.Task] = []
    started_at: float = time.time()


services = Services()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    safety.startup_check()
    eng = db.get_engine()
    db.log_event("info", "app", f"starting (safety: {safety.status()})", eng)
    if settings.run_collector:
        proxy = {"auto": True, "none": None}.get(settings.deriv_ws_proxy, settings.deriv_ws_proxy)
        services.client = DerivClient(settings.deriv_ws_url, settings.deriv_app_id, settings.deriv_max_rps, proxy=proxy)
        services.collector = Collector(services.client, eng, hub)
        services.tasks.append(asyncio.create_task(services.collector.run(), name="collector"))
    if settings.run_radar:
        services.radar = Radar(eng, hub)
        services.tasks.append(asyncio.create_task(services.radar.run(), name="radar"))
    if settings.run_worker:
        services.tasks.append(asyncio.create_task(worker_loop(eng, DynamicScheduler(eng), types=FAST_JOBS), name="w-fast"))
        services.tasks.append(asyncio.create_task(
            worker_loop(eng, None, types=SLOW_JOBS, poll=5, use_process=settings.slow_jobs_in_process), name="w-slow"))
    yield
    for t in services.tasks:
        t.cancel()
    if services.client:
        await services.client.stop()


app = FastAPI(title="AI Spike Radar — Deriv Boom & Crash (research only)", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                   allow_headers=["*"])
app.include_router(routes.router, prefix="/api")
routes.services = services


@app.websocket("/ws")
async def ws_live(ws: WebSocket, symbol: str | None = Query(default=None)) -> None:
    await ws.accept()
    q = hub.subscribe()
    try:
        await ws.send_json({"type": "hello", "safety": safety.status(), "radar": radar_summary(hub.radar)})
        while True:
            msg = await q.get()
            if msg.get("type") == "tick" and symbol and msg.get("symbol") != symbol:
                continue
            await ws.send_json(msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.unsubscribe(q)


# Serve the built frontend if present (single-container deployments)
_DIST = os.environ.get("FRONTEND_DIST", os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
if os.path.isdir(_DIST):
    from fastapi.staticfiles import StaticFiles
    app.mount("/assets", StaticFiles(directory=os.path.join(_DIST, "assets")), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):  # noqa: ANN201
        if path.startswith("api/"):
            raise HTTPException(404)
        f = os.path.join(_DIST, path)
        return FileResponse(f if path and os.path.isfile(f) else os.path.join(_DIST, "index.html"))
