"""Durable, resumable job queue stored in the database.

* Jobs are claimed with a lease. If the process dies, the lease expires and the job is re-queued on the next
  start (or by any other worker) — work resumes automatically after a server stop.
* A periodic scheduler enqueues recurring jobs; its last-enqueued times are stored in `schedule_state`,
  so restarts don't cause duplicate bursts or skipped runs.
* `dedupe_key` prevents enqueueing a job identical to one already queued/running.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from collections.abc import Callable

from sqlalchemy import and_, or_, select

from .. import db

log = logging.getLogger("jobs")

LEASE_SECONDS = 300          # short lease, renewed by a heartbeat while the job runs
HEARTBEAT_SECONDS = 60
HANDLERS: dict[str, Callable[[dict], dict | None]] = {}


def handler(name: str):
    def deco(fn):  # noqa: ANN001
        HANDLERS[name] = fn
        return fn
    return deco


def enqueue(type_: str, payload: dict | None = None, priority: int = 5, run_after: float = 0,
            dedupe_key: str | None = None, max_attempts: int = 3, engine=None) -> int | None:  # noqa: ANN001
    eng = engine or db.get_engine()
    with eng.begin() as conn:
        if dedupe_key:
            exists = conn.execute(select(db.jobs.c.id).where(and_(db.jobs.c.dedupe_key == dedupe_key,
                                                                  db.jobs.c.status.in_(["queued", "running"])))).first()
            if exists:
                return None
        r = conn.execute(db.jobs.insert().values(type=type_, payload=json.dumps(payload or {}), status="queued",
                                                 priority=priority, run_after=run_after, attempts=0,
                                                 max_attempts=max_attempts, dedupe_key=dedupe_key,
                                                 created_at=time.time()))
        return int(r.inserted_primary_key[0])


def recover_stale(engine=None) -> int:  # noqa: ANN001
    """Re-queue running jobs whose lease expired (crash / restart)."""
    eng = engine or db.get_engine()
    now = time.time()
    with eng.begin() as conn:
        r = conn.execute(db.jobs.update().where(and_(db.jobs.c.status == "running",
                                                     or_(db.jobs.c.lease_until.is_(None), db.jobs.c.lease_until < now)))
                         .values(status="queued", last_error="lease expired — resumed after restart"))
        return r.rowcount or 0


def claim(engine=None, types: list[str] | None = None) -> dict | None:  # noqa: ANN001
    eng = engine or db.get_engine()
    now = time.time()
    with eng.begin() as conn:
        q = select(db.jobs).where(and_(db.jobs.c.status == "queued", db.jobs.c.run_after <= now))
        if types:
            q = q.where(db.jobs.c.type.in_(types))
        q = q.order_by(db.jobs.c.priority, db.jobs.c.id).limit(1)
        if eng.dialect.name == "postgresql":
            q = q.with_for_update(skip_locked=True)
        row = conn.execute(q).mappings().first()
        if not row:
            return None
        r = conn.execute(db.jobs.update().where(and_(db.jobs.c.id == row["id"], db.jobs.c.status == "queued"))
                         .values(status="running", attempts=row["attempts"] + 1, started_at=now,
                                 lease_until=now + LEASE_SECONDS))
        if not r.rowcount:
            return None
        return {**dict(row), "attempts": row["attempts"] + 1}


def finish(job_id: int, ok: bool, result: dict | None = None, error: str | None = None, engine=None,  # noqa: ANN001
           retry: bool = True) -> None:
    eng = engine or db.get_engine()
    with eng.begin() as conn:
        row = conn.execute(select(db.jobs).where(db.jobs.c.id == job_id)).mappings().first()
        if not row:
            return
        if ok:
            vals = {"status": "done", "finished_at": time.time(), "lease_until": None,
                    "result": db.jdump(db.clean_json(result or {}))[:200000]}
        elif retry and row["attempts"] < row["max_attempts"]:
            vals = {"status": "queued", "run_after": time.time() + 60 * row["attempts"], "last_error": (error or "")[:8000],
                    "lease_until": None}
        else:
            vals = {"status": "failed", "finished_at": time.time(), "last_error": (error or "")[:8000],
                    "lease_until": None}
        conn.execute(db.jobs.update().where(db.jobs.c.id == job_id).values(**vals))


def heartbeat(job_id: int, engine=None) -> None:  # noqa: ANN001
    eng = engine or db.get_engine()
    with eng.begin() as conn:
        conn.execute(db.jobs.update().where((db.jobs.c.id == job_id) & (db.jobs.c.status == "running"))
                     .values(lease_until=time.time() + LEASE_SECONDS))


def execute(job: dict, engine=None) -> dict:  # noqa: ANN001
    """Run a claimed job and record the result (used in-thread and in the subprocess worker)."""
    fn = HANDLERS.get(job["type"])
    try:
        if fn is None:
            raise KeyError(f"no handler for job type {job['type']}")
        result = fn(json.loads(job["payload"] or "{}"))
        finish(job["id"], True, result, engine=engine)
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        finish(job["id"], False, error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}", engine=engine)
        db.log_event("error", "jobs", f"job {job['id']} {job['type']} failed: {e}", engine)
        job["status"] = "error"
    return job


def run_one(engine=None, types: list[str] | None = None) -> dict | None:  # noqa: ANN001
    job = claim(engine, types)
    if not job:
        return None
    return execute(job, engine)


def _subprocess_execute(job: dict) -> str:
    """Entry point in a separate process: own DB engine, own GIL."""
    from . import tasks  # noqa: F401 — registers handlers in the child
    return execute(job)["status"]


class Scheduler:
    """Enqueues recurring jobs based on persisted last-run times."""

    def __init__(self, specs: list[tuple[str, str, dict, float, int]], engine=None):  # noqa: ANN001
        # (schedule_name, job_type, payload, interval_seconds, priority)
        self.specs = specs
        self.engine = engine or db.get_engine()

    def tick(self) -> int:
        now = time.time()
        n = 0
        with self.engine.connect() as conn:
            last = {r.name: r.last_enqueued_at for r in conn.execute(select(db.schedule_state)).all()}
        for name, jtype, payload, interval, prio in self.specs:
            if now - (last.get(name) or 0) >= interval:
                if enqueue(jtype, payload, prio, dedupe_key=name, engine=self.engine) is not None:
                    n += 1
                db.upsert(self.engine, db.schedule_state, {"name": name, "last_enqueued_at": now}, ["name"])
        return n


async def _with_heartbeat(job_id: int, engine, coro):  # noqa: ANN001
    async def beat():  # noqa: ANN202
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            await asyncio.to_thread(heartbeat, job_id, engine)
    hb = asyncio.create_task(beat())
    try:
        return await coro
    finally:
        hb.cancel()


async def worker_loop(engine=None, scheduler: Scheduler | None = None, poll: float = 2.0,  # noqa: ANN001
                      types: list[str] | None = None, stop: asyncio.Event | None = None,
                      use_process: bool = False) -> None:
    """use_process=True runs each job in a separate process (no GIL contention with the API / radar)."""
    import concurrent.futures as cf
    import multiprocessing as mp
    pool = cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) if use_process else None
    loop = asyncio.get_running_loop()
    while stop is None or not stop.is_set():
        try:
            recovered = await asyncio.to_thread(recover_stale, engine)
            if recovered:
                db.log_event("warn", "jobs", f"re-queued {recovered} interrupted job(s) (expired lease)", engine)
            if scheduler:
                await asyncio.to_thread(scheduler.tick)
            job = await asyncio.to_thread(claim, engine, types)
            if job is None:
                await asyncio.sleep(poll)
                continue
            if pool is not None:
                try:
                    await _with_heartbeat(job["id"], engine, loop.run_in_executor(pool, _subprocess_execute, job))
                except cf.process.BrokenProcessPool:
                    finish(job["id"], False, error="worker process crashed", engine=engine)
                    pool = cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
            else:
                await _with_heartbeat(job["id"], engine, asyncio.to_thread(execute, job, engine))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("worker loop error")
            db.log_event("error", "jobs", f"worker loop: {e}", engine)
            await asyncio.sleep(poll)
