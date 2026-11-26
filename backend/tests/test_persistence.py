"""Append-only prediction log + durable, resumable job queue."""
import time

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError, IntegrityError, OperationalError

from app import db
from app.engine import jobs


def _pred(engine) -> int:
    with engine.begin() as c:
        return int(c.execute(db.predictions.insert().values(
            created_at=time.time(), symbol="X", horizon="1m", horizon_kind="sec", horizon_size=60, issued_epoch=1,
            expires_epoch=61, direction=1, probability=0.1, level="green", role="champion")).inserted_primary_key[0])


def test_predictions_cannot_be_updated_or_deleted(engine):
    pid = _pred(engine)
    for sql in (f"UPDATE predictions SET probability=0.9 WHERE id={pid}", f"DELETE FROM predictions WHERE id={pid}",
                f"UPDATE predictions SET level='red' WHERE id={pid}"):
        with pytest.raises((IntegrityError, OperationalError, DatabaseError)):
            with engine.begin() as c:
                c.execute(text(sql))
    row = db.exec_sql(f"SELECT probability, level FROM predictions WHERE id={pid}", engine=engine)[0]
    assert row["probability"] == 0.1 and row["level"] == "green"


def test_outcomes_insert_only(engine):
    pid = _pred(engine)
    db.insert_ignore(engine, db.prediction_outcomes, [{"prediction_id": pid, "evaluated_at": 1, "status": "miss",
                                                       "outcome": 0}])
    # a second "evaluation" (e.g. trying to flip a failed prediction into a hit) is ignored
    db.insert_ignore(engine, db.prediction_outcomes, [{"prediction_id": pid, "evaluated_at": 2, "status": "hit",
                                                       "outcome": 1}])
    assert db.exec_sql("SELECT status FROM prediction_outcomes", engine=engine)[0]["status"] == "miss"
    with pytest.raises((IntegrityError, OperationalError, DatabaseError)):
        with engine.begin() as c:
            c.execute(text(f"UPDATE prediction_outcomes SET outcome=1 WHERE prediction_id={pid}"))


def test_job_resumes_after_crash(engine):
    calls = []

    @jobs.handler("t_echo")
    def _echo(p):  # noqa: ANN001, ANN202
        calls.append(p["x"])
        return {"ok": p["x"]}

    jid = jobs.enqueue("t_echo", {"x": 7}, engine=engine)
    job = jobs.claim(engine)
    assert job["id"] == jid
    # simulate a crash: the job stays 'running' with an expired lease
    with engine.begin() as c:
        c.execute(db.jobs.update().where(db.jobs.c.id == jid).values(lease_until=time.time() - 1))
    assert jobs.claim(engine) is None             # not claimable while 'running'
    assert jobs.recover_stale(engine) == 1        # restart: lease expired -> re-queued
    done = jobs.run_one(engine)
    assert done["status"] == "done" and calls == [7]
    st = db.exec_sql(f"SELECT status, attempts FROM jobs WHERE id={jid}", engine=engine)[0]
    assert st["status"] == "done" and st["attempts"] == 2


def test_job_retry_then_fail_and_dedupe(engine):
    @jobs.handler("t_fail")
    def _fail(p):  # noqa: ANN001, ANN202
        raise RuntimeError("boom")

    jid = jobs.enqueue("t_fail", {}, max_attempts=2, dedupe_key="k1", engine=engine)
    assert jobs.enqueue("t_fail", {}, dedupe_key="k1", engine=engine) is None  # duplicate suppressed
    jobs.run_one(engine)
    with engine.begin() as c:
        c.execute(db.jobs.update().where(db.jobs.c.id == jid).values(run_after=0))
    jobs.run_one(engine)
    st = db.exec_sql(f"SELECT status, attempts, last_error FROM jobs WHERE id={jid}", engine=engine)[0]
    assert st["status"] == "failed" and st["attempts"] == 2 and "boom" in st["last_error"]


def test_scheduler_persists_state(engine):
    s = jobs.Scheduler([("every_hour", "t_echo", {"x": 1}, 3600, 5)], engine)
    assert s.tick() == 1
    assert jobs.Scheduler([("every_hour", "t_echo", {"x": 1}, 3600, 5)], engine).tick() == 0  # survives restart


async def test_subprocess_worker_runs_job():
    """Heavy jobs run in a separate process (own GIL) against the shared database."""
    import asyncio
    eng = db.make_engine()  # the env-configured DB, which the child process also opens
    db.set_engine(eng)
    jid = jobs.enqueue("evaluate", {}, engine=eng)
    stop = asyncio.Event()

    async def stopper():  # noqa: ANN202
        for _ in range(600):
            st = db.exec_sql(f"SELECT status FROM jobs WHERE id={jid}", engine=eng)[0]["status"]
            if st in ("done", "failed"):
                break
            await asyncio.sleep(0.1)
        stop.set()

    await asyncio.gather(jobs.worker_loop(eng, None, poll=0.1, types=["evaluate"], stop=stop, use_process=True), stopper())
    assert db.exec_sql(f"SELECT status FROM jobs WHERE id={jid}", engine=eng)[0]["status"] == "done"
