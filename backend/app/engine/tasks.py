"""Job handlers + recurring schedule."""
from __future__ import annotations

import time

from .. import db
from ..config import settings
from ..research.data import load_ticks
from .jobs import Scheduler, handler


def _sym(symbol: str) -> dict:
    r = db.exec_sql("SELECT * FROM symbols WHERE symbol=:s", {"s": symbol})
    if not r:
        raise ValueError(f"unknown symbol {symbol}")
    return dict(r[0])


@handler("scan_spikes")
def job_scan(payload: dict) -> dict:
    from ..research.spikes import scan_and_store
    s = _sym(payload["symbol"])
    return scan_and_store(s["symbol"], s["direction"])


@handler("research")
def job_research(payload: dict) -> dict:
    from ..research.run import run_research
    s = _sym(payload["symbol"])
    r = run_research(s["symbol"], s["direction"])
    return {"status": r.get("status"), "ticks": r.get("ticks")}


@handler("train")
def job_train(payload: dict) -> dict:
    from ..research.train import train_symbol
    s = _sym(payload["symbol"])
    r = train_symbol(s["symbol"], s["direction"], horizons=payload.get("horizons"))
    return {"status": r.get("status", "ok"), "run_id": r.get("run_id"), "spikes": r.get("spikes"),
            "horizons": {k: {"status": v.get("status"), "selected": v.get("selected"), "approved": v.get("approved"),
                             "registry": v.get("registry")} for k, v in r.get("horizons", {}).items()}}


@handler("arena")
def job_arena(payload: dict) -> dict:
    from ..research.arena import run_arena
    s = _sym(payload["symbol"])
    ep, q = load_ticks(s["symbol"])
    r = run_arena(s["symbol"], s["direction"], ep, q)
    return {"status": r.get("status"), "run_id": r.get("run_id"),
            "horizons": {k: {"val_significant": v.get("val_significant_fdr10"), "test_confirmed": v.get("test_confirmed")}
                         for k, v in r.get("horizons", {}).items()}}


@handler("evaluate")
def job_evaluate(payload: dict) -> dict:
    from .evaluate import evaluate_pending
    return evaluate_pending()


@handler("gaps")
def job_gaps(payload: dict) -> dict:
    import numpy as np
    from ..research.data import gap_threshold
    n = 0
    for s in db.exec_sql("SELECT symbol FROM symbols"):
        ep, _ = load_ticks(s["symbol"], start=int(time.time()) - 3 * 86400)
        if len(ep) < 3:
            continue
        d = np.diff(ep)
        thr = gap_threshold(ep)
        rows = [{"symbol": s["symbol"], "start_epoch": int(ep[i]), "end_epoch": int(ep[i + 1]), "seconds": float(d[i]),
                 "detected_at": time.time(), "filled": False, "source": "scan"} for i in np.where(d > thr)[0]]
        n += db.insert_ignore(db.get_engine(), db.data_gaps, rows)
    return {"new_gaps": n}


class DynamicScheduler(Scheduler):
    """Schedule rebuilt from the discovered symbols each tick."""

    def __init__(self, engine=None):  # noqa: ANN001
        super().__init__([], engine)

    def tick(self) -> int:
        syms = [r["symbol"] for r in db.exec_sql("SELECT symbol FROM symbols WHERE available = :a", {"a": True},
                                                  self.engine)]
        specs = [("evaluate", "evaluate", {}, settings.evaluate_interval, 1), ("gaps", "gaps", {}, 3600, 6)]
        for i, s in enumerate(syms):
            specs += [
                (f"scan:{s}", "scan_spikes", {"symbol": s}, settings.spike_scan_interval, 2),
                (f"research:{s}", "research", {"symbol": s}, settings.research_interval, 4),
                (f"train:{s}", "train", {"symbol": s}, settings.retrain_interval, 5),
                (f"arena:{s}", "arena", {"symbol": s}, settings.arena_interval, 7),
            ]
        self.specs = specs
        return super().tick()
