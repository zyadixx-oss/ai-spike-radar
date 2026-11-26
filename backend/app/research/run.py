"""Research job: spike statistics, memorylessness tests, pre-spike patterns vs controls, event study."""
from __future__ import annotations

import time

import numpy as np

from .. import db
from .data import load_ticks, segment_ids, tick_interval_stats
from .dataset import calibrate_scale
from .patterns import event_study, interarrival, pre_spike_patterns, spike_stats
from .spikes import detect, log_returns, sensitivity


def run_research(symbol: str, direction: int, engine=None, epochs=None, quotes=None, persist: bool = True) -> dict:  # noqa: ANN001
    eng = engine or (db.get_engine() if persist else None)
    if epochs is None:
        epochs, quotes = load_ticks(symbol, engine=eng)
    out: dict = {"symbol": symbol, "ticks": int(len(epochs)), "created_at": time.time()}
    if len(epochs) < 2000:
        out["status"] = "insufficient_data"
        return out
    seg = segment_ids(epochs)
    det = detect(epochs, quotes, direction)
    sp = np.array([e["index"] for e in det["events"]], dtype=np.int64)
    r = log_returns(quotes, seg)
    rc = np.where(np.nan_to_num(direction * r, nan=0.0) > det["threshold"], 0.0, np.nan_to_num(r, nan=0.0))
    sigma = calibrate_scale(quotes, seg, len(quotes))
    out.update({
        "status": "ok",
        "coverage": {"first_epoch": int(epochs[0]), "last_epoch": int(epochs[-1]), "segments": int(seg.max() + 1),
                     "tick_interval": tick_interval_stats(epochs), "days": float((epochs[-1] - epochs[0]) / 86400)},
        "definition": {"k": det["k"], "sigma": det["sigma"], "threshold_log_return": det["threshold"],
                       "size_cuts_abs_log_return": det["size_cuts"]},
        "threshold_sensitivity": sensitivity(quotes, seg, direction, det["sigma"]),
        "spike_stats": spike_stats(det["events"], epochs),
        "interarrival": interarrival(sp, seg, epochs),
        "patterns": pre_spike_patterns(epochs, quotes, seg, sp, rc, sigma),
        "event_study": event_study(quotes, seg, sp, sigma),
    })
    if persist and eng is not None:
        with eng.begin() as conn:
            conn.execute(db.research.insert().values(symbol=symbol, kind="research", created_at=time.time(),
                                                     payload=db.jdump(db.clean_json(out))))
    return out
