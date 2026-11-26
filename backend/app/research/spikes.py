"""Spike Detection Engine.

Definition (measurable, per symbol):
    r_i = ln(p_i / p_{i-1})  for consecutive ticks inside the same gap-free segment
    sigma = 1.4826 * MAD(r)   (robust scale; spikes are rare so they barely affect it)
    spike at tick i  <=>  direction * r_i > k * sigma
where direction = +1 for Boom (up spikes) and -1 for Crash (down spikes).
Consecutive spike ticks are merged into a single event (first tick = event time).
Size classes (small/medium/large) use per-symbol terciles of |r| of detected spikes.
"""
from __future__ import annotations

import time

import numpy as np

from .. import db
from ..config import settings
from .data import load_ticks, segment_ids


def robust_sigma(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    med = np.median(r)
    s = 1.4826 * float(np.median(np.abs(r - med)))
    if s <= 0:
        lo, hi = np.quantile(r, [0.01, 0.99])
        core = r[(r >= lo) & (r <= hi)]
        s = float(np.std(core)) if len(core) > 1 else 0.0
    return s


def log_returns(quotes: np.ndarray, seg: np.ndarray) -> np.ndarray:
    r = np.full(len(quotes), np.nan)
    if len(quotes) > 1:
        r[1:] = np.diff(np.log(quotes))
        r[1:][seg[1:] != seg[:-1]] = np.nan  # never a return across a data gap
    return r


def spike_threshold(quotes: np.ndarray, seg: np.ndarray, k: float | None = None) -> tuple[float, float]:
    r = log_returns(quotes, seg)
    sigma = robust_sigma(r)
    return (k or settings.spike_k) * sigma, sigma


def spike_mask(quotes: np.ndarray, seg: np.ndarray, direction: int, threshold: float) -> np.ndarray:
    """Boolean per tick: True on the FIRST tick of each spike event."""
    r = log_returns(quotes, seg)
    raw = np.nan_to_num(direction * r, nan=0.0) > threshold
    prev = np.concatenate([[False], raw[:-1]])
    prev_same_seg = np.concatenate([[False], seg[1:] == seg[:-1]])
    return raw & ~(prev & prev_same_seg)


def detect(epochs: np.ndarray, quotes: np.ndarray, direction: int, k: float | None = None,
           threshold: float | None = None) -> dict:
    seg = segment_ids(epochs)
    thr, sigma = spike_threshold(quotes, seg, k)
    if threshold is not None:
        thr = threshold
    m = spike_mask(quotes, seg, direction, thr)
    idx = np.where(m)[0]
    r = log_returns(quotes, seg)
    # event magnitude: sum of consecutive spike-tick returns
    raw = np.nan_to_num(direction * r, nan=0.0) > thr
    mags = []
    ends = []
    for i in idx:
        j = i
        while j + 1 < len(raw) and raw[j + 1] and seg[j + 1] == seg[i]:
            j += 1
        mags.append(float(np.nansum(r[i:j + 1])))
        ends.append(j)
    mags_a = np.abs(np.array(mags)) if mags else np.empty(0)
    if len(mags_a) >= 3:
        q1, q2 = np.quantile(mags_a, [1 / 3, 2 / 3])
    else:
        q1 = q2 = float("inf")
    events = []
    for n, (i, j) in enumerate(zip(idx, ends)):
        prev_i = idx[n - 1] if n > 0 else None
        same = prev_i is not None and seg[prev_i] == seg[i]
        a = abs(mags[n])
        events.append({
            "index": int(i), "epoch": int(epochs[i]), "price_before": float(quotes[i - 1]) if i > 0 else None,
            "price_after": float(quotes[j]), "ret": mags[n], "magnitude_sigma": a / sigma if sigma else None,
            "abs_move": float(abs(quotes[j] - quotes[i - 1])) if i > 0 else None, "direction": int(direction),
            "size_class": "small" if a <= q1 else ("medium" if a <= q2 else "large"),
            "ticks_since_prev": int(i - prev_i) if same else None,
            "secs_since_prev": float(epochs[i] - epochs[prev_i]) if same else None,
        })
    return {"events": events, "threshold": float(thr), "sigma": float(sigma), "k": float(k or settings.spike_k),
            "mask": m, "segments": seg, "size_cuts": [float(q1), float(q2)]}


def sensitivity(quotes: np.ndarray, seg: np.ndarray, direction: int, sigma: float,
                ks=(4, 6, 8, 12, 20, 40)) -> list[dict]:
    out = []
    for k in ks:
        m = spike_mask(quotes, seg, direction, k * sigma)
        out.append({"k": k, "count": int(m.sum()), "per_1000_ticks": float(m.sum() / max(len(quotes), 1) * 1000)})
    return out


MIN_TICKS_FOR_DEFINITION = 5000  # below this, every scan is a full scan (threshold not yet reliable)


def _last_full_scan(eng, symbol: str) -> dict | None:  # noqa: ANN001
    r = db.exec_sql("SELECT payload, created_at FROM research WHERE symbol=:s AND kind='spike_scan' "
                    "ORDER BY id DESC LIMIT 1", {"s": symbol}, eng)
    return {**(db.jloads(r[0]["payload"], {}) or {}), "_at": r[0]["created_at"]} if r else None


def scan_and_store(symbol: str, direction: int, engine=None, full_every: float = 6 * 3600) -> dict:  # noqa: ANN001
    """Detect spikes and persist. Full rescan (threshold + size classes re-estimated) every `full_every`
    seconds; in between, only new ticks are scanned with the stored definition."""
    eng = engine or db.get_engine()
    last = _last_full_scan(eng, symbol)
    incremental_ok = bool(last and time.time() - last["_at"] < full_every and last.get("last_epoch"))
    if incremental_ok:
        # A full rescan is required when history older than the last full scan has arrived since
        # (e.g. backfill finished after the first scan) or when that scan saw too few ticks to
        # estimate the threshold reliably.
        r = db.exec_sql("SELECT MIN(epoch) AS a FROM ticks WHERE symbol=:s", {"s": symbol}, eng)
        min_epoch = r[0]["a"] if r else None
        if (last.get("first_epoch") is None or (min_epoch is not None and min_epoch < last["first_epoch"])
                or (last.get("n_ticks") or 0) < MIN_TICKS_FOR_DEFINITION):
            incremental_ok = False
    if incremental_ok:
        ep, q = load_ticks(symbol, start=int(last["last_epoch"]) - 3600, engine=eng)
        if len(ep) < 3:
            return {"symbol": symbol, "mode": "incremental", "new": 0}
        res = detect(ep, q, direction, threshold=last["threshold"])
        q1, q2 = last.get("size_cuts") or [float("inf"), float("inf")]
        now = time.time()
        rows = []
        for e in res["events"]:
            a = abs(e["ret"])
            rows.append({"symbol": symbol, "epoch": e["epoch"], "price_before": e["price_before"],
                         "price_after": e["price_after"], "ret": e["ret"],
                         "magnitude_sigma": a / last["sigma"] if last.get("sigma") else None, "abs_move": e["abs_move"],
                         "direction": e["direction"], "size_class": "small" if a <= q1 else ("medium" if a <= q2 else "large"),
                         "ticks_since_prev": e["ticks_since_prev"], "secs_since_prev": e["secs_since_prev"],
                         "threshold": last["threshold"], "detected_at": now})
        n = db.insert_ignore(eng, db.spikes, rows)
        return {"symbol": symbol, "mode": "incremental", "new": n}
    ep, q = load_ticks(symbol, engine=eng)
    if len(ep) < 100:
        return {"symbol": symbol, "ticks": int(len(ep)), "spikes": 0, "note": "insufficient ticks"}
    res = detect(ep, q, direction)
    now = time.time()
    rows = [{"symbol": symbol, "epoch": e["epoch"], "price_before": e["price_before"], "price_after": e["price_after"],
             "ret": e["ret"], "magnitude_sigma": e["magnitude_sigma"], "abs_move": e["abs_move"],
             "direction": e["direction"], "size_class": e["size_class"], "ticks_since_prev": e["ticks_since_prev"],
             "secs_since_prev": e["secs_since_prev"], "threshold": res["threshold"], "detected_at": now}
            for e in res["events"]]
    # size classes are recomputed as history grows: replace rows for this symbol
    with eng.begin() as conn:
        conn.execute(db.spikes.delete().where(db.spikes.c.symbol == symbol))
    db.insert_ignore(eng, db.spikes, rows)
    sens = sensitivity(q, res["segments"], direction, res["sigma"])
    with eng.begin() as conn:
        conn.execute(db.research.insert().values(symbol=symbol, kind="spike_scan", created_at=now, payload=db.jdump(
            db.clean_json({"threshold": res["threshold"], "sigma": res["sigma"], "size_cuts": res["size_cuts"],
                           "first_epoch": int(ep[0]), "last_epoch": int(ep[-1]), "n_ticks": int(len(ep)),
                           "spikes": len(rows)}))))
    return {"symbol": symbol, "ticks": int(len(ep)), "spikes": len(rows), "threshold": res["threshold"],
            "sigma": res["sigma"], "k": res["k"], "size_cuts": res["size_cuts"], "sensitivity": sens}
