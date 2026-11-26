"""Tick data access and gap-aware segmentation."""
from __future__ import annotations

import numpy as np
from sqlalchemy import select

from .. import db
from ..config import settings


def load_ticks(symbol: str, start: int | None = None, end: int | None = None, engine=None,  # noqa: ANN001
               limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fast path: raw DB-API cursor (avoids per-row SQLAlchemy objects, which hold the GIL for long)."""
    eng = engine or db.get_engine()
    ph = "%s" if eng.dialect.name == "postgresql" else "?"
    sql = f"SELECT epoch, quote FROM ticks WHERE symbol = {ph}"
    params: list = [symbol]
    if start is not None:
        sql += f" AND epoch >= {ph}"
        params.append(int(start))
    if end is not None:
        sql += f" AND epoch <= {ph}"
        params.append(int(end))
    sql += " ORDER BY epoch"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    raw = eng.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
    finally:
        raw.close()
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0].astype(np.int64), arr[:, 1]


def load_recent_ticks(symbol: str, n: int, engine=None) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    eng = engine or db.get_engine()
    t = db.ticks
    q = select(t.c.epoch, t.c.quote).where(t.c.symbol == symbol).order_by(t.c.epoch.desc()).limit(n)
    with eng.connect() as conn:
        rows = conn.execute(q).all()
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    arr = np.asarray(rows[::-1], dtype=np.float64)
    return arr[:, 0].astype(np.int64), arr[:, 1]


def gap_threshold(epochs: np.ndarray) -> float:
    if len(epochs) < 3:
        return settings.gap_min_seconds
    med = float(np.median(np.diff(epochs)))
    return max(settings.gap_min_seconds, settings.gap_factor * med)


def segment_ids(epochs: np.ndarray, thr: float | None = None) -> np.ndarray:
    """Integer segment id per tick; a new segment starts after every gap > thr seconds."""
    if len(epochs) == 0:
        return np.empty(0, dtype=np.int64)
    thr = gap_threshold(epochs) if thr is None else thr
    brk = np.concatenate([[0], (np.diff(epochs) > thr).astype(np.int64)])
    return np.cumsum(brk)


def segment_bounds(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each tick: index of its segment's first tick and last tick (inclusive)."""
    n = len(seg)
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    starts_mask = np.concatenate([[True], seg[1:] != seg[:-1]])
    start_idx = np.where(starts_mask)[0]
    end_idx = np.concatenate([start_idx[1:] - 1, [n - 1]])
    seg_ord = np.cumsum(starts_mask) - 1
    return start_idx[seg_ord], end_idx[seg_ord]


def tick_interval_stats(epochs: np.ndarray) -> dict:
    if len(epochs) < 3:
        return {"median": None, "mean": None}
    d = np.diff(epochs)
    seg = segment_ids(epochs)
    same = seg[1:] == seg[:-1]
    d = d[same] if same.any() else d
    return {"median": float(np.median(d)), "mean": float(np.mean(d))}
