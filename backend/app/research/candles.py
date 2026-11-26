"""Candle construction from ticks for any timeframe (5s … D1). No synthetic volume — only real tick counts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import db
from ..config import TIMEFRAMES


def build_candles(epochs: np.ndarray, quotes: np.ndarray, tf_seconds: int) -> pd.DataFrame:
    """OHLC candles bucketed by floor(epoch / tf). Columns: time, open, high, low, close, ticks, avail.

    `avail` = bucket end; a candle may only be used by any computation at time t >= avail (it is closed).
    """
    cols = ["time", "open", "high", "low", "close", "ticks", "avail"]
    if len(epochs) == 0:
        return pd.DataFrame(columns=cols)
    if tf_seconds <= 0:
        raise ValueError("tf_seconds must be > 0 (use raw ticks for the 'tick' timeframe)")
    bucket = (epochs // tf_seconds) * tf_seconds
    df = pd.DataFrame({"b": bucket, "q": quotes})
    g = df.groupby("b", sort=True)["q"]
    out = pd.DataFrame({
        "time": g.first().index.values.astype(np.int64),
        "open": g.first().values, "high": g.max().values, "low": g.min().values,
        "close": g.last().values, "ticks": g.size().values.astype(np.int64),
    })
    out["avail"] = out["time"] + tf_seconds
    return out[cols]


def tick_bars(epochs: np.ndarray, quotes: np.ndarray, n: int = 1) -> pd.DataFrame:
    """For the 'tick' timeframe the chart shows each tick as a degenerate candle (or n-tick bars)."""
    if len(epochs) == 0:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "ticks", "avail"])
    idx = np.arange(len(epochs)) // max(n, 1)
    df = pd.DataFrame({"i": idx, "t": epochs, "q": quotes})
    g = df.groupby("i", sort=True)
    out = pd.DataFrame({
        "time": g["t"].first().values.astype(np.int64), "open": g["q"].first().values,
        "high": g["q"].max().values, "low": g["q"].min().values, "close": g["q"].last().values,
        "ticks": g.size().values.astype(np.int64),
    })
    out["avail"] = g["t"].last().values.astype(np.int64)
    return out


def native_candles(symbol: str, granularity: int, start: int | None = None, end: int | None = None,
                   engine=None) -> pd.DataFrame:  # noqa: ANN001
    sql = "SELECT epoch AS time, open, high, low, close FROM candles_native WHERE symbol=:s AND granularity=:g"
    params: dict = {"s": symbol, "g": granularity}
    if start is not None:
        sql += " AND epoch >= :a"
        params["a"] = start
    if end is not None:
        sql += " AND epoch <= :b"
        params["b"] = end
    sql += " ORDER BY epoch"
    rows = db.exec_sql(sql, params, engine)
    df = pd.DataFrame([dict(r) for r in rows], columns=["time", "open", "high", "low", "close"])
    if len(df):
        df["ticks"] = np.nan  # unknown — Deriv candles carry no tick count; never invented
        df["avail"] = df["time"] + granularity
    return df


def candles_for_chart(symbol: str, tf: str, limit: int = 1500, end: int | None = None, engine=None) -> tuple[pd.DataFrame, str]:  # noqa: ANN001
    """Return candles for the chart: built from ticks when coverage allows, else Deriv native candles."""
    from .data import load_ticks
    tf_s = TIMEFRAMES[tf]
    eng = engine or db.get_engine()
    if tf_s == 0:
        rows = db.exec_sql("SELECT epoch, quote FROM ticks WHERE symbol=:s" + (" AND epoch<=:e" if end else "")
                           + " ORDER BY epoch DESC LIMIT :n", {"s": symbol, "n": limit, "e": end}, eng)
        rows = rows[::-1]
        ep = np.array([r["epoch"] for r in rows], dtype=np.int64)
        q = np.array([r["quote"] for r in rows], dtype=float)
        return tick_bars(ep, q, 1), "ticks"
    span = tf_s * limit
    end_e = end
    if end_e is None:
        r = db.exec_sql("SELECT MAX(epoch) AS m FROM ticks WHERE symbol=:s", {"s": symbol}, eng)
        end_e = r[0]["m"] if r and r[0]["m"] is not None else None
    built = pd.DataFrame()
    if end_e is not None:
        ep, q = load_ticks(symbol, start=int(end_e) - span, end=int(end_e), engine=eng)
        built = build_candles(ep, q, tf_s)
    if tf_s >= 60:
        nat = native_candles(symbol, tf_s, engine=eng) if tf_s in (60, 120, 180, 300, 600, 900, 1800, 3600, 14400, 86400) else pd.DataFrame()
        if len(nat) and (len(built) < min(limit, len(nat))):
            # combine: native for older history, tick-built where we have ticks (tick-built wins on overlap)
            if len(built):
                nat = nat[nat["time"] < built["time"].min()]
            merged = pd.concat([nat, built], ignore_index=True).sort_values("time").tail(limit)
            return merged.reset_index(drop=True), "deriv_candles+ticks" if len(built) else "deriv_candles"
    return built.tail(limit).reset_index(drop=True), "ticks"
