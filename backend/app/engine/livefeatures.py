"""Live feature computation with train/serve parity: the SAME functions as training, applied to the
latest tick only. Candle indicators are recomputed only when a candle of that timeframe closes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAMES
from ..research.candles import build_candles
from ..research.data import load_ticks
from ..research.features import MAX_TICK_LOOKBACK, candle_feature_names, candle_indicators, tick_features
from ..research.spikes import log_returns

TAIL = MAX_TICK_LOOKBACK + 600
CANDLE_LOOKBACK = 160          # candles per timeframe
MAX_LOOKBACK_SECONDS = 8 * 86400


class SymbolFeatureState:
    def __init__(self, symbol: str, direction: int, engine=None):  # noqa: ANN001
        self.symbol, self.direction, self.engine = symbol, direction, engine
        self._candle_cache: dict[tuple[str, float, float], tuple[int, pd.Series, int]] = {}
        self._m1_cache: tuple[int, pd.DataFrame] | None = None

    def _candles(self, tf: str, now_epoch: int, recent: list[tuple[int, float]]) -> pd.DataFrame:
        tf_s = TIMEFRAMES[tf]
        start = now_epoch - min(CANDLE_LOOKBACK * tf_s, MAX_LOOKBACK_SECONDS)
        ep, q = load_ticks(self.symbol, start=start, end=now_epoch, engine=self.engine)
        if recent:
            re = np.array([r[0] for r in recent], dtype=np.int64)
            rq = np.array([r[1] for r in recent], dtype=float)
            m = re > (ep[-1] if len(ep) else -1)
            ep, q = np.concatenate([ep, re[m]]), np.concatenate([q, rq[m]])
        return build_candles(ep, q, tf_s)

    def candle_row(self, tf: str, now_epoch: int, recent, sigma: float, threshold: float) -> dict:  # noqa: ANN001
        tf_s = TIMEFRAMES[tf]
        bucket = now_epoch // tf_s
        key = (tf, sigma, threshold)
        cached = self._candle_cache.get(key)
        if cached is None or cached[0] != bucket:
            c = self._candles(tf, now_epoch, recent)
            closed = c[c["avail"] <= now_epoch] if len(c) else c
            if len(closed):
                ind = candle_indicators(closed.reset_index(drop=True), self.direction)
                row, avail = ind.iloc[-1], int(closed["avail"].iloc[-1])
            else:
                row, avail = pd.Series(dtype=float), now_epoch
            self._candle_cache[key] = (bucket, row, avail)
            cached = self._candle_cache[key]
        _, row, avail = cached
        out = {f"{tf}_{k}": float(v) if v is not None and np.isfinite(v) else np.nan for k, v in row.items()}
        for name in candle_feature_names(tf):
            out.setdefault(name, np.nan)
        out[f"{tf}_age"] = (now_epoch - avail) / tf_s
        return out

    def m1_candles(self, now_epoch: int, recent, days: int = 3) -> pd.DataFrame:  # noqa: ANN001
        minute = now_epoch // 60
        if self._m1_cache is None or self._m1_cache[0] != minute:
            ep, q = load_ticks(self.symbol, start=now_epoch - days * 86400, end=now_epoch, engine=self.engine)
            c = build_candles(ep, q, 60)
            self._m1_cache = (minute, c[c["avail"] <= now_epoch].reset_index(drop=True) if len(c) else c)
        return self._m1_cache[1]

    def features(self, recent: list[tuple[int, float]], sigma: float, threshold: float,
                 timeframes: list[str]) -> tuple[pd.DataFrame, dict]:
        """Feature row at the latest tick + side info (sequence window for deep models)."""
        ep = np.array([r[0] for r in recent], dtype=np.int64)
        q = np.array([r[1] for r in recent], dtype=float)
        tail_ep, tail_q = ep[-TAIL:], q[-TAIL:]
        tf_row = tick_features(tail_ep, tail_q, self.direction, sigma, threshold, np.array([len(tail_ep) - 1])).iloc[0].to_dict()
        # ticks-since-spike over the full buffer (may exceed the rolling tail)
        from ..research.data import segment_ids
        seg = segment_ids(ep)
        r = log_returns(q, seg)
        spk = np.nan_to_num(self.direction * r, nan=0.0) > threshold
        starts = spk & ~np.concatenate([[False], spk[:-1]])
        last_seg_start = int(np.where(seg == seg[-1])[0][0])
        sp_idx = np.where(starts[last_seg_start:])[0]
        n = len(ep)
        if len(sp_idx):
            li = last_seg_start + int(sp_idx[-1])
            tf_row["t_ticks_since_spike"] = float(n - 1 - li)
            tf_row["t_log_ticks_since_spike"] = float(np.log1p(n - 1 - li))
            tf_row["t_secs_since_spike"] = float(ep[-1] - ep[li])
        else:
            tf_row["t_ticks_since_spike"] = tf_row["t_log_ticks_since_spike"] = tf_row["t_secs_since_spike"] = np.nan
        row = dict(tf_row)
        now = int(ep[-1])
        for tf in timeframes:
            if tf != "tick":
                row.update(self.candle_row(tf, now, recent, sigma, threshold))
        rc = np.where(spk, 0.0, np.nan_to_num(r, nan=0.0))
        side = {"rc": rc, "spk": spk.astype(np.float32), "seg_start_last": last_seg_start, "epoch": now}
        return pd.DataFrame([row]), side
