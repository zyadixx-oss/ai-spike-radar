"""Dataset assembly: causal features at sampled evaluation ticks + future labels for every horizon."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import HORIZONS, TIMEFRAMES, settings
from .candles import build_candles
from .data import segment_bounds, segment_ids, tick_interval_stats
from .features import build_features, feature_columns
from .labels import make_labels
from .spikes import log_returns, robust_sigma, spike_mask

WARMUP_TICKS = 200


@dataclass
class Dataset:
    symbol: str
    direction: int
    epochs: np.ndarray          # all ticks
    quotes: np.ndarray
    seg: np.ndarray
    sigma: float
    threshold: float
    spike_idx: np.ndarray       # tick indices of spike starts
    idx: np.ndarray             # evaluation tick indices (sampled)
    X: pd.DataFrame             # features at idx
    feature_cols: list[str]
    timeframes: list[str]
    mean_dt: float
    labels: dict = field(default_factory=dict)  # horizon -> (y, valid, lead)
    fit_end_index: int = 0       # last tick index used to estimate sigma/threshold

    @property
    def row_epochs(self) -> np.ndarray:
        return self.epochs[self.idx]

    def horizon_seconds(self, key: str) -> float:
        kind, size = next((k, s) for h, k, s in HORIZONS if h == key)
        return float(size) if kind == "sec" else size * self.mean_dt * 1.5

    def spike_free_returns(self) -> tuple[np.ndarray, np.ndarray]:
        r = log_returns(self.quotes, self.seg)
        spk = np.nan_to_num(self.direction * r, nan=0.0) > self.threshold
        return np.where(spk, 0.0, np.nan_to_num(r, nan=0.0)), spk.astype(np.float32)


def calibrate_scale(quotes: np.ndarray, seg: np.ndarray, end_index: int) -> float:
    r = log_returns(quotes[:end_index], seg[:end_index])
    return robust_sigma(r)


def build_dataset(symbol: str, direction: int, epochs: np.ndarray, quotes: np.ndarray,
                  timeframes: list[str] | None = None, max_rows: int | None = None,
                  stride: int | None = None, fit_frac: float = 0.6, sigma: float | None = None,
                  threshold: float | None = None, horizons: list[str] | None = None) -> Dataset:
    timeframes = timeframes or settings.feature_timeframes
    max_rows = max_rows or settings.max_train_rows
    n = len(epochs)
    seg = segment_ids(epochs)
    fit_end = max(int(n * fit_frac), 2)
    if sigma is None:
        sigma = calibrate_scale(quotes, seg, fit_end)  # estimated on the earliest (training) part only
    if threshold is None:
        threshold = settings.spike_k * sigma
    m = spike_mask(quotes, seg, direction, threshold)
    spike_idx = np.where(m)[0]
    seg_start, _ = segment_bounds(seg)
    pos = np.arange(n) - seg_start
    stride = stride or max(settings.sample_stride, int(math.ceil(n / max_rows)))
    cand = np.arange(0, n, stride)
    idx = cand[pos[cand] >= WARMUP_TICKS]
    X = build_features(epochs, quotes, direction, sigma, threshold, timeframes, idx)
    st = tick_interval_stats(epochs)
    ds = Dataset(symbol, direction, epochs, quotes, seg, sigma, threshold, spike_idx, idx,
                 X, feature_columns(timeframes), timeframes, st["mean"] or 1.0, fit_end_index=fit_end)
    for key, kind, size in HORIZONS:
        if horizons and key not in horizons:
            continue
        ds.labels[key] = make_labels(epochs, spike_idx, idx, "tick" if kind == "tick" else "sec", size, seg)
    ds.labels["__next_tick"] = make_labels(epochs, spike_idx, idx, "tick", 1, seg)
    return ds


def train_intervals(ds: Dataset, end_epoch: float) -> np.ndarray:
    """Inter-spike intervals (ticks) fully observed before end_epoch, same segment only."""
    s = ds.spike_idx[ds.epochs[ds.spike_idx] <= end_epoch]
    if len(s) < 2:
        return np.empty(0)
    same = ds.seg[s[1:]] == ds.seg[s[:-1]]
    return np.diff(s)[same].astype(float)


def m1_candles(ds: Dataset) -> pd.DataFrame:
    return build_candles(ds.epochs, ds.quotes, TIMEFRAMES["M1"])
