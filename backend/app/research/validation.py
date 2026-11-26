"""Time-series validation with purging and embargo.

Layout (by time):   |------------- DEV (60%) -------------|purge|-- VAL (20%) --|purge|-- TEST (20%) --|
  * Model selection: purged walk-forward folds INSIDE DEV only.
  * Calibration + signal thresholds: VAL.
  * TEST is touched once, as a pass/fail acceptance gate — never used to choose between models.
Purge: rows whose label window reaches into the next block are dropped. Embargo: extra rows after each
boundary are dropped so overlapping feature windows don't correlate blocks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray
    name: str


def _purge_mask(epochs: np.ndarray, boundary_epoch: float, horizon_sec: float) -> np.ndarray:
    """True for rows that are safe to keep before boundary (their label window ends before it)."""
    return epochs + horizon_sec < boundary_epoch


def dev_val_test(epochs: np.ndarray, horizon_sec: float, embargo_sec: float,
                 fracs: tuple[float, float] = (0.6, 0.8)) -> dict[str, np.ndarray]:
    n = len(epochs)
    i1, i2 = int(n * fracs[0]), int(n * fracs[1])
    b1, b2 = epochs[i1] if i1 < n else np.inf, epochs[i2] if i2 < n else np.inf
    rows = np.arange(n)
    dev = rows[(rows < i1) & _purge_mask(epochs, b1, horizon_sec)]
    val = rows[(rows >= i1) & (rows < i2) & (epochs >= b1 + embargo_sec) & _purge_mask(epochs, b2, horizon_sec)]
    test = rows[(rows >= i2) & (epochs >= b2 + embargo_sec)]
    return {"dev": dev, "val": val, "test": test}


def walk_forward(epochs: np.ndarray, rows: np.ndarray, n_folds: int, horizon_sec: float,
                 embargo_sec: float, min_train_frac: float = 0.4) -> list[Split]:
    """Expanding-window walk-forward splits over `rows` (sorted by time)."""
    rows = np.asarray(rows)
    n = len(rows)
    if n < 10:
        return []
    start = int(n * min_train_frac)
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    out = []
    for k in range(n_folds):
        a, b = edges[k], edges[k + 1]
        if b - a < 5:
            continue
        boundary = epochs[rows[a]]
        tr = rows[:a]
        tr = tr[_purge_mask(epochs[tr], boundary, horizon_sec)]
        te = rows[a:b]
        te = te[epochs[te] >= boundary + embargo_sec]
        if b < n:
            te = te[_purge_mask(epochs[te], epochs[rows[b]], 0)]
        if len(tr) and len(te):
            out.append(Split(tr, te, f"wf{k + 1}"))
    return out


def assert_no_overlap(train_epochs: np.ndarray, test_epochs: np.ndarray, horizon_sec: float) -> None:
    """Hard check used in tests & pipeline: no training label window may reach the test period."""
    if len(train_epochs) and len(test_epochs):
        if train_epochs.max() + horizon_sec >= test_epochs.min():
            raise AssertionError("leakage: training label window overlaps the test period")
