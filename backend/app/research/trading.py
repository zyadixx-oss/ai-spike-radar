"""VIRTUAL trade simulation for evaluating signals. Never sends orders anywhere (see app/safety.py).

Trade model: on a signal at tick i, enter at quotes[i] in the spike direction (long Boom / short Crash);
exit on the first spike within the horizon (at the post-jump tick) or at horizon end. Costs are a
configurable % per round trip (assumption — Deriv's exact spread is not published per instrument).
Overlapping trades are not allowed (next entry only after the previous exit).
"""
from __future__ import annotations

import numpy as np


def exit_indices(epochs: np.ndarray, spike_idx: np.ndarray, idx: np.ndarray, kind: str, size: int,
                 seg: np.ndarray) -> np.ndarray:
    n = len(epochs)
    if kind == "tick":
        end = np.minimum(idx + size, n - 1)
    else:
        end = np.searchsorted(epochs, epochs[idx] + size, side="right") - 1
    # keep exit inside the same segment
    end = np.where(seg[np.clip(end, 0, n - 1)] == seg[idx], end, idx)
    sp = np.sort(spike_idx)
    pos = np.searchsorted(sp, idx, side="right")
    has = pos < len(sp)
    nxt = np.where(has, sp[np.clip(pos, 0, max(len(sp) - 1, 0))] if len(sp) else idx, np.iinfo(np.int64).max)
    return np.where(nxt <= end, nxt, end).astype(np.int64)


def simulate(idx: np.ndarray, signal: np.ndarray, quotes: np.ndarray, exits: np.ndarray, direction: int,
             cost_pct: float) -> dict:
    sel = np.where(signal)[0]
    pnl, last_exit = [], -1
    for r in sel:
        i = idx[r]
        if i <= last_exit:
            continue
        e = exits[r]
        if e <= i:
            continue
        ret = direction * (quotes[e] - quotes[i]) / quotes[i] * 100 - cost_pct
        pnl.append(ret)
        last_exit = e
    p = np.asarray(pnl)
    if len(p) == 0:
        return {"trades": 0, "net_return_pct": 0.0, "profit_factor": None, "max_drawdown_pct": 0.0,
                "win_rate": None, "avg_trade_pct": None, "virtual_only": True}
    wins, losses = p[p > 0].sum(), -p[p < 0].sum()
    eq = np.cumsum(p)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq])) - np.concatenate([[0], eq])))
    return {"trades": int(len(p)), "net_return_pct": float(p.sum()),
            "profit_factor": float(wins / losses) if losses > 0 else (None if wins == 0 else float("inf")),
            "max_drawdown_pct": dd, "win_rate": float(np.mean(p > 0)), "avg_trade_pct": float(p.mean()),
            "cost_pct_per_trade": cost_pct, "virtual_only": True}
