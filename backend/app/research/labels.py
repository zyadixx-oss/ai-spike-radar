"""Labels: did a spike START strictly after tick i and within the horizon? Uses the future BY DESIGN —
labels are never fed into features. Points whose horizon window is not fully observed (segment ends /
data gap) are marked invalid and excluded."""
from __future__ import annotations

import numpy as np

from .data import segment_bounds, segment_ids


def next_spike_index(n: int, spike_idx: np.ndarray) -> np.ndarray:
    """For every tick i: index of the first spike-start strictly after i (or -1)."""
    spike_idx = np.sort(np.asarray(spike_idx, dtype=np.int64))
    pos = np.searchsorted(spike_idx, np.arange(n), side="right")
    out = np.full(n, -1, dtype=np.int64)
    ok = pos < len(spike_idx)
    out[ok] = spike_idx[pos[ok]]
    return out


def make_labels(epochs: np.ndarray, spike_idx: np.ndarray, idx: np.ndarray, kind: str, size: int,
                seg: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y, valid, lead) at eval indices idx. lead = seconds until the spike (nan if none)."""
    n = len(epochs)
    seg = segment_ids(epochs) if seg is None else seg
    _, seg_end = segment_bounds(seg)
    nxt = next_spike_index(n, spike_idx)[idx]
    has = (nxt >= 0)
    nxt_c = np.where(has, nxt, 0)
    same_seg = has & (seg[nxt_c] == seg[idx])
    if kind == "tick":
        hit = same_seg & ((nxt_c - idx) <= size)
        observed = (seg_end[idx] - idx) >= size
    else:
        hit = same_seg & ((epochs[nxt_c] - epochs[idx]) <= size)
        observed = (epochs[seg_end[idx]] - epochs[idx]) >= size
    valid = hit | observed
    lead = np.where(hit, (epochs[nxt_c] - epochs[idx]).astype(float), np.nan)
    return hit.astype(np.int8), valid, lead
