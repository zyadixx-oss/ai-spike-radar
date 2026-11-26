"""TEST-ONLY tick simulator. Never used by the application, never written to the app database.

Used for (a) unit tests of mechanics and (b) statistical control tests:
  * memoryless=True  -> spikes are i.i.d. per tick: NO model should be approved (negative control)
  * memoryless=False -> a planted, detectable pre-spike pattern (volatility compression regime raises the
                        spike hazard): the pipeline SHOULD find and approve it (power check)
"""
from __future__ import annotations

import numpy as np


def simulate(n: int = 200_000, mean_gap: int = 300, direction: int = 1, memoryless: bool = True, seed: int = 0,
             start_epoch: int = 1_700_000_000, gaps: list[tuple[int, int]] | None = None):
    rng = np.random.default_rng(seed)
    sigma = 1e-4
    vol = np.full(n, sigma)
    hazard = np.full(n, 1.0 / mean_gap)
    if not memoryless:
        # compression regimes: ~1/3 of time; hazard x6 inside, reduced outside (same overall mean)
        state = np.zeros(n, bool)
        i = 0
        while i < n:
            calm = rng.integers(200, 600)
            comp = rng.integers(80, 200)
            state[i + calm:i + calm + comp] = True
            i += calm + comp
        vol = np.where(state, sigma * 0.35, sigma)
        frac = state.mean()
        h_in = 6.0 / mean_gap
        h_out = max((1.0 / mean_gap - frac * h_in) / (1 - frac), 1e-6)
        hazard = np.where(state, h_in, h_out)
    spikes = rng.random(n) < hazard
    r = rng.normal(-0.2 * sigma * direction, vol)
    r[spikes] = direction * rng.uniform(40, 160, spikes.sum()) * sigma
    q = 5000.0 * np.exp(np.cumsum(r))
    ep = start_epoch + np.arange(n, dtype=np.int64)
    if gaps:
        ep = ep.copy()
        for at, secs in gaps:
            ep[at:] += secs
    return ep, q, np.where(spikes)[0]
