"""Candles, spike detection, gaps, metrics, discovery."""
import numpy as np
import pytest

from app.deriv.discovery import match_boom_crash
from app.research import metrics as M
from app.research.candles import build_candles
from app.research.data import segment_ids
from app.research.spikes import detect
from tests.mock_deriv import ACTIVE
from tests.sim import simulate


def test_candles_ohlc_and_availability():
    ep = np.array([0, 1, 2, 5, 6, 9, 10, 14], dtype=np.int64) + 1000
    q = np.array([1, 3, 2, 5, 4, 6, 7, 8], dtype=float)
    c = build_candles(ep, q, 5)
    assert c["time"].tolist() == [1000, 1005, 1010]
    assert c["open"].tolist() == [1, 5, 7] and c["close"].tolist() == [2, 6, 8]
    assert c["high"].tolist() == [3, 6, 8] and c["low"].tolist() == [1, 4, 7]
    assert c["ticks"].tolist() == [3, 3, 2]
    assert c["avail"].tolist() == [1005, 1010, 1015]  # usable only once closed
    assert "volume" not in c.columns  # never invent volume


def test_spike_detection_recovers_simulated_spikes():
    ep, q, true_sp = simulate(n=40000, mean_gap=250, seed=5)
    d = detect(ep, q, 1)
    found = {e["index"] for e in d["events"]}
    recall = len(found & set(true_sp)) / len(true_sp)
    precision = len(found & set(true_sp)) / max(len(found), 1)
    assert recall > 0.97 and precision > 0.99
    assert {e["size_class"] for e in d["events"]} == {"small", "medium", "large"}
    # crash direction finds nothing on boom data
    assert len(detect(ep, q, -1)["events"]) == 0


def test_no_spike_or_return_across_gap():
    ep, q, _ = simulate(n=5000, mean_gap=10 ** 9, seed=6, gaps=[(2500, 3600)])
    q = q.copy()
    q[2500:] *= 1.05  # price jump hidden inside the data gap
    seg = segment_ids(ep)
    assert seg[2499] != seg[2500]
    assert all(e["index"] != 2500 for e in detect(ep, q, 1)["events"])


def test_discovery_by_display_name_only():
    found, missing = match_boom_crash(ACTIVE)
    assert [f["symbol"] for f in found] == ["TBOOM1000", "TCRASH500"]  # codes taken from API, never assumed
    assert found[0]["direction"] == 1 and found[1]["direction"] == -1
    assert "Boom 300 Index" in missing and len(missing) == 8


def test_metrics_and_trials():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 0])
    p = np.array([.1, .9, .2, .8, .6, .3, .1, .2])
    assert M.brier(y, p) < M.brier(y, np.full(8, y.mean()))
    assert M.bss(y, p, y.mean()) > 0
    lo, hi = M.wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    q = M.bh_fdr(np.array([0.01, 0.04, 0.03, 0.5]))
    assert (q >= np.array([0.01, 0.04, 0.03, 0.5])).all()
    # clustered signals collapse to one trial per horizon bucket
    ep = np.arange(100, dtype=float)
    trials, par = M.thinned_trials(ep, np.arange(0, 100), 10)
    assert len(trials) == 10 and set(par.tolist()) == {0, 1}


def test_trial_stats_negative_control():
    """Persistent random signal + memoryless spikes: trial-based p-values must be ~uniform (not tiny)."""
    rng = np.random.default_rng(0)
    pvals = []
    for k in range(40):
        n = 20000
        sp = rng.random(n) < 1 / 100
        y = np.array([sp[i + 1:i + 61].any() for i in range(n)], dtype=int)
        sig = np.repeat(rng.random(n // 200) < 0.3, 200)  # persistent, clustered signal
        st = M.trial_stats(y, np.arange(n, dtype=float), sig, np.arange(n - 61), 60)
        pvals.append(st["p_value"])
    assert np.mean(np.array(pvals) < 0.05) <= 0.15


@pytest.mark.parametrize("k", [6, 8, 12])
def test_threshold_sensitivity_stable(k):
    ep, q, true_sp = simulate(n=20000, mean_gap=200, seed=8)
    assert abs(len(detect(ep, q, 1, k=k)["events"]) - len(true_sp)) <= 3
