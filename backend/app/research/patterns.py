"""Pre-spike pattern research with control windows (to avoid "patterns" that appear everywhere).

For every spike s, the window ENDS at the last tick before the jump (s-1). Control windows end at random
ticks c that are NOT followed by a spike within the next CONTROL_GAP ticks. Window statistics are compared
with Mann-Whitney U tests, Benjamini-Hochberg FDR, and a replication check: a pattern found in the first
half of the history must show the same direction (p < 0.05) in the second half.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chisquare, kstest, mannwhitneyu

from . import metrics as M
from .data import segment_bounds

TIME_WINDOWS = {"5s": 5, "10s": 10, "30s": 30, "1m": 60, "3m": 180, "5m": 300, "10m": 600, "15m": 900, "30m": 1800}
TICK_WINDOWS = {"50t": 50, "100t": 100, "200t": 200, "500t": 500}
CONTROL_GAP = 30
STATS = ["ret", "vol", "range", "up_ratio", "entropy", "absmax", "accel", "compression"]


def _window_stats(lp: np.ndarray, rc: np.ndarray, lo: int, hi: int, sigma: float, rc_long: np.ndarray | None) -> dict:
    w = rc[lo + 1:hi + 1]  # returns inside window (exclude the return INTO the first tick)
    if len(w) < 3:
        return {}
    seg_lp = lp[lo:hi + 1]
    up = float(np.mean(w > 0))
    ent = 0.0 if up in (0.0, 1.0) else float(-(up * np.log2(up) + (1 - up) * np.log2(1 - up)))
    half = len(w) // 2
    vol = float(np.std(w))
    out = {"ret": float(w.sum() / sigma), "vol": vol / sigma, "range": float((seg_lp.max() - seg_lp.min()) / sigma),
           "up_ratio": up, "entropy": ent, "absmax": float(np.abs(w).max() / sigma),
           "accel": float((w[half:].sum() - w[:half].sum()) / sigma)}
    if rc_long is not None and len(rc_long) > len(w) and np.std(rc_long) > 0:
        out["compression"] = vol / float(np.std(rc_long))
    return out


def _collect(points: np.ndarray, epochs, lp, rc, seg_start, sigma) -> dict:  # noqa: ANN001
    res: dict[str, dict[str, list]] = {}
    for a in points:
        s0 = seg_start[a]
        for name, W in TIME_WINDOWS.items():
            lo = int(np.searchsorted(epochs, epochs[a] - W, side="left"))
            if lo < s0:
                continue
            lo4 = int(np.searchsorted(epochs, epochs[a] - 4 * W, side="left"))
            st = _window_stats(lp, rc, lo, a, sigma, rc[lo4 + 1:a + 1] if lo4 >= s0 else None)
            for k, v in st.items():
                res.setdefault(name, {}).setdefault(k, []).append(v)
        for name, W in TICK_WINDOWS.items():
            lo = a - W
            if lo < s0:
                continue
            lo4 = a - 4 * W
            st = _window_stats(lp, rc, lo, a, sigma, rc[lo4 + 1:a + 1] if lo4 >= s0 else None)
            for k, v in st.items():
                res.setdefault(name, {}).setdefault(k, []).append(v)
    return res


def pre_spike_patterns(epochs: np.ndarray, quotes: np.ndarray, seg: np.ndarray, spike_idx: np.ndarray,
                       rc: np.ndarray, sigma: float, seed: int = 0, max_events: int = 3000) -> dict:
    rng = np.random.default_rng(seed)
    seg_start, _ = segment_bounds(seg)
    lp = np.log(quotes)
    n = len(epochs)
    pre = spike_idx - 1
    pre = pre[(pre > 0) & (seg[np.clip(pre, 0, n - 1)] == seg[spike_idx])]
    if len(pre) > max_events:
        pre = np.sort(rng.choice(pre, max_events, replace=False))
    # controls: not followed by a spike within CONTROL_GAP ticks, and not the pre-spike tick itself
    is_sp = np.zeros(n, bool)
    is_sp[spike_idx] = True
    nxt_win = np.convolve(is_sp[::-1].astype(int), np.ones(CONTROL_GAP + 1, int), "full")[:n][::-1]
    cand = np.where((nxt_win - is_sp) == 0)[0]
    cand = cand[cand > 0]
    if len(cand) == 0 or len(pre) < 10:
        return {"status": "insufficient_spikes", "events": int(len(pre))}
    ctrl = np.sort(rng.choice(cand, min(len(cand), 3 * len(pre)), replace=False))
    half_e = np.median(epochs[pre])
    groups = {"pre": pre, "ctrl": ctrl}
    stats = {g: _collect(p, epochs, lp, rc, seg_start, sigma) for g, p in groups.items()}
    halves = {}
    for h, cond in (("A", lambda e: e <= half_e), ("B", lambda e: e > half_e)):
        halves[h] = {g: _collect(p[cond(epochs[p])], epochs, lp, rc, seg_start, sigma) for g, p in groups.items()}
    rows = []
    for win in list(TIME_WINDOWS) + list(TICK_WINDOWS):
        for st in STATS:
            a = np.asarray(stats["pre"].get(win, {}).get(st, []))
            b = np.asarray(stats["ctrl"].get(win, {}).get(st, []))
            if len(a) < 10 or len(b) < 10:
                continue
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            auc = float(u / (len(a) * len(b)))
            rep = {}
            for h in ("A", "B"):
                ah = np.asarray(halves[h]["pre"].get(win, {}).get(st, []))
                bh = np.asarray(halves[h]["ctrl"].get(win, {}).get(st, []))
                if len(ah) >= 10 and len(bh) >= 10:
                    uh, ph = mannwhitneyu(ah, bh, alternative="two-sided")
                    rep[h] = {"auc": float(uh / (len(ah) * len(bh))), "p": float(ph)}
            replicated = bool(rep.get("A") and rep.get("B") and rep["A"]["p"] < 0.05 and rep["B"]["p"] < 0.05
                              and (rep["A"]["auc"] - 0.5) * (rep["B"]["auc"] - 0.5) > 0)
            rows.append({"window": win, "stat": st, "n_pre": int(len(a)), "n_ctrl": int(len(b)),
                         "pre_median": float(np.median(a)), "ctrl_median": float(np.median(b)), "auc": auc,
                         "p_value": float(p), "halves": rep, "replicated": replicated})
    q = M.bh_fdr(np.array([r["p_value"] for r in rows])) if rows else []
    for r, qq in zip(rows, q):
        r["q_value"] = float(qq)
        r["significant"] = bool(qq < 0.05)
        r["robust"] = bool(qq < 0.05 and r["replicated"])
    rows.sort(key=lambda r: r["p_value"])
    return {"status": "ok", "events": int(len(pre)), "controls": int(len(ctrl)), "control_gap_ticks": CONTROL_GAP,
            "tests": rows, "n_significant": int(sum(r["significant"] for r in rows)),
            "n_robust": int(sum(r["robust"] for r in rows)),
            "interpretation": ("robust = FDR q<0.05 AND same-direction p<0.05 in both halves of history. "
                               "AUC 0.5 = no difference between pre-spike and control windows.")}


def event_study(quotes: np.ndarray, seg: np.ndarray, spike_idx: np.ndarray, sigma: float, before: int = 200,
                after: int = 200, max_events: int = 2000) -> dict:
    lp = np.log(quotes)
    n = len(quotes)
    paths = []
    for s in spike_idx[-max_events:]:
        lo, hi = s - before, s + after
        if lo < 0 or hi >= n or seg[lo] != seg[s] or seg[hi] != seg[s]:
            continue
        paths.append((lp[lo:hi + 1] - lp[s - 1]) / sigma)
    if not paths:
        return {"status": "insufficient", "events": 0}
    P = np.vstack(paths)
    return {"status": "ok", "events": int(len(P)), "offsets": list(range(-before, after + 1, 5)),
            "mean": P.mean(axis=0)[::5].tolist(), "p25": np.quantile(P, 0.25, axis=0)[::5].tolist(),
            "p75": np.quantile(P, 0.75, axis=0)[::5].tolist(), "units": "log-price change vs tick before spike, in sigma"}


def interarrival(spike_idx: np.ndarray, seg: np.ndarray, epochs: np.ndarray) -> dict:
    """Is spike timing memoryless? KS vs exponential with same mean, binned hazard vs constant (chi-square)."""
    if len(spike_idx) < 30:
        return {"status": "insufficient", "spikes": int(len(spike_idx))}
    same = seg[spike_idx[1:]] == seg[spike_idx[:-1]]
    iv = np.diff(spike_idx)[same].astype(float)
    iv_s = np.diff(epochs[spike_idx])[same].astype(float)
    if len(iv) < 30:
        return {"status": "insufficient", "spikes": int(len(spike_idx))}
    mean = float(iv.mean())
    ks = kstest(iv, "expon", args=(0, mean))
    # empirical discrete hazard in quantile bins
    edges = np.unique(np.quantile(iv, np.linspace(0, 1, 11)))
    hz = []
    for a, b in zip(edges[:-1], edges[1:]):
        at_risk = float(np.sum(np.clip(np.minimum(iv, b) - a, 0, None)))  # ticks spent at risk in (a, b]
        ev = float(np.sum((iv > a) & (iv <= b)))
        hz.append({"from": float(a), "to": float(b), "events": ev, "at_risk_ticks": at_risk,
                   "hazard_per_tick": ev / at_risk if at_risk else None})
    obs = np.array([h["events"] for h in hz])
    exp = np.array([h["at_risk_ticks"] for h in hz]) / mean
    exp = exp * obs.sum() / exp.sum() if exp.sum() else exp
    chi = chisquare(obs, exp) if len(obs) > 1 and (exp > 0).all() else None
    from scipy.stats import weibull_min
    c, _, scale = weibull_min.fit(iv, floc=0)
    rng = np.random.default_rng(0)
    cs = []
    for _ in range(100):
        smp = rng.choice(iv, len(iv), replace=True)
        cs.append(weibull_min.fit(smp, floc=0)[0])
    c_ci = M.ci(np.array(cs))
    memoryless = bool(ks.pvalue > 0.05 and (chi is None or chi.pvalue > 0.05) and c_ci[0] <= 1 <= c_ci[1])
    return {"status": "ok", "intervals": int(len(iv)), "mean_ticks": mean, "median_ticks": float(np.median(iv)),
            "cv": float(iv.std() / mean), "mean_seconds": float(iv_s.mean()),
            "ks_vs_exponential": {"stat": float(ks.statistic), "p_value": float(ks.pvalue)},
            "hazard_bins": hz, "hazard_flatness_chi2": ({"stat": float(chi.statistic), "p_value": float(chi.pvalue)}
                                                        if chi is not None else None),
            "weibull_shape": float(c), "weibull_shape_ci95": c_ci, "weibull_scale": float(scale),
            "consistent_with_memoryless": memoryless,
            "interpretation": ("Weibull shape ~1, KS p>0.05 and flat hazard => time since the last spike carries no "
                               "information (a 'late' spike is NOT more likely). Shape>1 => hazard rises with time.")}


def spike_stats(events: list[dict], epochs: np.ndarray) -> dict:
    if not events:
        return {"count": 0}
    mags = np.array([abs(e["ret"]) for e in events])
    sig = np.array([e["magnitude_sigma"] or np.nan for e in events])
    hours = np.array([(e["epoch"] // 3600) % 24 for e in events])
    counts = np.bincount(hours, minlength=24)
    hours_ticks = np.bincount(((epochs // 3600) % 24).astype(int), minlength=24).astype(float)
    exp = hours_ticks / hours_ticks.sum() * counts.sum() if hours_ticks.sum() else None
    chi = chisquare(counts, exp) if exp is not None and (exp > 0).all() else None
    classes = {k: int(sum(e["size_class"] == k for e in events)) for k in ("small", "medium", "large")}
    return {"count": len(events), "per_1000_ticks": len(events) / max(len(epochs), 1) * 1000,
            "abs_log_return_quantiles": dict(zip(["p10", "p25", "p50", "p75", "p90", "max"],
                                                 np.quantile(mags, [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist())),
            "sigma_multiple_median": float(np.nanmedian(sig)), "size_classes": classes,
            "by_hour_utc": counts.tolist(),
            "hour_uniformity_chi2": {"stat": float(chi.statistic), "p_value": float(chi.pvalue)} if chi else None}
