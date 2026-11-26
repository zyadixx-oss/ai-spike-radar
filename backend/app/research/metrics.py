"""Evaluation metrics, calibration diagnostics and block-bootstrap confidence intervals."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def bss(y: np.ndarray, p: np.ndarray, p_ref: np.ndarray | float) -> float:
    ref = brier(y, np.broadcast_to(p_ref, y.shape).astype(float))
    return float(1 - brier(y, p) / ref) if ref > 0 else float("nan")


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, s))


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else float("nan")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    m = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def binom_sf_pvalue(k: int, n: int, p0: float) -> float:
    """One-sided P(X >= k) under Binomial(n, p0)."""
    from scipy.stats import binom
    if n == 0:
        return 1.0
    return float(binom.sf(k - 1, n, min(max(p0, 1e-12), 1 - 1e-12)))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp and prec + rec) else 0.0
    far = fp / (fp + tn) if fp + tn else float("nan")
    lo, hi = wilson(tp, tp + fp)
    return {"threshold": float(thr), "n_predictions": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "precision_ci": [lo, hi], "recall": rec, "f1": f1, "false_alarm_rate": far,
            "false_discovery_rate": (fp / (tp + fp)) if tp + fp else float("nan")}


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    if len(y) == 0:
        return {"ece": float("nan"), "bins": []}
    qs = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(qs) < 2:
        qs = np.array([p.min(), p.max() + 1e-12])
    b = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, len(qs) - 2)
    out, ece = [], 0.0
    for k in range(len(qs) - 1):
        m = b == k
        if not m.any():
            continue
        mp, my = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(mp - my)
        out.append({"p_mean": mp, "y_rate": my, "n": int(m.sum())})
    return {"ece": float(ece), "bins": out}


def block_bootstrap(stat_fn, n: int, block: int, iters: int, seed: int = 0) -> np.ndarray:
    """Moving-block bootstrap over row indices; stat_fn(indices) -> float."""
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    nb = int(math.ceil(n / block))
    vals = []
    for _ in range(iters):
        starts = rng.integers(0, max(n - block + 1, 1), nb)
        ix = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        v = stat_fn(ix)
        if v == v:
            vals.append(v)
    return np.asarray(vals)


def ci(vals: np.ndarray, alpha: float = 0.05) -> list[float]:
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]


def event_detection(spike_epochs: np.ndarray, alert_epochs: np.ndarray, horizon_sec: float,
                    period: tuple[float, float]) -> dict:
    """Share of spikes (inside period) preceded by >=1 alert within horizon, and mean lead time."""
    se = spike_epochs[(spike_epochs >= period[0] + horizon_sec) & (spike_epochs <= period[1])]
    if len(se) == 0:
        return {"spikes": 0, "detected": 0, "detected_share": float("nan"), "mean_lead_sec": float("nan")}
    ae = np.sort(alert_epochs)
    det, leads = 0, []
    for s in se:
        lo = np.searchsorted(ae, s - horizon_sec, side="left")
        hi = np.searchsorted(ae, s, side="left")
        if hi > lo:
            det += 1
            leads.append(s - ae[lo])  # earliest alert inside the window
    return {"spikes": int(len(se)), "detected": det, "detected_share": det / len(se),
            "mean_lead_sec": float(np.mean(leads)) if leads else float("nan")}


def full_report(y: np.ndarray, p: np.ndarray, base_rate: float, thr: float | None = None) -> dict:
    r = {"n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()) if len(y) else float("nan"),
         "brier": brier(y, p), "brier_base": brier(y, np.full(len(y), base_rate)),
         "bss_vs_base_rate": bss(y, p, base_rate), "pr_auc": pr_auc(y, p), "roc_auc": roc_auc(y, p),
         "log_loss": log_loss(y, p), "calibration": reliability(y, p)}
    if thr is not None:
        r["at_threshold"] = threshold_metrics(y, p, thr)
    return r


def thinned_trials(epochs: np.ndarray, signal_rows: np.ndarray, horizon_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Turn (possibly clustered) signal rows into ~independent trials: the first signal in each
    horizon-length time bucket. Returns (trial_rows, bucket_parity). Trials within the same parity class
    have non-overlapping label windows, so their outcomes don't share a future spike."""
    if len(signal_rows) == 0:
        return signal_rows, signal_rows
    h = max(float(horizon_sec), 1e-9)
    b = np.floor(epochs[signal_rows] / h).astype(np.int64)
    _, first = np.unique(b, return_index=True)
    return signal_rows[first], b[first] % 2


def trial_stats(y: np.ndarray, epochs: np.ndarray, mask: np.ndarray, rows: np.ndarray, horizon_sec: float,
                base: float | None = None) -> dict:
    """Signal evaluation that is robust to overlapping labels (see thinned_trials)."""
    rows = np.asarray(rows)
    base = float(y[rows].mean()) if base is None and len(rows) else (base if base is not None else float("nan"))
    sig_rows = rows[mask[rows]]
    trials, par = thinned_trials(epochs, sig_rows, horizon_sec)
    n, tp = int(len(trials)), int(y[trials].sum()) if len(trials) else 0
    pv, lbs = [], []
    for k in (0, 1):
        t = trials[par == k]
        nk, tk = int(len(t)), int(y[t].sum()) if len(t) else 0
        pv.append(binom_sf_pvalue(tk, nk, base) if nk else 1.0)
        lbs.append(wilson(tk, nk)[0] if nk else float("nan"))
    prec = tp / n if n else float("nan")
    lo, hi = wilson(tp, n)
    return {"n_signals": n, "raw_signal_rows": int(len(sig_rows)), "tp": tp, "fp": n - tp, "precision": prec,
            "precision_ci": [lo, hi], "base_rate": base, "lift": prec / base if n and base > 0 else float("nan"),
            "p_value": float(max(pv)), "lb_conservative": float(np.nanmin(lbs)) if not np.all(np.isnan(lbs)) else float("nan")}
