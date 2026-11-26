"""AI Strategy Arena — automatically builds and tests rule-based strategies.

Each candidate rule = (family, timeframe, feature, side, quantile) e.g. "M5 bb_squeeze_rank in lowest 5%".
Quantile cut-offs are computed on DEV only. Protocol per horizon:
   DEV  : screening (lift > 1, one-sided binomial p < 0.05, n >= 30)
   VAL  : Benjamini-Hochberg FDR over all screened rules; parameters/timeframe chosen here
   TEST : reported once for the val-selected rules (never used for choosing)
Then the best validated singles are combined pairwise (AND) and run through the same protocol.
"""
from __future__ import annotations

import itertools
import time
import uuid

import numpy as np

from .. import db
from ..config import HORIZON_MAP, TIMEFRAMES, settings
from . import metrics as M
from .dataset import build_dataset
from .features import STRATEGY_FAMILIES, TICK_FAMILIES
from .models import IntervalHazard
from .trading import exit_indices, simulate
from .validation import dev_val_test

ARENA_HORIZONS = ["30s", "1m", "5m", "100t"]
QUANTILES = (0.01, 0.05, 0.10, 0.20)
ALL_TFS = [tf for tf in TIMEFRAMES if tf != "tick"]


def _evaluate(mask: np.ndarray, y: np.ndarray, rows: np.ndarray, epochs: np.ndarray, hsec: float) -> dict:
    """Trials = first signal per horizon-length bucket (non-overlapping labels => valid binomial test).
    Recall is row-based: share of positive rows covered by the raw signal."""
    st = M.trial_stats(y, epochs, mask, rows, hsec)
    yr = y[rows]
    s = mask[rows]
    rec = float((s & (yr == 1)).sum() / max(int(yr.sum()), 1))
    prec = st["precision"]
    st["recall"] = rec
    st["f1"] = 2 * prec * rec / (prec + rec) if st["n_signals"] and (prec + rec) else 0.0
    return st


def _candidate_rules(ds, dev_rows: np.ndarray):  # noqa: ANN001
    """Yield (family, tf, feature, side, q, mask)."""
    X = ds.X
    for fam, cols in STRATEGY_FAMILIES.items():
        if fam == "hazard_rate":
            continue
        pairs = [("tick", c) for c in cols] if fam in TICK_FAMILIES else \
            [(tf, f"{tf}_{c}") for tf in ds.timeframes for c in cols]
        for tf, col in pairs:
            if col not in X.columns:
                continue
            x = X[col].to_numpy(dtype=float)
            xd = x[dev_rows]
            xd = xd[np.isfinite(xd)]
            if len(xd) < 100:
                continue
            uniq = np.unique(xd)
            if len(uniq) <= 2 and set(uniq.tolist()) <= {0.0, 1.0}:
                yield fam, tf, col, "flag", None, np.nan_to_num(x, nan=0.0) == 1.0
                continue
            for q in QUANTILES:
                hi_t, lo_t = np.quantile(xd, 1 - q), np.quantile(xd, q)
                with np.errstate(invalid="ignore"):
                    yield fam, tf, col, "high", q, (x >= hi_t) & np.isfinite(x)
                    yield fam, tf, col, "low", q, (x <= lo_t) & np.isfinite(x)


def run_arena(symbol: str, direction: int, epochs: np.ndarray, quotes: np.ndarray, engine=None,  # noqa: ANN001
              horizons: list[str] | None = None, persist: bool = True, timeframes: list[str] | None = None) -> dict:
    eng = engine or (db.get_engine() if persist else None)
    horizons = horizons or ARENA_HORIZONS
    run_id = f"ar_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    t0 = time.time()
    tfs = timeframes or ALL_TFS
    if len(epochs) < 5000:
        return {"symbol": symbol, "run_id": run_id, "status": "insufficient_data", "ticks": int(len(epochs)), "horizons": {}}
    ds = build_dataset(symbol, direction, epochs, quotes, timeframes=tfs, horizons=horizons)
    out: dict = {"symbol": symbol, "run_id": run_id, "rows": int(len(ds.idx)), "spikes": int(len(ds.spike_idx)),
                 "horizons": {}, "timeframes": tfs}
    if len(ds.spike_idx) < settings.min_spikes_for_training:
        out["status"] = "insufficient_data"
        return out
    exp_rows: list[dict] = []
    now = time.time()
    for key in horizons:
        y_all, valid, _ = ds.labels[key]
        rows_all = np.where(valid)[0]
        y = y_all.astype(int)
        hsec = ds.horizon_seconds(key)
        ep = ds.row_epochs
        sp = dev_val_test(ds.row_epochs[rows_all], hsec, 500 * ds.mean_dt)
        dev, val, test = rows_all[sp["dev"]], rows_all[sp["val"]], rows_all[sp["test"]]
        if y[dev].sum() < 30 or y[val].sum() < 10:
            out["horizons"][key] = {"status": "insufficient_data"}
            continue
        kind, size = HORIZON_MAP[key]
        exits = exit_indices(ds.epochs, ds.spike_idx, ds.idx, "tick" if kind == "tick" else "sec", size, ds.seg)
        screened: list[dict] = []
        masks: dict[int, np.ndarray] = {}
        n_tested = 0
        fam_counts: dict[str, int] = {}

        def consider(fam, tf, col, side, q, mask, combo=False):  # noqa: ANN001
            nonlocal n_tested
            n_tested += 1
            fk = "combinations" if combo else fam
            fam_counts[fk] = fam_counts.get(fk, 0) + 1
            d = _evaluate(mask, y, dev, ep, hsec)
            if d["n_signals"] >= 20 and d["lift"] > 1 and d["p_value"] < 0.05:
                v = _evaluate(mask, y, val, ep, hsec)
                rec = {"family": fam, "timeframe": tf, "feature": col, "side": side, "q": q, "combo": combo,
                       "dev": d, "val": v}
                masks[len(screened)] = mask
                screened.append(rec)

        for fam, tf, col, side, q, mask in _candidate_rules(ds, dev):
            consider(fam, tf, col, side, q, mask)
        # hazard family: empirical hazard of ticks-since-spike fit on DEV for this horizon
        hz = IntervalHazard().fit(ds.X.iloc[dev], y[dev], None)
        hs = hz.score(ds.X, None)
        for q in QUANTILES:
            consider("hazard_rate", "tick", "hazard", "high", q, hs >= np.quantile(hs[dev], 1 - q))
        # FDR over validation p-values of every screened rule
        qv = M.bh_fdr(np.array([s["val"]["p_value"] for s in screened])) if screened else np.array([])
        for s, qq in zip(screened, qv):
            s["val"]["q_value"] = float(qq)
            s["val_significant"] = bool(qq < 0.10 and s["val"]["n_signals"] >= 15 and s["val"]["lift"] > 1)
        # best per family, chosen on VALIDATION only
        best_per_family: dict[str, int] = {}
        best_key: dict[str, tuple] = {}
        for i, s in enumerate(screened):
            f = s["family"]
            lift = s["val"]["lift"] if np.isfinite(s["val"]["lift"]) else 0.0
            key_s = (s["val_significant"], -s["val"].get("q_value", 1.0), lift)
            if f not in best_key or key_s > best_key[f]:
                best_key[f], best_per_family[f] = key_s, i
        # combinations of the top validated singles
        top = sorted(range(len(screened)), key=lambda i: (-(screened[i]["val"]["lift"] or 0)
                                                          if np.isfinite(screened[i]["val"]["lift"]) else 0))[:8]
        single_count = len(screened)
        for a, b in itertools.combinations(top, 2):
            sa, sb = screened[a], screened[b]
            if sa["feature"] == sb["feature"]:
                continue
            consider(f"{sa['family']}+{sb['family']}", f"{sa['timeframe']}+{sb['timeframe']}",
                     f"{sa['feature']}&{sb['feature']}", f"{sa['side']}&{sb['side']}", None,
                     masks[a] & masks[b], combo=True)
        combos = list(range(single_count, len(screened)))
        if combos:
            qc = M.bh_fdr(np.array([screened[i]["val"]["p_value"] for i in combos]))
            for i, qq in zip(combos, qc):
                screened[i]["val"]["q_value"] = float(qq)
                screened[i]["val_significant"] = bool(qq < 0.10 and screened[i]["val"]["n_signals"] >= 15
                                                     and screened[i]["val"]["lift"] > 1)
        best_combo = max(combos, key=lambda i: (screened[i]["val_significant"], screened[i]["val"]["lift"]
                                                if np.isfinite(screened[i]["val"]["lift"]) else 0), default=None)
        # TEST once for the val-selected rules
        report_ids = set(best_per_family.values()) | ({best_combo} if best_combo is not None else set())
        for i in report_ids:
            s = screened[i]
            s["test"] = _evaluate(masks[i], y, test, ep, hsec)
            s["test"]["confirmed"] = bool(s["test"]["p_value"] < 0.05 and s["test"]["lift"] > 1)
            sig = masks[i].copy()
            sig_test = np.zeros_like(sig)
            sig_test[test] = sig[test]
            s["virtual_trading_test"] = simulate(ds.idx, sig_test, ds.quotes, exits, direction,
                                                 settings.sim_cost_per_trade_pct)
            s["selected"] = True
        # timeframe ranking (validation-based, with test confirmation of each TF's best rule)
        tf_rank: dict[str, dict] = {}
        for i, s in enumerate(screened):
            if s["combo"]:
                continue
            tf = s["timeframe"]
            r = tf_rank.setdefault(tf, {"timeframe": tf, "screened": 0, "val_significant": 0, "best_val_lift": None,
                                        "best_rule": None, "_i": None})
            r["screened"] += 1
            r["val_significant"] += int(s["val_significant"])
            lift = s["val"]["lift"] if np.isfinite(s["val"]["lift"]) else 0
            if s["val_significant"] and (r["best_val_lift"] is None or lift > r["best_val_lift"]):
                r["best_val_lift"], r["_i"] = lift, i
                r["best_rule"] = f"{s['feature']} {s['side']} {s['q'] or ''}".strip()
        for r in tf_rank.values():
            i = r.pop("_i")
            if i is not None:
                tst = screened[i].get("test") or _evaluate(masks[i], y, test, ep, hsec)
                r["test_lift"], r["test_p"] = tst["lift"], tst["p_value"]
                r["test_confirmed"] = bool(tst["p_value"] < 0.05 and tst["lift"] > 1)
        ranking = sorted(tf_rank.values(), key=lambda r: (-(r["val_significant"]), -(r["best_val_lift"] or 0)))
        families = []
        for f, i in best_per_family.items():
            s = screened[i]
            families.append({k: s.get(k) for k in ("family", "timeframe", "feature", "side", "q", "dev", "val", "test",
                                                    "val_significant", "virtual_trading_test")})
        all_test = np.zeros(len(ds.idx), bool)
        all_test[test] = True
        benchmark = simulate(ds.idx, all_test, ds.quotes, exits, direction, settings.sim_cost_per_trade_pct)
        benchmark["note"] = "enter at every opportunity on TEST (no signal) — a strategy must beat this"
        n_val_sig = int(sum(s["val_significant"] for s in screened))
        confirmed = [f for f in families if f.get("test", {}) and f["test"].get("confirmed") and f["val_significant"]]
        out["horizons"][key] = {
            "status": "ok", "rules_tested": n_tested, "rules_per_family": fam_counts, "screened_on_dev": len(screened),
            "val_significant_fdr10": n_val_sig, "test_confirmed": len(confirmed),
            "base_rate_test": float(y[test].mean()) if len(test) else None,
            "families": sorted(families, key=lambda f: (-(f["val_significant"]), -(f["val"]["lift"] or 0))),
            "best_combo": ({k: screened[best_combo].get(k) for k in ("family", "timeframe", "feature", "side", "dev",
                                                                       "val", "test", "val_significant",
                                                                       "virtual_trading_test")}
                           if best_combo is not None else None),
            "timeframe_ranking": ranking, "trading_benchmark_test": benchmark,
            "verdict": ("validated & test-confirmed rules exist" if confirmed else
                        "no rule survived validation FDR + holdout confirmation — no evidence of an exploitable "
                        "pre-spike pattern for this horizon"),
        }
        for i, s in enumerate(screened):
            for split in ("dev", "val", "test"):
                m = s.get(split)
                if not m:
                    continue
                exp_rows.append({"run_id": run_id, "symbol": symbol, "timeframe": s["timeframe"][:8],
                                 "family": s["family"][:48], "params": db.jdump({"feature": s["feature"],
                                                                                 "side": s["side"], "q": s["q"]}),
                                 "horizon": key, "split": split, "n_signals": m["n_signals"], "tp": m["tp"],
                                 "fp": m["fp"], "precision": _f(m["precision"]), "recall": _f(m["recall"]),
                                 "f1": _f(m["f1"]), "lift": _f(m["lift"]), "p_value": _f(m["p_value"]),
                                 "q_value": _f(m.get("q_value")), "base_rate": _f(m["base_rate"]),
                                 "trading": db.jdump(db.clean_json(s.get("virtual_trading_test"))) if split == "test" else None,
                                 "is_combo": s["combo"], "selected": bool(s.get("selected")), "created_at": now})
    out["seconds"] = time.time() - t0
    out["status"] = "ok"
    if persist and eng is not None:
        with eng.begin() as conn:
            for i in range(0, len(exp_rows), 2000):
                conn.execute(db.experiments.insert(), exp_rows[i:i + 2000])
            conn.execute(db.research.insert().values(symbol=symbol, kind="arena", created_at=now,
                                                     payload=db.jdump(db.clean_json(out))))
    return out


def _f(x):  # noqa: ANN001, ANN202
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None
