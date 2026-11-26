"""Training & backtesting pipeline for one symbol.

For each horizon:
 1. dev / val / test split by time (purged + embargoed).
 2. Purged walk-forward inside DEV for every candidate (baselines + models). Each fold calibrates on the
    tail of its own training window. -> out-of-sample metrics used for MODEL SELECTION.
 3. Selected model and all baselines are refit on DEV and calibrated on VAL. Signal-level thresholds are
    derived from VAL (Wilson lower bound of precision vs base rate) — no arbitrary cut-offs.
 4. TEST is evaluated once, as an acceptance gate: the model must beat the base rate AND the best
    baseline (chosen on VAL) with block-bootstrap 95% CI lower bounds > 0. Otherwise it is NOT approved.
 5. Champion/challenger: an approved champion is only replaced by a model that is significantly better on
    a recent window that neither model trained on.
"""
from __future__ import annotations

import io
import logging
import time
import traceback
import uuid

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select

from .. import db
from ..config import HORIZON_MAP, HORIZONS, settings
from . import metrics as M
from .dataset import Dataset, build_dataset, m1_candles, train_intervals
from .deep import SkipModel, sequence_windows, torch_available
from .features import build_features
from .models import Calibrator, Ctx, baseline_factories, lambda_per_tick, model_factories
from .validation import assert_no_overlap, dev_val_test, walk_forward

log = logging.getLogger("train")

LEVELS = ["gray", "green", "yellow", "orange", "red"]


# ---------------------------------------------------------------------------- helpers
def _ctx_for(ds: Dataset, rows: np.ndarray, key: str, base_extra: dict, fit_end_epoch: float | None = None) -> Ctx:
    kind, size = HORIZON_MAP[key]
    ex = {}
    for k, v in base_extra.items():
        ex[k] = v[rows] if isinstance(v, np.ndarray) and len(v) == len(ds.idx) else v
    ep = ds.row_epochs[rows]
    if fit_end_epoch is not None:
        mask = ds.epochs <= fit_end_epoch
        ex["lambda_per_tick"] = lambda_per_tick(int(np.sum(ds.epochs[ds.spike_idx] <= fit_end_epoch)), int(mask.sum()))
        ex["train_intervals"] = train_intervals(ds, fit_end_epoch)
    return Ctx(ep, "tick" if kind == "tick" else "sec", size, ds.mean_dt, ex)


def _fit_predict(factory, ds: Dataset, X: pd.DataFrame, y: np.ndarray, key: str, extra: dict,
                 fit_rows: np.ndarray, cal_rows: np.ndarray, score_rows_list: list[np.ndarray]):
    model = factory()
    fit_end = float(ds.row_epochs[fit_rows].max())
    ctx_fit = _ctx_for(ds, fit_rows, key, extra, fit_end)
    model.fit(X.iloc[fit_rows], y[fit_rows], ctx_fit)
    run_ctx = dict(lambda_per_tick=ctx_fit.extra.get("lambda_per_tick"), train_intervals=ctx_fit.extra.get("train_intervals"))

    def score(rows):  # noqa: ANN001
        c = _ctx_for(ds, rows, key, extra)
        c.extra.update(run_ctx)
        return np.asarray(model.score(X.iloc[rows], c), dtype=float)

    cal = Calibrator().fit(score(cal_rows), y[cal_rows])
    outs = [(cal.predict(s), s) for s in (score(r) for r in score_rows_list)]
    return model, cal, outs, run_ctx


def _split_fit_cal(epochs_rows: np.ndarray, rows: np.ndarray, horizon_sec: float, frac: float = 0.8):
    k = int(len(rows) * frac)
    fit, cal = rows[:k], rows[k:]
    if len(cal) and len(fit):
        boundary = epochs_rows[cal[0]]
        fit = fit[epochs_rows[fit] + horizon_sec < boundary]
    return fit, cal


def level_thresholds(p: np.ndarray, y: np.ndarray, base_rate: float, epochs: np.ndarray | None = None,
                     horizon_sec: float | None = None, min_n: int = 30) -> dict:
    """Derive yellow/orange/red probability cut-offs from VALIDATION calibrated probabilities.
    Significance uses non-overlapping trials (see metrics.trial_stats) when epochs/horizon are given."""
    p = np.round(np.asarray(p, dtype=float), 6)
    uniq = np.unique(p)
    if len(uniq) > 200:
        uniq = np.unique(np.round(np.quantile(p, np.linspace(0.5, 0.999, 120)), 6))
    uniq = uniq[uniq > base_rate]  # a signal level must at least mean "above the normal rate"
    table = []
    rows = np.arange(len(p))
    for t in uniq:
        sel = p >= t
        if epochs is not None and horizon_sec:
            st = M.trial_stats(y, epochs, sel, rows, horizon_sec, base_rate)
            n, tp = st["n_signals"], st["tp"]
            if n < min_n:
                continue
            lo, hi = st["lb_conservative"], st["precision_ci"][1]
            prec = st["precision"]
        else:
            n, tp = int(sel.sum()), int((sel & (y == 1)).sum())
            if n < min_n:
                continue
            lo, hi = M.wilson(tp, n)
            prec = tp / n
        rec = int((sel & (y == 1)).sum()) / max(int(y.sum()), 1)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        table.append({"t": float(t), "n": n, "tp": tp, "precision": prec, "lb": lo, "ub": hi, "recall": rec,
                      "f1": f1, "lift": prec / base_rate if base_rate else None})
    sig = [r for r in table if r["lb"] > base_rate]
    out: dict = {"base_rate": base_rate, "yellow": None, "orange": None, "red": None, "table": table[-60:],
                 "rule": "cuts considered only above the base rate. yellow: lowest cut with Wilson-95% "
                         "LB(precision) > base rate; orange: F1-optimal significant cut; red: lowest cut above orange "
                         "whose LB(precision) >= orange's precision (red additionally requires an approved model)"}
    if not sig:
        return out
    out["yellow"] = min(r["t"] for r in sig)
    best = max(sig, key=lambda r: (r["f1"], -r["t"]))
    out["orange"] = max(best["t"], out["yellow"])
    reds = [r for r in table if r["t"] > out["orange"] and r["lb"] >= best["precision"] and r["n"] >= 20]
    out["red"] = min(r["t"] for r in reds) if reds else None
    for lvl in ("yellow", "orange", "red"):
        t = out[lvl]
        if t is not None:
            r = next(x for x in table if x["t"] == t)
            out[f"{lvl}_val"] = {k: r[k] for k in ("n", "precision", "lb", "ub", "recall", "lift")}
    return out


def level_for(p: float, thr: dict | None, approved: bool) -> str:
    if thr is None or p is None or not np.isfinite(p):
        return "gray"
    if not approved:
        # Unapproved models may at most flag "worth watching" when validation showed significant lift
        return "yellow" if thr.get("yellow") is not None and p >= thr["yellow"] else "green"
    for lvl in ("red", "orange", "yellow"):
        t = thr.get(lvl)
        if t is not None and p >= t:
            return lvl
    return "green"


def _boot_ci(y, p_a, p_b, block, iters, fn):  # noqa: ANN001
    vals = M.block_bootstrap(lambda ix: fn(y[ix], p_a[ix], p_b[ix]), len(y), block, iters)
    return M.ci(vals), float(np.mean(vals)) if len(vals) else float("nan")


def _bss_pair(y, p, ref):  # noqa: ANN001
    b_ref = np.mean((ref - y) ** 2)
    return 1 - np.mean((p - y) ** 2) / b_ref if b_ref > 0 else np.nan


def _prauc_diff(y, p, ref):  # noqa: ANN001
    if y.sum() == 0:
        return np.nan
    return M.pr_auc(y, p) - M.pr_auc(y, ref)


# ---------------------------------------------------------------------------- main
def train_horizon(ds: Dataset, key: str, model_names: list[str] | None = None, run_id: str = "",
                  extra_base: dict | None = None) -> dict:
    y_all, valid, lead = ds.labels[key]
    rows_all = np.where(valid)[0]
    y = y_all.astype(int)
    X = ds.X[ds.feature_cols]
    ep_rows = ds.row_epochs
    hsec = ds.horizon_seconds(key)
    embargo = 500 * ds.mean_dt
    sp = dev_val_test(ep_rows[rows_all], hsec, embargo)
    dev, val, test = rows_all[sp["dev"]], rows_all[sp["val"]], rows_all[sp["test"]]
    res: dict = {"symbol": ds.symbol, "horizon": key, "run_id": run_id, "rows": {"dev": len(dev), "val": len(val),
                 "test": len(test)}, "positives": {"dev": int(y[dev].sum()), "val": int(y[val].sum()),
                 "test": int(y[test].sum())}, "status": "ok", "candidates": {}, "notes": []}
    if len(ds.spike_idx) < settings.min_spikes_for_training or y[dev].sum() < 30 or y[val].sum() < 10 or y[test].sum() < 10:
        res["status"] = "insufficient_data"
        res["notes"].append(f"need >= {settings.min_spikes_for_training} spikes and >=30/10/10 positives in "
                            f"dev/val/test (have spikes={len(ds.spike_idx)}, {res['positives']})")
        return res
    assert_no_overlap(ep_rows[dev], ep_rows[val], hsec)
    assert_no_overlap(ep_rows[val], ep_rows[test], hsec)
    extra = dict(extra_base or {})
    extra["y_next_tick"] = ds.labels["__next_tick"][0].astype(int)

    factories = {**baseline_factories(), **model_factories(settings.enable_deep_models)}
    if model_names:
        factories = {k: v for k, v in factories.items() if k in model_names or k in baseline_factories()}

    folds = walk_forward(ep_rows, dev, 3, hsec, embargo)
    # ---- walk-forward (selection) ----
    for name, fac in factories.items():
        t0 = time.time()
        fold_res = []
        try:
            for f in folds:
                fit, cal = _split_fit_cal(ep_rows, f.train, hsec)
                assert_no_overlap(ep_rows[f.train], ep_rows[f.test], hsec)
                _, _, outs, _ = _fit_predict(fac, ds, X, y, key, extra, fit, cal, [f.test])
                p, s = outs[0]
                yt = y[f.test]
                base = float(y[f.train].mean())
                fold_res.append({"fold": f.name, "n": len(yt), "positives": int(yt.sum()), "brier": M.brier(yt, p),
                                 "bss_vs_base_rate": M.bss(yt, p, base), "pr_auc": M.pr_auc(yt, s),
                                 "roc_auc": M.roc_auc(yt, s), "prevalence": float(yt.mean())})
            ok = [r for r in fold_res if np.isfinite(r["bss_vs_base_rate"])]
            res["candidates"][name] = {
                "kind": fac().kind if hasattr(fac(), "kind") else "model", "status": "ok", "folds": fold_res,
                "wf_bss_mean": float(np.mean([r["bss_vs_base_rate"] for r in ok])) if ok else float("nan"),
                "wf_pr_auc_mean": float(np.nanmean([r["pr_auc"] for r in fold_res])) if fold_res else float("nan"),
                "wf_brier_mean": float(np.mean([r["brier"] for r in fold_res])) if fold_res else float("nan"),
                "fit_seconds": time.time() - t0, "complexity": getattr(fac(), "complexity", 3)}
        except SkipModel as e:
            res["candidates"][name] = {"kind": "model", "status": "skipped", "reason": str(e)}
        except Exception as e:  # noqa: BLE001 - a failing model must not stop the pipeline
            res["candidates"][name] = {"kind": "model", "status": "failed", "reason": f"{type(e).__name__}: {e}",
                                       "trace": traceback.format_exc(limit=3)}
    bl = baseline_factories()
    ranked = sorted([(n, c) for n, c in res["candidates"].items()
                     if c["status"] == "ok" and n not in bl and np.isfinite(c["wf_bss_mean"])],
                    key=lambda kv: (-kv[1]["wf_bss_mean"], kv[1]["complexity"]))
    best_baseline_wf = max((c["wf_bss_mean"] for n, c in res["candidates"].items()
                            if n in bl and c["status"] == "ok" and np.isfinite(c["wf_bss_mean"])), default=0.0)
    res["wf_ranking"] = [n for n, _ in ranked]
    res["best_baseline_wf_bss"] = best_baseline_wf
    if not ranked:
        res["status"] = "no_model"
        return res
    champion_name = ranked[0][0]
    challengers = [n for n, _ in ranked[1:3]]
    res["selected"] = champion_name
    res["selected_beats_baselines_in_wf"] = bool(ranked[0][1]["wf_bss_mean"] > best_baseline_wf)

    # ---- final fits on DEV, calibration on VAL, one-time TEST ----
    finals: dict[str, dict] = {}
    for name in list(bl) + [champion_name] + challengers:
        try:
            model, cal, outs, run_ctx = _fit_predict(factories[name], ds, X, y, key, extra, dev, val, [dev, val, test])
            (p_dev, _), (p_val, s_val), (p_test, s_test) = outs
            finals[name] = {"model": model, "cal": cal, "p_dev": p_dev, "p_val": p_val, "s_val": s_val,
                            "p_test": p_test, "s_test": s_test, "run_ctx": run_ctx}
        except Exception as e:  # noqa: BLE001
            res["candidates"].setdefault(name, {})["final_error"] = f"{type(e).__name__}: {e}"
    if champion_name not in finals:
        res["status"] = "champion_failed"
        return res
    base_rate = float(y[dev].mean())
    yv, yt = y[val], y[test]
    # best baseline chosen on VALIDATION brier (never on test)
    bl_ok = [n for n in bl if n in finals]
    best_bl = min(bl_ok, key=lambda n: M.brier(yv, finals[n]["p_val"]))
    res["best_baseline"] = best_bl
    ch = finals[champion_name]
    thr = level_thresholds(ch["p_val"], yv, float(yv.mean()) if len(yv) else base_rate, ep_rows[val], hsec)
    block = max(50, int(hsec / max(ds.mean_dt, 1e-9) / max(settings.sample_stride, 1)))
    iters = settings.bootstrap_iters
    ci_base, m_base = _boot_ci(yt, ch["p_test"], np.full(len(yt), base_rate), block, iters, _bss_pair)
    ci_bl, m_bl = _boot_ci(yt, ch["p_test"], finals[best_bl]["p_test"], block, iters, _bss_pair)
    ci_pr, m_pr = _boot_ci(yt, ch["s_test"], finals[best_bl]["s_test"], block, iters, _prauc_diff)
    accept = {
        "bss_vs_base_rate": {"mean": m_base, "ci95": ci_base, "pass": bool(ci_base[0] > 0)},
        "bss_vs_best_baseline": {"baseline": best_bl, "mean": m_bl, "ci95": ci_bl, "pass": bool(ci_bl[0] > 0)},
        "pr_auc_gain_vs_best_baseline": {"mean": m_pr, "ci95": ci_pr, "pass": bool(ci_pr[0] > 0)},
        "enough_test_positives": {"value": int(yt.sum()), "pass": bool(yt.sum() >= 15)},
        "significant_validation_threshold": {"pass": thr["yellow"] is not None},
    }
    approved = all(v["pass"] for v in accept.values())
    res["acceptance"] = accept
    res["approved"] = approved
    res["thresholds"] = thr

    # reports for champion + baselines (+ challengers on val only)
    orange = thr.get("orange") or thr.get("yellow")
    rep = {}
    for name, f in finals.items():
        is_main = name == champion_name or name in bl
        rep[name] = {"in_sample": M.full_report(y[dev], f["p_dev"], base_rate),
                     "validation": M.full_report(yv, f["p_val"], base_rate, orange)}
        if is_main:
            rep[name]["test"] = M.full_report(yt, f["p_test"], base_rate, orange)
            rep[name]["test"]["pr_auc"] = M.pr_auc(yt, f["s_test"])
    # event-level metrics for champion on TEST at the orange/yellow cut
    if orange is not None:
        alert_ep = ep_rows[test][ch["p_test"] >= orange]
        spike_ep = ds.epochs[ds.spike_idx]
        rep[champion_name]["test"]["event_detection"] = M.event_detection(
            spike_ep, alert_ep, hsec, (float(ep_rows[test].min()), float(ep_rows[test].max())))
        rep[champion_name]["test"]["event_detection"]["note"] = (
            f"evaluation rows are sampled every {int(np.median(np.diff(ds.idx))) if len(ds.idx) > 1 else 1} ticks; "
            "detected share is a lower bound")
        lead_t = lead[test][(ch["p_test"] >= orange) & (yt == 1)]
        rep[champion_name]["test"]["mean_warning_lead_sec"] = float(np.nanmean(lead_t)) if len(lead_t) else None
        from .trading import exit_indices, simulate
        kind, size = HORIZON_MAP[key]
        ex = exit_indices(ds.epochs, ds.spike_idx, ds.idx[test], "tick" if kind == "tick" else "sec", size, ds.seg)
        rep[champion_name]["test"]["virtual_trading"] = simulate(
            ds.idx[test], ch["p_test"] >= orange, ds.quotes, ex, ds.direction, settings.sim_cost_per_trade_pct)
        rep[champion_name]["test"]["virtual_trading_benchmark"] = {
            **simulate(ds.idx[test], np.ones(len(test), bool), ds.quotes, ex, ds.direction,
                       settings.sim_cost_per_trade_pct), "note": "enter at every opportunity (no signal)"}
    res["reports"] = rep
    res["base_rate"] = base_rate
    res["split_epochs"] = {k: [float(ep_rows[v].min()), float(ep_rows[v].max())] for k, v in
                           (("dev", dev), ("val", val), ("test", test))}
    # feature reference stats for live "reasons"
    Xd = X.iloc[dev]
    res["feature_ref"] = {"median": Xd.median(numeric_only=True).to_dict(),
                          "iqr": (Xd.quantile(0.75, numeric_only=True) - Xd.quantile(0.25, numeric_only=True)).to_dict()}
    imp = {}
    m = ch["model"]
    if hasattr(m, "importances"):
        imp = m.importances(list(X.columns))
    res["importances"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:25])
    res["_finals"] = finals
    res["_challengers"] = challengers
    return res


def dump_artifact(obj: dict) -> bytes:
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=3)
    return buf.getvalue()


def load_artifact(engine, model_id: int) -> dict:  # noqa: ANN001
    with engine.connect() as conn:
        r = conn.execute(select(db.model_artifacts.c.data).where(db.model_artifacts.c.model_id == model_id)).first()
    if r is None:
        raise FileNotFoundError(f"artifact for model {model_id} not found")
    return joblib.load(io.BytesIO(r[0]))


def _save_artifact(ds: Dataset, key: str, name: str, f: dict, res: dict, role: str) -> bytes:
    kind, size = HORIZON_MAP[key]
    return dump_artifact({"symbol": ds.symbol, "direction": ds.direction, "horizon": key, "kind": kind, "size": size,
                 "model_name": name, "model": f["model"], "calibrator": f["cal"], "run_ctx": f["run_ctx"],
                 "feature_cols": ds.feature_cols, "timeframes": ds.timeframes, "sigma": ds.sigma,
                 "threshold": ds.threshold, "mean_dt": ds.mean_dt, "thresholds": res.get("thresholds"),
                 "base_rate": res.get("base_rate"), "feature_ref": res.get("feature_ref"),
                 "importances": res.get("importances"), "approved": bool(res.get("approved")) and role == "champion",
                 "role": role})


def _current_champion(engine, symbol: str, key: str):  # noqa: ANN001
    with engine.connect() as conn:
        return conn.execute(select(db.models).where((db.models.c.symbol == symbol) & (db.models.c.horizon == key)
                                                    & (db.models.c.role == "champion"))
                            .order_by(db.models.c.id.desc())).mappings().first()


def compare_with_champion(engine, ds: Dataset, key: str, res: dict, old) -> dict:  # noqa: ANN001
    """Paired comparison on the new TEST window, only if the old champion never trained on it."""
    test_start = res["split_epochs"]["test"][0]
    tr = db.jloads(old["train_range"], {}) or {}
    if not tr or tr.get("val_end", float("inf")) >= test_start:
        return {"fair": False, "reason": "old champion's training/validation data overlaps the new test window"}
    art = load_artifact(engine, old["id"])
    y_all, valid, _ = ds.labels[key]
    rows = np.where(valid)[0]
    sp = dev_val_test(ds.row_epochs[rows], ds.horizon_seconds(key), 500 * ds.mean_dt)
    test = rows[sp["test"]]
    Xo = build_features(ds.epochs, ds.quotes, ds.direction, art["sigma"], art["threshold"], art["timeframes"], ds.idx[test])
    ctx = Ctx(ds.row_epochs[test], "tick" if art["kind"] == "tick" else "sec", art["size"], ds.mean_dt,
              dict(art.get("run_ctx") or {}))
    if art["model_name"] in ("lstm", "temporal_cnn", "transformer"):
        rc, spk = ds.spike_free_returns()
        from .data import segment_bounds
        ctx.extra["seq"] = sequence_windows(rc, spk, segment_bounds(ds.seg)[0], ds.idx[test], art["sigma"])
    if art["model_name"] == "hidden_markov":
        art["model"].refresh(m1_candles(ds))
    p_old = art["calibrator"].predict(art["model"].score(Xo[art["feature_cols"]], ctx))
    p_new = res["_finals"][res["selected"]]["p_test"]
    yt = y_all[test].astype(int)
    d = M.block_bootstrap(lambda ix: float(np.mean((p_old[ix] - yt[ix]) ** 2 - (p_new[ix] - yt[ix]) ** 2)),
                          len(yt), 100, settings.bootstrap_iters)
    lo, hi = M.ci(d)
    return {"fair": True, "brier_old": M.brier(yt, p_old), "brier_new": M.brier(yt, p_new),
            "brier_improvement_ci95": [lo, hi], "new_significantly_better": bool(lo > 0)}


def register(engine, ds: Dataset, key: str, res: dict) -> dict:  # noqa: ANN001
    """Persist backtests + models with champion/challenger rules."""
    now = time.time()
    run_id = res["run_id"]
    bt_rows = []
    for name, c in res["candidates"].items():
        bt_rows.append({"run_id": run_id, "symbol": ds.symbol, "horizon": key,
                        "kind": "baseline" if name in baseline_factories() else "model", "name": name,
                        "split": "walk_forward", "metrics": db.jdump(db.clean_json({k: v for k, v in c.items()
                                                                                  if k != "trace"})),
                        "created_at": now})
    for name, rep in (res.get("reports") or {}).items():
        for split, m in rep.items():
            bt_rows.append({"run_id": run_id, "symbol": ds.symbol, "horizon": key,
                            "kind": "baseline" if name in baseline_factories() else "model", "name": name,
                            "split": split, "metrics": db.jdump(db.clean_json(m)), "created_at": now})
    summary = {k: v for k, v in res.items() if not k.startswith("_") and k not in ("candidates", "reports")}
    bt_rows.append({"run_id": run_id, "symbol": ds.symbol, "horizon": key, "kind": "summary", "name": "summary",
                    "split": "summary", "metrics": db.jdump(db.clean_json(summary)), "created_at": now})
    with engine.begin() as conn:
        conn.execute(db.backtests.insert(), bt_rows)
    out = {"registered": []}
    if res.get("status") != "ok" or "_finals" not in res:
        return out
    champ = res["selected"]
    train_range = {"dev_start": res["split_epochs"]["dev"][0], "dev_end": res["split_epochs"]["dev"][1],
                   "val_end": res["split_epochs"]["val"][1], "test_start": res["split_epochs"]["test"][0],
                   "test_end": res["split_epochs"]["test"][1]}
    old = _current_champion(engine, ds.symbol, key)
    decision = {"action": "promote", "reason": "no current champion"}
    if old is not None:
        if not old["approved"]:
            decision = {"action": "promote", "reason": "current champion is not approved"}
        else:
            cmp_ = compare_with_champion(engine, ds, key, res, old)
            decision = {"comparison": cmp_}
            if cmp_.get("fair") and cmp_.get("new_significantly_better") and res.get("approved"):
                decision.update(action="promote", reason="new model significantly better on unseen window")
            else:
                decision.update(action="keep", reason="improvement not proven on data unseen by both models")
    metrics_payload = db.jdump(db.clean_json({"acceptance": res.get("acceptance"), "decision": decision,
                                              "wf": res["candidates"].get(champ),
                                              "test": res["reports"][champ].get("test"),
                                              "validation": res["reports"][champ].get("validation"),
                                              "best_baseline": res.get("best_baseline"),
                                              "base_rate": res.get("base_rate"), "run_id": run_id}))
    role = "champion" if decision["action"] == "promote" else "challenger"
    blob = _save_artifact(ds, key, champ, res["_finals"][champ], res, role)
    with engine.begin() as conn:
        if role == "champion" and old is not None:
            conn.execute(db.models.update().where(db.models.c.id == old["id"]).values(role="retired"))
        mid = conn.execute(db.models.insert().values(
            symbol=ds.symbol, horizon=key, model_name=champ, role=role,
            approved=bool(res.get("approved")) and role == "champion", created_at=now, path="db",
            metrics=metrics_payload, thresholds=db.jdump(db.clean_json(res.get("thresholds"))),
            feature_cols=db.jdump(ds.feature_cols), train_range=db.jdump(train_range),
            notes=decision.get("reason"))).inserted_primary_key[0]
        conn.execute(db.model_artifacts.insert().values(model_id=mid, data=blob, created_at=now))
        out["registered"].append({"id": mid, "name": champ, "role": role})
        # keep at most a few challengers per symbol/horizon for shadow testing
        conn.execute(db.models.update().where((db.models.c.symbol == ds.symbol) & (db.models.c.horizon == key)
                                              & (db.models.c.role == "challenger")
                                              & (db.models.c.created_at < now)).values(role="retired"))
    yv = _val_y(ds, key)
    for name in res.get("_challengers", []):
        if name not in res["_finals"]:
            continue
        blob_c = _save_artifact(ds, key, name, res["_finals"][name], {**res, "approved": False}, "challenger")
        with engine.begin() as conn:
            mid = conn.execute(db.models.insert().values(
                symbol=ds.symbol, horizon=key, model_name=name, role="challenger", approved=False,
                created_at=now, path="db", metrics=db.jdump(db.clean_json({"wf": res["candidates"].get(name),
                                                                        "validation": res["reports"].get(name, {}).get("validation")})),
                thresholds=db.jdump(db.clean_json(level_thresholds(
                    res["_finals"][name]["p_val"], yv, float(yv.mean()) if len(yv) else res["base_rate"],
                    ds.row_epochs[_val_rows(ds, key)], ds.horizon_seconds(key)))),
                feature_cols=db.jdump(ds.feature_cols), train_range=db.jdump(train_range),
                notes="shadow challenger")).inserted_primary_key[0]
            conn.execute(db.model_artifacts.insert().values(model_id=mid, data=blob_c, created_at=now))
            out["registered"].append({"id": mid, "name": name, "role": "challenger"})
    return out


def _val_rows(ds: Dataset, key: str) -> np.ndarray:
    _, valid, _ = ds.labels[key]
    rows = np.where(valid)[0]
    sp = dev_val_test(ds.row_epochs[rows], ds.horizon_seconds(key), 500 * ds.mean_dt)
    return rows[sp["val"]]


def _val_y(ds: Dataset, key: str) -> np.ndarray:
    return ds.labels[key][0][_val_rows(ds, key)].astype(int)


def train_symbol(symbol: str, direction: int, engine=None, horizons: list[str] | None = None,  # noqa: ANN001
                 model_names: list[str] | None = None, epochs: np.ndarray | None = None,
                 quotes: np.ndarray | None = None, persist: bool = True) -> dict:
    from .data import load_ticks, segment_bounds
    eng = engine or db.get_engine()
    if epochs is None:
        epochs, quotes = load_ticks(symbol, engine=eng)
    run_id = f"tr_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    out: dict = {"symbol": symbol, "run_id": run_id, "ticks": int(len(epochs)), "horizons": {}}
    if len(epochs) < 5000:
        out["status"] = "insufficient_data"
        return out
    horizons = horizons or [h for h, _, _ in HORIZONS]
    ds = build_dataset(symbol, direction, epochs, quotes, horizons=horizons)
    extra: dict = {"m1_candles": m1_candles(ds)}
    if settings.enable_deep_models and torch_available():
        rc, spk = ds.spike_free_returns()
        extra["seq"] = sequence_windows(rc, spk, segment_bounds(ds.seg)[0], ds.idx, ds.sigma)
    out["spikes"] = int(len(ds.spike_idx))
    out["rows"] = int(len(ds.idx))
    for key in horizons:
        t0 = time.time()
        try:
            res = train_horizon(ds, key, model_names, run_id, extra)
            if persist:
                res["registry"] = register(eng, ds, key, res)
            out["horizons"][key] = {k: v for k, v in res.items() if not k.startswith("_")}
        except Exception as e:  # noqa: BLE001
            out["horizons"][key] = {"status": "error", "error": f"{type(e).__name__}: {e}",
                                    "trace": traceback.format_exc(limit=5)}
            db.log_event("error", "train", f"{symbol} {key}: {e}", eng)
        out["horizons"][key]["seconds"] = time.time() - t0
    return out
