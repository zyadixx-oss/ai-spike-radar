"""REST API. Read-only research endpoints + manual job enqueueing. No endpoint can trade."""
from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from .. import db, safety
from ..config import HORIZONS, TIMEFRAMES, settings
from ..engine.evaluate import forward_stats
from ..engine.jobs import enqueue
from ..engine.radar import radar_summary
from ..live import hub
from ..research.candles import candles_for_chart
from ..research.deep import torch_available

router = APIRouter()
services = None  # injected by main


def _latest_research(symbol: str, kind: str) -> dict | None:
    r = db.exec_sql("SELECT payload, created_at FROM research WHERE symbol=:s AND kind=:k ORDER BY id DESC LIMIT 1",
                    {"s": symbol, "k": kind})
    if not r:
        return None
    p = db.jloads(r[0]["payload"], {}) or {}
    p["_created_at"] = r[0]["created_at"]
    return p


def _symbols() -> list[dict]:
    return [dict(r) for r in db.exec_sql("SELECT * FROM symbols ORDER BY family, n")]


# ------------------------------------------------------------------------------------------ meta
@router.get("/safety")
def get_safety() -> dict:
    return safety.status()


@router.post("/trade")
def trade_forbidden() -> dict:
    """Exists only to document & test that trading is impossible."""
    try:
        safety.assert_no_trading("api-trade")
    except safety.TradingForbiddenError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    return {}


@router.get("/config")
def get_config() -> dict:
    return {"timeframes": list(TIMEFRAMES), "horizons": [{"key": k, "kind": kd, "size": s} for k, kd, s in HORIZONS],
            "spike_k": settings.spike_k, "feature_timeframes": settings.feature_timeframes,
            "sim_cost_per_trade_pct": settings.sim_cost_per_trade_pct, "alert_min_level": settings.alert_min_level,
            "telegram_enabled": bool(settings.telegram_bot_token and settings.telegram_chat_id),
            "deep_models": {"enabled": settings.enable_deep_models, "torch_available": torch_available()}}


@router.get("/symbols")
def get_symbols() -> dict:
    missing = services.collector.missing if services and services.collector else []
    return {"symbols": _symbols(), "missing": missing}


@router.get("/health")
def get_health() -> dict:
    now = time.time()
    syms = []
    for s in _symbols():
        st = db.exec_sql("SELECT COUNT(*) AS n, MIN(epoch) AS a, MAX(epoch) AS b FROM ticks WHERE symbol=:s",
                         {"s": s["symbol"]})[0]
        gaps = db.exec_sql("SELECT COUNT(*) AS n, COALESCE(SUM(seconds),0) AS secs FROM data_gaps WHERE symbol=:s",
                           {"s": s["symbol"]})[0]
        bf = db.exec_sql("SELECT * FROM backfill_state WHERE symbol=:s", {"s": s["symbol"]})
        sp = db.exec_sql("SELECT COUNT(*) AS n FROM spikes WHERE symbol=:s", {"s": s["symbol"]})[0]["n"]
        syms.append({"symbol": s["symbol"], "display_name": s["display_name"], "available": s["available"],
                     "ticks": st["n"], "first_epoch": st["a"], "last_epoch": st["b"],
                     "last_tick_age_sec": (now - st["b"]) if st["b"] else None, "gaps": gaps["n"],
                     "gap_seconds": gaps["secs"], "spikes": sp,
                     "coverage_days": ((st["b"] - st["a"]) / 86400) if st["a"] else 0,
                     "backfill": dict(bf[0]) if bf else None, "live_buffer": len(hub.recent(s["symbol"]))})
    jobs = {r["status"]: r["n"] for r in db.exec_sql("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")}
    models = {r["role"]: r["n"] for r in db.exec_sql("SELECT role, COUNT(*) AS n FROM models GROUP BY role")}
    approved = db.exec_sql("SELECT COUNT(*) AS n FROM models WHERE role='champion' AND approved = :a", {"a": True})[0]["n"]
    events = [dict(r) for r in db.exec_sql("SELECT * FROM system_events ORDER BY id DESC LIMIT 40")]
    deriv = services.client.health() if services and services.client else {"connected": False, "disabled": True}
    return {"now": now, "uptime_sec": now - (services.started_at if services else now), "safety": safety.status(),
            "deriv": deriv, "symbols": syms, "missing_symbols": services.collector.missing if services and services.collector else [],
            "jobs": jobs, "models": models, "approved_champions": approved,
            "radar": services.radar.last_cycle if services and services.radar else None,
            "predictions": db.exec_sql("SELECT COUNT(*) AS n FROM predictions")[0]["n"],
            "outcomes": db.exec_sql("SELECT COUNT(*) AS n FROM prediction_outcomes")[0]["n"],
            "database": settings.database_url.split("://")[0], "torch_available": torch_available(), "events": events}


@router.get("/events")
def get_events(limit: int = 200, level: str | None = None) -> list[dict]:
    if level:
        return [dict(r) for r in db.exec_sql("SELECT * FROM system_events WHERE level=:l ORDER BY id DESC LIMIT :n",
                                             {"l": level, "n": limit})]
    return [dict(r) for r in db.exec_sql("SELECT * FROM system_events ORDER BY id DESC LIMIT :n", {"n": limit})]


# ------------------------------------------------------------------------------------------ radar
@router.get("/radar")
def get_radar() -> dict:
    return {"updated_at": services.radar.last_cycle.get("at") if services and services.radar else None,
            "rows": radar_summary(hub.radar), "safety": safety.status()}


@router.get("/radar/{symbol}")
def get_radar_symbol(symbol: str) -> dict:
    r = hub.radar.get(symbol)
    if r is None:
        raise HTTPException(404, "no radar state for symbol (no live data yet?)")
    return db.clean_json(r)


# ------------------------------------------------------------------------------------------ charts
@router.get("/candles")
def get_candles(symbol: str, tf: str = "M1", limit: int = Query(1000, le=5000), end: int | None = None) -> dict:
    if tf not in TIMEFRAMES:
        raise HTTPException(400, f"unknown timeframe {tf}")
    df, source = candles_for_chart(symbol, tf, limit, end)
    rows = [{"time": int(r.time), "open": float(r.open), "high": float(r.high), "low": float(r.low),
             "close": float(r.close), "ticks": (int(r.ticks) if r.ticks == r.ticks else None)}
            for r in df.itertuples()] if len(df) else []
    return {"symbol": symbol, "tf": tf, "source": source, "candles": rows}


@router.get("/markers")
def get_markers(symbol: str, start: int, end: int, horizon: str | None = None) -> dict:
    spikes = [dict(r) for r in db.exec_sql(
        "SELECT epoch, ret, magnitude_sigma, size_class, direction FROM spikes WHERE symbol=:s AND epoch BETWEEN :a AND :b "
        "ORDER BY epoch", {"s": symbol, "a": start, "b": end})]
    q = ("SELECT p.id, p.issued_epoch, p.expires_epoch, p.horizon, p.probability, p.base_rate, p.level, p.role, "
         "p.model_name, o.status, o.spike_epoch FROM predictions p LEFT JOIN prediction_outcomes o ON o.prediction_id=p.id "
         "WHERE p.symbol=:s AND p.issued_epoch BETWEEN :a AND :b AND p.role='champion' AND p.level IN ('yellow','orange','red')")
    params = {"s": symbol, "a": start, "b": end}
    if horizon:
        q += " AND p.horizon=:h"
        params["h"] = horizon
    preds = [dict(r) for r in db.exec_sql(q + " ORDER BY p.issued_epoch LIMIT 3000", params)]
    return {"spikes": spikes, "predictions": preds}


# ------------------------------------------------------------------------------------------ spikes & research
@router.get("/spikes")
def get_spikes(symbol: str | None = None, size_class: str | None = None, limit: int = Query(200, le=5000),
               offset: int = 0) -> dict:
    where, params = [], {"n": limit, "o": offset}
    if symbol:
        where.append("symbol=:s")
        params["s"] = symbol
    if size_class:
        where.append("size_class=:c")
        params["c"] = size_class
    w = (" WHERE " + " AND ".join(where)) if where else ""
    rows = [dict(r) for r in db.exec_sql(f"SELECT * FROM spikes{w} ORDER BY epoch DESC LIMIT :n OFFSET :o", params)]
    total = db.exec_sql(f"SELECT COUNT(*) AS n FROM spikes{w}", params)[0]["n"]
    return {"total": total, "rows": rows}


@router.get("/research/{symbol}")
def get_research(symbol: str) -> dict:
    return {"research": _latest_research(symbol, "research")}


@router.get("/research")
def get_research_all() -> dict:
    out = {}
    for s in _symbols():
        r = _latest_research(s["symbol"], "research")
        if r:
            out[s["symbol"]] = {"display_name": s["display_name"], "spike_stats": r.get("spike_stats"),
                                "interarrival": {k: v for k, v in (r.get("interarrival") or {}).items() if k != "hazard_bins"},
                                "patterns": {k: v for k, v in (r.get("patterns") or {}).items() if k != "tests"},
                                "coverage": r.get("coverage"), "_created_at": r.get("_created_at")}
    return out


# ------------------------------------------------------------------------------------------ arena / rankings
@router.get("/arena")
def get_arena(symbol: str | None = None) -> dict:
    syms = [symbol] if symbol else [s["symbol"] for s in _symbols()]
    return {s: _latest_research(s, "arena") for s in syms}


@router.get("/arena/experiments")
def get_experiments(symbol: str, horizon: str | None = None, split: str = "val", limit: int = Query(300, le=5000),
                    selected_only: bool = False) -> list[dict]:
    run = db.exec_sql("SELECT run_id FROM experiments WHERE symbol=:s ORDER BY id DESC LIMIT 1", {"s": symbol})
    if not run:
        return []
    q = "SELECT * FROM experiments WHERE symbol=:s AND run_id=:r AND split=:sp"
    params = {"s": symbol, "r": run[0]["run_id"], "sp": split, "n": limit}
    if horizon:
        q += " AND horizon=:h"
        params["h"] = horizon
    if selected_only:
        q += " AND selected = :t"
        params["t"] = True
    rows = [dict(r) for r in db.exec_sql(q + " ORDER BY p_value ASC LIMIT :n", params)]
    for r in rows:
        r["params"] = db.jloads(r["params"], {})
        r["trading"] = db.jloads(r["trading"], None)
    return rows


@router.get("/rankings/timeframes")
def get_tf_rankings() -> dict:
    out = {}
    for s in _symbols():
        a = _latest_research(s["symbol"], "arena")
        if not a:
            continue
        out[s["symbol"]] = {"display_name": s["display_name"], "run_id": a.get("run_id"),
                            "_created_at": a.get("_created_at"),
                            "horizons": {k: {"ranking": v.get("timeframe_ranking"), "verdict": v.get("verdict"),
                                             "status": v.get("status")} for k, v in (a.get("horizons") or {}).items()}}
    return out


def _latest_backtests(symbol: str | None = None) -> list[dict]:
    params: dict = {}
    q = ("SELECT b.* FROM backtests b JOIN (SELECT symbol, horizon, MAX(id) AS mid FROM backtests "
         "WHERE kind='summary' GROUP BY symbol, horizon) s ON s.symbol=b.symbol AND s.horizon=b.horizon "
         "JOIN backtests bs ON bs.id = s.mid AND bs.run_id = b.run_id")
    if symbol:
        q += " WHERE b.symbol=:s"
        params["s"] = symbol
    rows = [dict(r) for r in db.exec_sql(q, params)]
    for r in rows:
        r["metrics"] = db.jloads(r["metrics"], {})
    return rows


@router.get("/rankings/models")
def get_model_rankings(symbol: str | None = None) -> dict:
    rows = _latest_backtests(symbol)
    out: dict = {}
    for r in rows:
        g = out.setdefault(r["symbol"], {}).setdefault(r["horizon"], {"models": [], "summary": None, "run_id": r["run_id"]})
        if r["kind"] == "summary":
            g["summary"] = r["metrics"]
        elif r["split"] == "walk_forward":
            m = r["metrics"]
            g["models"].append({"name": r["name"], "kind": r["kind"], "status": m.get("status"), "reason": m.get("reason"),
                                "wf_bss_mean": m.get("wf_bss_mean"), "wf_pr_auc_mean": m.get("wf_pr_auc_mean"),
                                "wf_brier_mean": m.get("wf_brier_mean"), "folds": m.get("folds"),
                                "fit_seconds": m.get("fit_seconds"), "complexity": m.get("complexity")})
    for sym, hz in out.items():
        for h, g in hz.items():
            g["models"].sort(key=lambda m: -(m["wf_bss_mean"] if m["wf_bss_mean"] is not None else -9))
    champs = [dict(r) for r in db.exec_sql("SELECT id, symbol, horizon, model_name, role, approved, created_at, notes "
                                           "FROM models WHERE role IN ('champion','challenger') ORDER BY symbol, horizon")]
    return {"rankings": out, "registry": champs}


@router.get("/backtests")
def get_backtests(symbol: str, horizon: str | None = None) -> dict:
    rows = _latest_backtests(symbol)
    out: dict = {}
    for r in rows:
        if horizon and r["horizon"] != horizon:
            continue
        g = out.setdefault(r["horizon"], {"summary": None, "results": {}})
        if r["kind"] == "summary":
            g["summary"] = r["metrics"]
        else:
            g["results"].setdefault(r["name"], {"kind": r["kind"]})[r["split"]] = r["metrics"]
    return {"symbol": symbol, "horizons": out}


@router.get("/models")
def get_models(symbol: str | None = None) -> list[dict]:
    q = "SELECT id, symbol, horizon, model_name, role, approved, created_at, metrics, thresholds, train_range, notes FROM models"
    params = {}
    if symbol:
        q += " WHERE symbol=:s"
        params["s"] = symbol
    rows = [dict(r) for r in db.exec_sql(q + " ORDER BY id DESC LIMIT 500", params)]
    for r in rows:
        for k in ("metrics", "thresholds", "train_range"):
            r[k] = db.jloads(r[k], None)
    return rows


# ------------------------------------------------------------------------------------------ predictions
@router.get("/predictions")
def get_predictions(symbol: str | None = None, horizon: str | None = None, role: str | None = None,
                    level: str | None = None, status: str | None = None, limit: int = Query(200, le=2000),
                    offset: int = 0) -> dict:
    where, params = [], {"n": limit, "o": offset}
    for col, val, key in (("p.symbol", symbol, "s"), ("p.horizon", horizon, "h"), ("p.role", role, "r"),
                          ("p.level", level, "l")):
        if val:
            where.append(f"{col}=:{key}")
            params[key] = val
    if status == "pending":
        where.append("o.prediction_id IS NULL")
    elif status:
        where.append("o.status=:st")
        params["st"] = status
    w = (" WHERE " + " AND ".join(where)) if where else ""
    base = f"FROM predictions p LEFT JOIN prediction_outcomes o ON o.prediction_id=p.id{w}"
    rows = [dict(r) for r in db.exec_sql(f"SELECT p.*, o.status AS outcome_status, o.outcome, o.spike_epoch, "
                                         f"o.lead_time_sec, o.evaluated_at {base} ORDER BY p.id DESC LIMIT :n OFFSET :o",
                                         params)]
    for r in rows:
        r["reasons"] = db.jloads(r["reasons"], [])
        r["details"] = db.jloads(r["details"], {})
    total = db.exec_sql(f"SELECT COUNT(*) AS n {base}", params)[0]["n"]
    return {"total": total, "rows": rows, "immutable": True}


@router.get("/forward")
def get_forward(role: str | None = None, symbol: str | None = None) -> list[dict]:
    return db.clean_json(forward_stats(role=role, symbol=symbol))


@router.get("/shadow")
def get_shadow() -> dict:
    return db.clean_json({"shadow": forward_stats(role="shadow"), "champion": forward_stats(role="champion"),
                          "baseline": forward_stats(role="baseline")})


@router.get("/alerts")
def get_alerts(limit: int = 100) -> list[dict]:
    return [dict(r) for r in db.exec_sql("SELECT * FROM alerts ORDER BY id DESC LIMIT :n", {"n": limit})]


# ------------------------------------------------------------------------------------------ jobs
@router.get("/jobs")
def get_jobs(limit: int = 100) -> list[dict]:
    rows = [dict(r) for r in db.exec_sql("SELECT id, type, payload, status, priority, attempts, max_attempts, last_error, "
                                         "created_at, started_at, finished_at FROM jobs ORDER BY id DESC LIMIT :n",
                                         {"n": limit})]
    return rows


@router.post("/jobs/{job_type}")
def post_job(job_type: str, symbol: str | None = None) -> dict:
    allowed = {"scan_spikes", "research", "train", "arena", "evaluate", "gaps"}
    if job_type not in allowed:
        raise HTTPException(400, f"job type must be one of {sorted(allowed)}")
    payload = {"symbol": symbol} if symbol else {}
    jid = enqueue(job_type, payload, 3, dedupe_key=f"manual:{job_type}:{symbol}")
    return {"job_id": jid, "queued": jid is not None}


# ------------------------------------------------------------------------------------------ reports
@router.get("/reports/summary")
def get_summary() -> dict:
    """Direct answers to the research questions, per symbol, from stored evidence only."""
    rank = get_model_rankings()["rankings"]
    fwd = forward_stats(role="champion")
    out = []
    for s in _symbols():
        sym = s["symbol"]
        res = _latest_research(sym, "research") or {}
        arena = _latest_research(sym, "arena") or {}
        hz = rank.get(sym, {})
        approved = {h: g["summary"] for h, g in hz.items() if g.get("summary") and g["summary"].get("approved")}
        best_test = None
        for h, g in hz.items():
            sm = g.get("summary") or {}
            acc = (sm.get("acceptance") or {}).get("bss_vs_base_rate") or {}
            cand = {"horizon": h, "model": sm.get("selected"), "bss": acc.get("mean"), "ci95": acc.get("ci95"),
                    "approved": bool(sm.get("approved"))}
            if cand["bss"] is not None and (best_test is None or (cand["approved"], cand["bss"]) >
                                            (best_test["approved"], best_test["bss"])):
                best_test = cand  # approved horizons always rank above unapproved ones
        tf_best = []
        strat_best = []
        for h, v in (arena.get("horizons") or {}).items():
            rk = [r for r in (v.get("timeframe_ranking") or []) if r.get("val_significant")]
            if rk:
                tf_best.append({"horizon": h, **{k: rk[0].get(k) for k in ("timeframe", "best_rule", "best_val_lift",
                                                                           "test_lift", "test_confirmed")}})
            for f in (v.get("families") or [])[:3]:
                if f.get("val_significant"):
                    strat_best.append({"horizon": h, "family": f["family"], "timeframe": f["timeframe"],
                                       "feature": f["feature"], "side": f["side"], "q": f["q"],
                                       "val_lift": (f.get("val") or {}).get("lift"),
                                       "test_lift": (f.get("test") or {}).get("lift"),
                                       "test_confirmed": (f.get("test") or {}).get("confirmed")})
        pats = [t for t in ((res.get("patterns") or {}).get("tests") or []) if t.get("robust")][:6]
        f_sym = [f for f in fwd if f["symbol"] == sym]
        lead, false_alarm = [], []
        bts = _latest_backtests(sym)
        for h, g in hz.items():
            sel = (g.get("summary") or {}).get("selected")
            t = next((b["metrics"] for b in bts if b["horizon"] == h and b["name"] == sel and b["split"] == "test"), None)
            if not t:
                continue
            ed = t.get("event_detection") or {}
            at = t.get("at_threshold") or {}
            lead.append({"horizon": h, "model": sel, "mean_warning_lead_sec": t.get("mean_warning_lead_sec"),
                         "spikes_detected_share": ed.get("detected_share"), "spikes": ed.get("spikes")})
            false_alarm.append({"horizon": h, "model": sel, "precision": at.get("precision"),
                                "false_discovery_rate": at.get("false_discovery_rate"),
                                "false_alarm_rate": at.get("false_alarm_rate"), "alerts": at.get("n_predictions")})
        out.append({
            "symbol": sym, "display_name": s["display_name"],
            "data": res.get("coverage"), "spikes": (res.get("spike_stats") or {}).get("count"),
            "q1_better_than_chance": {"approved_horizons": list(approved), "best_test": best_test,
                                      "answer": ("yes — approved on holdout for " + ", ".join(approved)) if approved
                                      else ("no reliable evidence yet" if hz else "not yet tested")},
            "q_memoryless": (res.get("interarrival") or {}).get("consistent_with_memoryless"),
            "interarrival": {k: (res.get("interarrival") or {}).get(k) for k in
                             ("mean_ticks", "weibull_shape", "weibull_shape_ci95", "ks_vs_exponential",
                              "hazard_flatness_chi2")},
            "q2_best_timeframe": tf_best or "no timeframe produced a validated rule",
            "q3_patterns": pats or "no robust (FDR + replicated) pre-spike pattern found",
            "q4_best_strategy": strat_best or "no strategy survived validation FDR",
            "q5_lead_time": lead or "no model with a validated alert threshold yet",
            "q6_false_alarms": {"holdout": false_alarm,
                                "forward": [{"horizon": f["horizon"], "alerts": f["alerts"],
                                             "alert_precision": f["alert_precision"],
                                             "false_alarm_rate": f["false_alarm_rate"]} for f in f_sym]},
            "q7_forward": [{"horizon": f["horizon"], "n": f["n"], "bss_vs_base_rate": f["bss_vs_base_rate"],
                            "observed_rate": f["observed_rate"], "mean_probability": f["mean_probability"]} for f in f_sym],
        })
    return db.clean_json({"generated_at": time.time(), "symbols": out, "safety": safety.status(),
                          "disclaimer": "Research only. Probabilities are statistical estimates, never certainties."})


@router.get("/stats/overview")
def get_overview() -> dict:
    sym = _symbols()
    n_ticks = db.exec_sql("SELECT COUNT(*) AS n FROM ticks")[0]["n"]
    n_spikes = db.exec_sql("SELECT COUNT(*) AS n FROM spikes")[0]["n"]
    preds = db.exec_sql("SELECT COUNT(*) AS n FROM predictions")[0]["n"]
    outc = {r["status"]: r["n"] for r in db.exec_sql("SELECT status, COUNT(*) AS n FROM prediction_outcomes GROUP BY status")}
    rec = [dict(r) for r in db.exec_sql("SELECT symbol, epoch, size_class, magnitude_sigma FROM spikes ORDER BY epoch DESC LIMIT 15")]
    approved = [dict(r) for r in db.exec_sql("SELECT symbol, horizon, model_name FROM models WHERE role='champion' AND approved = :a",
                                             {"a": True})]
    per = [dict(r) for r in db.exec_sql("SELECT symbol, COUNT(*) AS n FROM spikes GROUP BY symbol")]
    return {"symbols": len(sym), "ticks": n_ticks, "spikes": n_spikes, "predictions": preds, "outcomes": outc,
            "recent_spikes": rec, "approved_models": approved, "spikes_per_symbol": per,
            "radar": radar_summary(hub.radar), "safety": safety.status(),
            "deriv_connected": bool(services and services.client and services.client.is_connected)}


_ = np  # keep numpy import (used by type coercions in some deployments)
