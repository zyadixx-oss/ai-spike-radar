"""Evaluate expired predictions against what actually happened (forward + shadow testing).

Outcomes go to `prediction_outcomes` (insert-only); the prediction row itself is never modified.
A prediction whose window contains a data gap is marked `invalid` (not counted as hit or miss).
"""
from __future__ import annotations

import time

import numpy as np

from .. import db
from ..config import settings
from ..research import metrics as M


def _pending(engine, limit: int = 3000):  # noqa: ANN001
    return db.exec_sql(
        "SELECT p.* FROM predictions p LEFT JOIN prediction_outcomes o ON o.prediction_id = p.id "
        "WHERE o.prediction_id IS NULL ORDER BY p.id LIMIT :n", {"n": limit}, engine)


def evaluate_pending(engine=None) -> dict:  # noqa: ANN001
    eng = engine or db.get_engine()
    rows = _pending(eng)
    latest = {r["symbol"]: r["m"] for r in db.exec_sql("SELECT symbol, MAX(epoch) AS m FROM ticks GROUP BY symbol",
                                                        engine=eng)}
    done = {"hit": 0, "miss": 0, "invalid": 0, "waiting": 0}
    out_rows = []
    for p in rows:
        sym, last = p["symbol"], latest.get(p["symbol"])
        if last is None:
            done["waiting"] += 1
            continue
        det = db.jloads(p["details"], {}) or {}
        thr = det.get("threshold")
        if p["horizon_kind"] == "sec":
            if last < p["expires_epoch"] + 2:
                done["waiting"] += 1
                continue
            tk = db.exec_sql("SELECT epoch, quote FROM ticks WHERE symbol=:s AND epoch>=:a AND epoch<=:b ORDER BY epoch",
                             {"s": sym, "a": p["issued_epoch"], "b": p["expires_epoch"]}, eng)
        else:
            tk = db.exec_sql("SELECT epoch, quote FROM ticks WHERE symbol=:s AND epoch>=:a ORDER BY epoch LIMIT :n",
                             {"s": sym, "a": p["issued_epoch"], "n": p["horizon_size"] + 1}, eng)
            if len(tk) < p["horizon_size"] + 1:
                done["waiting"] += 1
                continue
        status, outcome, spike_epoch, lead, note = _judge(tk, p, thr)
        done[status] += 1
        out_rows.append({"prediction_id": p["id"], "evaluated_at": time.time(), "status": status, "outcome": outcome,
                         "spike_epoch": spike_epoch, "lead_time_sec": lead, "note": note})
    db.insert_ignore(eng, db.prediction_outcomes, out_rows)
    return done


def _judge(tk, p, thr):  # noqa: ANN001, ANN202
    if not tk or tk[0]["epoch"] != p["issued_epoch"]:
        return "invalid", None, None, None, "issue tick not found in stored data"
    ep = np.array([t["epoch"] for t in tk], dtype=np.int64)
    q = np.array([t["quote"] for t in tk], dtype=float)
    if len(ep) > 1 and np.max(np.diff(ep)) > max(settings.gap_min_seconds, 10 * max(np.median(np.diff(ep)), 1)):
        return "invalid", None, None, None, "data gap inside prediction window"
    if p["horizon_kind"] == "sec" and len(ep) and (p["expires_epoch"] - ep[-1]) > settings.gap_min_seconds:
        return "invalid", None, None, None, "window not fully covered by data"
    if thr is None:
        return "invalid", None, None, None, "no spike threshold recorded"
    r = np.diff(np.log(q))
    hit = np.where(p["direction"] * r > thr)[0]
    if len(hit):
        se = int(ep[hit[0] + 1])
        return "hit", 1, se, float(se - p["issued_epoch"]), None
    return "miss", 0, None, None, None


def forward_stats(engine=None, role: str | None = None, symbol: str | None = None) -> list[dict]:  # noqa: ANN001
    """Live forward-test performance per (symbol, horizon, role, model) — includes an honest calibration check."""
    eng = engine or db.get_engine()
    where, params = ["o.status != 'invalid'"], {}
    if role:
        where.append("p.role = :r")
        params["r"] = role
    if symbol:
        where.append("p.symbol = :s")
        params["s"] = symbol
    rows = db.exec_sql("SELECT p.symbol, p.horizon, p.role, p.model_name, p.level, p.probability, p.base_rate, "
                       "o.outcome, o.lead_time_sec FROM predictions p JOIN prediction_outcomes o ON o.prediction_id=p.id "
                       f"WHERE {' AND '.join(where)}", params, eng)
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["symbol"], r["horizon"], r["role"], r["model_name"]), []).append(r)
    out = []
    for (s, h, role_, name), rs in groups.items():
        y = np.array([r["outcome"] for r in rs], dtype=float)
        p = np.array([r["probability"] for r in rs], dtype=float)
        b = np.array([r["base_rate"] if r["base_rate"] is not None else np.nan for r in rs], dtype=float)
        b = np.where(np.isfinite(b), b, np.nanmean(p))
        alerts = np.array([r["level"] in ("orange", "red") for r in rs])
        flagged = np.array([r["level"] in ("yellow", "orange", "red") for r in rs])
        tp = int((alerts & (y == 1)).sum())
        leads = [r["lead_time_sec"] for r in rs if r["level"] in ("orange", "red") and r["outcome"] == 1]
        out.append({
            "symbol": s, "horizon": h, "role": role_, "model_name": name, "n": len(rs), "hits": int(y.sum()),
            "observed_rate": float(y.mean()), "mean_probability": float(p.mean()),
            "brier": M.brier(y, p), "brier_base": M.brier(y, b), "bss_vs_base_rate": M.bss(y, p, b),
            "pr_auc": M.pr_auc(y, p), "alerts": int(alerts.sum()), "alerts_correct": tp,
            "alert_precision": tp / alerts.sum() if alerts.sum() else None,
            "alert_precision_ci": M.wilson(tp, int(alerts.sum())) if alerts.sum() else None,
            "flagged": int(flagged.sum()), "flagged_precision": float(y[flagged].mean()) if flagged.any() else None,
            "recall": tp / y.sum() if y.sum() else None,
            "false_alarm_rate": float((alerts & (y == 0)).sum() / max((y == 0).sum(), 1)),
            "mean_lead_sec": float(np.mean(leads)) if leads else None,
            "calibration": M.reliability(y, p, bins=5),
        })
    out.sort(key=lambda r: (r["symbol"], r["horizon"], r["role"]))
    return out
