"""Live Spike Radar / Prediction Engine.

Every RADAR_INTERVAL seconds, for each symbol and horizon:
  * champion model (if any) -> calibrated probability -> level (thresholds from validation)
  * challengers -> shadow predictions (logged, never shown as signals)
  * no model -> memoryless base-rate probability, level gray
Predictions are written BEFORE their outcome is known into an append-only table (DB triggers forbid
UPDATE/DELETE). A new prediction is issued when the previous one for the same stream has expired
(non-overlapping windows, so the forward test is clean) or when the level escalates.
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
from sqlalchemy import select

from .. import db
from ..config import HORIZONS, settings
from ..live import LiveHub, hub as default_hub
from ..research.deep import sequence_windows
from ..research.models import Ctx
from ..research.train import level_for, load_artifact
from .alerts import AlertManager
from .livefeatures import SymbolFeatureState

log = logging.getLogger("radar")
LEVEL_ORDER = {"gray": 0, "green": 1, "yellow": 2, "orange": 3, "red": 4}
DEEP = {"lstm", "temporal_cnn", "transformer"}


class ModelCache:
    def __init__(self, engine):  # noqa: ANN001
        self.engine = engine
        self.models: dict[int, dict] = {}
        self.index: dict[tuple[str, str], dict[str, list[int]]] = {}
        self.loaded_at = 0.0
        self.errors: dict[int, str] = {}

    def refresh(self, force: bool = False) -> None:
        if not force and time.time() - self.loaded_at < 60:
            return
        with self.engine.connect() as conn:
            rows = conn.execute(select(db.models).where(db.models.c.role.in_(["champion", "challenger"]))).mappings().all()
        idx: dict[tuple[str, str], dict[str, list[int]]] = {}
        keep = set()
        for r in rows:
            keep.add(r["id"])
            if r["id"] not in self.models and r["id"] not in self.errors:
                try:
                    art = load_artifact(self.engine, r["id"])
                    art["db"] = dict(r)
                    art["metrics"] = db.jloads(r["metrics"], {})
                    self.models[r["id"]] = art
                except Exception as e:  # noqa: BLE001
                    self.errors[r["id"]] = f"{type(e).__name__}: {e}"
                    db.log_event("error", "radar", f"cannot load model {r['id']}: {e}", self.engine)
                    continue
            if r["id"] in self.models:
                self.models[r["id"]]["db"]["role"] = r["role"]
                idx.setdefault((r["symbol"], r["horizon"]), {}).setdefault(r["role"], []).append(r["id"])
        for mid in list(self.models):
            if mid not in keep:
                del self.models[mid]
        self.index, self.loaded_at = idx, time.time()


class Radar:
    def __init__(self, engine=None, live_hub: LiveHub | None = None, alerts: AlertManager | None = None):  # noqa: ANN001
        self.engine = engine or db.get_engine()
        self.hub = live_hub or default_hub
        self.cache = ModelCache(self.engine)
        self.alerts = alerts or AlertManager(self.engine)
        self.states: dict[str, SymbolFeatureState] = {}
        self.last_logged: dict[tuple, dict] = {}
        self.symbol_info: dict[str, dict] = {}
        self.context: dict[str, dict] = {}
        self.context_at = 0.0
        self.last_cycle: dict = {}

    # -------------------------------------------------------------- context (slow-changing info)
    def refresh_context(self) -> None:
        if time.time() - self.context_at < 300 and self.symbol_info:
            return
        syms = db.exec_sql("SELECT * FROM symbols WHERE available = :a", {"a": True}, self.engine)
        self.symbol_info = {r["symbol"]: dict(r) for r in syms}
        ctx: dict[str, dict] = {}
        for s in self.symbol_info:
            sz = db.exec_sql("SELECT size_class, COUNT(*) AS n, AVG(abs_move) AS avg_move FROM spikes WHERE symbol=:s "
                             "GROUP BY size_class", {"s": s}, self.engine)
            med = db.exec_sql("SELECT abs_move FROM spikes WHERE symbol=:s ORDER BY abs_move", {"s": s}, self.engine)
            moves = [r["abs_move"] for r in med if r["abs_move"] is not None]
            thr = db.exec_sql("SELECT threshold FROM spikes WHERE symbol=:s LIMIT 1", {"s": s}, self.engine)
            cnt = db.exec_sql("SELECT COUNT(*) AS n FROM spikes WHERE symbol=:s", {"s": s}, self.engine)[0]["n"]
            tk = db.exec_sql("SELECT COUNT(*) AS n FROM ticks WHERE symbol=:s", {"s": s}, self.engine)[0]["n"]
            arena = db.exec_sql("SELECT payload FROM research WHERE symbol=:s AND kind='arena' ORDER BY id DESC LIMIT 1",
                                {"s": s}, self.engine)
            best_tf = {}
            if arena:
                pl = db.jloads(arena[0]["payload"], {}) or {}
                for hk, hv in (pl.get("horizons") or {}).items():
                    rk = [r for r in (hv.get("timeframe_ranking") or []) if r.get("val_significant")]
                    best_tf[hk] = ({"timeframe": rk[0]["timeframe"], "rule": rk[0].get("best_rule"),
                                    "test_confirmed": rk[0].get("test_confirmed")} if rk else None)
            ctx[s] = {"size": {"classes": {r["size_class"]: r["n"] for r in sz},
                               "median_abs_move": float(np.median(moves)) if moves else None,
                               "p25_abs_move": float(np.quantile(moves, 0.25)) if moves else None,
                               "p75_abs_move": float(np.quantile(moves, 0.75)) if moves else None,
                               "note": "historical distribution — no model of spike size has shown predictive value"},
                      "threshold": thr[0]["threshold"] if thr else None,
                      "lambda_per_tick": (cnt / tk) if tk else None, "spikes": cnt, "ticks": tk, "best_tf": best_tf}
        self.context, self.context_at = ctx, time.time()

    # -------------------------------------------------------------- scoring
    def _score(self, art: dict, X, side: dict, state: SymbolFeatureState, recent) -> float:  # noqa: ANN001
        ctx = Ctx(np.array([side["epoch"]]), "tick" if art["kind"] == "tick" else "sec", art["size"], art["mean_dt"],
                  dict(art.get("run_ctx") or {}))
        name = art["model_name"]
        if name in DEEP:
            rc, spk = side["rc"], side["spk"]
            seg_start = np.full(len(rc), side["seg_start_last"])
            ctx.extra["seq"] = sequence_windows(rc, spk, seg_start, np.array([len(rc) - 1]), art["sigma"])
        if name == "hidden_markov":
            art["model"].refresh(state.m1_candles(side["epoch"], recent))
        s = art["model"].score(X[art["feature_cols"]], ctx)
        return float(art["calibrator"].predict(np.asarray(s, dtype=float))[0])

    def _reasons(self, art: dict, X) -> list[dict]:  # noqa: ANN001
        imp = art.get("importances") or {}
        ref = art.get("feature_ref") or {}
        med, iqr = ref.get("median", {}), ref.get("iqr", {})
        out = []
        for f, w in list(imp.items())[:12]:
            if f not in X.columns:
                continue
            v = X[f].iloc[0]
            if v is None or not np.isfinite(v):
                continue
            sc = iqr.get(f) or 0
            z = (v - (med.get(f) or 0)) / sc if sc else 0.0
            out.append({"feature": f, "value": float(v), "typical": med.get(f), "z": float(z), "importance": float(w),
                        "score": float(abs(z) * w)})
        out.sort(key=lambda r: -r["score"])
        return out[:4]

    def _perf(self, art: dict, level: str) -> dict:
        m = art.get("metrics") or {}
        test = m.get("test") or {}
        thr = art.get("thresholds") or {}
        lv = thr.get(f"{level}_val") if level in ("yellow", "orange", "red") else None
        acc = m.get("acceptance") or {}
        return {"approved": bool(art["db"].get("approved")), "test_bss_vs_base_rate": test.get("bss_vs_base_rate"),
                "test_pr_auc": test.get("pr_auc"), "test_brier": test.get("brier"),
                "level_precision_val": lv.get("precision") if lv else None,
                "false_alarm_share_val": (1 - lv["precision"]) if lv and lv.get("precision") is not None else None,
                "test_at_threshold": test.get("at_threshold"), "acceptance": acc,
                "event_detection": test.get("event_detection")}

    def _predict_stream(self, sym: str, info: dict, key: str, kind: str, size: int, X_by_params: dict,
                        recent) -> dict:  # noqa: ANN001
        ids = self.cache.index.get((sym, key), {})
        ctx = self.context.get(sym, {})
        direction = info["direction"]
        out = {"symbol": sym, "display_name": info["display_name"], "horizon": key, "horizon_kind": kind,
               "horizon_size": size, "direction": direction, "direction_label": "up" if direction > 0 else "down",
               "expected_size": ctx.get("size"), "best_timeframe": (ctx.get("best_tf") or {}).get(
                   {"30s": "30s", "1m": "1m", "5m": "5m", "100t": "100t"}.get(key, "1m")),
               "shadow": []}
        lam = ctx.get("lambda_per_tick")
        mean_dt = 1.0
        for mids in ids.values():
            for mid in mids:
                mean_dt = self.cache.models[mid].get("mean_dt", 1.0)
        h_ticks = size if kind == "tick" else size / max(mean_dt, 1e-9)
        base_p = (1 - (1 - lam) ** h_ticks) if lam else None
        out["base_rate"] = base_p
        champ = (ids.get("champion") or [None])[0]
        if champ is None:
            out.update(probability=base_p, level="gray", model_name="base_rate (memoryless)", model_id=None,
                       role="baseline", approved=False, reasons=[{"text": "no trained model yet — showing the "
                                                                          "memoryless base rate"}],
                       performance=None, threshold=ctx.get("threshold"))
            return out
        art = self.cache.models[champ]
        try:
            X, side, state = X_by_params[(art["sigma"], art["threshold"], tuple(art["timeframes"]))]
            p = self._score(art, X, side, state, recent)
            approved = bool(art["db"].get("approved"))
            thr = art.get("thresholds") or {}
            level = level_for(p, thr, approved)
            out.update(probability=p, level=level, model_name=art["model_name"], model_id=champ, role="champion",
                       approved=approved, reasons=self._reasons(art, X), performance=self._perf(art, level),
                       base_rate=art.get("base_rate") or base_p, threshold=art["threshold"],
                       thresholds={k: thr.get(k) for k in ("yellow", "orange", "red")})
        except Exception as e:  # noqa: BLE001 — model failure => gray, radar continues
            out.update(probability=base_p, level="gray", model_name=art["model_name"], model_id=champ,
                       role="baseline", approved=False, error=f"{type(e).__name__}: {e}",
                       reasons=[{"text": f"model failed: {type(e).__name__}"}], performance=None,
                       threshold=ctx.get("threshold"))
        for mid in ids.get("challenger", []):
            ca = self.cache.models[mid]
            try:
                X, side, state = X_by_params[(ca["sigma"], ca["threshold"], tuple(ca["timeframes"]))]
                pc = self._score(ca, X, side, state, recent)
                out["shadow"].append({"model_id": mid, "model_name": ca["model_name"], "probability": pc,
                                      "level": level_for(pc, ca.get("thresholds"), False), "threshold": ca["threshold"]})
            except Exception as e:  # noqa: BLE001
                out["shadow"].append({"model_id": mid, "model_name": ca["model_name"], "error": str(e)})
        return out

    # -------------------------------------------------------------- logging
    def _should_log(self, stream: tuple, issued_epoch: int, n_ticks_seen: int, kind: str, size: int, level: str) -> bool:
        last = self.last_logged.get(stream)
        if last is None:
            return True
        expired = (issued_epoch >= last["epoch"] + size) if kind == "sec" else (n_ticks_seen - last["tickno"] >= size)
        escalated = LEVEL_ORDER.get(level, 0) > LEVEL_ORDER.get(last["level"], 0) and LEVEL_ORDER.get(level, 0) >= 2
        return expired or escalated

    def _log(self, rows: list[dict]) -> list[int]:
        if not rows:
            return []
        ids = []
        with self.engine.begin() as conn:
            for r in rows:
                ids.append(int(conn.execute(db.predictions.insert().values(**r)).inserted_primary_key[0]))
        return ids

    def cycle(self) -> dict:
        t_start = time.perf_counter()
        self.cache.refresh()
        self.refresh_context()
        now = time.time()
        radar: dict[str, dict] = {}
        to_log: list[dict] = []
        log_meta: list[tuple] = []
        for sym, info in self.symbol_info.items():
            recent = self.hub.recent(sym)
            if len(recent) < 600:
                radar[sym] = {"symbol": sym, "display_name": info["display_name"], "status": "insufficient_live_data",
                              "level": "gray", "horizons": {}}
                continue
            last_epoch = recent[-1][0]
            stale = (now - last_epoch) > 30
            st = self.states.setdefault(sym, SymbolFeatureState(sym, info["direction"], self.engine))
            # one feature computation per distinct (sigma, threshold, timeframes) among loaded models
            params = set()
            for (s2, _h), roles in self.cache.index.items():
                if s2 != sym:
                    continue
                for mids in roles.values():
                    for mid in mids:
                        a = self.cache.models[mid]
                        params.add((a["sigma"], a["threshold"], tuple(a["timeframes"])))
            X_by = {}
            for prm in params:
                try:
                    X, side = st.features(recent, prm[0], prm[1], list(prm[2]))
                    X_by[prm] = (X, side, st)
                except Exception as e:  # noqa: BLE001
                    db.log_event("error", "radar", f"{sym} live features failed: {e}", self.engine)
            hz = {}
            for key, kind, size in HORIZONS:
                pr = self._predict_stream(sym, info, key, kind, size, X_by, recent)
                if stale:
                    pr["level"] = "gray"
                    pr["stale"] = True
                hz[key] = pr
                n_seen = self.hub.count.get(sym, 0)
                streams = [(sym, key, pr.get("role"), pr.get("model_id"), pr)] + \
                          [(sym, key, "shadow", s["model_id"], {**pr, **s, "role": "shadow", "approved": False,
                                                                 "reasons": [], "performance": None})
                           for s in pr.get("shadow", []) if "probability" in s]
                for _, _, role, mid, p in streams:
                    if p.get("probability") is None or stale:
                        continue
                    stream = (sym, key, role, mid)
                    if self._should_log(stream, last_epoch, n_seen, kind, size, p["level"]):
                        to_log.append({
                            "created_at": now, "symbol": sym, "horizon": key, "horizon_kind": kind, "horizon_size": size,
                            "issued_epoch": int(last_epoch), "expires_epoch": int(last_epoch + size) if kind == "sec" else None,
                            "direction": info["direction"], "probability": float(p["probability"]),
                            "base_rate": p.get("base_rate"), "level": p["level"], "model_id": mid,
                            "model_name": p.get("model_name"), "role": role, "approved": bool(p.get("approved")),
                            "reasons": db.jdump(db.clean_json(p.get("reasons") or [])),
                            "details": db.jdump(db.clean_json({"threshold": p.get("threshold"),
                                                               "performance": p.get("performance"),
                                                               "best_timeframe": p.get("best_timeframe"),
                                                               "expected_size": p.get("expected_size"),
                                                               "escalation": stream in self.last_logged}))})
                        log_meta.append((stream, {"epoch": int(last_epoch), "tickno": n_seen, "level": p["level"]}, p))
            best = max(hz.values(), key=lambda p: (LEVEL_ORDER.get(p["level"], 0),
                                                   (p.get("probability") or 0) / (p.get("base_rate") or 1)))
            radar[sym] = {"symbol": sym, "display_name": info["display_name"], "family": info["family"],
                          "direction": info["direction"], "status": "stale" if stale else "live",
                          "last_epoch": last_epoch, "last_quote": recent[-1][1], "level": best["level"],
                          "top_horizon": best["horizon"], "horizons": hz}
        ids = self._log(to_log)
        for pid, (stream, meta, p) in zip(ids, log_meta):
            self.last_logged[stream] = meta
            if stream[2] == "champion":
                self.alerts.consider(pid, p, meta["epoch"])
        self.hub.radar = radar
        self.last_cycle = {"at": now, "logged": len(ids), "symbols": len(radar),
                           "duration_sec": round(time.perf_counter() - t_start, 3)}
        return radar

    async def run(self, stop: asyncio.Event | None = None) -> None:
        while stop is None or not stop.is_set():
            t0 = time.time()
            try:
                radar = await asyncio.to_thread(self.cycle)
                # publish on the event-loop thread (asyncio queues are not thread-safe)
                self.hub.publish({"type": "radar", "data": db.clean_json(radar_summary(radar))})
                await self.alerts.flush_telegram()
            except Exception as e:  # noqa: BLE001
                log.exception("radar cycle failed")
                db.log_event("error", "radar", f"cycle failed: {e}", self.engine)
            await asyncio.sleep(max(0.2, settings.radar_interval - (time.time() - t0)))


def radar_summary(radar: dict) -> list[dict]:
    rows = []
    for sym, r in radar.items():
        hz = r.get("horizons", {})
        rows.append({"symbol": sym, "display_name": r.get("display_name"), "level": r.get("level"),
                     "direction": r.get("direction"), "family": r.get("family"),
                     "status": r.get("status"), "last_quote": r.get("last_quote"), "last_epoch": r.get("last_epoch"),
                     "top_horizon": r.get("top_horizon"),
                     "horizons": {k: {"p": v.get("probability"), "base": v.get("base_rate"), "level": v.get("level"),
                                      "model": v.get("model_name"), "approved": v.get("approved")}
                                  for k, v in hz.items()}})
    rows.sort(key=lambda x: (-LEVEL_ORDER.get(x["level"] or "gray", 0),
                             -max([(h["p"] or 0) / (h["base"] or 1) for h in x["horizons"].values()] or [0])))
    return rows
