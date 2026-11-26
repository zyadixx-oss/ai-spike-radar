"""End-to-end on a TEST database: ticks -> spikes -> training -> live radar -> immutable predictions ->
evaluation -> forward stats -> alerts; plus model-failure and stale-data handling."""
import numpy as np
import pytest

from app import db
from app.engine.alerts import AlertManager
from app.engine.evaluate import evaluate_pending, forward_stats
from app.engine.radar import Radar
from app.live import LiveHub
from app.research.spikes import scan_and_store
from app.research.train import train_symbol
from tests.sim import simulate

N_TRAIN = 90_000


@pytest.fixture()
def world(engine, monkeypatch):
    import time as _t
    ep, q, _ = simulate(n=N_TRAIN + 3000, mean_gap=150, memoryless=False, seed=31,
                        start_epoch=int(_t.time()) - N_TRAIN - 3000)
    db.upsert(engine, db.symbols, {"symbol": "TBOOM", "display_name": "Boom 1000 Index", "family": "boom", "n": 1000,
                                   "direction": 1, "available": True}, ["symbol"])
    db.insert_ignore(engine, db.ticks, [{"symbol": "TBOOM", "epoch": int(e), "quote": float(x)}
                                        for e, x in zip(ep[:N_TRAIN], q[:N_TRAIN])])
    return ep, q


def test_end_to_end(engine, world, monkeypatch):
    ep, q = world
    s = scan_and_store("TBOOM", 1, engine)
    assert s["spikes"] > 300
    out = train_symbol("TBOOM", 1, engine=engine, horizons=["30s", "100t"],
                       model_names=["lightgbm", "logistic_regression"])
    roles = db.exec_sql("SELECT horizon, role, approved FROM models", engine=engine)
    assert {r["horizon"] for r in roles if r["role"] == "champion"} == {"30s", "100t"}, out
    hub = LiveHub(maxlen=20000)
    hub.seed("TBOOM", ep[N_TRAIN - 20000:N_TRAIN].tolist(), q[N_TRAIN - 20000:N_TRAIN].tolist())
    # pretend the live feed is current
    monkeypatch.setattr("app.engine.radar.time.time", lambda: float(ep[N_TRAIN - 1]) + 1)
    radar = Radar(engine, hub, AlertManager(engine))
    r = radar.cycle()["TBOOM"]
    assert r["status"] == "live"
    for key in ("30s", "100t"):
        h = r["horizons"][key]
        assert h["role"] == "champion" and 0 <= h["probability"] <= 1 and h["level"] in ("green", "yellow", "orange", "red")
        assert h["model_name"] in ("lightgbm", "logistic_regression")
        assert h["performance"] is not None and "acceptance" in h["performance"]
    assert r["horizons"]["5m"]["role"] == "baseline" and r["horizons"]["5m"]["level"] == "gray"
    n0 = db.exec_sql("SELECT COUNT(*) AS n FROM predictions", engine=engine)[0]["n"]
    assert n0 >= 9
    # stream more ticks, keep cycling
    for i in range(N_TRAIN, N_TRAIN + 2500):
        hub.add_tick("TBOOM", int(ep[i]), float(q[i]))
        if i % 25 == 0:
            db.insert_ignore(engine, db.ticks, [{"symbol": "TBOOM", "epoch": int(e), "quote": float(x)}
                                                for e, x in zip(ep[i - 25:i + 1], q[i - 25:i + 1])])
            monkeypatch.setattr("app.engine.radar.time.time", lambda i=i: float(ep[i]) + 1)
            if i % 100 == 0:
                radar.cycle()
    db.insert_ignore(engine, db.ticks, [{"symbol": "TBOOM", "epoch": int(e), "quote": float(x)}
                                        for e, x in zip(ep[N_TRAIN:], q[N_TRAIN:])])
    res = evaluate_pending(engine)
    assert res["hit"] + res["miss"] > 20, res
    fs = forward_stats(engine, role="champion")
    assert fs and all(0 <= f["observed_rate"] <= 1 for f in fs)
    # outcomes are consistent with the stored ticks: re-evaluating cannot change anything
    evaluate_pending(engine)
    assert db.exec_sql("SELECT COUNT(*) AS n FROM prediction_outcomes", engine=engine)[0]["n"] == res["hit"] + res["miss"] + res["invalid"]
    # alerts (if any) are de-duplicated per symbol/horizon window
    al = db.exec_sql("SELECT dedupe_key FROM alerts", engine=engine)
    assert len(al) == len({a["dedupe_key"] for a in al})


def test_model_failure_degrades_to_gray(engine, world, monkeypatch):
    ep, q = world
    scan_and_store("TBOOM", 1, engine)
    train_symbol("TBOOM", 1, engine=engine, horizons=["100t"], model_names=["logistic_regression"])
    hub = LiveHub(maxlen=20000)
    hub.seed("TBOOM", ep[N_TRAIN - 20000:N_TRAIN].tolist(), q[N_TRAIN - 20000:N_TRAIN].tolist())
    monkeypatch.setattr("app.engine.radar.time.time", lambda: float(ep[N_TRAIN - 1]) + 1)
    radar = Radar(engine, hub, AlertManager(engine))
    radar.cache.refresh(force=True)
    for art in radar.cache.models.values():
        def boom(*a, **k):  # noqa: ANN002, ANN003, ANN202
            raise RuntimeError("corrupted model")
        monkeypatch.setattr(art["model"], "score", boom)
    radar.cache.loaded_at = 1e18  # keep the sabotaged models
    h = radar.cycle()["TBOOM"]["horizons"]["100t"]
    assert h["level"] == "gray" and "corrupted" in h["error"]


def test_stale_feed_is_gray(engine, world, monkeypatch):
    ep, q = world
    hub = LiveHub(maxlen=20000)
    hub.seed("TBOOM", ep[N_TRAIN - 5000:N_TRAIN].tolist(), q[N_TRAIN - 5000:N_TRAIN].tolist())
    monkeypatch.setattr("app.engine.radar.time.time", lambda: float(ep[N_TRAIN - 1]) + 600)
    r = Radar(engine, hub, AlertManager(engine)).cycle()["TBOOM"]
    assert r["status"] == "stale" and all(h["level"] == "gray" for h in r["horizons"].values())
    assert db.exec_sql("SELECT COUNT(*) AS n FROM predictions", engine=engine)[0]["n"] == 0
    assert np.isfinite(r["last_quote"])


def test_champion_replaced_only_on_proven_improvement(engine, world):
    """Second training run on more data: the old champion is compared on a window it never saw."""
    ep, q = world
    from app.research.data import load_ticks
    e_all, q_all = load_ticks("TBOOM", engine=engine)
    cut = 60_000
    train_symbol("TBOOM", 1, engine=engine, horizons=["100t"], model_names=["logistic_regression"],
                 epochs=e_all[:cut], quotes=q_all[:cut])
    first = db.exec_sql("SELECT id, approved FROM models WHERE role='champion'", engine=engine)
    assert len(first) == 1
    out = train_symbol("TBOOM", 1, engine=engine, horizons=["100t"], model_names=["lightgbm"])
    reg = out["horizons"]["100t"]["registry"]["registered"]
    champs = db.exec_sql("SELECT id, model_name, role, notes FROM models WHERE role='champion'", engine=engine)
    assert len(champs) == 1  # never two champions for one stream
    summary = db.exec_sql("SELECT metrics FROM models WHERE id=:i", {"i": reg[0]["id"]}, engine=engine)[0]
    dec = db.jloads(summary["metrics"], {})["decision"]
    if first[0]["approved"]:
        assert "comparison" in dec and dec["comparison"]["fair"] is True
        assert (champs[0]["id"] == reg[0]["id"]) == bool(dec["comparison"]["new_significantly_better"]
                                                         and out["horizons"]["100t"]["approved"])
    else:
        assert champs[0]["id"] == reg[0]["id"] and dec["reason"] == "current champion is not approved"
    # artifacts are stored in the database (shared by API and worker processes)
    n_art = db.exec_sql("SELECT COUNT(*) AS n FROM model_artifacts", engine=engine)[0]["n"]
    assert n_art == db.exec_sql("SELECT COUNT(*) AS n FROM models", engine=engine)[0]["n"]
