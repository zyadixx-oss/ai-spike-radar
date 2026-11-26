"""Statistical control tests of the whole research pipeline (TEST data only, never in the app DB).

Negative control: memoryless spikes -> no model may be approved, no strategy may pass validation+holdout.
Positive control: planted pre-spike regime -> the pipeline must find it (otherwise it is too blunt to be useful).
"""
import numpy as np
import pytest

from app.research.arena import run_arena
from app.research.train import train_symbol
from tests.sim import simulate

MODELS = ["logistic_regression", "lightgbm", "random_forest", "discrete_hazard", "temporal_cnn"]


@pytest.fixture(scope="module")
def memoryless():
    return simulate(n=100_000, mean_gap=150, memoryless=True, seed=21)[:2]


@pytest.fixture(scope="module")
def planted():
    return simulate(n=100_000, mean_gap=150, memoryless=False, seed=22)[:2]


def test_negative_control_no_model_approved(memoryless):
    ep, q = memoryless
    out = train_symbol("SIM", 1, epochs=ep, quotes=q, horizons=["1m", "100t"], model_names=MODELS, persist=False)
    for key, h in out["horizons"].items():
        assert h["status"] == "ok", h
        assert h["approved"] is False, (key, h.get("acceptance"))
        # baselines and every model were evaluated out-of-sample
        assert {"random", "base_rate", "geometric", "interval_hazard", "weibull_survival"} <= set(h["candidates"])


def test_positive_control_pattern_found(planted):
    ep, q = planted
    out = train_symbol("SIM", 1, epochs=ep, quotes=q, horizons=["1m", "100t"], model_names=MODELS, persist=False)
    approved = [k for k, h in out["horizons"].items() if h.get("approved")]
    assert approved, {k: h.get("acceptance") for k, h in out["horizons"].items()}
    h = out["horizons"][approved[0]]
    assert h["thresholds"]["yellow"] is not None
    assert h["thresholds"]["yellow"] > h["thresholds"]["base_rate"]
    t = h["reports"][h["selected"]]["test"]
    assert t["bss_vs_base_rate"] > 0 and t["pr_auc"] > t["prevalence"]


def test_arena_negative_control(memoryless):
    ep, q = memoryless
    out = run_arena("SIM", 1, ep, q, persist=False, horizons=["1m", "100t"])
    for key, h in out["horizons"].items():
        assert h["test_confirmed"] == 0, key
        assert h["rules_tested"] > 1000
        assert len(h["rules_per_family"]) >= 23


def test_arena_positive_control(planted):
    ep, q = planted
    out = run_arena("SIM", 1, ep, q, persist=False, horizons=["1m"])
    h = out["horizons"]["1m"]
    assert h["val_significant_fdr10"] > 0 and h["test_confirmed"] > 0
    assert any(f["family"] in ("candle_compression", "atr_compression", "bollinger_squeeze", "statistical_anomaly",
                               "tick_velocity", "volatility_regime") for f in h["families"][:5] if f["val_significant"])
    assert np.isfinite(h["trading_benchmark_test"]["net_return_pct"])
