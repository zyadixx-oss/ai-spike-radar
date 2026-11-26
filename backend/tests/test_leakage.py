"""No look-ahead: features at tick i must not change when the future is altered or removed."""
import numpy as np
import pytest

from app.config import HORIZONS
from app.research.dataset import build_dataset
from app.research.features import build_features
from app.research.labels import make_labels
from app.research.validation import assert_no_overlap, dev_val_test, walk_forward
from tests.sim import simulate

TFS = ["5s", "15s", "M1", "M5", "M15"]


@pytest.fixture(scope="module")
def series():
    ep, q, _ = simulate(n=20000, mean_gap=200, seed=3, gaps=[(9000, 400)])
    return ep, q


def test_features_unchanged_by_future(series):
    ep, q = series
    sigma, thr = 1e-4, 8e-4
    idx = np.array([1500, 5000, 8999, 9001, 12000, 15000])
    full = build_features(ep, q, 1, sigma, thr, TFS, idx)
    for k, i in enumerate(idx):
        trunc = build_features(ep[:i + 1], q[:i + 1], 1, sigma, thr, TFS, np.array([i]))
        a, b = full.iloc[k].to_numpy(dtype=float), trunc.iloc[0].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-12), (
            i, [c for c, x, y in zip(full.columns, a, b) if not (np.isclose(x, y) or (np.isnan(x) and np.isnan(y)))])


def test_future_perturbation_does_not_move_features(series):
    ep, q = series
    i = 10000
    q2 = q.copy()
    q2[i + 1:] *= np.exp(np.random.default_rng(0).normal(0, 0.01, len(q) - i - 1))  # wreck the future
    a = build_features(ep, q, 1, 1e-4, 8e-4, TFS, np.array([i]))
    b = build_features(ep, q2, 1, 1e-4, 8e-4, TFS, np.array([i]))
    assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


def test_labels_do_use_future_and_respect_gaps(series):
    ep, q = series
    sp = np.array([100, 500, 8990, 9010])
    idx = np.array([50, 450, 8950, 8995])
    y, valid, lead = make_labels(ep, sp, idx, "tick", 60)
    assert y.tolist() == [1, 1, 1, 0]
    # tick 8995 is just before the gap at 9000: its 60-tick window is not observed -> invalid
    assert valid.tolist() == [True, True, True, False]


def test_purged_splits_never_overlap(series):
    ep, q = series
    ds = build_dataset("SIM", 1, ep, q, timeframes=["M1"], horizons=["1m", "100t"])
    for key in ("1m", "100t"):
        hsec = ds.horizon_seconds(key)
        e = ds.row_epochs
        sp = dev_val_test(e, hsec, 300)
        assert_no_overlap(e[sp["dev"]], e[sp["val"]], hsec)
        assert_no_overlap(e[sp["val"]], e[sp["test"]], hsec)
        for f in walk_forward(e, sp["dev"], 3, hsec, 300):
            assert_no_overlap(e[f.train], e[f.test], hsec)


def test_overlap_detector_catches_leak():
    with pytest.raises(AssertionError):
        assert_no_overlap(np.array([0, 100]), np.array([150, 200]), 60)


def test_scale_estimated_on_training_part_only(series):
    """sigma/threshold come from the first 60% of ticks: changing the last 40% must not change them."""
    ep, q = series
    ds1 = build_dataset("SIM", 1, ep, q, timeframes=["M1"], horizons=["1m"])
    q2 = q.copy()
    cut = int(len(q) * 0.6) + 1
    q2[cut:] = q2[cut] * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 5e-4, len(q) - cut)))
    ds2 = build_dataset("SIM", 1, ep, q2, timeframes=["M1"], horizons=["1m"])
    assert ds1.sigma == pytest.approx(ds2.sigma) and ds1.threshold == pytest.approx(ds2.threshold)
    assert set(h for h, _, _ in HORIZONS) >= {"1m"}
