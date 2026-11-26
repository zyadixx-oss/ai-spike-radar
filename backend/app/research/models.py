"""Candidate models and reference baselines.

Every estimator exposes:  fit(X, y, ctx) -> self ;  score(X, ctx) -> np.ndarray (higher = more likely)
Raw scores are mapped to calibrated probabilities by `Calibrator` fitted on the VALIDATION block.
`ctx` carries row-aligned side information (epochs, horizon, per-row next-tick labels for hazard models…).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class Ctx:
    epochs: np.ndarray
    horizon_kind: str
    horizon_size: int
    mean_dt: float
    extra: dict = field(default_factory=dict)

    def subset(self, rows: np.ndarray) -> "Ctx":
        ex = {k: (v[rows] if isinstance(v, np.ndarray) and len(v) == len(self.epochs) else v)
              for k, v in self.extra.items()}
        return Ctx(self.epochs[rows], self.horizon_kind, self.horizon_size, self.mean_dt, ex)

    @property
    def horizon_ticks(self) -> float:
        return float(self.horizon_size) if self.horizon_kind == "tick" else self.horizon_size / max(self.mean_dt, 1e-9)


class Calibrator:
    """Isotonic calibration (falls back to Platt when the validation block has too few positives)."""

    def __init__(self):
        self.kind = "identity"
        self.iso: IsotonicRegression | None = None
        self.lr: LogisticRegression | None = None
        self.const: float | None = None

    def fit(self, s: np.ndarray, y: np.ndarray) -> "Calibrator":
        s = np.nan_to_num(s, nan=np.nanmedian(s) if np.isfinite(s).any() else 0.0)
        if y.sum() == 0 or y.sum() == len(y) or np.ptp(s) == 0:
            self.kind, self.const = "constant", float(y.mean()) if len(y) else 0.0
        elif y.sum() >= 30:
            self.kind = "isotonic"
            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(s, y)
        else:
            self.kind = "platt"
            self.lr = LogisticRegression(C=1.0).fit(s.reshape(-1, 1), y)
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        s = np.nan_to_num(np.asarray(s, dtype=float), nan=0.0)
        if self.kind == "constant":
            return np.full(len(s), self.const)
        if self.kind == "isotonic":
            return self.iso.predict(s)
        if self.kind == "platt":
            return self.lr.predict_proba(s.reshape(-1, 1))[:, 1]
        return np.clip(s, 0, 1)


# ============================================================================ baselines
class BaseRate:
    name, kind, complexity = "base_rate", "baseline", 0

    def fit(self, X, y, ctx):  # noqa: ANN001
        self.p = float(np.mean(y)) if len(y) else 0.0
        return self

    def score(self, X, ctx):  # noqa: ANN001
        return np.full(len(X), self.p)


class RandomGuess:
    """Uninformative ranking (random scores); calibrated probability equals the base rate."""
    name, kind, complexity = "random", "baseline", 0

    def fit(self, X, y, ctx):  # noqa: ANN001
        self.p = float(np.mean(y)) if len(y) else 0.0
        return self

    def score(self, X, ctx):  # noqa: ANN001
        rng = np.random.default_rng(len(X))
        return rng.random(len(X))


class Geometric:
    """Memoryless model: constant per-tick spike probability lambda; P(spike within h) = 1-(1-lambda)^h."""
    name, kind, complexity = "geometric", "baseline", 0

    def fit(self, X, y, ctx):  # noqa: ANN001
        self.lam = float(ctx.extra.get("lambda_per_tick", np.nan))
        if not np.isfinite(self.lam):
            p = max(float(np.mean(y)), 1e-9)
            self.lam = 1 - (1 - p) ** (1 / max(ctx.horizon_ticks, 1))
        return self

    def score(self, X, ctx):  # noqa: ANN001
        return np.full(len(X), 1 - (1 - self.lam) ** ctx.horizon_ticks)


class IntervalHazard:
    """Empirical conditional probability given ticks-since-last-spike (quantile bins, Laplace smoothing)."""
    name, kind, complexity = "interval_hazard", "baseline", 1
    col = "t_ticks_since_spike"

    def fit(self, X, y, ctx):  # noqa: ANN001
        x = X[self.col].to_numpy(dtype=float)
        self.prior = float(np.mean(y)) if len(y) else 0.0
        ok = np.isfinite(x)
        self.edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, 21))) if ok.sum() > 50 else np.array([0, np.inf])
        b = np.clip(np.searchsorted(self.edges, x[ok], side="right") - 1, 0, max(len(self.edges) - 2, 0))
        nb = max(len(self.edges) - 1, 1)
        k = np.bincount(b, weights=y[ok], minlength=nb)
        n = np.bincount(b, minlength=nb)
        a = 10.0
        self.rate = (k + a * self.prior) / (n + a)
        self.nan_rate = float(np.mean(y[~ok])) if (~ok).sum() > 30 else self.prior
        return self

    def score(self, X, ctx):  # noqa: ANN001
        x = X[self.col].to_numpy(dtype=float)
        out = np.full(len(x), self.nan_rate)
        ok = np.isfinite(x)
        b = np.clip(np.searchsorted(self.edges, x[ok], side="right") - 1, 0, len(self.rate) - 1)
        out[ok] = self.rate[b]
        return out


class WeibullSurvival:
    """Survival model on inter-spike intervals: P(T <= s+h | T > s) = 1 - S(s+h)/S(s)."""
    name, kind, complexity = "weibull_survival", "baseline", 1
    col = "t_ticks_since_spike"

    def fit(self, X, y, ctx):  # noqa: ANN001
        from scipy.stats import weibull_min
        iv = np.asarray(ctx.extra.get("train_intervals", []), dtype=float)
        iv = iv[np.isfinite(iv) & (iv > 0)]
        self.prior = float(np.mean(y)) if len(y) else 0.0
        if len(iv) >= 20:
            c, _, scale = weibull_min.fit(iv, floc=0)
            self.c, self.scale = float(c), float(scale)
        else:
            self.c, self.scale = 1.0, float(np.mean(iv)) if len(iv) else 1e9
        return self

    def score(self, X, ctx):  # noqa: ANN001
        s = X[self.col].to_numpy(dtype=float)
        h = ctx.horizon_ticks

        def logS(t):  # noqa: ANN001
            return -np.power(np.clip(t, 0, None) / self.scale, self.c)
        p = 1 - np.exp(logS(s + h) - logS(s))
        return np.where(np.isfinite(s), p, self.prior)


# ============================================================================ supervised models
class _SK:
    kind = "model"

    def __init__(self, est, name: str, complexity: int, cols: list[str] | None = None):
        self.est, self.name, self.complexity, self.cols = est, name, complexity, cols

    def _X(self, X: pd.DataFrame):  # noqa: ANN202
        return X[self.cols].to_numpy(dtype=np.float32) if self.cols else X.to_numpy(dtype=np.float32)

    def fit(self, X, y, ctx):  # noqa: ANN001
        if y.sum() == 0:
            raise ValueError("no positive labels in training data")
        self.est.fit(self._X(X), y)
        return self

    def score(self, X, ctx):  # noqa: ANN001
        return self.est.predict_proba(self._X(X))[:, 1]

    def importances(self, cols: list[str]) -> dict[str, float]:
        est = self.est[-1] if hasattr(self.est, "steps") else self.est
        cols = self.cols or cols
        if hasattr(est, "feature_importances_"):
            v = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            v = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
        else:
            return {}
        v = v[:len(cols)]
        s = v.sum() or 1.0
        return {c: float(x / s) for c, x in zip(cols, v)}


def logistic():
    return _SK(make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=1000)), "logistic_regression", 1)


def random_forest():
    return _SK(make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=40,
                                                    max_features="sqrt", n_jobs=-1, random_state=0)),
               "random_forest", 3)


class DropAllNaN(BaseEstimator, TransformerMixin):
    """Drop columns that are entirely NaN in the training data (works around a binning bug in HistGB)."""

    def fit(self, X, y=None):  # noqa: ANN001
        self.keep_ = ~np.all(np.isnan(np.asarray(X, dtype=float)), axis=0)
        return self

    def transform(self, X):  # noqa: ANN001
        return np.asarray(X)[:, self.keep_]


def gradient_boosting():
    return _SK(make_pipeline(DropAllNaN(), HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0,
        random_state=0)), "gradient_boosting", 3)


def xgboost_model():
    from xgboost import XGBClassifier
    return _SK(XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.6, min_child_weight=20, tree_method="hist", n_jobs=4,
                             eval_metric="logloss", random_state=0, verbosity=0), "xgboost", 3)


def lightgbm_model():
    from lightgbm import LGBMClassifier
    return _SK(LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.6, verbose=-1,
                              random_state=0, n_jobs=4), "lightgbm", 3)


class IsolationForestModel:
    """Unsupervised anomaly score (fit on features only); calibration maps anomaly -> probability."""
    name, kind, complexity = "isolation_forest", "model", 2

    def fit(self, X, y, ctx):  # noqa: ANN001
        self.est = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                 IsolationForest(n_estimators=150, random_state=0, n_jobs=-1))
        self.est.fit(X.to_numpy(dtype=np.float32))
        return self

    def score(self, X, ctx):  # noqa: ANN001
        return -self.est.score_samples(X.to_numpy(dtype=np.float32))


class DiscreteHazard:
    """Discrete-time survival model: per-tick hazard h(x) by logistic regression on next-tick spike labels,
    aggregated over the horizon: P = 1 - (1 - h)^H."""
    name, kind, complexity = "discrete_hazard", "model", 1
    cols = ["t_log_ticks_since_spike", "t_vol_50", "t_vol_500", "t_compress_50_500", "t_spikes_2000",
            "t_ret_50", "t_entropy_200"]

    def fit(self, X, y, ctx):  # noqa: ANN001
        y1 = ctx.extra.get("y_next_tick")
        if y1 is None or np.sum(y1) < 5:
            raise ValueError("discrete hazard needs next-tick labels")
        cols = [c for c in self.cols if c in X.columns]
        self.use = cols
        self.est = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                 LogisticRegression(C=0.1, max_iter=1000))
        self.est.fit(X[cols].to_numpy(dtype=np.float32), np.asarray(y1))
        return self

    def score(self, X, ctx):  # noqa: ANN001
        h = self.est.predict_proba(X[self.use].to_numpy(dtype=np.float32))[:, 1]
        return 1 - np.power(1 - h, ctx.horizon_ticks)


class HMMRegime:
    """Gaussian HMM on M1 candle (log-range, return) sequence. States are FORWARD-FILTERED (causal — never
    the smoothed forward-backward posteriors), joined to rows as-of candle close, then logistic regression."""
    name, kind, complexity = "hidden_markov", "model", 2
    n_states = 3

    def _obs(self, c: pd.DataFrame) -> np.ndarray:
        rng = np.log((c["high"] - c["low"]).clip(lower=1e-12) / c["close"])
        ret = np.log(c["close"] / c["open"])
        o = np.column_stack([rng.to_numpy(), ret.to_numpy()])
        return np.nan_to_num(o, nan=0.0)

    def _filter(self, obs: np.ndarray) -> np.ndarray:
        from scipy.stats import multivariate_normal
        K = self.n_states
        ll = np.column_stack([multivariate_normal(self.means[k], np.diag(self.covs[k]), allow_singular=True)
                              .logpdf(obs) for k in range(K)])
        alpha = np.zeros((len(obs), K))
        prev = self.start
        for t in range(len(obs)):
            pred = prev if t == 0 else alpha[t - 1] @ self.trans
            a = np.log(np.clip(pred, 1e-300, None)) + ll[t]
            a = np.exp(a - a.max())
            alpha[t] = a / a.sum()
        return alpha

    def _state_feats(self, epochs: np.ndarray) -> np.ndarray:
        c = self.candles
        m = pd.merge_asof(pd.DataFrame({"e": epochs.astype(np.int64), "o": np.arange(len(epochs))}).sort_values("e"),
                          pd.DataFrame({"a": c["avail"].to_numpy(np.int64), **{f"s{k}": self.alpha[:, k]
                                                                               for k in range(self.n_states)}}),
                          left_on="e", right_on="a", direction="backward").sort_values("o")
        return m[[f"s{k}" for k in range(self.n_states)]].to_numpy(dtype=float)

    def fit(self, X, y, ctx):  # noqa: ANN001
        from hmmlearn.hmm import GaussianHMM
        c = ctx.extra.get("m1_candles")
        if c is None or len(c) < 200:
            raise ValueError("HMM needs >= 200 M1 candles")
        train_c = c[c["avail"] <= ctx.epochs.max()]
        if len(train_c) < 200:
            raise ValueError("HMM needs >= 200 training M1 candles")
        hmm = GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=50, random_state=0)
        hmm.fit(self._obs(train_c))
        self.means, self.covs = hmm.means_, np.array([np.diag(cv) for cv in hmm.covars_])
        self.trans, self.start = hmm.transmat_, hmm.startprob_
        self.candles = c.reset_index(drop=True)
        self.alpha = self._filter(self._obs(self.candles))
        self.extra_cols = [col for col in ("t_vol_50", "t_vol_500", "t_log_ticks_since_spike") if col in X.columns]
        Z = np.column_stack([self._state_feats(ctx.epochs), X[self.extra_cols].to_numpy(dtype=float)])
        self.est = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                 LogisticRegression(C=0.1, max_iter=1000)).fit(Z, y)
        return self

    def refresh(self, candles: pd.DataFrame) -> None:
        """Live use: re-filter on an updated candle series with the FIXED fitted parameters."""
        self.candles = candles.reset_index(drop=True)
        self.alpha = self._filter(self._obs(self.candles))

    def score(self, X, ctx):  # noqa: ANN001
        Z = np.column_stack([self._state_feats(ctx.epochs), X[self.extra_cols].to_numpy(dtype=float)])
        return self.est.predict_proba(Z)[:, 1]


def baseline_factories():
    return {"random": RandomGuess, "base_rate": BaseRate, "geometric": Geometric,
            "interval_hazard": IntervalHazard, "weibull_survival": WeibullSurvival}


def model_factories(enable_deep: bool = True):
    f = {"logistic_regression": logistic, "random_forest": random_forest, "gradient_boosting": gradient_boosting,
         "xgboost": xgboost_model, "lightgbm": lightgbm_model, "isolation_forest": IsolationForestModel,
         "discrete_hazard": DiscreteHazard, "hidden_markov": HMMRegime}
    if enable_deep:
        from .deep import deep_factories
        f.update(deep_factories())
    return f


def lambda_per_tick(n_spikes: int, n_ticks: int) -> float:
    return n_spikes / n_ticks if n_ticks else float("nan")


def safe_log(x: float) -> float:
    return math.log(x) if x > 0 else float("-inf")
