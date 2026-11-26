"""Causal feature engineering.

Every feature value at tick i is a function of ticks[0..i] only (verified by tests/test_leakage.py).
* Tick features: bounded rolling windows that are invalidated (NaN) when the window crosses a data gap.
* Candle features: computed on CLOSED candles only, joined with merge_asof on the candle's close time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAMES
from .candles import build_candles
from .data import segment_bounds, segment_ids
from .spikes import log_returns

TICK_WINDOWS = (10, 30, 50, 100, 200, 500)
MAX_TICK_LOOKBACK = 2000


# ---------------------------------------------------------------------------- tick features
def _roll(s: pd.Series, w: int, fn: str) -> np.ndarray:
    r = s.rolling(w, min_periods=w)
    return getattr(r, fn)().to_numpy() if fn != "std" else r.std(ddof=0).to_numpy()


def tick_feature_columns() -> list[str]:
    cols = []
    for w in TICK_WINDOWS:
        cols += [f"t_vol_{w}", f"t_ret_{w}", f"t_up_{w}"]
    cols += ["t_absmax_50", "t_absmax_200", "t_compress_10_100", "t_compress_50_500", "t_accel_10", "t_accel_50",
             "t_entropy_50", "t_entropy_200", "t_runlen", "t_z_last", "t_rsi_14", "t_dd_high_200", "t_du_low_200",
             "t_dt_50", "t_spikes_500", "t_spikes_2000", "t_ticks_since_spike", "t_log_ticks_since_spike",
             "t_secs_since_spike"]
    return cols


def tick_features(epochs: np.ndarray, quotes: np.ndarray, direction: int, sigma: float, threshold: float,
                  idx: np.ndarray | None = None) -> pd.DataFrame:
    """Tick features at indices `idx` (default: all). sigma/threshold must come from TRAINING data."""
    n = len(epochs)
    idx = np.arange(n) if idx is None else np.asarray(idx, dtype=np.int64)
    if n < 2 or len(idx) == 0:
        return pd.DataFrame(np.nan, index=range(len(idx)), columns=tick_feature_columns())
    seg = segment_ids(epochs)
    seg_start, _ = segment_bounds(seg)
    pos = np.arange(n) - seg_start  # ticks since segment start (0-based)
    r = log_returns(quotes, seg)
    is_spk_tick = np.nan_to_num(direction * r, nan=0.0) > threshold
    spk_start = is_spk_tick & ~np.concatenate([[False], is_spk_tick[:-1] & (seg[1:] == seg[:-1])])
    rc = np.where(is_spk_tick, 0.0, np.nan_to_num(r, nan=0.0))  # spike-free returns
    s_rc = pd.Series(rc)
    sig = sigma if sigma and sigma > 0 else 1.0
    out: dict[str, np.ndarray] = {}

    def valid(w: int) -> np.ndarray:
        return pos >= w  # window of returns [i-w+1..i] all inside segment (r at seg start is undefined)

    for w in TICK_WINDOWS:
        v = valid(w)
        out[f"t_vol_{w}"] = np.where(v, _roll(s_rc, w, "std") / sig, np.nan)[idx]
        out[f"t_ret_{w}"] = np.where(v, _roll(s_rc, w, "sum") / sig, np.nan)[idx]
        out[f"t_up_{w}"] = np.where(v, _roll(pd.Series((rc > 0).astype(float)), w, "mean"), np.nan)[idx]
    s_abs = pd.Series(np.abs(rc))
    for w in (50, 200):
        out[f"t_absmax_{w}"] = np.where(valid(w), _roll(s_abs, w, "max") / sig, np.nan)[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["t_compress_10_100"] = out["t_vol_10"] / out["t_vol_100"]
        out["t_compress_50_500"] = out["t_vol_50"] / out["t_vol_500"]
    for w in (10, 50):
        ret_w = np.where(valid(2 * w), _roll(s_rc, w, "sum") / sig, np.nan)
        prev = np.concatenate([np.full(w, np.nan), ret_w[:-w]]) if n > w else np.full(n, np.nan)
        out[f"t_accel_{w}"] = (ret_w - prev)[idx]
    for w in (50, 200):
        p = out[f"t_up_{w}"]
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -(p * np.log2(np.clip(p, 1e-12, 1)) + (1 - p) * np.log2(np.clip(1 - p, 1e-12, 1)))
        out[f"t_entropy_{w}"] = np.where(np.isnan(p), np.nan, h)
    # current run length of same-sign returns (causal)
    sgn = np.sign(rc)
    change = np.concatenate([[True], (sgn[1:] != sgn[:-1]) | (seg[1:] != seg[:-1])])
    grp = np.cumsum(change)
    run = pd.Series(np.ones(n)).groupby(grp).cumsum().to_numpy() * np.where(sgn == 0, 0, sgn)
    out["t_runlen"] = run[idx]
    vol200 = np.where(valid(200), _roll(s_rc, 200, "std"), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["t_z_last"] = (rc / vol200)[idx]
    gain = pd.Series(np.clip(rc, 0, None))
    loss = pd.Series(np.clip(-rc, 0, None))
    ag, al = _roll(gain, 14, "mean"), _roll(loss, 14, "mean")
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + ag / al)
    rsi = np.where(al == 0, 100.0, rsi)
    out["t_rsi_14"] = np.where(valid(14), rsi, np.nan)[idx]
    lq = pd.Series(np.log(quotes))
    v200 = pos >= 199
    out["t_dd_high_200"] = np.where(v200, (lq.to_numpy() - _roll(lq, 200, "max")) / sig, np.nan)[idx]
    out["t_du_low_200"] = np.where(v200, (lq.to_numpy() - _roll(lq, 200, "min")) / sig, np.nan)[idx]
    dt = np.concatenate([[np.nan], np.diff(epochs).astype(float)])
    out["t_dt_50"] = np.where(valid(50), _roll(pd.Series(dt), 50, "mean"), np.nan)[idx]
    s_sp = pd.Series(spk_start.astype(float))
    for w in (500, 2000):
        out[f"t_spikes_{w}"] = np.where(pos >= w - 1, _roll(s_sp, w, "sum"), np.nan)[idx]
    last = np.maximum.accumulate(np.where(spk_start, np.arange(n), -1))
    known = last >= seg_start
    ts = np.where(known, np.arange(n) - last, np.nan)
    out["t_ticks_since_spike"] = ts[idx]
    out["t_log_ticks_since_spike"] = np.log1p(ts)[idx]
    last_c = np.clip(last, 0, None)
    out["t_secs_since_spike"] = np.where(known, epochs - epochs[last_c], np.nan)[idx].astype(float)
    df = pd.DataFrame(out)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df[tick_feature_columns()]


# ---------------------------------------------------------------------------- candle indicators
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def candle_indicators(c: pd.DataFrame, direction: int) -> pd.DataFrame:
    """Indicators on CLOSED candles. Row k depends on candles[0..k] only."""
    o, h, l, cl = (c[k].astype(float) for k in ("open", "high", "low", "close"))
    f: dict[str, pd.Series] = {}
    prev_c = cl.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr14 = _wilder(tr, 14)
    atr50 = tr.rolling(50, min_periods=50).mean()
    # Bollinger
    ma20 = cl.rolling(20, min_periods=20).mean()
    sd20 = cl.rolling(20, min_periods=20).std(ddof=0)
    bbw = 4 * sd20 / ma20
    f["bb_width"] = bbw
    bmin, bmax = bbw.rolling(100, min_periods=30).min(), bbw.rolling(100, min_periods=30).max()
    f["bb_squeeze_rank"] = (bbw - bmin) / (bmax - bmin)
    f["bb_pctb"] = (cl - (ma20 - 2 * sd20)) / (4 * sd20)
    # ATR compression
    f["atr_pct"] = atr14 / cl
    f["atr_ratio"] = atr14 / atr50
    # RSI / StochRSI
    d = cl.diff()
    ag, al = _wilder(d.clip(lower=0), 14), _wilder((-d).clip(lower=0), 14)
    rsi = 100 - 100 / (1 + ag / al)
    f["rsi"] = rsi
    rmin, rmax = rsi.rolling(14, min_periods=14).min(), rsi.rolling(14, min_periods=14).max()
    f["stochrsi"] = (rsi - rmin) / (rmax - rmin)
    # MACD
    macd = _ema(cl, 12) - _ema(cl, 26)
    f["macd_hist"] = (macd - _ema(macd, 9)) / cl
    # ADX
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    pdi = 100 * _wilder(pdm, 14) / atr14
    ndi = 100 * _wilder(ndm, 14) / atr14
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    f["adx"] = _wilder(dx, 14)
    f["di_diff"] = pdi - ndi
    # Keltner
    ema20 = _ema(cl, 20)
    atr10 = _wilder(tr, 10)
    f["keltner_pos"] = (cl - ema20) / (2 * atr10)
    f["bb_in_kc"] = ((2 * sd20) < (1.5 * atr10)).astype(float).where(sd20.notna() & atr10.notna())
    # Donchian
    hh20, ll20 = h.rolling(20, min_periods=20).max(), l.rolling(20, min_periods=20).min()
    f["donchian_pos"] = (cl - ll20) / (hh20 - ll20)
    f["donchian_width"] = (hh20 - ll20) / cl
    # EMA distance
    f["ema50_dist"] = (cl - _ema(cl, 50)) / atr14
    f["ema_fast_slow"] = (_ema(cl, 9) - _ema(cl, 21)) / atr14
    # Momentum
    f["roc10"] = cl.pct_change(10)
    # Candle compression / anatomy
    rng = h - l
    f["range_ratio"] = rng / rng.rolling(20, min_periods=20).mean()
    f["compression_5"] = rng.rolling(5, min_periods=5).mean() / rng.rolling(50, min_periods=50).mean()
    f["body_ratio"] = (cl - o).abs() / rng.replace(0, np.nan)
    f["upper_wick"] = (h - np.maximum(o, cl)) / rng.replace(0, np.nan)
    f["lower_wick"] = (np.minimum(o, cl) - l) / rng.replace(0, np.nan)
    # Price action: consecutive candles against the spike direction (the "drift")
    against = ((cl - o) * direction < 0).astype(int)
    grp = (against != against.shift()).cumsum()
    f["consec_drift"] = against.groupby(grp).cumsum().astype(float)
    # Liquidity sweep: wick takes prior 20-bar extreme on the drift side and closes back inside
    prior_ll, prior_hh = l.shift(1).rolling(20, min_periods=20).min(), h.shift(1).rolling(20, min_periods=20).max()
    if direction > 0:
        sweep = ((l < prior_ll) & (cl > prior_ll)).astype(float)
    else:
        sweep = ((h > prior_hh) & (cl < prior_hh)).astype(float)
    f["liq_sweep"] = sweep.where(prior_ll.notna())
    f["liq_sweep_recent"] = sweep.rolling(5, min_periods=5).max()
    # ICT: premium/discount within the 50-bar dealing range + displacement candle
    hh50, ll50 = h.rolling(50, min_periods=50).max(), l.rolling(50, min_periods=50).min()
    f["ict_range_pos"] = (cl - ll50) / (hh50 - ll50)
    f["ict_displacement"] = ((cl - o).abs() > 2 * atr14).astype(float).where(atr14.notna())
    # CRT (candle range theory): current candle trades beyond previous candle's extreme then closes inside it
    ph, pl = h.shift(1), l.shift(1)
    crt_bull = ((l < pl) & (cl > pl) & (cl < ph)).astype(float)
    crt_bear = ((h > ph) & (cl < ph) & (cl > pl)).astype(float)
    f["crt"] = (crt_bull if direction > 0 else crt_bear).where(ph.notna())
    # Fair value gaps (3-candle imbalance) — counts in last 20 candles
    bull_fvg = (l > h.shift(2)).astype(float)
    bear_fvg = (h < l.shift(2)).astype(float)
    f["fvg_bull_20"] = bull_fvg.rolling(20, min_periods=20).sum()
    f["fvg_bear_20"] = bear_fvg.rolling(20, min_periods=20).sum()
    # Market structure: lower-lows minus higher-highs over 10 bars; break of structure on drift side
    ll_cnt = (l < l.shift(1)).astype(float).rolling(10, min_periods=10).sum()
    hh_cnt = (h > h.shift(1)).astype(float).rolling(10, min_periods=10).sum()
    f["structure"] = (hh_cnt - ll_cnt) * direction
    f["bos_drift"] = ((cl < prior_ll) if direction > 0 else (cl > prior_hh)).astype(float).where(prior_ll.notna())
    # Volatility regime: log range percentile over 100 bars
    lr = np.log(rng.replace(0, np.nan) / cl)
    f["vol_regime"] = lr.rolling(100, min_periods=30).rank(pct=True)
    # Real activity: ticks per candle (NOT volume)
    if "ticks" in c:
        tk = c["ticks"].astype(float)
        f["tick_count_ratio"] = tk / tk.rolling(20, min_periods=20).mean()
    out = pd.DataFrame(f, index=c.index)
    return out.replace([np.inf, -np.inf], np.nan)


CANDLE_FEATURES = None  # filled lazily


def candle_feature_names(tf: str) -> list[str]:
    global CANDLE_FEATURES
    if CANDLE_FEATURES is None:
        dummy = pd.DataFrame({"open": np.ones(60), "high": np.ones(60) * 1.01, "low": np.ones(60) * 0.99,
                              "close": np.ones(60), "ticks": np.ones(60)})
        CANDLE_FEATURES = list(candle_indicators(dummy, 1).columns) + ["age"]
    return [f"{tf}_{c}" for c in CANDLE_FEATURES]


def candle_features_at(epochs_eval: np.ndarray, candles: pd.DataFrame, tf: str, direction: int) -> pd.DataFrame:
    """Join closed-candle indicators onto evaluation epochs (backward as-of on candle close time)."""
    names = candle_feature_names(tf)
    if len(candles) == 0:
        return pd.DataFrame(np.nan, index=range(len(epochs_eval)), columns=names)
    ind = candle_indicators(candles.reset_index(drop=True), direction)
    ind.columns = [f"{tf}_{c}" for c in ind.columns]
    ind["avail"] = candles["avail"].to_numpy().astype(np.int64)
    ind = ind.sort_values("avail")
    left = pd.DataFrame({"epoch": np.asarray(epochs_eval, dtype=np.int64), "_o": np.arange(len(epochs_eval))})
    m = pd.merge_asof(left.sort_values("epoch"), ind, left_on="epoch", right_on="avail", direction="backward",
                      allow_exact_matches=True)
    tf_s = TIMEFRAMES[tf]
    m[f"{tf}_age"] = (m["epoch"] - m["avail"]) / tf_s
    m = m.sort_values("_o")
    return m[names].reset_index(drop=True)


def build_features(epochs: np.ndarray, quotes: np.ndarray, direction: int, sigma: float, threshold: float,
                   timeframes: list[str], idx: np.ndarray | None = None) -> pd.DataFrame:
    idx = np.arange(len(epochs)) if idx is None else np.asarray(idx, dtype=np.int64)
    parts = [tick_features(epochs, quotes, direction, sigma, threshold, idx)]
    ev = epochs[idx]
    for tf in timeframes:
        if tf == "tick":
            continue
        c = build_candles(epochs, quotes, TIMEFRAMES[tf])
        parts.append(candle_features_at(ev, c, tf, direction))
    df = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    df.insert(0, "epoch", ev)
    df.insert(0, "idx", idx)
    return df


def feature_columns(timeframes: list[str]) -> list[str]:
    cols = tick_feature_columns()
    for tf in timeframes:
        if tf != "tick":
            cols += candle_feature_names(tf)
    return cols


# Family -> list of (feature-suffix, timeframe-scoped?) used by the Strategy Arena and reason texts
STRATEGY_FAMILIES: dict[str, list[str]] = {
    "bollinger_squeeze": ["bb_squeeze_rank", "bb_width"],
    "atr_compression": ["atr_ratio", "atr_pct"],
    "rsi": ["rsi"],
    "stoch_rsi": ["stochrsi"],
    "macd": ["macd_hist"],
    "adx": ["adx", "di_diff"],
    "keltner": ["keltner_pos", "bb_in_kc"],
    "donchian": ["donchian_pos", "donchian_width"],
    "ema": ["ema50_dist", "ema_fast_slow"],
    "momentum": ["roc10"],
    "price_action": ["consec_drift", "body_ratio", "lower_wick", "upper_wick"],
    "candle_compression": ["range_ratio", "compression_5"],
    "liquidity_sweep": ["liq_sweep", "liq_sweep_recent"],
    "ict": ["ict_range_pos", "ict_displacement"],
    "crt": ["crt"],
    "fair_value_gap": ["fvg_bull_20", "fvg_bear_20"],
    "market_structure": ["structure", "bos_drift"],
    "volatility_regime": ["vol_regime"],
    # tick-level families
    "tick_velocity": ["t_ret_10", "t_ret_50", "t_ret_200"],
    "tick_acceleration": ["t_accel_10", "t_accel_50"],
    "statistical_anomaly": ["t_z_last", "t_absmax_50", "t_compress_10_100", "t_compress_50_500"],
    "spike_interval": ["t_ticks_since_spike", "t_spikes_2000"],
    "entropy": ["t_entropy_50", "t_entropy_200"],
    "hazard_rate": ["hazard"],  # computed from training data (empirical hazard of ticks-since-spike)
}
TICK_FAMILIES = {"tick_velocity", "tick_acceleration", "statistical_anomaly", "spike_interval", "entropy",
                 "hazard_rate"}
