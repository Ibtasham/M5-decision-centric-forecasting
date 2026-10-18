"""
common.py
=========
Data loading, series sampling, and shared helpers for the experiment driver.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paths import PROC, RESULTS


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_all():
    Y = np.load(PROC / "Y.npy")
    meta = pd.read_parquet(PROC / "meta.parquet")
    cal = pd.read_parquet(PROC / "calendar.parquet")
    prices = np.load(PROC / "P.npy")
    pw = np.array(json.load(open(PROC / "price_weeks.json")), dtype=np.int64)
    splits = json.load(open(PROC / "splits.json"))
    return Y, meta, cal, prices, pw, splits


def stratified_sample(meta: pd.DataFrame, n: int, seed: int = 0,
                      strata_cols=("sbc_class",), drop_dead=True) -> np.ndarray:
    """
    Proportional stratified sample of series indices.

    Proportional (rather than balanced) sampling keeps the demand-pattern mix of
    the full M5 panel, so aggregate cost conclusions transfer. We additionally
    report every result broken out by stratum, which is where the interesting
    heterogeneity lives.
    """
    m = meta.copy()
    if drop_dead:
        m = m[m["total_units"] > 0]
    rng = np.random.default_rng(seed)
    keys = m[strata_cols[0]].astype(str)
    for c in strata_cols[1:]:
        keys = keys + "|" + m[c].astype(str)
    picked = []
    total = len(keys)
    for _, idx in keys.groupby(keys).groups.items():
        idx = np.asarray(sorted(idx))
        k = max(1, int(round(n * len(idx) / total)))
        picked.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    out = np.sort(np.concatenate(picked).astype(np.int64))
    if len(out) > n:                     # trim proportionally from the largest strata
        keep = rng.choice(len(out), size=n, replace=False)
        out = np.sort(out[keep])
    return out


# --------------------------------------------------------------------------
# look-ahead-free demand-pattern descriptors
# --------------------------------------------------------------------------
def history_stats(Y: np.ndarray, t: int) -> dict:
    """
    Demand-pattern descriptors computed from Y[:, :t] ONLY (information available
    when the decision is taken at day t). Mirrors the definitions used for the
    descriptive full-panel statistics in 01_prepare_data.py, but restricted to the
    history: the series' active span starts at its first sale; ADI is the mean
    interval between non-zero days; CV^2 is the squared coefficient of variation
    of daily demand over the active span.
    """
    hist = Y[:, :t].astype(np.float32)
    n = hist.shape[0]
    nz = hist > 0
    cnt = nz.sum(1)
    has = cnt > 0
    first = np.where(has, nz.argmax(1), t)
    last = np.where(has, t - 1 - nz[:, ::-1].argmax(1), -1)
    span = np.maximum(t - first, 1).astype(np.float64)
    tot = hist.sum(1).astype(np.float64)
    mean = tot / span
    sumsq = (hist.astype(np.float64) ** 2).sum(1)
    var = np.maximum(sumsq / span - mean ** 2, 0.0)
    sd = np.sqrt(var)
    with np.errstate(invalid="ignore", divide="ignore"):
        adi = np.where(cnt > 1, (last - first) / np.maximum(cnt - 1, 1), np.nan)
        cv2 = np.where(mean > 0, var / np.maximum(mean, 1e-12) ** 2, np.nan)
    return dict(has_history=has, first_sale_t=first.astype(np.float32),
                n_active_days=span.astype(np.float32), total_units=tot.astype(np.float32),
                mean_daily=mean.astype(np.float32), sd_daily=sd.astype(np.float32),
                adi=adi.astype(np.float32), cv2=cv2.astype(np.float32),
                zero_share=(1.0 - cnt / span).astype(np.float32),
                max_daily=hist.max(1).astype(np.float32))


def groups_at(Y: np.ndarray, t: int, min_pool: int = 30) -> np.ndarray:
    """
    Calibration pools at origin t, from history only: SBC class (ADI/CV^2 over the
    history) x scale tercile (mean daily demand over the history). Series with no
    sale before t borrow the pool of low-volume lumpy items; pools with fewer than
    `min_pool` series are merged into the lumpy pool of the same scale. Replaces the full-panel pooling that
    used descriptors computed over validation and test days as well.
    """
    st = history_stats(Y, t)
    a = np.nan_to_num(st["adi"], nan=999.0)
    c = np.nan_to_num(st["cv2"], nan=0.0)
    cls = np.select([(a < 1.32) & (c < 0.49), (a < 1.32) & (c >= 0.49),
                     (a >= 1.32) & (c < 0.49)], [0, 1, 2], default=3)
    has = st["has_history"]
    scale = np.zeros(len(cls), dtype=np.int64)
    if has.sum() >= 3:
        mu = st["mean_daily"][has]
        rank = pd.Series(mu).rank(method="first").to_numpy()
        scale[has] = np.minimum((3 * (rank - 1) / len(mu)).astype(np.int64), 2)
    g = cls * 3 + scale
    # no sale yet: pool with the most typical new item (lumpy, low volume)
    g = np.where(has, g, 3 * 3 + 0)
    # merge thin pools (< min_pool series) into the lumpy pool of the same scale
    for code in np.unique(g):
        if (g == code).sum() < min_pool:
            g[g == code] = 3 * 3 + (code % 3)
    return pd.Categorical(g).codes.astype(np.int32)


# --------------------------------------------------------------------------
# prices
# --------------------------------------------------------------------------
def week_index(pw: np.ndarray, cal: pd.DataFrame):
    """Map day index t -> column of the price matrix."""
    wk_ix = {int(w): i for i, w in enumerate(pw)}
    wks = cal["wm_yr_wk"].to_numpy()
    cols = np.array([wk_ix.get(int(w), -1) for w in wks], dtype=np.int64)
    # forward-fill any week not present in the price file
    last = 0
    for i in range(len(cols)):
        if cols[i] < 0:
            cols[i] = last
        else:
            last = cols[i]
    return cols


def sell_price_at(prices: np.ndarray, wcol: np.ndarray, t: int) -> np.ndarray:
    """(n,) current sell price, with NaNs filled by the series median."""
    p = prices[:, wcol[t]].astype(np.float64)
    if np.isnan(p).any():
        med = np.nanmedian(prices, axis=1)
        med = np.where(np.isnan(med), 3.47, med)
        p = np.where(np.isnan(p), med, p)
    return np.maximum(p, 0.01)


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def rss_mb() -> float:
    """Resident set size of this process in MB (read from /proc/self/statm)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * 4096 / 1e6
    except Exception:
        return float("nan")


def mem_avail_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return float(line.split()[1]) / 1024
    except Exception:
        pass
    return float("nan")


def save_json(obj, name: str, subdir: str = ""):
    d = Path(RESULTS) / subdir if subdir else Path(RESULTS)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / name, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return d / name


def save_df(df: pd.DataFrame, name: str, subdir: str = ""):
    d = Path(RESULTS) / subdir if subdir else Path(RESULTS)
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    (df.to_csv(p, index=False) if name.endswith(".csv") else df.to_parquet(p, index=False))
    return p


def naive_scale(Y: np.ndarray, t0: int, window: int = 364):
    """
    M5-style scaling factors from the 1-step seasonal-naive error on the trailing
    `window` days. Returns (sq_scale, abs_scale) per series.

    Intermittent series can be *exactly* flat over a long stretch, which drives
    the naive scale to zero and makes any scaled error explode (this is why
    bottom-level M5 RMSSE values are often reported in the tens). We therefore
    return the raw scales and let the caller mask degenerate series rather than
    silently flooring them, and we always report the share of series excluded.
    """
    hist = Y[:, max(t0 - window, 1):t0].astype(np.float64)
    d = np.diff(hist, axis=1)
    return (d ** 2).mean(1), np.abs(d).mean(1)


FLAT_TOL = 1e-9


def scaled_errors(actual: np.ndarray, pred: np.ndarray,
                  sq_scale: np.ndarray, abs_scale: np.ndarray) -> dict:
    """
    RMSSE and MASE with degenerate (flat-history) series excluded rather than
    floored. `actual`/`pred` are (n, H).
    """
    ok_sq = sq_scale > FLAT_TOL
    ok_ab = abs_scale > FLAT_TOL
    num_sq = np.mean((actual - pred) ** 2, axis=1)
    num_ab = np.mean(np.abs(actual - pred), axis=1)
    rmsse = float(np.mean(np.sqrt(num_sq[ok_sq] / sq_scale[ok_sq]))) if ok_sq.any() else np.nan
    mase = float(np.mean(num_ab[ok_ab] / abs_scale[ok_ab])) if ok_ab.any() else np.nan
    return dict(rmsse=rmsse, mase=mase,
                pct_scaled_sq=float(ok_sq.mean()), pct_scaled_ab=float(ok_ab.mean()),
                n_valid_sq=int(ok_sq.sum()), n_valid_ab=int(ok_ab.sum()))


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Weighted absolute percentage error -- scale-free and never degenerate."""
    d = float(np.abs(actual).sum())
    return float(np.abs(actual - pred).sum() / d) if d > 0 else np.nan
