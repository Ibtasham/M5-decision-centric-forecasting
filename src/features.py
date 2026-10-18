"""
features.py
===========
Feature construction for the machine-learning forecasters.

Two flavours are produced from the same code path:

1. `build_train_frame`  -- a long panel over the fitting window, used to train a
   global LightGBM model (one model for all series, the standard M5 recipe).
2. `build_origin_frame` -- features evaluated at a single forecast origin t0 for
   horizons h = 1..H, used both to train (recursively over past origins) and to
   predict.

All features at origin t0 with horizon h use information available strictly
before t0 (lags) or calendar information known in advance (events, SNAP,
weekday, price). No target leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Lag / window features that do not depend on the horizon
LAGS = (7, 14, 28, 56, 91, 182, 364)
ROLLS = ((7, 7), (14, 7), (28, 14), (56, 28), (91, 28), (182, 56), (364, 91))


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float32), where=b > 0)


def series_static_features(Y: np.ndarray, t0: int) -> dict[str, np.ndarray]:
    """Features summarising the history Y[:, :t0] for each series."""
    hist = Y[:, :t0].astype(np.float32)
    n = hist.shape[0]
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = hist > 0
        cnt = nz.sum(1).astype(np.float32)
        mu = hist.mean(1)
        sd = hist.std(1)
        out["s_mean"] = mu
        out["s_sd"] = sd
        out["s_cv2"] = _safe_div(sd ** 2, mu ** 2)
        out["s_zero_share"] = 1.0 - cnt / max(hist.shape[1], 1)
        out["s_max"] = hist.max(1)
        out["s_p90"] = np.percentile(hist, 90, axis=1) if n else hist.max(1)
        # ADI on the trailing 364 days (the practical operational definition)
        w = min(364, hist.shape[1])
        tail = hist[:, -w:]
        tnz = tail > 0
        tc = tnz.sum(1).astype(np.float32)
        out["s_adi"] = np.where(tc > 1, w / np.maximum(tc, 1e-6), np.nan)
        # demand-size statistics (non-zero observations only); guard series whose
        # trailing window contains no demand at all
        sizes = np.where(tnz, tail, np.nan)
        has_any = tnz.any(1)
        sz_mean = np.zeros(n, dtype=np.float32)
        sz_sd = np.zeros(n, dtype=np.float32)
        if has_any.any():
            sub = sizes[has_any]
            with np.errstate(invalid="ignore"):
                sz_mean[has_any] = np.nanmean(sub, 1)
                sz_sd[has_any] = np.nanstd(sub, 1)
        out["s_size_mean"] = sz_mean
        out["s_size_cv2"] = _safe_div(sz_sd ** 2, sz_mean ** 2)
        # trend: recent vs older mean
        r = min(56, hist.shape[1]); o = min(182, hist.shape[1])
        recent = hist[:, -r:].mean(1)
        older = hist[:, -o:-r].mean(1) if o > r else recent
        out["s_trend_ratio"] = _safe_div(recent, older)
        out["s_recent_mean56"] = recent
        # volatility ratio: dispersion of the recent window vs the longer history
        long_sd = hist[:, -o:].std(1)
        out["s_vol_ratio"] = _safe_div(hist[:, -r:].std(1), long_sd)
    return {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for k, v in out.items()}


def build_origin_frame(Y: np.ndarray, t0: int, H: int,
                       cal: pd.DataFrame, price_at: np.ndarray,
                       static: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """
    Long panel with one row per (series, horizon) for a single origin t0.

    Y          : (n_series, n_days) uint16 demand panel
    t0         : forecast origin; predictions cover t0 .. t0+H-1 (0-based day idx)
    H          : horizon length in days
    cal        : calendar DataFrame indexed by day position t
    price_at   : (n_series, H) float32 sell price for the horizon days
    static     : precomputed static features (optional; computed if None)
    """
    n = Y.shape[0]
    if static is None:
        static = series_static_features(Y, t0)

    # ---- lag / rolling features (constant across h for a given origin) -------
    hist = Y[:, :t0].astype(np.float32)
    lag_feats = {}
    for L in LAGS:
        lag_feats[f"lag_{L}"] = hist[:, t0 - L] if t0 - L >= 0 else np.zeros(n, np.float32)
    for (w, step) in ROLLS:
        s = max(t0 - w, 0)
        blk = hist[:, s:t0]
        lag_feats[f"rmean_{w}"] = blk.mean(1)
        lag_feats[f"rsd_{w}"] = blk.std(1)
        lag_feats[f"rmax_{w}"] = blk.max(1) if blk.size else np.zeros(n, np.float32)
        nz = (blk > 0).sum(1).astype(np.float32)
        lag_feats[f"rnz_{w}"] = nz / max(blk.shape[1], 1)
        if step <= w // 2:
            half = blk.shape[1] // 2
            if half > 0:
                lag_feats[f"rtrend_{w}"] = _safe_div(blk[:, half:].mean(1),
                                                     blk[:, :half].mean(1))
    # weekday-of-history mean: mean demand on the same weekday over last 8 weeks
    dow_hist = np.zeros((n, 7), dtype=np.float32)
    w8 = min(56, t0)
    if w8 > 0:
        blk = hist[:, t0 - w8:t0]
        dows = cal["wday"].to_numpy()[t0 - w8:t0] - 1
        for d in range(7):
            sel = dows == d
            if sel.any():
                dow_hist[:, d] = blk[:, sel].mean(1)

    # ---- assemble the (n_series x H) long frame -----------------------------
    h_idx = np.arange(H, dtype=np.int32)
    cal_slice = cal.iloc[t0:t0 + H]
    assert len(cal_slice) == H, f"calendar too short for origin {t0} + H {H}"

    def bcast(arr1d):
        """(n,) -> (n*H,) by repeating each series value across its H horizons."""
        return np.broadcast_to(np.asarray(arr1d, dtype=np.float32)[:, None],
                               (n, H)).ravel().copy()

    def bcast_day(arrH):
        """(H,) -> (n*H,) by tiling the same day pattern for every series."""
        return np.broadcast_to(np.asarray(arrH, dtype=np.float32)[None, :],
                               (n, H)).ravel().copy()

    parts: list[tuple[str, np.ndarray]] = []
    for name, arr in static.items():
        parts.append((name, bcast(arr)))
    for name, arr in lag_feats.items():
        parts.append((name, bcast(arr)))

    # horizon-dependent features
    parts.append(("h", bcast_day(h_idx + 1)))
    parts.append(("sell_price", price_at.ravel().copy()))
    parts.append(("log_price", np.log1p(price_at).ravel().copy()))

    dow = (cal_slice["wday"].to_numpy() - 1).astype(np.int64)
    parts.append(("dow_mean", dow_hist[:, dow].ravel().copy()))

    for c in ["wday", "month", "snap_CA", "snap_TX", "snap_WI"]:
        parts.append((c, bcast_day(cal_slice[c].to_numpy())))
    for c in ["event_type_1", "event_type_2"]:
        codes = pd.Categorical(cal_slice[c].astype(str).to_numpy()).codes.astype(np.float32)
        parts.append((c, bcast_day(codes)))
    parts.append(("is_event",
                  bcast_day((cal_slice["event_name_1"].astype(str) != "None")
                            .to_numpy().astype(np.float32))))

    feat_names = [nm for nm, _ in parts]
    X = np.column_stack([a for _, a in parts]).astype(np.float32)
    del parts

    df = pd.DataFrame(X, columns=feat_names)
    df.insert(0, "series_ix", np.repeat(np.arange(n, dtype=np.int32), H))
    df.insert(1, "t", np.tile(np.arange(t0, t0 + H, dtype=np.int32), n))
    df.insert(2, "horizon", np.tile(h_idx + 1, n))
    return df, feat_names


def iter_train_origins(Y: np.ndarray, origins: list[int], H: int,
                       cal: pd.DataFrame, prices: np.ndarray,
                       price_weeks: np.ndarray):
    """
    Yield (X, y, fnames) for one training origin at a time.

    Streaming the origins instead of stacking them keeps peak memory bounded by
    a single origin's design matrix, which matters because the sandbox has a
    1 GB cap and a full panel would need several hundred MB just for X.
    """
    for t0 in origins:
        price_at = price_window(prices, price_weeks, cal, t0, H)
        # static features must be derived from the SAME Y as the design matrix
        df, fnames = build_origin_frame(Y, t0, H, cal, price_at)
        tcol = df["t"].to_numpy()
        six = df["series_ix"].to_numpy()
        y = Y[six, tcol].astype(np.float32)
        X = df[fnames].to_numpy(dtype=np.float32)
        del df
        yield X, y, fnames


def build_train_panel(Y: np.ndarray, origins: list[int], H: int,
                      cal: pd.DataFrame, prices: np.ndarray,
                      price_weeks: np.ndarray):
    """Stack the per-origin design matrices (memory-hungry; prefer the iterator)."""
    Xs, ys = [], []
    fnames = None
    for X, y, fn in iter_train_origins(Y, origins, H, cal, prices, price_weeks):
        Xs.append(X); ys.append(y); fnames = fn
    return np.vstack(Xs), np.concatenate(ys), None, fnames


def price_window(prices: np.ndarray, price_weeks: np.ndarray, cal: pd.DataFrame,
                 t0: int, H: int) -> np.ndarray:
    """(n_series, H) sell price for days t0..t0+H-1, forward-filled."""
    wks = cal["wm_yr_wk"].to_numpy()[t0:t0 + H]
    wk_ix = {int(w): i for i, w in enumerate(price_weeks)}
    cols = np.array([wk_ix.get(int(w), -1) for w in wks], dtype=np.int64)
    P = prices
    if (cols < 0).any():
        # nearest earlier week
        good = np.where(cols >= 0, cols, 0)
        out = P[:, good]
        return np.ascontiguousarray(out, dtype=np.float32)
    out = P[:, cols]
    # forward-fill missing prices per series along the horizon
    if np.isnan(out).any():
        last = np.where(np.isnan(out[:, :1]), np.nanmedian(P, axis=1)[:, None], out[:, :1])
        for j in range(out.shape[1]):
            m = np.isnan(out[:, j])
            if m.any():
                out[m, j] = last[m, 0]
            last = out[:, j:j + 1]
    return np.ascontiguousarray(out, dtype=np.float32)
