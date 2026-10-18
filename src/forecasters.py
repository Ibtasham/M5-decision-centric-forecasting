"""
forecasters.py
==============
The forecaster suite compared in the study.

Unified interface (every class implements both methods):

    fit(Y, t0, ctx)             -> self        # may be a no-op for closed-form methods
    predict(Y, t0, H, ctx)      -> np.ndarray

  * point forecasters        return shape (n_series, H)
  * distributional forecasters return shape (n_tau, n_series, H), tau ascending

`ctx` is a dict with keys: cal, prices, price_weeks, groups, meta.

Members
-------
Statistical / intermittent-demand arm
  naive    last observed value, carried flat over the horizon
  snaive   mean of the same weekday over the trailing 4 weeks
  ma28     28-day trailing mean, shaped by the weekly seasonality profile
  croston  Croston (1972)
  sba      Syntetos-Boylan Approximation (Syntetos & Boylan, 2005)
  tsb      Teunter-Syntetos-Babai (2011)
  hw       damped-trend Holt-Winters, weekly seasonality, hyper-parameters
           shared across series and selected by grid search on a subsample
  comb     equal-weight combination of snaive/ma28/sba/tsb -- the family of
           approaches that dominated the M5 accuracy task

Machine-learning arm
  lgbm_mean  global LightGBM, squared-loss objective (point forecast)
  lgbm_q     global LightGBM trained once per quantile tau (distributional)

Calibration arm (used for the cost-gap decomposition)
  cal_emp    the point forecast's level combined with the empirical demand
             distribution pooled within demand-pattern groups. Distributional
             decisions without distributional training -- what a practitioner
             would actually do.
"""
from __future__ import annotations

import gc
import itertools
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import features as FEAT

TAUS = np.array([0.50, 0.70, 0.80, 0.90, 0.95, 0.99], dtype=np.float32)
M_SEASON = 7


# ==========================================================================
# helpers
# ==========================================================================
def weekday_profile(hist: np.ndarray, t0: int, cal: pd.DataFrame, weeks: int = 12):
    """(n, 7) mean demand by weekday over the trailing `weeks` weeks."""
    n, T = hist.shape
    w = min(weeks * 7, T)
    blk = hist[:, T - w:]
    dows = (cal["wday"].to_numpy()[t0 - w:t0] - 1).astype(np.int64)
    prof = np.zeros((n, 7), dtype=np.float32)
    seen = np.zeros(7, dtype=bool)
    for d in range(7):
        sel = dows == d
        if sel.any():
            prof[:, d] = blk[:, sel].mean(1)
            seen[d] = True
    if (~seen).any():
        gm = blk.mean(1, keepdims=True)
        prof[:, ~seen] = np.broadcast_to(gm, (n, int((~seen).sum())))
    return np.maximum(prof, 0.0)


def seasonality_shape(hist, t0, cal, weeks=12):
    """(n, 7) multiplicative weekday factors normalised to mean 1."""
    prof = weekday_profile(hist, t0, cal, weeks)
    return prof / np.maximum(prof.mean(1, keepdims=True), 1e-6)


def horizon_dow(cal, t0, H):
    return (cal["wday"].to_numpy()[t0:t0 + H] - 1).astype(np.int64)


# ==========================================================================
# intermittent-demand baselines (Croston / SBA / TSB)
# ==========================================================================
def croston_family(Y: np.ndarray, t0: int, H: int, cal: pd.DataFrame,
                   method: str = "croston", alpha: float = 0.10) -> np.ndarray:
    """Vectorised (over series) Croston / SBA / TSB, shaped by weekday factors."""
    hist = Y[:, :t0].astype(np.float32)
    n, T = hist.shape
    nz = hist > 0

    w0 = min(28, T)
    p = np.full(n, float(w0), dtype=np.float32)     # mean inter-demand interval
    zz = hist[:, :w0].mean(1).astype(np.float32)    # mean demand size
    prob = nz[:, :w0].mean(1).astype(np.float32)    # TSB demand probability
    first = np.flatnonzero(nz[:, :w0].any(1))
    if first.size:
        cnt = nz[first, :w0].sum(1).astype(np.float32)
        p[first] = np.where(cnt > 0, nz[first, :w0].argmax(1) + 1, w0).astype(np.float32)
        tot = hist[first, :w0].sum(1)
        zz[first] = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
        prob[first] = cnt / w0

    since = np.zeros(n, dtype=np.float32)
    a = np.float32(alpha)
    for t in range(w0, T):
        d = hist[:, t]
        since += 1.0
        hit = np.flatnonzero(d > 0)
        if hit.size:
            p[hit] += a * (since[hit] - p[hit])
            zz[hit] += a * (d[hit] - zz[hit])
            since[hit] = 0.0
        if method == "tsb":
            prob += a * ((d > 0).astype(np.float32) - prob)

    if method == "sba":
        rate = (1.0 - a / 2.0) * zz / np.maximum(p, 1e-6)
    elif method == "tsb":
        rate = prob * zz
    else:
        rate = zz / np.maximum(p, 1e-6)
    rate = np.maximum(np.nan_to_num(rate, nan=0.0, posinf=0.0), 0.0)

    shape = seasonality_shape(hist, t0, cal)
    out = rate[:, None] * shape[:, horizon_dow(cal, t0, H)]
    return np.ascontiguousarray(np.maximum(out, 0.0), dtype=np.float32)


class StatForecaster:
    """Closed-form statistical baselines."""

    family = "statistical"
    kind = "point"

    def __init__(self, name: str):
        self.name = name

    def fit(self, Y, t0, ctx):
        return self

    def predict(self, Y, t0, H, ctx):
        cal = ctx["cal"]
        hist = Y[:, :t0].astype(np.float32)
        n = hist.shape[0]
        dow = horizon_dow(cal, t0, H)

        if self.name == "naive":
            return np.repeat(hist[:, -1:], H, axis=1).astype(np.float32)

        if self.name == "ma28":
            w = min(28, hist.shape[1])
            v = hist[:, -w:].mean(1)
            shape = seasonality_shape(hist, t0, cal)
            return np.maximum(v[:, None] * shape[:, dow], 0).astype(np.float32)

        if self.name == "snaive":
            w = min(28, hist.shape[1])
            blk = hist[:, -w:]
            hist_dow = (cal["wday"].to_numpy()[t0 - w:t0] - 1).astype(np.int64)
            out = np.zeros((n, H), dtype=np.float32)
            gm = blk.mean(1)
            for h in range(H):
                m = hist_dow == dow[h]
                out[:, h] = blk[:, m].mean(1) if m.any() else gm
            return np.maximum(out, 0)

        if self.name in ("croston", "sba", "tsb"):
            return croston_family(Y, t0, H, cal, method=self.name)

        raise ValueError(f"unknown statistical forecaster {self.name}")


# ==========================================================================
# damped-trend Holt-Winters with weekly seasonality
# ==========================================================================
HW_GRID = dict(alpha=[0.05, 0.15, 0.35], beta=[0.0, 0.03],
               gamma=[0.0, 0.05, 0.15], phi=[0.90, 0.98],
               seasonal=["additive"])


def hw_recursion(y, alpha, beta, gamma, phi, m=M_SEASON, seasonal="additive"):
    """Vectorised over series. y is (n, T). Returns (level, trend, seas)."""
    n, T = y.shape
    seas = np.zeros((n, T + m), dtype=np.float32)
    seed = y[:, :m].mean(1, keepdims=True)
    if seasonal == "additive":
        seas[:, :m] = y[:, :m] - seed
    else:
        seas[:, :m] = y[:, :m] / np.maximum(seed, 1e-6)
    level = y[:, :m].mean(1).astype(np.float32)
    trend = np.zeros(n, dtype=np.float32)
    if T > 2 * m:
        trend = ((y[:, m:2 * m].mean(1) - y[:, :m].mean(1)) / m).astype(np.float32)

    for t in range(T):
        st = seas[:, t]
        obs = y[:, t]
        if seasonal == "additive":
            new_level = alpha * (obs - st) + (1 - alpha) * (level + phi * trend)
            seas[:, t + m] = gamma * (obs - new_level) + (1 - gamma) * st
        else:
            new_level = alpha * (obs / np.maximum(st, 1e-6)) + (1 - alpha) * (level + phi * trend)
            seas[:, t + m] = gamma * (obs / np.maximum(level + phi * trend, 1e-6)) + (1 - gamma) * st
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    return level, trend, seas


def hw_forecast(level, trend, seas, T, H, phi, m=M_SEASON, seasonal="additive"):
    out = np.empty((level.shape[0], H), dtype=np.float32)
    cum = 0.0
    for h in range(1, H + 1):
        cum += phi ** h
        st = seas[:, T + ((h - 1) % m)]
        base = level + cum * trend
        out[:, h - 1] = base + st if seasonal == "additive" else base * np.maximum(st, 1e-6)
    return np.maximum(out, 0.0)


class HWForecaster:
    """Damped Holt-Winters with hyper-parameters shared across series, selected
    by grid search on a subsample (keeps the statistical arm tractable at scale)."""

    name = "hw"
    family = "statistical"
    kind = "point"

    def __init__(self, tune_series=600, window=448, val_len=28, seed=0):
        self.tune_series = tune_series
        self.window = window
        self.val_len = val_len
        self.seed = seed
        self.params = None

    @staticmethod
    def _grid():
        keys = list(HW_GRID)
        return [dict(zip(keys, v)) for v in itertools.product(*[HW_GRID[k] for k in keys])]

    def fit(self, Y, t0, ctx):
        n = Y.shape[0]
        rng = np.random.default_rng(self.seed)
        sub = rng.choice(n, size=min(self.tune_series, n), replace=False)
        need = self.window + self.val_len
        W = min(need, t0)
        y = Y[sub, t0 - W:t0].astype(np.float32)
        ytr, yva = y[:, :-self.val_len], y[:, -self.val_len:]
        best, bp = np.inf, self._grid()[0]
        for prm in self._grid():
            try:
                lev, trd, seas = hw_recursion(ytr, prm["alpha"], prm["beta"],
                                              prm["gamma"], prm["phi"],
                                              seasonal=prm["seasonal"])
                f = hw_forecast(lev, trd, seas, ytr.shape[1], self.val_len,
                                prm["phi"], seasonal=prm["seasonal"])
                err = float(np.sqrt(np.mean((f - yva) ** 2)))
            except Exception:
                err = np.inf
            if err < best:
                best, bp = err, prm
        self.params, self.tune_rmse = bp, best
        return self

    def predict(self, Y, t0, H, ctx):
        prm = self.params
        W = min(self.window, t0)
        y = Y[:, max(t0 - W, 0):t0].astype(np.float32)
        lev, trd, seas = hw_recursion(y, prm["alpha"], prm["beta"], prm["gamma"],
                                      prm["phi"], seasonal=prm["seasonal"])
        return hw_forecast(lev, trd, seas, y.shape[1], H, prm["phi"],
                           seasonal=prm["seasonal"])


# ==========================================================================
# combination baseline
# ==========================================================================
class CombForecaster:
    """Equal-weight average of statistical baselines (M5-accuracy winning family)."""

    name = "comb"
    family = "statistical"
    kind = "point"
    COMPONENTS = ["snaive", "ma28", "sba", "tsb"]

    def __init__(self):
        self.parts = [StatForecaster(c) for c in self.COMPONENTS]

    def fit(self, Y, t0, ctx):
        return self

    def predict(self, Y, t0, H, ctx):
        acc = None
        for f in self.parts:
            p = f.predict(Y, t0, H, ctx)
            acc = p if acc is None else acc + p
        return (acc / len(self.parts)).astype(np.float32)


# ==========================================================================
# LightGBM
# ==========================================================================
@dataclass
class LGBSpec:
    objective: str = "regression"
    n_estimators: int = 350
    learning_rate: float = 0.06
    num_leaves: int = 63
    min_child_samples: int = 60
    subsample: float = 0.8
    colsample: float = 0.7
    reg_lambda: float = 1.0


def _lgb_params(spec, alpha=None):
    p = dict(n_estimators=spec.n_estimators, learning_rate=spec.learning_rate,
             num_leaves=spec.num_leaves, min_child_samples=spec.min_child_samples,
             subsample=spec.subsample, subsample_freq=1,
             colsample_bytree=spec.colsample, reg_lambda=spec.reg_lambda,
             n_jobs=2, verbosity=-1, random_state=0)
    if alpha is None:
        p["objective"] = spec.objective
    else:
        p["objective"] = "quantile"
        p["alpha"] = float(alpha)
    return p


class LGBMForecaster:
    """Global LightGBM model (one model for all series) -- the standard M5 recipe.

    quantiles=None  -> one squared-loss point model
    quantiles=[...] -> one model per tau, giving a full quantile surface
    """

    family = "ml"

    def __init__(self, spec: LGBSpec, quantiles=None, train_origins=None,
                 H=28, name="lgbm", max_train_rows=None):
        self.spec = spec
        self.quantiles = list(quantiles) if quantiles is not None else None
        self.H = H
        self.train_origins = train_origins
        self.max_train_rows = int(max_train_rows or 0)
        self.train_subset = None
        self.name = name if quantiles is None else f"{name}_q"
        self.kind = "point" if quantiles is None else "dist"
        self.models = {}
        self.fnames = []

    def fit(self, Y, t0, ctx):
        """
        Fit a single global model on the stacked multi-origin training panel --
        the standard M5 recipe, and exactly equivalent to what a practitioner
        would do.

        Memory is bounded by subsampling *series* (not rows) so that the panel
        stays a proper time-series design: if n_series x H x n_origins exceeds
        `max_train_rows`, a random subset of series is used for fitting while
        prediction still covers every series. Feature count is the other lever;
        the design matrix is float32 throughout.
        """
        import lightgbm as lgb
        origins = self.train_origins or [t0 - 4 * 28, t0 - 3 * 28, t0 - 2 * 28, t0 - 28]
        origins = [o for o in origins if o > 364]
        n = Y.shape[0]
        per_origin = n * self.H
        if self.max_train_rows and per_origin * len(origins) > self.max_train_rows:
            keep = max(self.max_train_rows // (self.H * len(origins)), 64)
            keep = min(keep, n)
            sel = np.sort(np.random.default_rng(0).choice(n, keep, replace=False))
            Yf = np.ascontiguousarray(Y[sel])
            log_msg = (f"panel subsampled to {keep}/{n} series "
                       f"({keep*self.H*len(origins)} rows)")
        else:
            Yf, sel, log_msg = Y, None, f"full panel ({per_origin*len(origins)} rows)"
        self.train_subset = sel

        t_start = time.time()
        # the price matrix must be subsetted in lock-step with the series, or the
        # design matrix silently mixes row spaces
        prices_f = (np.ascontiguousarray(ctx["prices"][sel]) if sel is not None
                    else ctx["prices"])
        X, y, _, fnames = FEAT.build_train_panel(
            Yf, origins, self.H, ctx["cal"], prices_f, ctx["price_weeks"])
        self.fnames = fnames
        self.n_train_rows = int(len(X))
        self.train_msg = log_msg
        targets = [None] if self.quantiles is None else list(self.quantiles)
        for tau in targets:
            key = "mean" if tau is None else f"q{tau:.2f}"
            m = lgb.LGBMRegressor(**_lgb_params(self.spec, tau))
            m.fit(X, y)
            self.models[key] = m
            del m
            gc.collect()
        self.fit_seconds = time.time() - t_start
        del X, y
        gc.collect()
        return self

    def _design(self, Y, t0, H, ctx):
        price_at = FEAT.price_window(ctx["prices"], ctx["price_weeks"], ctx["cal"], t0, H)
        df, fnames = FEAT.build_origin_frame(
            Y, t0, H, ctx["cal"], price_at,
            static=FEAT.series_static_features(Y, t0))
        X = df[self.fnames].to_numpy(dtype=np.float32)
        del df
        return X

    def predict(self, Y, t0, H, ctx):
        X = self._design(Y, t0, H, ctx)
        n = Y.shape[0]
        if self.quantiles is None:
            p = self.models["mean"].predict(X).reshape(n, H)
            del X
            return np.maximum(p, 0).astype(np.float32)
        out = np.empty((len(self.quantiles), n, H), dtype=np.float32)
        for i, tau in enumerate(self.quantiles):
            out[i] = np.maximum(self.models[f"q{tau:.2f}"].predict(X).reshape(n, H), 0)
        del X
        gc.collect()
        # independently trained quantiles can cross; enforce monotonicity in tau.
        # crossing_rate[i] = share of forecasts where tau_i is pulled up to
        # tau_{i-1} by the running maximum (i.e. the raw model for tau_i was lower)
        self.crossing_rate = [0.0] + [float((out[i] < np.maximum.accumulate(out[:i], axis=0)[-1]).mean())
                                      for i in range(1, len(self.quantiles))]
        return np.maximum.accumulate(out, axis=0)


# ==========================================================================
# empirical calibration of a point forecast
# ==========================================================================
def empirical_protection_quantiles(Y: np.ndarray, t0: int, protection: int,
                                   taus: np.ndarray, groups: np.ndarray,
                                   level_scale: np.ndarray | None = None,
                                   n_windows: int = 12) -> np.ndarray:
    """
    Empirical quantiles of *cumulative* demand over a protection interval of
    `protection` days, pooled within demand-pattern groups.

    This is the distributional object a replenishment decision actually needs:
    the order-up-to level must cover demand over (review period + lead time), and
    for lumpy retail demand that distribution is strongly right-skewed with a
    large mass at zero. A normal approximation of it is badly misspecified.

    Level anchoring. Group-pooled quantiles ignore that an individual SKU may be
    running well above or below its group. If `level_scale` is supplied (a
    per-series ratio of current forecast level to trailing historical level) the
    pooled quantiles are rescaled series-by-series:

        Q_i(tau) = level_scale_i * empq_g(tau)

    Returns (len(taus), n).
    """
    n = Y.shape[0]
    sums = []
    for k in range(n_windows):
        e = t0 - k * protection
        b = e - protection
        if b < 0:
            break
        sums.append(Y[:, b:e].astype(np.float64).sum(1))
    if not sums:
        sums = [Y[:, max(t0 - protection, 0):t0].astype(np.float64).sum(1)]
    W = np.stack(sums, axis=1)                       # (n, n_windows)

    Q = np.empty((len(taus), n), dtype=np.float64)
    for g in np.unique(groups):
        sel = groups == g
        vals = W[sel].ravel()
        q = np.quantile(vals, taus) if vals.size >= 8 else np.quantile(W.ravel(), taus)
        Q[:, sel] = q[:, None]
    if level_scale is not None:
        Q = Q * np.maximum(level_scale, 0.0)[None, :]
    Q = np.maximum.accumulate(np.maximum(Q, 0.0), axis=0)
    return Q.astype(np.float64)


def level_scale_from_forecast(Y: np.ndarray, t0: int, point_fc: np.ndarray,
                              lookback: int = 84, clip=(0.25, 4.0)) -> np.ndarray:
    """Per-series ratio (forecast level) / (trailing historical level), clipped."""
    w = min(lookback, t0)
    hist_mean = Y[:, t0 - w:t0].astype(np.float64).mean(1)
    fc_mean = np.maximum(point_fc.mean(1).astype(np.float64), 0.0)
    scale = np.divide(fc_mean, hist_mean, out=np.ones_like(fc_mean),
                      where=hist_mean > 1e-6)
    return np.clip(scale, clip[0], clip[1])


def empirical_quantile_surface(point_fc: np.ndarray, Y: np.ndarray, t0: int, H: int,
                               groups: np.ndarray, taus: np.ndarray,
                               lookback: int = 504, n_windows: int = 12) -> np.ndarray:
    """
    Daily-horizon quantile surface for a calibrated point forecaster.

    The level comes from the point forecast; the shape comes from the empirical
    distribution of daily demand pooled within demand-pattern groups. tau=0.5
    reproduces the point-forecast level and the other quantiles spread
    multiplicatively around it.

    This exists so that calibrated forecasters can be scored with the same
    probabilistic accuracy metrics as the ML quantile models. The *decision*
    path uses `empirical_protection_quantiles`, which is defined on the
    protection interval and is what actually drives the order-up-to level.

    Returns (len(taus), n, H) float32.
    """
    n = Y.shape[0]
    w = min(max(lookback, 28), max(t0, 28))
    hist = Y[:, t0 - w:t0].astype(np.float64)

    shape = np.ones((n, len(taus)), dtype=np.float64)
    for g in np.unique(groups):
        sel = groups == g
        if not sel.any():
            continue
        vals = hist[sel].ravel()
        pool = vals if vals.size >= 50 else hist.ravel()
        q = np.quantile(pool, taus)
        med = float(np.median(pool))
        denom = med if med > 1e-9 else max(float(pool.mean()), 1e-9)
        shape[sel] = (q / denom)[None, :]

    level = np.maximum(point_fc.mean(1, keepdims=True).astype(np.float64), 1e-6)
    Q = np.empty((len(taus), n, H), dtype=np.float32)
    for i in range(len(taus)):
        Q[i] = (level * shape[:, i:i + 1]).astype(np.float32)
    return np.maximum.accumulate(np.maximum(Q, 0.0), axis=0).astype(np.float32)


class CalibratedForecaster:
    """Wraps any point forecaster and emits a calibrated quantile surface."""

    family = "calibration"
    kind = "dist"

    def __init__(self, base, name="cal", lookback=504, n_windows=6, taus=None):
        self.base = base
        self.name = name
        self.lookback = lookback
        self.n_windows = n_windows
        self.taus = (np.asarray(taus, dtype=np.float32) if taus is not None
                     else TAUS.copy())

    def fit(self, Y, t0, ctx):
        self.base.fit(Y, t0, ctx)
        return self

    def predict(self, Y, t0, H, ctx):
        p = self.base.predict(Y, t0, H, ctx)      # base is always a point forecaster
        assert p.ndim == 2, "CalibratedForecaster requires a point-forecast base"
        return empirical_quantile_surface(p, Y, t0, H, ctx["groups"], self.taus,
                                          self.lookback, self.n_windows)


# ==========================================================================
# registry
# ==========================================================================
STAT_NAMES = ["naive", "snaive", "ma28", "croston", "sba", "tsb"]


def build_registry(H=28, train_origins=None, lgbm_estimators=350,
                   quantiles=None, include=("stat", "ml", "cal")):
    """name -> forecaster instance."""
    reg = {}
    if "stat" in include:
        for nm in STAT_NAMES:
            reg[nm] = StatForecaster(nm)
        reg["hw"] = HWForecaster()
        reg["comb"] = CombForecaster()
    if "ml" in include:
        reg["lgbm_mean"] = LGBMForecaster(LGBSpec(n_estimators=lgbm_estimators),
                                          None, train_origins, H, name="lgbm_mean")
        q = quantiles if quantiles is not None else TAUS.tolist()
        reg["lgbm_q"] = LGBMForecaster(LGBSpec(n_estimators=lgbm_estimators),
                                       q, train_origins, H, name="lgbm_q")
    if "cal" in include:
        tq = np.asarray(quantiles if quantiles is not None else TAUS, dtype=np.float32)
        reg["cal_comb"] = CalibratedForecaster(CombForecaster(), name="cal_comb",
                                               taus=tq)
    return reg
