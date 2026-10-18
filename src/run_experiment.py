"""
run_experiment.py
=================
Main driver for the study "Beyond Forecast Accuracy: Demand Forecasting That
Actually Improves Inventory Decisions".

Pipeline
--------
  stage 0  load the M5 panel, draw a stratified sample, derive the cost model
  stage 1  FORECAST  at every rolling origin of a block, produce point forecasts
           (statistical baselines + global LightGBM) and protection-interval
           quantile surfaces (LightGBM quantile regression with several
           aggregation rules, and empirically calibrated baselines);
           score accuracy (RMSSE, MASE, MAE, RMSE, bias, pinball)
  stage 2  REGIME    label every (series, period) with a demand-regime class
  stage 3  POLICY    run the lost-sales periodic-review simulator for each
           forecaster x policy x critical-ratio combination
  stage 4  SELECT    per-series pairing selection on validation (no test leakage)
  stage 5  persist tidy CSVs for the analysis notebook and the paper's tables

Protocol
--------
  TRAIN  t in [0, 1456)     forecaster fitting only
  VAL    t in [1456, 1824)  13 x 28-day periods -- used for all selection/tuning
  TEST   t in [1824, 1969)  5 x 28-day periods  -- reporting only

ML models are fitted once per block on data strictly before the block; features
still refresh at every origin. Per-series policy selection uses validation
outcomes only, so the test block carries no selection leakage.

Usage
-----
  python3 run_experiment.py --n_series 4000 --blocks val --out pilot
  python3 run_experiment.py --n_series 8000 --blocks val,test --out main
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS                                     # noqa: E402
import common as CM                                            # noqa: E402
import features as FEAT                                        # noqa: E402
import forecasters as FC                                       # noqa: E402
import simulation as SIM                                       # noqa: E402

H = 28                      # forecast horizon (days) = one replenishment cycle
REVIEW = 7                  # periodic review interval (days)
QUANTILES = [0.50, 0.75, 0.90, 0.95, 0.99, 0.995]
POINT_METHODS = ["naive", "snaive", "ma28", "croston", "sba", "tsb", "hw",
                 "comb", "lgbm_mean"]
DIST_METHODS = ["lgbm_q", "cal_comb", "cal_sba", "cal_lgbm"]
REVIEW_DAILY = 1
PERIOD = 28                 # period length used for the regime breakdown
ECON_FOR_REGIME = "margin"  # only this economic setting is broken out by regime


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_cache(cache, outdir, bname):
    """Persist the forecast cache so that policy-side sweeps (lead time, ordering
    cost, holding rate, review interval) do not have to refit any model."""
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / f"fcache_{bname}.npz",
        origins=np.array(cache["origins"]), t_start=cache["t_start"],
        t_end=cache["t_end"], protection=cache["protection"],
        point_methods=np.array(list(cache["point"].keys()), dtype=object),
        **{f"P_{k}": v for k, v in cache["point"].items()},
        dist_methods=np.array(list(cache["pi_q"].keys()), dtype=object),
        **{f"Q_{k}": v for k, v in cache["pi_q"].items()},
        G=cache["G"], Qd=cache["Qd"])
    pd.DataFrame(cache["acc"]).to_csv(d / f"acc_{bname}.csv", index=False)
    pd.DataFrame(cache["pinball"]).to_csv(d / f"pin_{bname}.csv", index=False)
    log(f"  cached forecasts -> {d}/fcache_{bname}.npz")


def load_cache(d, bname, n):
    f = Path(d) / f"fcache_{bname}.npz"
    if not f.exists():
        log(f"  no forecast cache at {f}; refitting")
        return None
    z = np.load(f, allow_pickle=True)
    cache = dict(origins=[int(x) for x in z["origins"]],
                 t_start=int(z["t_start"]), t_end=int(z["t_end"]),
                 protection=int(z["protection"]),
                 point={k: z[f"P_{k}"] for k in [str(x) for x in z["point_methods"]]},
                 pi_q={k: z[f"Q_{k}"] for k in [str(x) for x in z["dist_methods"]]},
                 G=z["G"], Qd=z["Qd"], acc=[], pinball=[])
    if cache["point"] and next(iter(cache["point"].values())).shape[1] != n:
        log(f"  cache was built for a different sample size; refitting")
        return None
    log(f"  reused forecast cache from {f} ({len(cache['origins'])} origins)")
    return cache


# ==========================================================================
# stage 0
# ==========================================================================
def setup(args):
    Y_all, meta_all, cal, prices_all, pw, splits = CM.load_all()
    ix = CM.stratified_sample(meta_all, args.n_series, seed=args.seed)
    Y = np.ascontiguousarray(Y_all[ix])
    prices = np.ascontiguousarray(prices_all[ix])
    meta = meta_all.iloc[ix].reset_index(drop=True)
    del Y_all, prices_all
    gc.collect()
    n = len(Y)
    wcol = CM.week_index(pw, cal)

    scale = pd.qcut(meta["mean_daily"].rank(method="first"), 3,
                    labels=["small", "mid", "large"])
    gstr = meta["sbc_class"].astype(str) + "|" + scale.astype(str)
    groups = pd.Categorical(gstr).codes.astype(np.int32)
    meta["group"] = groups
    meta["scale_band"] = scale.astype(str).to_numpy()

    # Look-ahead-free descriptors (history before the validation block only), used
    # as meta-classifier features. The full-panel columns (adi, cv2, ...) are kept
    # for descriptive stratification of results only.
    lf = CM.history_stats(Y, splits["val"][0])
    for k in ["first_sale_t", "n_active_days", "total_units", "mean_daily", "sd_daily",
              "adi", "cv2", "zero_share", "max_daily"]:
        meta["lf_" + k] = lf[k]
    meta["lf_p99_daily"] = np.percentile(Y[:, :splits["val"][0]], 99, axis=1).astype(np.float32)

    log(f"sample n={n} | SBC classes {meta.sbc_class.value_counts().to_dict()}")
    log(f"calibration pools: {len(np.unique(groups))}")
    ctx = dict(cal=cal, prices=prices, price_weeks=pw, groups=groups,
               meta=meta, wcol=wcol)
    return Y, meta, ctx, splits, n


# ==========================================================================
# stage 1  forecasting
# ==========================================================================
def quantile_pi_surface(Q_daily: np.ndarray, protection: int, mode: str,
                        lam: float = 1.0) -> np.ndarray:
    """
    (n_tau, n, H) daily quantiles -> (n_tau, n) protection-interval quantiles.

    Summing marginal quantiles is only exact under comonotonicity. Three rules:
      comonotonic  S(tau) = sum_h Q_h(tau)                     (upper bound)
      sqrt         S(tau) = m + (sum_h Q_h(tau) - m) * sqrt(P)/P
                   with m = sum_h Q_h(0.5), i.e. the variance-aggregation rule
      blend        linear interpolation between the two, weight lam
    """
    Qp = Q_daily[:, :, :protection]
    cs = np.cumsum(Qp, axis=2)[:, :, -1]                     # comonotonic
    if mode == "comonotonic":
        out = cs
    elif mode == "sqrt":
        m = np.cumsum(Qp[0][:, :protection], axis=1)[:, -1]   # median-path sum
        out = m[None, :] + (cs - m[None, :]) * (np.sqrt(protection) / protection)
    elif mode == "blend":
        m = np.cumsum(Qp[0][:, :protection], axis=1)[:, -1]
        sq = m[None, :] + (cs - m[None, :]) * (np.sqrt(protection) / protection)
        out = lam * cs + (1 - lam) * sq
    else:
        raise ValueError(mode)
    return np.maximum.accumulate(np.maximum(out, 0.0), axis=0).astype(np.float32)


def forecast_block(Y, ctx, block_name, block, n, args):
    t_start, t_end = block
    origins = list(range(t_start, t_end - H + 1, H))
    protection = args.review + args.lead
    log(f"[{block_name}] {len(origins)} origins {origins[0]}..{origins[-1]}  "
        f"protection interval {protection}d")

    train_origins = [o for o in (t_start - k * H for k in (4, 3, 2, 1)) if o > 364]
    spec = FC.LGBSpec(n_estimators=args.lgbm_estimators)
    lgbm_mean = FC.LGBMForecaster(spec, None, train_origins, H, name="lgbm_mean",
                                  max_train_rows=args.max_train_rows)
    lgbm_q = FC.LGBMForecaster(FC.LGBSpec(n_estimators=args.lgbm_estimators),
                               QUANTILES, train_origins, H, name="lgbm_q",
                               max_train_rows=args.max_train_rows)
    stat_objs = {nm: FC.StatForecaster(nm) for nm in
                 ["naive", "snaive", "ma28", "croston", "sba", "tsb"]}
    hw = FC.HWForecaster(tune_series=args.hw_tune, window=args.hw_window)
    comb = FC.CombForecaster()

    t0 = time.time()
    lgbm_mean.fit(Y, t_start, ctx)
    log(f"  fit lgbm_mean {lgbm_mean.fit_seconds:.0f}s ({lgbm_mean.n_train_rows} rows, "
        f"{len(lgbm_mean.fnames)} features; {lgbm_mean.train_msg})")
    log(f"    mem after lgbm_mean: rss {CM.rss_mb():.0f} MB, avail {CM.mem_avail_mb():.0f} MB")
    lgbm_q.fit(Y, t_start, ctx)
    log(f"  fit lgbm_q    {lgbm_q.fit_seconds:.0f}s ({len(QUANTILES)} quantiles, "
        f"{lgbm_q.n_train_rows} rows)")
    log(f"    mem after lgbm_q: rss {CM.rss_mb():.0f} MB, avail {CM.mem_avail_mb():.0f} MB")
    hw.fit(Y, t_start, ctx)
    log(f"  fit hw        {hw.params} tune_rmse={hw.tune_rmse:.3f}")
    for o in stat_objs.values():
        o.fit(Y, t_start, ctx)
    log(f"  total fitting {time.time()-t0:.0f}s")

    qarr = np.array(QUANTILES, dtype=np.float32)
    cache = dict(
        origins=origins, protection=protection, t_start=t_start, t_end=t_end,
        point={m: np.zeros((len(origins), n, H), dtype=np.float32) for m in POINT_METHODS},
        pi_q={m: np.zeros((len(origins), len(QUANTILES), n), dtype=np.float32)
              for m in DIST_METHODS},
        acc=[], pinball=[],
        feature_importance=(lgbm_mean.models["mean"].booster_
                            .feature_importance(importance_type="gain").tolist()
                            if hasattr(lgbm_mean.models["mean"], "booster_") else None),
        fnames=lgbm_mean.fnames,
        G=np.zeros((len(origins), n), dtype=np.int32),
        Qd=np.zeros((len(origins), len(QUANTILES), n, H), dtype=np.float32),
        crossing=[],
    )

    for oi, t in enumerate(origins):
        ts = time.time()
        sp = CM.sell_price_at(ctx["prices"], ctx["wcol"], t)
        ctx_o = dict(ctx, sell_price=sp)

        pf = {nm: o.predict(Y, t, H, ctx_o) for nm, o in stat_objs.items()}
        pf["hw"] = hw.predict(Y, t, H, ctx_o)
        pf["comb"] = comb.predict(Y, t, H, ctx_o)
        pf["lgbm_mean"] = lgbm_mean.predict(Y, t, H, ctx_o)
        for nm in POINT_METHODS:
            cache["point"][nm][oi] = pf[nm]

        # distributional arm 1: LightGBM quantile regression (daily surface)
        Ql = lgbm_q.predict(Y, t, H, ctx_o)
        cache["Qd"][oi] = Ql
        cache["crossing"].append(lgbm_q.crossing_rate)
        cache["pi_q"]["lgbm_q"][oi] = quantile_pi_surface(Ql, protection, args.pi_mode,
                                                          lam=args.pi_lam)

        # distributional arm 2: empirical protection-interval quantiles pooled
        # within demand-pattern groups, level-anchored on a point forecast
        # pools are rebuilt at every origin from the history before t only
        g = CM.groups_at(Y, t)
        cache["G"][oi] = g
        for base_name, key in [("comb", "cal_comb"), ("sba", "cal_sba"),
                               ("lgbm_mean", "cal_lgbm")]:
            sc = FC.level_scale_from_forecast(Y, t, pf[base_name])
            cache["pi_q"][key][oi] = FC.empirical_protection_quantiles(
                Y, t, protection, qarr, g, level_scale=sc).astype(np.float32)

        # ---- accuracy on this origin ---------------------------------------
        act = Y[:, t:t + H].astype(np.float64)
        sq_sc, ab_sc = CM.naive_scale(Y, t)
        for nm in POINT_METHODS:
            p = pf[nm].astype(np.float64)
            se = CM.scaled_errors(act, p, sq_sc, ab_sc)
            cache["acc"].append(dict(
                block=block_name, origin=int(t), period=oi, method=nm,
                mae=float(np.abs(p - act).mean()),
                rmse=float(np.sqrt(((p - act) ** 2).mean())),
                bias=float((p - act).mean()),
                wape=CM.wape(act, p),
                rmsse=se["rmsse"], mase=se["mase"],
                pct_scaled=se["pct_scaled_sq"],
                mean_fc=float(p.mean()), mean_act=float(act.mean())))
        Qcal = FC.empirical_quantile_surface(pf["comb"], Y, t, H, g, qarr).astype(np.float64)
        for nm, Q in [("lgbm_q", Ql.astype(np.float64)), ("cal_comb", Qcal)]:
            rec = dict(block=block_name, origin=int(t), period=oi, method=nm)
            for i, tau in enumerate(QUANTILES):
                q = Q[i]
                rec[f"pinball_{tau:.2f}"] = float(np.mean(
                    np.maximum(tau * (act - q), (tau - 1) * (act - q))))
                rec[f"coverage_{tau:.2f}"] = float((act <= q).mean())
            rec["pinball_mean"] = float(np.mean(
                [rec[f"pinball_{tau:.2f}"] for tau in QUANTILES]))
            cache["pinball"].append(rec)

        if oi % 4 == 0:
            log(f"    origin {t} ({time.time()-ts:.1f}s) act_mean={act.mean():.3f} "
                f"comb_MAE={cache['acc'][-2]['mae']:.3f} "
                f"rss={CM.rss_mb():.0f}MB avail={CM.mem_avail_mb():.0f}MB")
        del Ql, Qcal, act, pf
        gc.collect()
    return cache


def rebuild_pi_q(cache, Y, protection, args):
    """
    Protection-interval quantile surfaces for the protection interval of THIS
    configuration (P = review + lead). Sensitivity runs that change the lead time
    or the review interval must resize the interval; reading the surfaces cached
    for the baseline P would silently under-protect every quantile policy.
    """
    qarr = np.array(QUANTILES, dtype=np.float64)
    out = {}
    nO = len(cache["origins"])
    out["lgbm_q"] = np.stack([quantile_pi_surface(cache["Qd"][oi], protection,
                                                  args.pi_mode, lam=args.pi_lam)
                              for oi in range(nO)])
    for base_name, key in [("comb", "cal_comb"), ("sba", "cal_sba"),
                           ("lgbm_mean", "cal_lgbm")]:
        arr = []
        for oi, t in enumerate(cache["origins"]):
            sc = FC.level_scale_from_forecast(Y, t, cache["point"][base_name][oi])
            arr.append(FC.empirical_protection_quantiles(
                Y, t, protection, qarr, cache["G"][oi], level_scale=sc))
        out[key] = np.stack(arr).astype(np.float32)
    return out


# ==========================================================================
# stage 2  demand-regime labels
# ==========================================================================
def regime_labels(Y, origins, baseline=364, recent=56):
    lab = np.empty((len(origins), Y.shape[0]), dtype=object)
    stats = np.zeros((len(origins), Y.shape[0], 4), dtype=np.float32)
    for oi, t in enumerate(origins):
        b0, b1 = max(t - baseline - recent, 0), max(t - recent, 0)
        base = Y[:, b0:b1].astype(np.float64)
        rec = Y[:, b1:t].astype(np.float64)
        if base.shape[1] < 28 or rec.shape[1] < 14:
            lab[oi] = "warmup"
            continue
        bm, rm = base.mean(1), rec.mean(1)
        bs, rs = base.std(1), rec.std(1)
        lvl = np.divide(rm, bm, out=np.full_like(rm, 10.0), where=bm > 1e-9)
        vol = np.divide(rs, bs, out=np.ones_like(rs), where=bs > 1e-9)
        lab[oi] = np.where(base.sum(1) <= 0, "new",
                  np.where(lvl >= 1.5, "level_up",
                  np.where(lvl <= 0.67, "level_down",
                  np.where(vol >= 1.5, "vol_up", "stable"))))
        stats[oi] = np.stack([bm, rm, np.minimum(lvl, 10), np.minimum(vol, 10)], 1)
    return lab.astype(str), stats


P1SWEEP = {}
Z_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]
SWEEP_METHODS_FULL = ["naive", "snaive", "ma28", "croston", "sba", "tsb", "hw", "comb",
                      "lgbm_mean"]
SWEEP_METHODS_SMALL = ["lgbm_mean", "tsb", "hw", "comb"]


def run_p1_sweep(Y, cache, demand, t_start, T, local_origins, h, pen, sp, protection,
                 args, n):
    """
    Tuned normal-theory extension. The default P1 fixes z at the critical-ratio
    normal quantile and sigma at the trailing-56-day standard deviation measured
    once at the start of the block. Here every ingredient a practitioner could
    tune is tuned:
      wk_static  weekly review, sigma fixed at block start, z swept
      wk_dyn     weekly review, sigma refreshed at every forecast origin
      day_static daily review base-stock, protection interval L+1, z swept
      day_dyn    daily review base-stock, sigma refreshed, z swept
      sS_static / sS_dyn   (s,S) with s=0.5 S, daily review, S = normal-theory level
                           for the weekly protection interval (same structure as P3)
    z is selected on VAL only (selection happens in the analysis script).
    """
    origins = cache["origins"]
    rev_week = np.arange(0, T, args.review)
    rev_day = np.arange(0, T, REVIEW_DAILY)
    sig_static = np.maximum(Y[:, max(t_start - 56, 0):t_start].astype(np.float64).std(1), 1e-3)
    sig_dyn = np.stack([np.maximum(Y[:, max(o - 56, 0):o].astype(np.float64).std(1), 1e-3)
                        for o in origins])
    rows, tot = [], {}
    variants = [("wk_static", rev_week, protection, False, SWEEP_METHODS_FULL),
                ("sS_static", rev_day, protection, False, SWEEP_METHODS_SMALL),
                ("sS_dyn", rev_day, protection, True, SWEEP_METHODS_SMALL),
                ("wk_dyn", rev_week, protection, True, SWEEP_METHODS_SMALL),
                ("day_static", rev_day, args.lead + 1, False, SWEEP_METHODS_SMALL),
                ("day_dyn", rev_day, args.lead + 1, True, SWEEP_METHODS_SMALL)]
    for vname, rev, prot, dyn, methods in variants:
        for nm in methods:
            for z in Z_GRID:
                lv = np.stack([SIM.normal_theory_target(
                    cache["point"][nm][oi], sig_dyn[oi] if dyn else sig_static, prot, z)
                    for oi in range(len(origins))])
                S = SIM.build_S_surface(T, local_origins, lv, n)
                if vname.startswith("sS_"):       # (s,S) with a normal-theory S, daily review
                    r = SIM.simulate_sS(demand, S, args.s_frac * S, args.lead, h, pen,
                                        args.K, review_days=rev)
                else:
                    r = SIM.simulate(demand, S, rev, lead=args.lead, h=h, p=pen, K=args.K)
                key = f"{vname}|{nm}|{z:g}"
                tot[key] = r["total"].astype(np.float64)
                rows.append(dict(variant=vname, method=nm, z=z,
                                 total_cost=float(r["total"].sum()),
                                 holding_cost=float(r["holding"].sum()),
                                 penalty_cost=float(r["penalty"].sum()),
                                 ordering_cost=float(r["ordering"].sum()),
                                 fill_rate=float(r["sold"].sum() / max(r["demand"].sum(), 1e-9)),
                                 csl_period=float(r["csl_period"].mean()),
                                 avg_on_hand=float(r["avg_on_hand"].mean())))
                del S, r
        log(f"    p1 sweep {vname} done")
        gc.collect()
    return rows, tot


# ==========================================================================
# stage 3  policies
# ==========================================================================
def run_policies(Y, ctx, cache, meta, block_name, n, args, econ_list):
    block = block_name
    """
    Run every forecaster x policy combination under each economic setting.

    Each *economic setting* fixes the cost model (holding rate, stockout penalty,
    fixed order cost). Within a setting, every policy is tuned to its own optimal
    critical ratio CR = p / (p + h*R) given its review frequency R, so policies
    are compared at their respective optima under identical economics rather than
    at an arbitrary common service target.
    """
    t_start, t_end = cache["t_start"], cache["t_end"]
    T = t_end - t_start
    origins = cache["origins"]
    local_origins = [o - t_start for o in origins]
    protection = args.review + args.lead
    pi_q = rebuild_pi_q(cache, Y, protection, args)
    demand = np.ascontiguousarray(Y[:, t_start:t_end].astype(np.float64).T)
    rev_week = np.arange(0, T, args.review)
    rev_day = np.arange(0, T, REVIEW_DAILY)
    sp = CM.sell_price_at(ctx["prices"], ctx["wcol"], t_start)
    sigma = np.maximum(Y[:, max(t_start - 56, 0):t_start].astype(np.float64).std(1), 1e-3)
    qarr = np.array(QUANTILES, dtype=np.float64)

    ser, agg, combo_names, combo_costs = [], [], [], []
    percost, perunits = {}, {}
    # per-period breakdown is only stored for the primary economic setting

    for econ in econ_list:
        c, h, pen = SIM.derive_costs(sp, penalty_mode=econ["mode"],
                                     penalty_mult=(econ["mult"] if econ.get("mult")
                                                   is not None else args.penalty_mult),
                                     target_cr=econ.get("cr"),
                                     review_days=args.review,
                                     hold_rate=args.hold_rate,
                                     cost_ratio=args.cost_ratio)
        keep_per = (econ["name"] == ECON_FOR_REGIME)
        cr7 = SIM.critical_ratio(h, pen, args.review)         # periodic review
        cr1 = SIM.critical_ratio(h, pen, REVIEW_DAILY)        # daily review
        cr_med = float(np.median(cr7))
        tau7 = SIM.tau_index_for_cr(qarr, cr_med)
        tau1 = SIM.tau_index_for_cr(qarr, float(np.median(cr1)))
        from scipy.stats import norm
        z7 = float(norm.ppf(min(max(cr_med, 1e-4), 1 - 1e-4)))
        log(f"  [{block_name}|{econ['name']}] CR(R={args.review})={cr_med:.4f} -> tau={QUANTILES[tau7]}, "
            f"z={z7:.2f} | CR(R=1)={np.median(cr1):.4f} -> tau={QUANTILES[tau1]} | "
            f"penalty median {np.median(pen):.3f}, h/day median {np.median(h):.5f}")

        # ---- P1: point forecast + normal-theory safety stock (weekly review) --
        for nm in POINT_METHODS:
            lv = np.stack([SIM.normal_theory_target(cache["point"][nm][oi], sigma,
                                                    protection, z7)
                           for oi in range(len(origins))])
            S = SIM.build_S_surface(T, local_origins, lv, n)
            r = SIM.simulate(demand, S, rev_week, lead=args.lead, h=h, p=pen, K=args.K)
            _record(ser, agg, combo_names, combo_costs, block_name, econ["name"],
                    nm, "P1_base_stock_normal", QUANTILES[tau7], r, meta, T, sp, n,
                    percost if keep_per else None, perunits if keep_per else None)
            del S
        gc.collect()

        # ---- extension: how much of the gap survives a *tuned* normal-theory rule?
        if args.p1_sweep and econ["name"] == ECON_FOR_REGIME:
            sweep_rows, sweep_tot = run_p1_sweep(Y, cache, demand, t_start, T, local_origins,
                                                 h, pen, sp, protection, args, n)
            P1SWEEP[block_name] = (sweep_rows, sweep_tot)

        # ---- P2: order-up-to at a quantile of the protection-interval demand
        #      distribution. We sweep every tau so that the cost-service frontier
        #      can be traced and the service-constrained optimum identified,
        #      rather than committing to a single (contestable) service target.
        for nm in DIST_METHODS:
            for tix, tau in enumerate(QUANTILES):
                lv = np.stack([pi_q[nm][oi][tix] for oi in range(len(origins))])
                S = SIM.build_S_surface(T, local_origins, lv, n)
                r = SIM.simulate(demand, S, rev_week, lead=args.lead, h=h, p=pen,
                                 K=args.K)
                _record(ser, agg, combo_names, combo_costs, block_name, econ["name"],
                        nm, f"P2_quantile_base_stock@{tau:g}", tau, r, meta, T, sp, n,
                        percost if keep_per else None, perunits if keep_per else None)
                del S
            gc.collect()

        # ---- P3: (s, S) with a fixed ordering cost, weekly vs daily review ----
        for nm in ["lgbm_q", "cal_comb"]:
            for rev_label, rev_days, R in [("weekly", rev_week, args.review),
                                           ("daily", rev_day, REVIEW_DAILY)]:
                for tix, tau in enumerate(QUANTILES):
                    lv = np.stack([pi_q[nm][oi][tix]
                                   for oi in range(len(origins))])
                    S = SIM.build_S_surface(T, local_origins, lv, n)
                    sl = args.s_frac * S
                    r = SIM.simulate_sS(demand, S, sl, args.lead, h, pen, args.K,
                                        review_days=rev_days)
                    _record(ser, agg, combo_names, combo_costs, block_name,
                            econ["name"], nm, f"P3_sS_{rev_label}@{tau:g}", tau,
                            r, meta, T, sp, n,
                            percost if keep_per else None,
                            perunits if keep_per else None)
                    del S, sl
                gc.collect()

        # ---- P4: DDMRP-style buffered policy (level only, no distribution) ----
        for nm in ["comb", "lgbm_mean"]:
            Sf, sf = SIM.policy_ddmrp([cache["point"][nm][oi] for oi in range(len(origins))],
                                      protection, local_origins, T, n,
                                      lt_days=args.lead, buffer_frac=args.ddmrp_buffer,
                                      review=args.review)
            r = SIM.simulate_sS(demand, Sf, sf, args.lead, h, pen, args.K,
                                review_days=rev_day)
            _record(ser, agg, combo_names, combo_costs, block_name, econ["name"],
                    nm, "P4_ddmrp", np.nan, r, meta, T, sp, n,
                    percost if keep_per else None, perunits if keep_per else None)
            del Sf, sf
        gc.collect()

    costM = np.stack(combo_costs, axis=0) if combo_costs else np.zeros((0, n), np.float32)
    del combo_costs
    gc.collect()
    return (ser, agg, costM, [k for _, k in combo_names], [e for e, _ in combo_names],
            percost, perunits)


def _record(ser, agg, names, costs, block, econ, method, policy, tau, r, meta,
            T, sp, n, percost=None, perunits=None):
    key = f"{econ}|{method}|{policy}"
    names.append((econ, key))
    costs.append(np.asarray(r["total"], dtype=np.float32))
    ser.append(_series_rows(block, econ, method, policy, tau, r, meta))
    agg.append(_agg_row(block, econ, method, policy, tau, r, T, sp))
    if percost is not None and perunits is not None:
        # 28-day period breakdown of cost and units short, for regime analysis
        nper = max(T // PERIOD, 1)
        dc = np.asarray(r["day_cost"], dtype=np.float64)
        ds = np.asarray(r["day_short"], dtype=np.float64)
        cp = np.zeros((nper, n)); sp_ = np.zeros((nper, n))
        for k in range(nper):
            sl = slice(k * PERIOD, min((k + 1) * PERIOD, T))
            cp[k] = dc[sl].sum(0)
            sp_[k] = ds[sl].sum(0)
        percost[key] = cp.astype(np.float32)
        perunits[key] = sp_.astype(np.float32)


def _series_rows(block, econ, method, policy, tau, r, meta):
    df = pd.DataFrame({
        "series_ix": np.arange(len(r["total"])),
        "block": block, "econ": econ, "method": method, "policy": policy, "tau": tau,
        "total": r["total"], "holding": r["holding"], "penalty": r["penalty"],
        "ordering": r["ordering"], "fill_rate": r["fill_rate"],
        "lost_units": r["lost"], "demand_units": r["demand"],
        "avg_on_hand": r["avg_on_hand"], "n_orders": r["n_orders"],
        "short_day_share": r["short_day_share"],
    })
    for c in ["sbc_class", "scale_band", "cat_id", "state_id", "dept_id"]:
        df[c] = meta[c].to_numpy()
    for c in ["adi", "cv2", "zero_share", "mean_daily", "sbc_class_adi"]:
        if c in meta.columns:
            df[c] = meta[c].to_numpy()
    return df


def _agg_row(block, econ, method, policy, tau, r, T, sp):
    """
    Window-level KPIs. Normalisation choices matter for interpretability, so we
    report several: absolute cost, cost per unit sold, inventory-related cost as
    a share of revenue and of COGS, and the holding / stockout / ordering split.
    """
    dem_ser = np.asarray(r["demand"], dtype=np.float64)      # per-series units (n,)
    dem = float(dem_ser.sum())
    revenue = float((dem_ser * np.asarray(sp, dtype=np.float64)).sum())
    cogs = float(CM.COST_RATIO_DEFAULT * revenue) if hasattr(CM, "COST_RATIO_DEFAULT") \
        else float(0.60 * revenue)
    inv_cost = float(r["holding"].sum() + r["ordering"].sum())
    return dict(
        block=block, econ=econ, method=method, policy=policy, tau=tau,
        total_cost=float(r["total"].sum()),
        holding_cost=float(r["holding"].sum()),
        penalty_cost=float(r["penalty"].sum()),
        ordering_cost=float(r["ordering"].sum()),
        mean_cost_series=float(r["total"].mean()),
        median_cost_series=float(np.median(r["total"])),
        p90_cost_series=float(np.percentile(r["total"], 90)),
        fill_rate=float(r["sold"].sum() / max(dem, 1e-9)),
        csl_period=float(r["csl_period"].mean()),
        csl_window=float(r["csl_window"].mean()),
        short_day_share=float(r["short_day_share"].mean()),
        lost_units=float(r["lost"].sum()),
        demand_units=float(dem),
        avg_on_hand=float(r["avg_on_hand"].mean()),
        orders_per_series=float(np.nanmean(r["n_orders"])),
        revenue=revenue,
        cost_per_unit=float(r["total"].sum() / max(dem, 1e-9)),
        cost_pct_revenue=float(r["total"].sum() / max(revenue, 1e-9)),
        inv_cost_pct_cogs=float(inv_cost / max(cogs, 1e-9)),
        holding_pct_revenue=float(r["holding"].sum() / max(revenue, 1e-9)),
        penalty_pct_revenue=float(r["penalty"].sum() / max(revenue, 1e-9)),
        ordering_pct_revenue=float(r["ordering"].sum() / max(revenue, 1e-9)),
        days=T,
    )


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_series", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", type=str, default="val")
    ap.add_argument("--out", type=str, default="pilot")
    ap.add_argument("--lead", type=int, default=SIM.BASE_LEAD)
    ap.add_argument("--K", type=float, default=SIM.BASE_K)
    ap.add_argument("--s_frac", type=float, default=0.5)
    ap.add_argument("--econ", type=str, default="margin",
                    help="comma list of economic settings: margin, margin2, "
                         "cr75, cr90, cr95, cr99, free")
    ap.add_argument("--ddmrp_buffer", type=float, default=0.5)
    ap.add_argument("--review", type=int, default=REVIEW,
                    help="periodic review interval in days for base-stock policies")
    ap.add_argument("--cost_ratio", type=float, default=SIM.COST_RATIO,
                    help="purchase cost as a share of shelf price")
    ap.add_argument("--hold_rate", type=float, default=SIM.HOLD_RATE,
                    help="annual inventory holding rate as a share of unit cost")
    ap.add_argument("--penalty_mult", type=float, default=1.0,
                    help="multiplier on forgone margin for the stockout penalty")
    ap.add_argument("--reuse_forecasts", type=str, default="",
                    help="run directory whose cached forecasts should be reused "
                         "(skips all model fitting; used for policy-side sweeps)")
    ap.add_argument("--cache_forecasts", action="store_true",
                    help="persist the forecast cache for later reuse")
    ap.add_argument("--max_train_rows", type=int, default=260000,
                    help="cap on total training-panel rows; series are subsampled "
                         "to respect it (0 = no cap)")
    ap.add_argument("--lgbm_estimators", type=int, default=350)
    ap.add_argument("--hw_tune", type=int, default=400)
    ap.add_argument("--hw_window", type=int, default=336)
    ap.add_argument("--p1_sweep", action="store_true",
                    help="also run the tuned normal-theory extension (z sweep, "
                         "daily review, refreshed sigma) for the margin economy")
    ap.add_argument("--pi_mode", type=str, default="blend")
    ap.add_argument("--pi_lam", type=float, default=0.5)
    args = ap.parse_args()
    ECON_PRESETS = {
        # name: (penalty_mode, multiplier or None, target CR or None)
        "margin":  dict(name="margin",  mode="margin", mult=None),
        "margin2": dict(name="margin2", mode="margin", mult=2.0),
        "cr75":    dict(name="cr75",    mode="cr", cr=0.75),
        "cr90":    dict(name="cr90",    mode="cr", cr=0.90),
        "cr95":    dict(name="cr95",    mode="cr", cr=0.95),
        "cr99":    dict(name="cr99",    mode="cr", cr=0.99),
        "free":    dict(name="free",    mode="cr", cr=0.90),
    }
    args.econ_list = [ECON_PRESETS[e] for e in args.econ.split(",")]

    outdir = RESULTS / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    log(f"config: {vars(args)}")

    t_start = time.time()
    Y, meta, ctx, splits, n = setup(args)
    np.save(outdir / "sample_index.npy", np.arange(n))
    meta.to_parquet(outdir / "sample_meta.parquet", index=False)

    blocks = {b: tuple(splits[b]) for b in args.blocks.split(",")}
    all_ser, all_agg, all_regime, all_acc, all_pin = [], [], [], [], []
    caches = {}

    for bname, blk in blocks.items():
        log(f"=== BLOCK {bname} {blk} ===")
        cache = None
        if args.reuse_forecasts:
            cache = load_cache(RESULTS / args.reuse_forecasts, bname, n)
        if cache is None:
            cache = forecast_block(Y, ctx, bname, blk, n, args)
        if args.cache_forecasts:
            save_cache(cache, outdir, bname)
        caches[bname] = cache
        if cache["acc"]:
            all_acc.append(pd.DataFrame(cache["acc"]))
            all_pin.append(pd.DataFrame(cache["pinball"]))
        else:
            ad = Path(args.reuse_forecasts)
            all_acc.append(pd.read_csv(RESULTS / ad / f"acc_{bname}.csv"))
            all_pin.append(pd.read_csv(RESULTS / ad / f"pin_{bname}.csv"))

        reg, rstats = regime_labels(Y, cache["origins"])
        rd = []
        for oi, t in enumerate(cache["origins"]):
            rd.append(pd.DataFrame({
                "block": bname, "origin": t, "period": oi,
                "series_ix": np.arange(n), "regime": reg[oi],
                "base_mean": rstats[oi, :, 0], "recent_mean": rstats[oi, :, 1],
                "level_ratio": rstats[oi, :, 2], "vol_ratio": rstats[oi, :, 3]}))
        all_regime.append(pd.concat(rd, ignore_index=True))

        ser, agg, costM, keys, econs, percost, perunits = run_policies(
            Y, ctx, cache, meta, bname, n, args, args.econ_list)
        all_ser.append(pd.concat(ser, ignore_index=True))
        all_agg.append(pd.DataFrame(agg))
        np.save(outdir / f"costM_{bname}.npy", costM)
        # per-(series, period) cost for every pairing, so the analysis can do
        # split-half selection experiments without re-simulating
        np.savez_compressed(outdir / f"percost_{bname}.npz",
                            keys=np.array(list(percost.keys()), dtype=object),
                            **{f"k{i}": v for i, (_, v) in enumerate(percost.items())})

        # ---- regime-conditional cost table ---------------------------------
        # For each (pairing, period, series) we now have cost and units short.
        # Joining the regime label of that same (series, period) cell answers
        # RQ3 directly: does the best strategy change when demand shifts?
        if percost:
            nper = next(iter(percost.values())).shape[0]
            recs = []
            reg_b = all_regime[-1] if all_regime else None
            regmap = {}
            if reg_b is not None:
                for oi in range(min(nper, len(cache["origins"]))):
                    sub = reg_b[reg_b.period == oi]
                    regmap[oi] = sub.set_index("series_ix")["regime"]
            for key, cp in percost.items():
                econ_k, method_k, policy_k = key.split("|", 2)
                up = perunits[key]
                for oi in range(nper):
                    rg = regmap.get(oi)
                    if rg is None:
                        continue
                    rvals = rg.reindex(np.arange(n)).fillna("unknown").to_numpy()
                    d = pd.DataFrame({"regime": rvals, "cost": cp[oi],
                                      "short": up[oi]})
                    g = d.groupby("regime", as_index=False).agg(
                        cost=("cost", "sum"), short=("short", "sum"),
                        n=("cost", "size"), mean_cost=("cost", "mean"))
                    recs.extend(g.assign(block=bname, econ=econ_k, method=method_k,
                                         policy=policy_k, period=oi).to_dict("records"))
            if recs:
                pd.DataFrame(recs).to_parquet(outdir / f"regime_cost_{bname}.parquet",
                                              index=False)
                log(f"  regime-cost table: {len(recs)} rows")
        del percost, perunits
        if bname in P1SWEEP:
            rows_, tot_ = P1SWEEP[bname]
            pd.DataFrame(rows_).to_csv(outdir / f"p1sweep_{bname}.csv", index=False)
            np.savez_compressed(outdir / f"p1sweep_tot_{bname}.npz",
                                keys=np.array(list(tot_.keys()), dtype=object),
                                M=np.stack(list(tot_.values()), axis=0))
        if cache.get("crossing"):
            cr = np.mean(np.array(cache["crossing"]), axis=0).tolist()
            with open(outdir / f"quantile_crossing_{bname}.json", "w") as f:
                json.dump(dict(quantiles=QUANTILES, share_pulled_up_by_monotonisation=cr), f)
        with open(outdir / f"combos_{bname}.json", "w") as f:
            json.dump({"keys": keys, "econs": econs}, f)
        del costM, cache["point"], cache["pi_q"]
        gc.collect()
        log(f"  block {bname} done in {time.time()-t_start:.0f}s total")

    pd.concat(all_acc).to_csv(outdir / "accuracy.csv", index=False)
    pd.concat(all_pin).to_csv(outdir / "pinball.csv", index=False)
    pd.concat(all_regime).to_csv(outdir / "regimes.csv", index=False)
    pd.concat(all_agg).to_csv(outdir / "policy_agg.csv", index=False)
    # Series-level results are the largest artefact (~1M rows). Store one file per
    # block with categoricals and int32 indices so the analysis can load a single
    # block inside a 1 GB memory cap.
    allser = pd.concat(all_ser, ignore_index=True)
    for c in ["econ", "method", "policy", "block", "sbc_class", "scale_band",
              "cat_id", "state_id", "dept_id"]:
        if c in allser:
            allser[c] = allser[c].astype("category")
    allser["series_ix"] = allser["series_ix"].astype(np.int32)
    for c in allser.columns:
        if allser[c].dtype == np.float64:
            allser[c] = allser[c].astype(np.float32)
    allser.to_parquet(outdir / "policy_series.parquet", index=False,
                      compression="zstd")
    for b in allser.block.cat.categories:
        allser[allser.block == b].to_parquet(outdir / f"policy_series_{b}.parquet",
                                             index=False, compression="zstd")
    log(f"series table: {len(allser)} rows, "
        f"{allser.memory_usage(deep=True).sum()/1e6:.0f} MB in memory")
    del allser
    gc.collect()
    log(f"WROTE results to {outdir} | total {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
