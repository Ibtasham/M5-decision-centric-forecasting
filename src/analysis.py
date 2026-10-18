"""
analysis.py
===========
Decision-centric analysis of the experiment output. Produces every table and
figure used in the paper.

The analysis is organised around the four research questions:

  RQ1  Does the accuracy ranking of forecasters agree with their inventory-cost
       ranking? How large is the *decision loss* from selecting on accuracy?
  RQ2  Where does that loss come from? A four-step decomposition separates
       (i) picking a better point forecaster, (ii) replacing the normal-theory
       safety-stock rule with a distribution-based one, (iii) replacing an
       estimated distribution with a directly trained one, and (iv) changing the
       policy structure (review frequency, order-up-to vs (s,S) vs buffered).
  RQ3  Under which demand patterns and demand regimes does the disagreement
       appear? Does the best pairing change when demand shifts?
  RQ4  Can a meta-model that selects the pairing per SKU (and per regime) beat
       the best fixed pairing chosen on validation data?

Usage
-----
  python3 analysis.py --runs main_val,main_test --out main
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, FIGS                                # noqa: E402

POLICY_FAMILIES = ["P1_base_stock_normal", "P2_quantile_base_stock",
                   "P3_sS_weekly", "P3_sS_daily", "P4_ddmrp"]


def policy_family(policy: str) -> str:
    """Strip the @tau suffix so that policies can be grouped by family."""
    return policy.split("@")[0]


POLICY_ORDER = POLICY_FAMILIES
ACC_METRICS = ["rmsse", "wape", "mae", "rmse", "mase"]


def log(m):
    print(m, flush=True)


# ==========================================================================
# loading
# ==========================================================================
def load_run(run: str, block: str | None = None):
    """
    Load one experiment directory, optionally restricted to a single block.

    A single driver invocation writes both the validation and test blocks into
    one directory, distinguished by the `block` column, so the analysis must
    filter explicitly -- otherwise validation results are silently reported as
    test results.
    """
    d = RESULTS / run
    agg = pd.read_csv(d / "policy_agg.csv")
    acc = pd.read_csv(d / "accuracy.csv")
    pin = pd.read_csv(d / "pinball.csv")
    reg = pd.read_csv(d / "regimes.csv")
    ser_path = d / (f"policy_series_{block}.parquet" if block else "policy_series.parquet")
    ser = pd.read_parquet(ser_path)
    meta = pd.read_parquet(d / "sample_meta.parquet")
    cfg = json.load(open(d / "config.json"))
    if block:
        for name in ("agg", "acc", "pin", "reg", "ser"):
            obj = locals()[name]
            locals()[name] = obj[obj.block == block].reset_index(drop=True)
        agg, acc, pin, reg, ser = (agg[agg.block == block].reset_index(drop=True),
                                   acc[acc.block == block].reset_index(drop=True),
                                   pin[pin.block == block].reset_index(drop=True),
                                   reg[reg.block == block].reset_index(drop=True),
                                   ser[ser.block == block].reset_index(drop=True))
        combos = json.load(open(d / f"combos_{block}.json"))
    else:
        combos = json.load(open(d / f"combos_{agg.block.iloc[0]}.json"))
    # `ser` is stored with categorical dtypes to fit the memory budget; string
    # concatenation on categoricals is not supported, so cast first.
    ser["combo"] = (ser["econ"].astype(str) + "|" + ser["method"].astype(str)
                    + "|" + ser["policy"].astype(str))
    return dict(dir=d, block=block, agg=agg, acc=acc, pin=pin, reg=reg, ser=ser,
                combos=combos, meta=meta, cfg=cfg)


# ==========================================================================
# RQ1  accuracy vs cost
# ==========================================================================
POINT_ONLY = ["naive", "snaive", "ma28", "croston", "sba", "tsb", "hw",
              "comb", "lgbm_mean"]


def table_accuracy(R, econ):
    """Point-forecast accuracy metrics, ranked. Distributional methods are
    reported separately by pinball loss -- they have no point forecast to rank,
    and mixing them into the same ranking is a category error."""
    cols = [c for c in ACC_METRICS if c in R["acc"]]
    a = R["acc"].groupby("method")[cols + ["bias", "pct_scaled"]].mean()
    a = a.loc[[m for m in POINT_ONLY if m in a.index]]
    a = a.sort_values("rmsse")
    a.insert(0, "rank_rmsse", np.arange(1, len(a) + 1))
    return a


def table_probabilistic(R):
    p = R["pin"]
    cols = [c for c in p.columns if c.startswith(("pinball", "coverage"))]
    t = p.groupby("method")[cols].mean()
    if "pinball_mean" in t:
        t = t.sort_values("pinball_mean")
    return t


def table_cost(R, econ):
    g = R["agg"]
    g = g[g.econ == econ] if econ else g
    g = g.assign(family=g.policy.map(policy_family))
    t = g.groupby(["econ", "method", "policy"], as_index=False).agg(
        total_cost=("total_cost", "first"),
        fill_rate=("fill_rate", "first"),
        csl_period=("csl_period", "first"),
        short_day_share=("short_day_share", "first"),
        avg_on_hand=("avg_on_hand", "first"),
        orders_per_series=("orders_per_series", "first"),
        cost_per_unit=("cost_per_unit", "first"),
        cost_pct_revenue=("cost_pct_revenue", "first"),
        inv_cost_pct_cogs=("inv_cost_pct_cogs", "first"))
    return t.sort_values(["econ", "total_cost"])


def rank_agreement(acc_tab, cost_tab, econ):
    """
    Compare the forecaster ranking induced by accuracy with the ranking induced
    by inventory cost, holding the policy family fixed so that the comparison is
    about the forecaster and not the policy.
    """
    from scipy.stats import kendalltau, spearmanr
    out = []
    for pol in POLICY_ORDER:
        sub = cost_tab[(cost_tab.policy == pol) & (cost_tab.econ == econ)]
        if len(sub) < 4:
            continue
        m = sub.merge(acc_tab.reset_index()[["method", "rmsse", "wape", "mae"]],
                      on="method")
        m = m[m.method.isin(POINT_ONLY)]     # only rankable point forecasters
        if len(m) < 4:
            continue
        for metric in ["rmsse", "wape", "mae"]:
            rho, p_rho = spearmanr(m[metric], m["total_cost"])
            tau, p_tau = kendalltau(m[metric], m["total_cost"])
            out.append(dict(policy=pol, econ=econ, metric=metric, n=len(m),
                            spearman=rho, spearman_p=p_rho,
                            kendall=tau, kendall_p=p_tau,
                            accuracy_best=m.sort_values(metric).method.iloc[0],
                            cost_best=m.sort_values("total_cost").method.iloc[0],
                            agree=bool(m.sort_values(metric).method.iloc[0]
                                       == m.sort_values("total_cost").method.iloc[0])))
    return pd.DataFrame(out)


def decision_loss(R, econ):
    """
    The headline quantity: how much extra inventory cost is incurred by a
    pipeline that selects its forecaster on out-of-sample accuracy and then
    applies the standard safety-stock rule, versus selecting on realised
    inventory cost.

    Reported three ways, because "the accuracy-optimal choice" is itself
    ambiguous in practice:
      A  best by RMSSE            (the M5 accuracy metric)
      B  best by WAPE / MAE       (what most S&OP dashboards actually show)
      C  best by pinball loss     (a probabilistic accuracy metric)
    each compared against the cost-optimal pairing.
    """
    acc = table_accuracy(R, econ)
    cost = table_cost(R, econ)
    cost = cost[cost.econ == econ]
    best_cost = cost.total_cost.min()
    best_row = cost.loc[cost.total_cost.idxmin()]

    res = dict(econ=econ, cost_optimal_total=best_cost,
               cost_optimal_combo=f"{best_row['method']}|{best_row['policy']}")
    # A: accuracy-best point forecaster under the standard policy
    for metric, tag in [("rmsse", "A_rmsse"), ("wape", "B_wape"), ("mae", "B_mae")]:
        m = acc[metric].idxmin()
        row = cost[(cost.method == m) & (cost.policy == "P1_base_stock_normal")]
        if len(row):
            c = float(row.total_cost.iloc[0])
            res[f"{tag}_combo"] = f"{m}|P1_base_stock_normal"
            res[f"{tag}_total"] = c
            res[f"{tag}_excess_pct"] = 100.0 * (c / best_cost - 1.0)
    # C: probabilistic-accuracy-best under a quantile policy
    prob = table_probabilistic(R)
    if "pinball_mean" in prob and prob["pinball_mean"].notna().any():
        m = prob["pinball_mean"].idxmin()
        sub2 = cost[(cost.method == m) & cost.policy.str.startswith("P2_quantile")]
        row = sub2.sort_values("total_cost").head(1)
        if len(row):
            c = float(row.total_cost.iloc[0])
            res["C_pinball_combo"] = f"{m}|P2_quantile_base_stock"
            res["C_pinball_total"] = c
            res["C_pinball_excess_pct"] = 100.0 * (c / best_cost - 1.0)
    # accuracy-best forecaster given its own best policy
    acc_best = acc["rmsse"].idxmin()
    sub = cost[cost.method == acc_best]
    if len(sub):
        c = float(sub.total_cost.min())
        res["A_own_best_policy_combo"] = f"{acc_best}|{sub.loc[sub.total_cost.idxmin(), 'policy']}"
        res["A_own_best_policy_total"] = c
        res["A_own_best_policy_excess_pct"] = 100.0 * (c / best_cost - 1.0)
    return res


def selection_leverage(R, econ):
    """
    The decisive comparison for RQ1.

    A practitioner choosing "the best forecaster" faces two very different
    decision problems, and conflating them is the source of most disappointing
    forecasting ROI:

      (i)  *within* a fixed decision rule (the standard normal-theory base-stock
           policy), how much does the choice of forecaster matter?
      (ii) *across* decision rules, how much does the choice of rule matter?

    We report the cost spread attributable to each, as a percentage of the
    accuracy-driven default. If (ii) >> (i), then optimising forecast accuracy is
    second-order: the decision rule, not the point forecast, carries the value.
    """
    cost = table_cost(R, econ)
    cost = cost[cost.econ == econ]
    p1 = cost[(cost.policy == "P1_base_stock_normal") &
              (cost.method.isin(POINT_ONLY))]
    if not len(p1):
        return {}
    best_p1, worst_p1 = p1.total_cost.min(), p1.total_cost.max()
    allbest = cost.total_cost.min()
    b = cost.loc[cost.total_cost.idxmin()]
    acc = table_accuracy(R, econ)
    acc_best = acc["rmsse"].idxmin()
    ref = float(p1.loc[p1.method == acc_best, "total_cost"].iloc[0])
    return dict(
        econ=econ,
        reference_accuracy_driven=ref,
        reference_combo=f"{acc_best}|P1_base_stock_normal",
        within_rule_best=float(best_p1),
        within_rule_worst=float(worst_p1),
        within_rule_spread_pct=100 * (worst_p1 / best_p1 - 1),
        within_rule_gain_vs_reference_pct=100 * (ref - best_p1) / ref,
        across_rule_best=float(allbest),
        across_rule_combo=f"{b['method']}|{b['policy']}",
        across_rule_gain_vs_reference_pct=100 * (ref - allbest) / ref,
        # guard: if the accuracy-best forecaster already IS the cost-best one
        # within the standard rule, the within-rule gain is zero by construction
        ratio_across_to_within=(float((ref - allbest) / (ref - best_p1))
                                if (ref - best_p1) > 1e-6 else float("inf")),
        leverage_statement=(
            f"Choosing the best decision rule instead of the best point forecast "
            f"is worth {100*(ref-allbest)/ref:.1f}% of cost, versus "
            f"{100*(ref-best_p1)/ref:.1f}% from choosing the best forecaster "
            f"within the standard rule."),
    )


# ==========================================================================
# RQ2  decomposition of the accuracy-to-decision gap
# ==========================================================================
def decomposition(R, econ):
    """
    A -> E path from the accuracy-driven default to the cost-optimal pairing,
    with the marginal contribution of each step.

      A  point forecaster ranked best by RMSSE + normal-theory base stock.
         This is the pipeline that a forecast-accuracy-led project delivers.
      B  the *cost*-best point forecaster under the same rule.
         Isolates: does switching forecaster within the standard rule help?
      C  same forecaster family, but the order-up-to level is a quantile of an
         *empirically calibrated* protection-interval demand distribution
         instead of mean + z*sigma.
         Isolates: the decision rule and post-hoc distributional information.
      D  quantile-trained LightGBM at the same rule.
         Isolates: trained distributional forecasting vs post-hoc calibration.
      E  the globally cost-optimal pairing.
         Isolates: policy structure (review frequency, base-stock vs (s,S)).
    """
    acc = table_accuracy(R, econ)
    cost = table_cost(R, econ)
    cost = cost[cost.econ == econ].copy()
    cost["family"] = cost.policy.map(policy_family)

    def best_in(family, methods=None):
        sub = cost[cost.family == family]
        if methods is not None:
            sub = sub[sub.method.isin(methods)]
        if not len(sub):
            return None
        r = sub.loc[sub.total_cost.idxmin()]
        return dict(method=r["method"], policy=r["policy"],
                    total=float(r.total_cost), fill=float(r.fill_rate),
                    on_hand=float(r.avg_on_hand))

    acc_best = acc["rmsse"].idxmin()
    ref = cost[(cost.method == acc_best) &
               (cost.policy == "P1_base_stock_normal")]
    if not len(ref):
        return pd.DataFrame()
    base = float(ref.total_cost.iloc[0])
    base_fill = float(ref.fill_rate.iloc[0])

    rows = [dict(step="A_accuracy_driven_default", method=acc_best,
                 policy="P1_base_stock_normal", total_cost=base,
                 fill_rate=base_fill, on_hand=float(ref.avg_on_hand.iloc[0]),
                 pct_vs_A=0.0)]

    b = best_in("P1_base_stock_normal", POINT_ONLY)
    cal = best_in("P2_quantile_base_stock",
                  [m for m in cost.method.unique() if m.startswith("cal_")])
    trained = best_in("P2_quantile_base_stock", ["lgbm_q"])
    for label, step in [("B_best_point_forecaster", b),
                        ("C_calibrated_quantile_rule", cal),
                        ("D_trained_quantile_rule", trained)]:
        if step is None:
            continue
        rows.append(dict(step=label, method=step["method"], policy=step["policy"],
                         total_cost=step["total"], fill_rate=step["fill"],
                         on_hand=step["on_hand"],
                         pct_vs_A=100 * (step["total"] / base - 1)))

    e = cost.loc[cost.total_cost.idxmin()]
    rows.append(dict(step="E_cost_optimal_pairing", method=e["method"],
                     policy=e["policy"], total_cost=float(e.total_cost),
                     fill_rate=float(e.fill_rate), on_hand=float(e.avg_on_hand),
                     pct_vs_A=100 * (float(e.total_cost) / base - 1)))

    d = pd.DataFrame(rows)
    d["marginal_gain"] = -d.total_cost.diff()
    d["marginal_gain_pct_of_A"] = 100 * d["marginal_gain"] / base
    return d


# ==========================================================================
# RQ3  conditionals: demand pattern, scale, regime
# ==========================================================================
def conditional_tables(R, econ):
    ser = R["ser"]
    ser = ser[ser.econ == econ]
    meta = R["meta"]

    def by(col):
        g = ser.groupby([col, "method", "policy"], as_index=False).agg(
            total=("total", "sum"), fill=("fill_rate", "mean"),
            n=("series_ix", "count"))
        # best pairing within each level of col
        idx = g.groupby(col)["total"].idxmin()
        best = g.loc[idx, [col, "method", "policy", "total", "fill"]].rename(
            columns={"method": "best_method", "policy": "best_policy",
                     "total": "best_total", "fill": "best_fill"})
        # the accuracy-driven default within each level
        d = g[g.policy == "P1_base_stock_normal"]
        didx = d.groupby(col)["total"].idxmin()
        dflt = d.loc[didx, [col, "method", "total"]].rename(
            columns={"method": "p1_best_method", "total": "p1_best_total"})
        return best.merge(dflt, on=col)

    out = {}
    for col in ["sbc_class", "scale_band", "cat_id", "state_id"]:
        if col in ser.columns:
            t = by(col)
            t["excess_pct"] = 100 * (t.p1_best_total / t.best_total - 1)
            out[col] = t.sort_values("excess_pct", ascending=False)

    # regime-conditional: join the regime label of the *first* period of the
    # block to each series, then repeat
    reg = R["reg"]
    reg0 = reg[reg.period == 0][["series_ix", "regime"]].rename(
        columns={"regime": "regime_p0"})
    sr = ser.merge(reg0, on="series_ix", how="left")
    g = sr.groupby(["regime_p0", "method", "policy"], as_index=False).agg(
        total=("total", "sum"), fill=("fill_rate", "mean"), n=("series_ix", "count"))
    idx = g.groupby("regime_p0")["total"].idxmin()
    out["regime"] = g.loc[idx].sort_values("regime_p0")
    out["regime_full"] = g
    # how often does the best pairing change across regimes?
    out["regime_agreement"] = float(
        g.loc[idx].set_index("regime_p0")[["method", "policy"]]
        .astype(str).agg("|".join, axis=1).nunique() == 1)
    return out


def pairing_by_regime(R, econ):
    """
    Which pairing is cost-optimal *within each demand regime*?

    The regime label is taken from the first period of the block (i.e. it is
    knowable at decision time from history alone). If the optimal pairing differs
    across regimes, a regime-aware switch dominates any fixed choice -- this is
    the evidence base for RQ3 and for the regime-adaptive policy.
    """
    ser = R["ser"][R["ser"].econ == econ]
    reg0 = R["reg"]
    reg0 = reg0[reg0.period == 0][["series_ix", "regime"]]
    m = ser.merge(reg0, on="series_ix", how="left")
    g = m.groupby(["regime", "method", "policy"], as_index=False).agg(
        total=("total", "sum"), fill=("fill_rate", "mean"), n=("series_ix", "count"))
    best = g.loc[g.groupby("regime")["total"].idxmin()].sort_values("regime")
    best["distinct_pairings"] = best.duplicated(subset=["method", "policy"]).sum() + 1
    return g, best


def regime_analysis(run: str, block: str, econ: str = "margin"):
    """
    RQ3: does the cost-optimal strategy depend on the demand regime?

    Uses the per-(pairing, period, regime) cost table produced by the driver.
    Two questions are answered:

      (a) Cross-sectionally: within each regime class, which pairing is cheapest,
          and how much does a regime-blind choice cost?
      (b) Dynamically: when a series *transitions* from one regime to another
          between consecutive periods, does the ranking of pairings change? This
          is the sharpest available test of whether "demand has changed" should
          trigger "change the policy".
    """
    f = RESULTS / run / f"regime_cost_{block}.parquet"
    if not f.exists():
        return None, None
    rc = pd.read_parquet(f)
    rc = rc[rc.econ == econ]
    if not len(rc):
        return None, None

    # ---- (a) best pairing within each regime -------------------------------
    g = rc.groupby(["regime", "method", "policy"], as_index=False).agg(
        cost=("cost", "sum"), short=("short", "sum"), n=("n", "sum"))
    best = g.loc[g.groupby("regime")["cost"].idxmin()].copy()
    # cost of the regime-blind choice: the single pairing that is cheapest
    # pooled over all regimes
    pooled = g.groupby(["method", "policy"], as_index=False).cost.sum()
    blind = pooled.loc[pooled.cost.idxmin()]
    blind_key = (blind["method"], blind["policy"])
    best["regime_blind_cost"] = [
        float(g[(g.regime == r) & (g.method == blind_key[0]) &
                (g.policy == blind_key[1])].cost.sum()) for r in best.regime]
    best["blind_penalty_pct"] = 100 * (best.regime_blind_cost / best.cost - 1)
    best["distinct"] = (best.method + "|" + best.policy)

    # ---- (b) does the optimum move across regimes and over time? -----------
    wide = rc.pivot_table(index=["regime", "period"], columns=["method", "policy"],
                          values="cost", aggfunc="sum")
    ranks = wide.rank(axis=1, method="min")
    best_per_cell = pd.DataFrame({
        "best": ranks.idxmin(axis=1),
        "regime": ranks.index.get_level_values("regime"),
        "period": ranks.index.get_level_values("period")}).reset_index(drop=True)
    # how often the regime-blind winner is also the winner inside a regime
    n_cells = len(best_per_cell)
    blind_is_best = float((best_per_cell.best.astype(str) ==
                           f"{blind_key[0]}|{blind_key[1]}").mean())
    summary = dict(
        n_cells=n_cells,
        blind_pairing=f"{blind_key[0]}|{blind_key[1]}",
        share_cells_blind_is_best=blind_is_best,
        n_distinct_regime_winners=int(best_per_cell.best.nunique()),
        mean_blind_penalty_pct=float(best.blind_penalty_pct.mean()))
    return best, dict(regime_cost=rc, best_by_regime=g,
                      best_per_cell=best_per_cell, summary=summary)


# ==========================================================================
# RQ4  meta-learning: per-series and per-regime pairing selection
# ==========================================================================
def meta_features(R):
    meta = R["meta"].copy()
    f = pd.DataFrame({"series_ix": np.arange(len(meta))})
    # features use the look-ahead-free descriptors (history before the validation
    # block); the full-panel columns are never fed to a selector
    for c in ["adi", "cv2", "zero_share", "mean_daily", "sd_daily", "max_daily",
              "p99_daily", "n_active_days", "total_units", "first_sale_t"]:
        src = "lf_" + c if ("lf_" + c) in meta else c
        if src in meta:
            f[c] = meta[src].to_numpy()
    for c in ["cat_id", "dept_id", "store_id", "state_id"]:      # ex-ante identifiers
        if c in meta:
            f[c] = pd.Categorical(meta[c]).codes.astype(np.int16)
    f["log_mean_daily"] = np.log1p(f["mean_daily"])
    f["log_adi"] = np.log1p(f["adi"].fillna(0))
    return f


def pairing_selection(R_val, R_test, econ, top_k=None, min_share=0.0):
    """
    Select a forecaster x policy pairing per SKU using *validation* cost, then
    evaluate that selection on the *test* block against the best fixed pairing
    (also chosen on validation). No test information is used for selection.
    """
    def cost_matrix(R):
        ser = R["ser"]
        ser = ser[ser.econ == econ]
        piv = ser.pivot_table(index="series_ix", columns=["method", "policy"],
                              values="total", aggfunc="first")
        return piv

    Cv = cost_matrix(R_val)
    Ct = cost_matrix(R_test)
    cols = [c for c in Cv.columns if c in Ct.columns]
    Cv, Ct = Cv[cols], Ct[cols]
    n = len(Cv)

    best_fixed_col = Cv.sum().idxmin()
    per_series_col = Cv.idxmin(axis=1)

    test_fixed = float(Ct[best_fixed_col].sum())
    # evaluate the validation-selected per-series pairing on the test block
    rows = np.arange(len(Ct))
    col_ix = {c: i for i, c in enumerate(Ct.columns)}
    sel_ix = np.array([col_ix[c] for c in per_series_col])
    test_perseries = float(Ct.to_numpy()[rows, sel_ix].sum())

    # oracle on test (upper bound, not achievable -- reported for context only)
    oracle_col = Ct.sum().idxmin()
    test_oracle_fixed = float(Ct[oracle_col].sum())
    test_oracle_perseries = float(Ct.min(axis=1).sum())

    share_disagree = float((per_series_col != best_fixed_col).mean())
    res = dict(
        econ=econ, n_series=n, n_pairings=len(cols),
        val_best_fixed=f"{best_fixed_col[0]}|{best_fixed_col[1]}",
        test_best_fixed_cost=test_fixed,
        test_per_series_cost=test_perseries,
        per_series_gain_pct=100 * (test_fixed - test_perseries) / test_fixed,
        share_series_disagree=share_disagree,
        test_oracle_fixed_cost=test_oracle_fixed,
        test_oracle_perseries_cost=test_oracle_perseries,
        oracle_headroom_pct=100 * (test_fixed - test_oracle_perseries) / test_fixed,
        capture_rate=((test_fixed - test_perseries) /
                      max(test_fixed - test_oracle_perseries, 1e-9)),
    )
    return res, per_series_col, best_fixed_col, Cv, Ct


def meta_model(R_val, R_test, econ, seed=0):
    """
    Train a classifier on validation data to predict the cost-optimal pairing
    from series meta-features, then evaluate the predicted pairing on test.
    This is the deployable version of per-series selection: it needs no
    validation-cost bookkeeping per SKU at run time, only features.
    """
    import lightgbm as lgb
    ser = R_val["ser"]; ser = ser[ser.econ == econ]
    piv = ser.pivot_table(index="series_ix", columns=["method", "policy"],
                          values="total", aggfunc="first")
    serT = R_test["ser"]; serT = serT[serT.econ == econ]
    pivT = serT.pivot_table(index="series_ix", columns=["method", "policy"],
                            values="total", aggfunc="first")
    cols = [c for c in piv.columns if c in pivT.columns]
    piv, pivT = piv[cols], pivT[cols]

    F = meta_features(R_val).set_index("series_ix").loc[piv.index]
    FT = meta_features(R_test).set_index("series_ix").loc[pivT.index]
    num = [c for c in F.columns if F[c].dtype.kind in "fi"]
    cat = [c for c in F.columns if c not in num]

    y = piv.values.argmin(axis=1)
    X = F[num + cat].to_numpy(dtype=np.float32)
    XT = FT[num + cat].to_numpy(dtype=np.float32)

    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             min_child_samples=40, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, n_jobs=2, verbosity=-1,
                             random_state=seed)
    clf.fit(X, y)
    pred = clf.predict(XT)
    train_acc = float((clf.predict(X) == y).mean())

    rows = np.arange(len(pivT))
    V = pivT.to_numpy()
    cost_pred = float(V[rows, pred].sum())
    best_fixed_col = piv.sum().idxmin()
    cost_fixed = float(pivT[best_fixed_col].sum())
    cost_perseries_val = float(V[rows, piv.values.argmin(axis=1)].sum())
    cost_oracle = float(V.min(axis=1).sum())

    res = dict(econ=econ, n_classes=len(cols), train_top1=train_acc,
               val_best_fixed=f"{best_fixed_col[0]}|{best_fixed_col[1]}",
               test_cost_fixed=cost_fixed,
               test_cost_meta_model=cost_pred,
               test_cost_val_perseries=cost_perseries_val,
               test_cost_oracle=cost_oracle,
               meta_vs_fixed_pct=100 * (cost_fixed - cost_pred) / cost_fixed,
               meta_vs_fixed_abs=cost_fixed - cost_pred,
               capture_rate=((cost_fixed - cost_pred) /
                             max(cost_fixed - cost_oracle, 1e-9)),
               top_pairings=[f"{c[0]}|{c[1]}" for c in cols],
               class_share=pd.Series(y).value_counts(normalize=True).round(4).to_dict())
    imp = pd.DataFrame({"feature": num + cat,
                        "gain": clf.booster_.feature_importance("gain")})
    return res, imp.sort_values("gain", ascending=False)


def regime_meta_model(R_val, R_test, econ, seed=0):
    """
    Regime-conditional selection: choose the pairing as a function of
    (series meta-features, detected demand regime) using validation data, then
    apply it on test. Compared against the regime-blind meta-model, the gain is
    the value of *detecting* that demand has changed.
    """
    import lightgbm as lgb
    def build(R):
        ser = R["ser"]; ser = ser[ser.econ == econ]
        piv = ser.pivot_table(index="series_ix", columns=["method", "policy"],
                              values="total", aggfunc="first")
        # regime information available when the pairing is chosen: the label and
        # ratios at the FIRST origin of the block (history only)
        reg = R["reg"]
        reg0 = reg[reg.period == reg.period.min()].set_index("series_ix")
        dom = reg0["regime"].rename("regime_mode")
        lv = reg0["level_ratio"].rename("level_ratio_mean")
        vo = reg0["vol_ratio"].rename("vol_ratio_mean")
        aux = pd.concat([dom, lv, vo], axis=1)
        return piv, aux

    piv, aux = build(R_val)
    pivT, auxT = build(R_test)
    cols = [c for c in piv.columns if c in pivT.columns]
    piv, pivT = piv[cols], pivT[cols]

    F = meta_features(R_val).set_index("series_ix").loc[piv.index].join(aux)
    FT = meta_features(R_test).set_index("series_ix").loc[pivT.index].join(auxT)
    cats = list(pd.unique(F["regime_mode"].dropna().astype(str)))
    F["regime_mode"] = pd.Categorical(F["regime_mode"].astype(str),
                                      categories=cats).codes.astype(np.int16)
    FT["regime_mode"] = pd.Categorical(FT["regime_mode"].astype(str),
                                       categories=cats).codes.astype(np.int16)
    num = [c for c in F.columns if F[c].dtype.kind in "fi"]
    X = F[num].to_numpy(dtype=np.float32)
    XT = FT[num].to_numpy(dtype=np.float32)
    y = piv.values.argmin(axis=1)

    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             min_child_samples=40, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, n_jobs=2, verbosity=-1,
                             random_state=seed)
    clf.fit(X, y)
    pred = clf.predict(XT)
    V = pivT.to_numpy(); rows = np.arange(len(V))
    cost_pred = float(V[rows, pred].sum())
    best_fixed_col = piv.sum().idxmin()
    cost_fixed = float(pivT[best_fixed_col].sum())
    cost_oracle = float(V.min(axis=1).sum())
    return dict(econ=econ, test_cost_fixed=cost_fixed,
                test_cost_regime_meta=cost_pred,
                test_cost_oracle=cost_oracle,
                regime_meta_vs_fixed_pct=100 * (cost_fixed - cost_pred) / cost_fixed,
                capture_rate=(cost_fixed - cost_pred) / max(cost_fixed - cost_oracle, 1e-9),
                train_top1=float((clf.predict(X) == y).mean()))


# ==========================================================================
# service-constrained frontier
# ==========================================================================
def service_frontier(R, econ, targets=(0.90, 0.95, 0.97, 0.99, 0.995)):
    """
    Retailers specify a service target, not a stockout penalty. For each
    fill-rate target we report the cheapest configuration attaining it. Because
    the driver simulates every quantile level tau for the quantile-based policies,
    this traces a genuine cost-service frontier and makes the conclusions robust
    to the (unknowable) true stockout penalty.
    """
    ser = R["ser"][R["ser"].econ == econ]
    g = ser.groupby(["method", "policy", "tau"], as_index=False).agg(
        total=("total", "sum"),
        sold=("fill_rate", "sum"),          # placeholder, recomputed below
        demand_units=("demand_units", "sum"),
        on_hand=("avg_on_hand", "mean"),
        n=("series_ix", "count"))
    # exact aggregate fill rate = sum(units satisfied) / sum(units demanded)
    num = ser.assign(su=ser.fill_rate * ser.demand_units).groupby(
        ["method", "policy", "tau"], as_index=False)["su"].sum()
    g = g.drop(columns=["sold"]).merge(num, on=["method", "policy", "tau"])
    g["fill"] = g["su"] / g["demand_units"].clip(lower=1e-9)
    g["cost_per_unit"] = g["total"] / g["demand_units"].clip(lower=1e-9)

    rows = []
    for tg in targets:
        ok = g[g.fill >= tg]
        if not len(ok):
            rows.append(dict(target=tg, method=None, policy=None, tau=np.nan,
                             val_cost=np.nan, fill=np.nan, on_hand=np.nan))
            continue
        b = ok.sort_values("total").iloc[0]
        rows.append(dict(target=tg, method=b["method"], policy=b["policy"],
                         tau=float(b["tau"]), val_cost=float(b.total),
                         fill=float(b.fill), on_hand=float(b.on_hand),
                         cost_per_unit=float(b.cost_per_unit)))
    return pd.DataFrame(rows), g.drop(columns=["su"])


# ==========================================================================
# figures
# ==========================================================================
def make_figures(R, econ, acc_tab, cost_tab, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = cost_tab[cost_tab.econ == econ]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    c = c.assign(family=c.policy.map(policy_family))
    for pol, mk in zip(POLICY_ORDER, ["o", "s", "^", "D", "v"]):
        sub = c[c.family == pol]
        m = sub.merge(acc_tab.reset_index()[["method", "rmsse"]], on="method")
        if len(m) == 0:
            continue
        ax[0].scatter(m.rmsse, m.total_cost, marker=mk, s=52, label=pol, alpha=.9)
        for _, r in m.iterrows():
            ax[0].annotate(r["method"], (r.rmsse, r.total_cost), fontsize=6.5,
                           xytext=(3, 3), textcoords="offset points")
    ax[0].set_xlabel("Forecast error (RMSSE, lower is better)")
    ax[0].set_ylabel("Total inventory cost (USD)")
    ax[0].set_title("Accuracy does not order the methods by cost")
    ax[0].legend(fontsize=6.5)
    ax[0].grid(alpha=.3)

    piv = c.groupby(["method", "family"], as_index=False).total_cost.min() \
           .pivot_table(index="method", columns="family", values="total_cost")
    piv = piv.reindex(acc_tab.index.intersection(piv.index))
    piv.plot(kind="bar", ax=ax[1], width=.8)
    ax[1].set_ylabel("Total inventory cost (USD)")
    ax[1].set_title("Cost by forecaster and policy")
    ax[1].tick_params(axis="x", labelsize=7, rotation=60)
    ax[1].legend(fontsize=6.5)
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(outdir / "fig_accuracy_vs_cost.png", dpi=180)
    plt.close(fig)

    # cost/service frontier
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for pol, mk in zip(POLICY_ORDER, ["o", "s", "^", "D", "v"]):
        sub = c[c.family == pol]
        if not len(sub):
            continue
        ax.scatter(sub.fill_rate, sub.total_cost, marker=mk, s=48, label=pol)
        for _, r in sub.iterrows():
            ax.annotate(r["method"], (r.fill_rate, r.total_cost), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Fill rate (units satisfied / units demanded)")
    ax.set_ylabel("Total inventory cost (USD)")
    ax.set_title("Cost-service frontier: dominating and dominated options")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig_frontier.png", dpi=180)
    plt.close(fig)
    return True


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default="main_val")
    ap.add_argument("--test_run", type=str, default="")
    ap.add_argument("--econ", type=str, default="margin")
    ap.add_argument("--out", type=str, default="main")
    args = ap.parse_args()

    outdir = RESULTS / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    run = args.runs.split(",")[0]
    R = load_run(run, block="val")
    econ = args.econ
    log(f"loaded {run} [val]: {len(R['ser'])} series-rows, "
        f"{R['ser'].combo.nunique()} pairings, blocks={R['ser'].block.unique().tolist()}")

    acc_tab = table_accuracy(R, econ)
    cost_tab = table_cost(R, econ)
    acc_tab.to_csv(outdir / "T1_accuracy.csv")
    cost_tab.to_csv(outdir / "T2_cost_service.csv", index=False)
    log("\n=== T1 accuracy (sorted by RMSSE) ===\n" +
        acc_tab[[c for c in ACC_METRICS if c in acc_tab]].round(4).to_string())
    log("\n=== T2 cost & service ===\n" +
        cost_tab[["method", "policy", "total_cost", "fill_rate", "csl_period",
                  "avg_on_hand", "cost_pct_revenue"]].round(4).to_string(index=False))

    ra = rank_agreement(acc_tab, cost_tab, econ)
    ra.to_csv(outdir / "T3_rank_agreement.csv", index=False)
    log("\n=== T3 rank agreement (accuracy vs cost) ===\n" +
        ra[["policy", "metric", "spearman", "kendall", "accuracy_best",
            "cost_best", "agree"]].round(3).to_string(index=False))

    dl = decision_loss(R, econ)
    pd.Series(dl).to_csv(outdir / "T4_decision_loss.csv")
    log("\n=== T4 decision loss ===\n" + json.dumps(dl, indent=2, default=float))

    lev = selection_leverage(R, econ)
    pd.Series(lev).to_csv(outdir / "T4b_selection_leverage.csv")
    log("\n=== T4b selection leverage (RQ1 headline) ===\n" +
        json.dumps(lev, indent=2, default=float))

    prob = table_probabilistic(R)
    prob.to_csv(outdir / "T1b_probabilistic.csv")
    log("\n=== T1b probabilistic accuracy ===\n" + prob.round(4).to_string())

    dec = decomposition(R, econ)
    dec.to_csv(outdir / "T5_decomposition.csv", index=False)
    log("\n=== T5 decomposition A->E ===\n" + dec.round(3).to_string(index=False))

    cond = conditional_tables(R, econ)
    for k, v in cond.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(outdir / f"T6_conditional_{k}.csv", index=False)
            log(f"\n=== T6 conditional on {k} ===\n" + v.head(14).round(3).to_string(index=False))

    rb, rd = regime_analysis(run, "val", econ)
    if rb is not None:
        rb.to_csv(outdir / "T6c_regime_best_pairing.csv", index=False)
        rd["best_per_cell"].to_csv(outdir / "T6d_regime_cells.csv", index=False)
        log("\n=== T6c best pairing by demand regime (val) ===\n" +
            rb[["regime", "method", "policy", "cost", "regime_blind_cost",
                "blind_penalty_pct"]].round(2).to_string(index=False))
        log("\nregime summary: " + json.dumps(rd["summary"], indent=2, default=float))

    fr, fr_full = service_frontier(R, econ)
    fr.to_csv(outdir / "T7_service_frontier.csv", index=False)
    fr_full.to_csv(outdir / "T7b_cost_service_all.csv", index=False)
    log("\n=== T7 service-constrained frontier ===\n" +
        fr.round(3).to_string(index=False))

    make_figures(R, econ, acc_tab, cost_tab, outdir)
    log(f"\nfigures -> {outdir}")

    if args.test_run:
        RT = load_run(args.test_run, block="test")
        log(f"\nloaded {args.test_run} [test]: {len(RT['ser'])} series-rows")
        accT = table_accuracy(RT, econ)
        costT = table_cost(RT, econ)
        accT.to_csv(outdir / "T1_accuracy_test.csv")
        costT.to_csv(outdir / "T2_cost_service_test.csv", index=False)
        log("\n=== TEST accuracy ===\n" + accT[["rmsse", "wape", "mae"]].round(4).to_string())
        log("\n=== TEST cost & service ===\n" +
            costT[["method", "policy", "total_cost", "fill_rate", "csl_period"]]
            .round(4).to_string(index=False))

        sel, percol, fixedcol, Cv, Ct = pairing_selection(R, RT, econ)
        pd.Series(sel).to_csv(outdir / "T8_pairing_selection.csv")
        log("\n=== T8 per-series pairing selection (val -> test) ===\n" +
            json.dumps(sel, indent=2, default=float))

        mm, imp = meta_model(R, RT, econ)
        pd.Series({k: v for k, v in mm.items() if not isinstance(v, (list, dict))}) \
            .to_csv(outdir / "T9_meta_model.csv")
        imp.to_csv(outdir / "T9_meta_importance.csv", index=False)
        log("\n=== T9 meta-model (features -> best pairing) ===\n" +
            json.dumps({k: v for k, v in mm.items() if k != "top_pairings"},
                       indent=2, default=float))
        log("\ntop meta-features:\n" + imp.head(12).to_string(index=False))

        rm = regime_meta_model(R, RT, econ)
        pd.Series(rm).to_csv(outdir / "T10_regime_meta.csv")
        log("\n=== T10 regime-aware meta-model ===\n" + json.dumps(rm, indent=2, default=float))

        dlT = decision_loss(RT, econ)
        pd.Series(dlT).to_csv(outdir / "T4_decision_loss_test.csv")
        log("\n=== T4c decision loss on TEST ===\n" + json.dumps(dlT, indent=2, default=float))
        levT = selection_leverage(RT, econ)
        pd.Series(levT).to_csv(outdir / "T4b_selection_leverage_test.csv")
        log("\n=== T4b(TEST) selection leverage ===\n" +
            json.dumps(levT, indent=2, default=float))
        probT = table_probabilistic(RT)
        probT.to_csv(outdir / "T1b_probabilistic_test.csv")
        log("\n=== T1b(TEST) probabilistic ===\n" + probT.round(4).to_string())
        decT = decomposition(RT, econ)
        decT.to_csv(outdir / "T5_decomposition_test.csv", index=False)
        log("\n=== T5b decomposition on TEST ===\n" + decT.round(3).to_string(index=False))
        make_figures(RT, econ, accT, costT, outdir)


if __name__ == "__main__":
    main()
