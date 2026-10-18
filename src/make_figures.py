"""
make_figures.py
===============
Publication-quality figures for the manuscript. Reads only the CSV/JSON
artefacts written by analysis.py, selection.py, bootstrap.py and collect.py, so
the figures can never drift from the tables.

  fig1_leverage.png       the headline: forecaster choice vs decision-rule choice
  fig2_decomposition.png  the A->E waterfall with bootstrap CIs
  fig3_frontier.png       cost-service frontier over all 59 pairings
  fig4_selection.png      the eight selection strategies, val-B vs test
  fig5_sensitivity.png    default excess cost across the decision-environment suite
  fig6_accuracy_scatter.png  RMSSE vs cost, coloured by policy family
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS                                     # noqa: E402
from analysis import policy_family                            # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
    "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

FAM_COLOR = {"P1_base_stock_normal": "#c0392b",
             "P2_quantile_base_stock": "#2980b9",
             "P3_sS_weekly": "#27ae60",
             "P3_sS_daily": "#16a085",
             "P4_ddmrp": "#8e44ad"}
FAM_LABEL = {"P1_base_stock_normal": "P1 base stock, normal safety stock",
             "P2_quantile_base_stock": "P2 base stock at CR quantile",
             "P3_sS_weekly": "P3 (s,S), weekly review",
             "P3_sS_daily": "P3 (s,S), daily review",
             "P4_ddmrp": "P4 DDMRP-style buffer"}


def _lev(run, block):
    f = RESULTS / run / f"T4b_selection_leverage{'_test' if block == 'test' else ''}.csv"
    s = pd.read_csv(f, index_col=0).iloc[:, 0]
    # the value column mixes floats and strings, so coerce per key
    out = {}
    for k, v in s.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def fig1(run="main"):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    blocks = ["val", "test"]
    within = [_lev(run, b)["within_rule_gain_vs_reference_pct"] for b in blocks]
    across = [_lev(run, b)["across_rule_gain_vs_reference_pct"] for b in blocks]
    x = np.arange(2); w = 0.34
    b1 = ax.bar(x - w/2, within, w, label="Choose a better forecaster\n(inside the standard rule)",
                color="#95a5a6")
    b2 = ax.bar(x + w/2, across, w, label="Choose a better decision rule",
                color="#c0392b")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.1f}%", (r.get_x() + r.get_width()/2,
                                                   r.get_height()),
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["Validation block", "Test block"])
    ax.set_ylabel("Cost reduction vs the accuracy-driven default (%)")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=.95)
    ax.set_ylim(0, max(across) * 1.28)
    fig.savefig(RESULTS / run / "fig1_leverage.png")
    plt.close(fig)


def fig2(run="main"):
    bs = json.load(open(RESULTS / run / "T12_bootstrap.json"))
    dec = pd.read_csv(RESULTS / run / "T5_decomposition_test.csv")
    labels = ["A\naccuracy-\ndriven\ndefault", "B\nbest point\nforecaster",
              "C\ncalibrated\nquantile rule", "D\ntrained\nquantile rule",
              "E\ncost-optimal\npairing"]
    vals = dec.total_cost.to_numpy()
    step = list(bs["test"]["steps"].values())
    ci = {s["combo"]: s["ci95_pct"] for s in step}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                                  gridspec_kw=dict(width_ratios=[1.25, 1]))
    cols = ["#7f8c8d", "#95a5a6", "#c0392b", "#2980b9", "#16a085"]
    ax.bar(range(5), vals, color=cols)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:,.0f}", (i, v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Total inventory cost, TEST block (USD)")
    ax.set_ylim(0, max(vals) * 1.16)

    names = ["B vs A", "C vs A", "D vs A", "E vs A"]
    # saving = -(change of the step's cost relative to A); CI endpoints swap sign
    pts = [-s["change_b_vs_a_pct"] for s in step]
    lo = [-s["ci95_change_b_vs_a_pct"][1] for s in step]
    hi = [-s["ci95_change_b_vs_a_pct"][0] for s in step]
    y = np.arange(len(pts))[::-1]
    ax2.errorbar(pts, y, xerr=[np.array(pts) - np.array(lo), np.array(hi) - np.array(pts)],
                 fmt="o", color="#2c3e50", ecolor="#95a5a6", elinewidth=2, capsize=3, ms=5)
    ax2.axvline(0, color="#c0392b", lw=1.2, ls="--")
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("Cost saving vs the accuracy-driven default (%)\nwith 95% paired bootstrap CI")
    fig.savefig(RESULTS / run / "fig2_decomposition.png")
    plt.close(fig)


def fig3(run="main"):
    c = pd.read_csv(RESULTS / run / "T2_cost_service_test.csv")
    c = c[c.econ == "margin"].copy()
    c["family"] = c.policy.map(policy_family)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for fam, g in c.groupby("family"):
        ax.scatter(g.fill_rate, g.total_cost, s=34, alpha=.85,
                   color=FAM_COLOR.get(fam, "#333"), label=FAM_LABEL.get(fam, fam),
                   edgecolor="white", linewidth=.5)
    # mark the accuracy-driven default and the optimum
    dflt = c[(c.method == "lgbm_mean") & (c.policy == "P1_base_stock_normal")]
    if len(dflt):
        ax.scatter(dflt.fill_rate, dflt.total_cost, s=190, facecolor="none",
                   edgecolor="#c0392b", linewidth=2.0, zorder=5,
                   label="accuracy-driven default")
    b = c.loc[c.total_cost.idxmin()]
    ax.scatter([b.fill_rate], [b.total_cost], s=190, facecolor="none",
               edgecolor="#16a085", linewidth=2.0, zorder=5, label="cost optimum")
    ax.set_xlabel("Fill rate (units satisfied / units demanded)")
    ax.set_ylabel("Total inventory cost, TEST block (USD)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="upper left", framealpha=.95)
    fig.savefig(RESULTS / run / "fig3_frontier.png")
    plt.close(fig)


def fig4(run="main"):
    t = pd.read_csv(RESULTS / run / "T11_selection_strategies.csv")
    keep = [s for s in t.strategy if not s.startswith("S9")]
    t = t[t.strategy.isin(keep)].copy()
    t["lbl"] = t.strategy.str.replace(r"^S\d_", "", regex=True).str.replace("_", " ")
    order = t.sort_values("test_cost").lbl.tolist()
    t = t.set_index("lbl").loc[order].reset_index()
    x = np.arange(len(t)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    base_v = float(t.loc[t.strategy == "S1_fixed_pooled_best", "valB_cost"].iloc[0])
    base_t = float(t.loc[t.strategy == "S1_fixed_pooled_best", "test_cost"].iloc[0])
    ax.bar(x - w/2, 100 * (t.valB_cost / base_v - 1), w, color="#7f8c8d",
           label="held-out half of validation")
    ax.bar(x + w/2, 100 * (t.test_cost / base_t - 1), w, color="#2c3e50",
           label="test block")
    ax.axhline(0, color="#c0392b", lw=1.1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(t.lbl, rotation=32, ha="right", fontsize=7.5)
    ax.set_ylabel("Cost vs the best fixed pairing (%)\nlower is better")
    ax.legend(fontsize=7.5)
    fig.savefig(RESULTS / run / "fig4_selection.png")
    plt.close(fig)


def fig5():
    s = pd.read_csv(RESULTS / "S1_sensitivity.csv")
    s = s[s.block == "test"].copy()
    s["lbl"] = s.config.str.replace("^sens_", "", regex=True).replace("main", "baseline")
    s = s.sort_values("default_excess_pct")
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(x, s.default_excess_pct, color="#c0392b", alpha=.88,
           label="excess cost of the accuracy-driven default")
    ax.bar(x, s.across_rule_gain_pct, color="#2980b9", alpha=.88,
           label="gain from choosing the decision rule on cost")
    ax.bar(x, s.within_rule_gain_pct, color="#7f8c8d", alpha=.95,
           label="gain from choosing the forecaster on accuracy")
    for i, v in enumerate(s.default_excess_pct):
        ax.annotate(f"{v:.0f}%", (i, v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(s.lbl, rotation=32, ha="right", fontsize=7.5)
    ax.set_ylabel("Percent")
    ax.legend(fontsize=7.2, loc="upper left")
    fig.savefig(RESULTS / "fig5_sensitivity.png")
    plt.close(fig)


def fig6(run="main"):
    acc = pd.read_csv(RESULTS / run / "T1_accuracy_test.csv", index_col=0)
    cost = pd.read_csv(RESULTS / run / "T2_cost_service_test.csv")
    cost["family"] = cost.policy.map(policy_family)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for fam, g in cost.groupby("family"):
        m = g.merge(acc.reset_index()[["method", "rmsse"]], on="method")
        if not len(m):
            continue
        ax.scatter(m.rmsse, m.total_cost, s=40, alpha=.85,
                   color=FAM_COLOR.get(fam, "#333"), label=FAM_LABEL.get(fam, fam),
                   edgecolor="white", linewidth=.5)
    p1 = cost[cost.family == "P1_base_stock_normal"].merge(
        acc.reset_index()[["method", "rmsse"]], on="method")
    for _, r in p1.iterrows():
        ax.annotate(r["method"], (r.rmsse, r.total_cost), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points", color="#555")
    ax.set_xlabel("Forecast error (RMSSE, lower is better) — TEST block")
    ax.set_ylabel("Total inventory cost (USD)")
    ax.legend(fontsize=6.8, loc="center left", framealpha=.95)
    fig.savefig(RESULTS / run / "fig6_accuracy_scatter.png")
    plt.close(fig)


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "main"
    for fn in [fig1, fig2, fig3, fig4, fig6]:
        try:
            fn(run)
            print(f"ok {fn.__name__}")
        except Exception as e:                              # noqa: BLE001
            print(f"!! {fn.__name__}: {type(e).__name__}: {e}")
    try:
        fig5()
        print("ok fig5")
    except Exception as e:                                  # noqa: BLE001
        print(f"!! fig5: {type(e).__name__}: {e}")
