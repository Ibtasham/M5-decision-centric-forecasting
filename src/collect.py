"""
collect.py
==========
Assemble the paper's headline numbers from every experiment directory into a
single machine-readable summary. Run after run_experiment.py / analysis.py.

Produces results/paper_numbers.json plus two tidy CSVs:
  S1_sensitivity.csv   one row per configuration: cost-optimal pairing, the
                       accuracy-driven default's cost, and the excess it incurs
  S2_ranking_stability.csv  forecaster accuracy ranks per block, to show that
                       the accuracy ranking is itself unstable
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS                                     # noqa: E402
from analysis import (load_run, table_accuracy, table_cost, decision_loss,
                      selection_leverage, decomposition, policy_family,
                      POINT_ONLY)                              # noqa: E402

CONFIG_LABELS = {
    "main":            "baseline (L=2d, K=$0.50, R=7d, hold 25%, cost 60%, margin penalty)",
    "sens_lead7":      "lead time 7 days",
    "sens_lead14":     "lead time 14 days",
    "sens_K5":         "fixed order cost $5.00 per line",
    "sens_review14":   "review interval 14 days",
    "sens_holdrate40": "annual holding rate 40%",
    "sens_costratio75": "purchase cost 75% of price (25% margin)",
    "sens_penmult3":   "stockout penalty = 3x forgone margin",
    "sens_sfrac30":    "(s,S) reorder point 0.30 x S",
    "sens_ddmrp80":    "DDMRP buffer 0.80 x ADG x (L+R)",
    "seed1":           "independent sampling seed (robustness replication)",
}


def log(m):
    print(m, flush=True)


def one_config(run: str, econ: str = "margin", block: str = "test"):
    try:
        R = load_run(run, block=block)
    except Exception as e:                                  # noqa: BLE001
        log(f"  !! {run}: {type(e).__name__} {e}")
        return None
    cost = table_cost(R, econ)
    cost = cost[cost.econ == econ]
    if not len(cost):
        return None
    best = cost.loc[cost.total_cost.idxmin()]
    dl = decision_loss(R, econ)
    lev = selection_leverage(R, econ)
    acc = table_accuracy(R, econ)
    d = dict(
        config=run, label=CONFIG_LABELS.get(run, run), econ=econ, block=block,
        n_series=int(R["ser"].series_ix.nunique()),
        best_combo=f"{best['method']}|{best['policy']}",
        best_family=policy_family(best["policy"]),
        best_cost=float(best.total_cost),
        best_fill=float(best.fill_rate),
        best_csl=float(best.csl_period),
        best_on_hand=float(best.avg_on_hand),
        best_cost_pct_revenue=float(best.cost_pct_revenue),
        default_combo=dl.get("A_rmsse_combo"),
        default_cost=dl.get("A_rmsse_total"),
        default_fill=float(cost[(cost.method == dl.get("A_rmsse_combo", "").split("|")[0]) &
                                (cost.policy == "P1_base_stock_normal")].fill_rate.iloc[0])
        if len(cost[(cost.method == dl.get("A_rmsse_combo", "").split("|")[0]) &
                    (cost.policy == "P1_base_stock_normal")]) else np.nan,
        default_excess_pct=dl.get("A_rmsse_excess_pct"),
        within_rule_spread_pct=lev.get("within_rule_spread_pct"),
        within_rule_gain_pct=lev.get("within_rule_gain_vs_reference_pct"),
        across_rule_gain_pct=lev.get("across_rule_gain_vs_reference_pct"),
        accuracy_best_rmsse=acc["rmsse"].idxmin(),
        accuracy_best_wape=acc["wape"].idxmin(),
        accuracy_best_mae=acc["mae"].idxmin(),
    )
    # is the accuracy-best forecaster also the cost-best under the standard rule?
    p1 = cost[(cost.policy == "P1_base_stock_normal") & cost.method.isin(POINT_ONLY)]
    d["cost_best_under_P1"] = p1.loc[p1.total_cost.idxmin(), "method"] if len(p1) else None
    d["accuracy_cost_agree"] = bool(d["accuracy_best_rmsse"] == d["cost_best_under_P1"])
    return d


def main():
    rows = []
    for run in CONFIG_LABELS:
        if not (RESULTS / run).exists():
            continue
        for block in ["val", "test"]:
            r = one_config(run, "margin", block)
            if r:
                rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(RESULTS / "S1_sensitivity.csv", index=False)
    log("=== S1 sensitivity of the headline conclusion ===")
    cols = ["config", "block", "best_combo", "best_cost", "best_fill",
            "default_combo", "default_cost", "default_excess_pct",
            "within_rule_gain_pct", "across_rule_gain_pct",
            "accuracy_best_rmsse", "cost_best_under_P1", "accuracy_cost_agree"]
    log(S[cols].round(2).to_string(index=False))

    # ---- ranking stability across blocks ---------------------------------
    rk = []
    for run in ["main"]:
        for block in ["val", "test"]:
            R = load_run(run, block=block)
            a = table_accuracy(R, "margin")
            for m in a.index:
                rk.append(dict(run=run, block=block, method=m,
                               rmsse=float(a.loc[m, "rmsse"]),
                               wape=float(a.loc[m, "wape"]),
                               mae=float(a.loc[m, "mae"]),
                               rank_rmsse=int(a.loc[m, "rank_rmsse"])))
    Rk = pd.DataFrame(rk)
    if len(Rk):
        piv = Rk.pivot_table(index="method", columns="block", values="rank_rmsse")
        piv["rank_shift"] = piv.get("test") - piv.get("val")
        piv.to_csv(RESULTS / "S2_ranking_stability.csv")
        log("\n=== S2 stability of the accuracy ranking (val -> test) ===")
        log(piv.to_string())

    out = dict(
        sensitivity=rows,
        ranking_stability=Rk.to_dict("records") if len(Rk) else [],
    )
    with open(RESULTS / "paper_numbers.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    log(f"\nwrote {RESULTS/'paper_numbers.json'}")


if __name__ == "__main__":
    main()
