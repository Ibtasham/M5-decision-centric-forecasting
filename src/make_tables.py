"""
make_tables.py
==============
Render every results table of the manuscript as markdown from the CSV/JSON
artefacts, so that no table number is typed by hand.  Output: paper/tables/*.md
(assemble.py splices them into paper/manuscript.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, ROOT                                # noqa: E402

OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
M = RESULTS / "main"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(lines) + "\n"


def save(name, df):
    (OUT / f"{name}.md").write_text(md(df))


def n0(x):
    return f"{x:,.0f}"


def pc(x, d=1, sign=False):
    return f"{x:+.{d}f}%" if sign else f"{x:.{d}f}%"


def ci(lo, hi, d=1):
    return f"[{lo:+.{d}f}, {hi:+.{d}f}]"


def pretty(method, policy):
    fam = {"P1_base_stock_normal": "P1 normal base stock",
           "P2_quantile_base_stock": "P2 quantile base stock",
           "P3_sS_weekly": "P3 (s,S) weekly", "P3_sS_daily": "P3 (s,S) daily",
           "P4_ddmrp": "P4 DDMRP buffer"}
    f, _, tau = policy.partition("@")
    s = f"{method} + {fam[f]}"
    return s + (f" @ τ={tau}" if tau else "")


def main():
    agg = pd.read_csv(M / "policy_agg.csv")
    agg = agg[agg.econ == "margin"]
    bs = json.load(open(M / "T12_bootstrap.json"))

    # ---- Table 2: accuracy (TEST)
    a = pd.read_csv(M / "T1_accuracy_test.csv")
    best = {c: a[c].min() for c in ["wape", "mae", "rmse", "mase"]}
    rows = []
    for _, r in a.iterrows():
        def f(c, d=3):
            v = f"{r[c]:.{d}f}"
            return f"**{v}**" if c in best and abs(r[c] - best[c]) < 1e-12 else v
        rows.append({"Rank (RMSSE)": int(r.rank_rmsse), "Method": r.method, "RMSSE": f("rmsse"),
                     "WAPE": f("wape"), "MAE": f("mae"), "RMSE": f("rmse"), "MASE": f("mase"),
                     "Bias": f"{r.bias:+.3f}"})
    save("T2_accuracy", pd.DataFrame(rows))

    # ---- Table 3: cost-optimal vs accuracy-optimal (TEST)
    t = agg[agg.block == "test"].copy()
    t = t[~((t.method == "lgbm_q") & t.policy.str.endswith("@0.995"))]
    t = t.sort_values("total_cost")
    top = t.head(5)
    p1 = t[t.policy == "P1_base_stock_normal"]
    acc_default = p1[p1.method == "lgbm_mean"].iloc[0]
    best_p1 = p1.iloc[0]
    worst_p1 = p1.iloc[-1]
    sel = [(top.iloc[i], "") for i in range(5)]
    sel.append((acc_default, " *(accuracy-driven default)*"))
    if best_p1.method != "lgbm_mean":
        sel.append((best_p1, " *(best P1 forecaster on TEST)*"))
    sel.append((worst_p1, " *(worst)*"))
    rows = []
    for r, tag in sel:
        name = pretty(r.method, r.policy) + tag
        if r is top.iloc[0]:
            name = f"**{name}**"
        rows.append({"Pairing": name, "Total cost (USD)": n0(r.total_cost),
                     "Fill rate": f"{r.fill_rate:.4f}", "Period CSL": f"{r.csl_period:.4f}",
                     "Avg on-hand": f"{r.avg_on_hand:.1f}",
                     "Cost, % revenue": pc(100 * r.cost_pct_revenue, 2)})
    save("T3_pairings", pd.DataFrame(rows))

    # ---- Table 4: rank agreement (VAL, P1 family)
    ra = pd.read_csv(M / "T3_rank_agreement.csv")
    rows = [{"Metric": r.metric.upper() if r.metric != "rmsse" else "RMSSE",
             "Spearman ρ": f"{r.spearman:.3f}", "p": f"{r.spearman_p:.3f}",
             "Kendall τ": f"{r.kendall:.3f}", "p ": f"{r.kendall_p:.3f}",
             "Accuracy-best": r.accuracy_best, "Cost-best": r.cost_best,
             "Agree?": "Yes" if r.agree else "**No**"} for _, r in ra.iterrows()]
    save("T4_rank_agreement", pd.DataFrame(rows))

    # ---- Table 5: selection leverage
    lv = {b: pd.read_csv(M / f"T4b_selection_leverage{'_test' if b == 'test' else ''}.csv",
                         index_col=0).iloc[:, 0] for b in ["val", "test"]}
    stepB = {b: bs[b]["steps"]["B_best_point_forecaster"] for b in bs}
    rows = [
        {"Quantity": "Reference: accuracy-driven default (USD)",
         "VAL": n0(float(lv["val"]["reference_accuracy_driven"])),
         "TEST": n0(float(lv["test"]["reference_accuracy_driven"]))},
        {"Quantity": "Best forecaster *within* P1 (ex post)",
         "VAL": f"{n0(float(lv['val']['within_rule_best']))} ({stepB['val']['combo'].split('|')[0]})",
         "TEST": f"{n0(float(lv['test']['within_rule_best']))} ({stepB['test']['combo'].split('|')[0]})"},
        {"Quantity": "Gain from optimising the forecaster",
         "VAL": f"**{pc(-stepB['val']['change_b_vs_a_pct'], 2)}** (CI {ci(-stepB['val']['ci95_change_b_vs_a_pct'][1], -stepB['val']['ci95_change_b_vs_a_pct'][0])})",
         "TEST": f"**{pc(-stepB['test']['change_b_vs_a_pct'], 2)}** (CI {ci(-stepB['test']['ci95_change_b_vs_a_pct'][1], -stepB['test']['ci95_change_b_vs_a_pct'][0])})"},
        {"Quantity": "Worst forecaster within P1 (USD)",
         "VAL": n0(float(lv["val"]["within_rule_worst"])), "TEST": n0(float(lv["test"]["within_rule_worst"]))},
        {"Quantity": "Spread within P1 (worst ÷ best − 1)",
         "VAL": pc(float(lv["val"]["within_spread"]) if "within_spread" in lv["val"] else float(lv["val"]["within_rule_spread_pct"])),
         "TEST": pc(float(lv["test"]["within_rule_spread_pct"]))},
        {"Quantity": "Best pairing overall (USD)",
         "VAL": n0(float(lv["val"]["across_rule_best"])), "TEST": n0(float(lv["test"]["across_rule_best"]))},
        {"Quantity": "Gain from optimising the decision rule",
         "VAL": f"**{pc(-bs['val']['steps']['E_cost_optimal_pairing']['change_b_vs_a_pct'])}** (CI {ci(-bs['val']['steps']['E_cost_optimal_pairing']['ci95_change_b_vs_a_pct'][1], -bs['val']['steps']['E_cost_optimal_pairing']['ci95_change_b_vs_a_pct'][0])})",
         "TEST": f"**{pc(-bs['test']['steps']['E_cost_optimal_pairing']['change_b_vs_a_pct'])}** (CI {ci(-bs['test']['steps']['E_cost_optimal_pairing']['ci95_change_b_vs_a_pct'][1], -bs['test']['steps']['E_cost_optimal_pairing']['ci95_change_b_vs_a_pct'][0])})"},
    ]
    save("T5_leverage", pd.DataFrame(rows))

    # ---- Table 6: decomposition
    dec = {b: pd.read_csv(M / f"T5_decomposition{'_test' if b == 'test' else ''}.csv") for b in ["val", "test"]}
    names = ["A  Accuracy-driven default", "B  Best point forecaster, same rule",
             "C  Calibrated quantile rule", "D  Trained quantile rule", "E  Cost-optimal pairing"]
    steps = [None, "B_best_point_forecaster", "C_calibrated_quantile_rule",
             "D_trained_quantile_rule", "E_cost_optimal_pairing"]
    rows = []
    for i in range(5):
        rv, rt = dec["val"].iloc[i], dec["test"].iloc[i]
        row = {"Step": names[i],
               "VAL pairing": pretty(rv.method, rv.policy) if i else pretty(rv.method, rv.policy),
               "VAL cost": n0(rv.total_cost),
               "VAL vs A": "—" if i == 0 else f"{bs['val']['steps'][steps[i]]['change_b_vs_a_pct']:+.1f}% {ci(*bs['val']['steps'][steps[i]]['ci95_change_b_vs_a_pct'])}",
               "TEST pairing": pretty(rt.method, rt.policy),
               "TEST cost": n0(rt.total_cost),
               "TEST vs A": "—" if i == 0 else f"{bs['test']['steps'][steps[i]]['change_b_vs_a_pct']:+.1f}% {ci(*bs['test']['steps'][steps[i]]['ci95_change_b_vs_a_pct'])}",
               "Fill (TEST)": f"{rt.fill_rate:.3f}", "On-hand (TEST)": f"{rt.on_hand:.1f}"}
        rows.append(row)
    save("T6_decomposition", pd.DataFrame(rows))

    # ---- Table 7: tuned normal theory
    t13 = pd.read_csv(M / "T13_tuned_normal_theory.csv")
    keep = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    rows = []
    for i, r in t13.iterrows():
        nm = r.configuration.replace("Normal-theory base stock, ", "Normal theory, ")
        nm = nm.replace(" (selected on VAL)", "").replace("Accuracy-driven default (RMSSE forecaster, z at CR quantile, weekly)", "Accuracy-driven default (untuned)")
        rows.append({"Configuration": nm, "Selected on VAL": (pretty(*r.selected.split("|", 1)) if "|" in r.selected else r.selected),
                     "TEST cost": n0(r.test_cost), "Fill": f"{r.test_fill:.3f}",
                     "vs default": "—" if i == 0 else f"{r.vs_default_pct:+.1f}% {ci(r.vs_default_ci_lo, r.vs_default_ci_hi)}",
                     "Excess over cost-optimal": f"{r.excess_over_best_pct:+.1f}% {ci(r.excess_over_best_ci_lo, r.excess_over_best_ci_hi)}"})
    save("T7_tuned_normal", pd.DataFrame(rows))

    # ---- Table 8: strata (VAL)
    rows = []
    for lab, f, col in [("SBC class", "T6_conditional_sbc_class.csv", "sbc_class"),
                        ("Scale tercile", "T6_conditional_scale_band.csv", "scale_band"),
                        ("Category", "T6_conditional_cat_id.csv", "cat_id")]:
        s = pd.read_csv(M / f)
        for _, r in s.iterrows():
            rows.append({"Stratum": f"{lab}: {r[col]}", "Best pairing": pretty(r.best_method, r.best_policy),
                         "Best cost": n0(r.best_total), "Best P1 forecaster": r.p1_best_method,
                         "P1 cost": n0(r.p1_best_total), "Excess": pc(r.excess_pct, 1, True)})
    save("T8_strata", pd.DataFrame(rows))

    # ---- Table 9: regimes (VAL in-sample, TEST out-of-sample)
    rg = pd.read_csv(M / "T6c_regime_best_pairing.csv").set_index("regime")
    oos = pd.read_csv(M / "T14_regime_oos.csv").set_index("regime")
    rows = []
    for r in ["level_down", "level_up", "stable", "vol_up", "new"]:
        v = rg.loc[r]
        o = oos.loc[r]
        rows.append({"Regime": r + (" (no demand in baseline year)" if r == "new" else ""),
                     "VAL cells": f"{int(v.n):,}", "VAL best pairing": pretty(v.method, v.policy),
                     "VAL regime-blind penalty": pc(v.blind_penalty_pct),
                     "TEST cells": f"{int(o.test_cells):,}",
                     "TEST gain of regime-specific choice": f"{o.test_gain_of_regime_choice_pct:+.1f}%"})
    save("T9_regimes", pd.DataFrame(rows))

    # ---- Table 10: selection strategies
    s = pd.read_csv(M / "T11_selection_strategies.csv")
    nice = {"S1_fixed_pooled_best": ("S1 Fixed, pooled best", "one pairing, chosen on all of VAL"),
            "S2_fixed_split_half": ("S2 Fixed, split-half", "one pairing, chosen on VAL-A"),
            "S3_per_series_full": ("S3 Per-SKU, all of VAL", "argmin per SKU on VAL"),
            "S4_per_series_split_half": ("S4 Per-SKU, split-half", "argmin per SKU on VAL-A"),
            "S5_menu_policy_family": ("S5 Menu: policy families", "per SKU, forecaster fixed"),
            "S6_menu_forecaster": ("S6 Menu: forecasters", "per SKU, policy fixed"),
            "S7_shrunk_lambda_0.25": ("S7 Shrinkage λ=0.25", "blend SKU and pooled cost"),
            "S7_shrunk_lambda_0.5": ("S7 Shrinkage λ=0.5", ""),
            "S7_shrunk_lambda_0.9": ("S7 Shrinkage λ=0.9", ""),
            "S8_accuracy_driven_default": ("S8 Accuracy-driven default", "best RMSSE + P1"),
            "S9_oracle_valB": ("S9 Oracle on VAL-B", "*(unattainable)*"),
            "S9_oracle_test": ("S9 Oracle on TEST", "*(unattainable)*")}
    rows = []
    for k, (a_, b_) in nice.items():
        r = s[s.strategy == k].iloc[0]
        rows.append({"Strategy": a_, "Selection rule": b_, "VAL-B cost": n0(r.valB_cost),
                     "TEST cost": n0(r.test_cost), "vs S1 (VAL-B)": pc(r.vs_S1_valB_pct, 1, True),
                     "vs S1 (TEST)": pc(r.vs_S1_test_pct, 1, True), "Distinct choices": int(r.n_distinct)})
    save("T10_selection", pd.DataFrame(rows))

    # ---- Table 11: sensitivity
    sv = pd.read_csv(RESULTS / "S1_sensitivity.csv")
    sv = sv[sv.block == "test"].set_index("config")
    order = [("main", "Baseline (L=2, K=$0.50, R=7, hold 25%, cost 60%)"),
             ("sens_lead7", "Lead time 7 d"), ("sens_lead14", "Lead time 14 d"),
             ("sens_K5", "Order cost $5.00 per line"), ("sens_review14", "Review interval 14 d"),
             ("sens_holdrate40", "Holding rate 40%"), ("sens_costratio75", "Purchase cost 75% of price"),
             ("sens_penmult3", "Stockout penalty = 3× margin"), ("sens_sfrac30", "(s,S) reorder point 0.30×S"),
             ("sens_ddmrp80", "DDMRP buffer 0.80")]
    rows = []
    for k, lab in order:
        r = sv.loc[k]
        m_, p_ = r.best_combo.split("|")
        rows.append({"Configuration": lab, "Best pairing": pretty(m_, p_), "Best cost": n0(r.best_cost),
                     "Default cost": n0(r.default_cost), "Default excess": pc(r.default_excess_pct, 1, True),
                     "Within-rule gain": pc(r.within_rule_gain_pct), "Across-rule gain": pc(r.across_rule_gain_pct),
                     "Accuracy-best = cost-best in P1?": "Yes" if r.accuracy_cost_agree else "No"})
    save("T11_sensitivity", pd.DataFrame(rows))

    # ---- Table C1: seed robustness
    def seedcol(run):
        R = RESULTS / run
        b = json.load(open(R / "T12_bootstrap.json"))["test"]
        lvt = pd.read_csv(R / "T4b_selection_leverage_test.csv", index_col=0).iloc[:, 0]
        dd = b["default_vs_best"]
        return {
            "Cost-optimal pairing": pretty(*b["best_combo"].split("|", 1)),
            "Its total cost": n0(float(lvt["across_rule_best"])),
            "Accuracy-driven default": pretty(*b["default_combo"].split("|", 1)),
            "Its total cost ": n0(float(lvt["reference_accuracy_driven"])),
            "Excess cost of the default": f"{dd['point_pct']:+.1f}% {ci(*dd['ci95_pct'])}",
            "Gain from swapping the forecaster within P1": f"{-b['steps']['B_best_point_forecaster']['change_b_vs_a_pct']:.1f}% {ci(-b['steps']['B_best_point_forecaster']['ci95_change_b_vs_a_pct'][1], -b['steps']['B_best_point_forecaster']['ci95_change_b_vs_a_pct'][0])}",
            "Gain from swapping the decision rule": f"{-b['steps']['E_cost_optimal_pairing']['change_b_vs_a_pct']:.1f}%",
            "Spread within P1 (worst ÷ best − 1)": pc(float(lvt["within_rule_spread_pct"])),
            "Step C vs default": f"{b['steps']['C_calibrated_quantile_rule']['change_b_vs_a_pct']:+.1f}% {ci(*b['steps']['C_calibrated_quantile_rule']['ci95_change_b_vs_a_pct'])}",
            "Step D vs default": f"{b['steps']['D_trained_quantile_rule']['change_b_vs_a_pct']:+.1f}% {ci(*b['steps']['D_trained_quantile_rule']['ci95_change_b_vs_a_pct'])}"}
    c0, c1 = seedcol("main"), seedcol("seed1")
    save("TC1_seed", pd.DataFrame([{"Quantity": k.strip(), "Seed 0": c0[k], "Seed 1": c1[k]} for k in c0]))
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
