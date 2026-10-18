"""
p1_tuned.py
===========
How much of the accuracy-to-decision gap survives when the *normal-theory* rule is
given every degree of freedom a practitioner could tune?

The default P1 fixes the safety factor z at the critical-ratio normal quantile and
the dispersion sigma at the trailing-56-day standard deviation measured once at the
start of a block. A sceptical reader will ask whether the headline gap is just an
untuned straw man. run_experiment.py --p1_sweep therefore simulates, for the
margin economy,

  wk_static   weekly review, sigma fixed at block start, z swept over a grid
  wk_dyn      weekly review, sigma refreshed at every forecast origin, z swept
  day_static  daily-review base stock (protection interval L+1), z swept
  day_dyn     daily-review base stock, sigma refreshed, z swept

This script selects (forecaster, z) for every variant on VALIDATION cost only,
reports the selected configuration on TEST, and bootstraps the TEST cost against
the accuracy-driven default and against the cost-optimal pairing (paired over
series). Output: results/<run>/T13_tuned_normal_theory.csv and .json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS                                     # noqa: E402
from analysis import load_run                                  # noqa: E402
from bootstrap import paired_gap, combo_cost_matrix            # noqa: E402

LABEL = {"sS_static": "(s,S) daily review, sigma fixed, z tuned",
         "sS_dyn": "(s,S) daily review, sigma refreshed, z tuned",
         "wk_static": "weekly review, sigma fixed, z tuned",
         "wk_dyn": "weekly review, sigma refreshed, z tuned",
         "day_static": "daily review, sigma fixed, z tuned",
         "day_dyn": "daily review, sigma refreshed, z tuned"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    d = RESULTS / args.run

    sv = pd.read_csv(d / "p1sweep_val.csv")
    st = pd.read_csv(d / "p1sweep_test.csv")
    zv = np.load(d / "p1sweep_tot_val.npz", allow_pickle=True)
    zt = np.load(d / "p1sweep_tot_test.npz", allow_pickle=True)
    keys_t = [str(k) for k in zt["keys"]]
    Mt = zt["M"]
    kt = {k: i for i, k in enumerate(keys_t)}

    # comparators from the main grid (margin economy)
    Rv, Rt = load_run(args.run, block="val"), load_run(args.run, block="test")
    Mv_main, lab_v = combo_cost_matrix(Rv["ser"], "margin")
    Mt_main, lab_t = combo_cost_matrix(Rt["ser"], "margin")
    assert lab_v == lab_t
    L = {k: i for i, k in enumerate(lab_t)}
    tot_v = Mv_main.sum(0)
    best_key = lab_v[int(np.argmin(tot_v))]
    p2 = [k for k in lab_v if k.startswith("lgbm_q|P2_")]
    d_key = min(p2, key=lambda k: tot_v[L[k]])
    from analysis import table_accuracy
    acc_best = table_accuracy(Rv, "margin")["rmsse"].idxmin()
    default_key = f"{acc_best}|P1_base_stock_normal"

    rows, out = [], {"best_pairing": best_key, "trained_quantile_base_stock": d_key,
                     "default": default_key}
    A = Mt_main[:, L[default_key]]
    E = Mt_main[:, L[best_key]]
    D = Mt_main[:, L[d_key]]

    def add(name, val_cost, test_vec, test_row, detail):
        g_def = paired_gap(A, test_vec, args.reps)
        g_best = paired_gap(test_vec, E, args.reps)
        rows.append(dict(
            configuration=name, selected=detail,
            val_cost=val_cost, test_cost=float(test_vec.sum()),
            test_fill=test_row.get("fill_rate", np.nan),
            test_on_hand=test_row.get("avg_on_hand", np.nan),
            vs_default_pct=g_def["change_b_vs_a_pct"],
            vs_default_ci_lo=g_def["ci95_change_b_vs_a_pct"][0],
            vs_default_ci_hi=g_def["ci95_change_b_vs_a_pct"][1],
            excess_over_best_pct=g_best["point_pct"],
            excess_over_best_ci_lo=g_best["ci95_pct"][0],
            excess_over_best_ci_hi=g_best["ci95_pct"][1]))

    agg_t = Rt["agg"][Rt["agg"].econ == "margin"].set_index(["method", "policy"])
    agg_v = Rv["agg"][Rv["agg"].econ == "margin"].set_index(["method", "policy"])

    def main_row(key, agg):
        m, p = key.split("|", 1)
        return agg.loc[(m, p)].to_dict()

    add("Accuracy-driven default (RMSSE forecaster, z at CR quantile, weekly)",
        float(tot_v[L[default_key]]), A, main_row(default_key, agg_t), default_key)

    for var in ["wk_static", "wk_dyn", "sS_static", "sS_dyn", "day_static", "day_dyn"]:
        v = sv[sv.variant == var]
        b = v.loc[v.total_cost.idxmin()]
        key = f"{var}|{b['method']}|{b['z']:g}"
        trow = st[(st.variant == var) & (st.method == b["method"]) & (st.z == b["z"])].iloc[0]
        add(f"Normal-theory base stock, {LABEL[var]} (selected on VAL)",
            float(b["total_cost"]), Mt[kt[key]], trow.to_dict(),
            f"{b['method']}, z={b['z']:g}")
        # forecaster ranking after tuning z per forecaster (VAL), evaluated on TEST
        per = []
        for m, g in v.groupby("method"):
            bb = g.loc[g.total_cost.idxmin()]
            tr = st[(st.variant == var) & (st.method == m) & (st.z == bb["z"])].iloc[0]
            per.append(dict(method=m, z=float(bb["z"]), val_cost=float(bb["total_cost"]),
                            test_cost=float(tr["total_cost"])))
        per = pd.DataFrame(per).sort_values("val_cost")
        out[f"{var}_per_forecaster"] = per.to_dict("records")
        out[f"{var}_spread_val_pct"] = float(100 * (per.val_cost.max() / per.val_cost.min() - 1))
        out[f"{var}_spread_test_pct"] = float(100 * (per.test_cost.max() / per.test_cost.min() - 1))
        out[f"{var}_forecaster_gain_test_pct"] = float(
            100 * (per.test_cost.max() / per.test_cost.min() - 1))

    add("Trained quantile base stock (lgbm_q, P2, tau selected on VAL)",
        float(tot_v[L[d_key]]), D, main_row(d_key, agg_t), d_key)
    add("Cost-optimal pairing (selected on VAL)", float(tot_v[L[best_key]]), E,
        main_row(best_key, agg_t), best_key)

    df = pd.DataFrame(rows)
    df.to_csv(d / "T13_tuned_normal_theory.csv", index=False)
    with open(d / "T13_tuned_normal_theory.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 70)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
