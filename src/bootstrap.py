"""
bootstrap.py
============
Paired bootstrap inference for the paper's headline comparisons.

Cost differences between two inventory strategies evaluated on the *same* series
and the *same* realised demand are strongly paired, so we resample series with
replacement and take the difference of totals within each replicate. This gives
a confidence interval for the cost gap that accounts for cross-SKU dependence
without assuming any parametric form.

Reported:
  * default (accuracy-driven) vs cost-optimal pairing, per block
  * each step of the A->E decomposition, per block
  * best fixed pairing vs per-SKU selection learned on validation

Usage:  python3 bootstrap.py --run main --econ margin --reps 2000
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
from analysis import load_run, table_cost, table_accuracy, POINT_ONLY  # noqa: E402


def combo_cost_matrix(ser: pd.DataFrame, econ: str):
    """(n_combo, n_series) total-cost matrix over the window, plus combo labels."""
    s = ser[ser.econ.astype(str) == econ]
    piv = s.pivot_table(index="series_ix", columns=["method", "policy"],
                        values="total", aggfunc="first")
    labels = [f"{a}|{b}" for a, b in piv.columns]
    # orientation: rows = series, columns = pairings
    return piv.to_numpy(dtype=np.float64), labels


def paired_gap(a: np.ndarray, b: np.ndarray, reps: int, seed: int = 0):
    """
    Paired bootstrap over series.

    a, b : (n_series,) window-total cost of two strategies on the same series
           and the same realised demand.
    Returns the point estimate and a 95% CI for both the mean per-series cost
    difference and the percentage excess of a over b.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    n = a.size
    rng = np.random.default_rng(seed)
    d = a - b                                     # per-series difference
    ta, tb = a.sum(), b.sum()
    diffs = np.empty(reps); ratios = np.empty(reps); rel = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, n, n)
        diffs[r] = d[idx].mean()
        sa, sb = a[idx].sum(), b[idx].sum()
        ratios[r] = (sa - sb) / sb if sb > 0 else np.nan
        rel[r] = (sb - sa) / sa if sa > 0 else np.nan
    return dict(
        point_diff_per_series=float(d.mean()),
        ci95_diff=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        # "excess of a over b": how much dearer a is than b, in % of b
        point_pct=float(100 * (ta - tb) / tb),
        ci95_pct=[float(np.nanpercentile(ratios, 2.5) * 100),
                  float(np.nanpercentile(ratios, 97.5) * 100)],
        # "change of b relative to a": cost of b in % of a, minus 100 (negative = b cheaper)
        change_b_vs_a_pct=float(100 * (tb - ta) / ta),
        ci95_change_b_vs_a_pct=[float(np.nanpercentile(rel, 2.5) * 100),
                                float(np.nanpercentile(rel, 97.5) * 100)],
        p_two_sided=float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
        n_series=int(n),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--econ", default="margin")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="main")
    args = ap.parse_args()

    out = {}
    for block in ["val", "test"]:
        R = load_run(args.run, block=block)
        M, labels = combo_cost_matrix(R["ser"], args.econ)
        L = {lab: i for i, lab in enumerate(labels)}
        cost = table_cost(R, args.econ)
        cost = cost[cost.econ == args.econ]

        best = cost.loc[cost.total_cost.idxmin()]
        best_key = f"{best['method']}|{best['policy']}"
        acc = table_accuracy(R, args.econ)
        acc_best = acc["rmsse"].idxmin()
        default_key = f"{acc_best}|P1_base_stock_normal"

        res = dict(best_combo=best_key, default_combo=default_key)
        if best_key in L and default_key in L:
            res["default_vs_best"] = paired_gap(
                M[:, L[default_key]], M[:, L[best_key]], args.reps)

        # A->E decomposition steps
        p1 = cost[(cost.policy == "P1_base_stock_normal") & cost.method.isin(POINT_ONLY)]
        steps = {}
        if len(p1):
            b = p1.loc[p1.total_cost.idxmin()]
            steps["B_best_point_forecaster"] = f"{b['method']}|{b['policy']}"
        p2 = cost[cost.policy.str.startswith("P2_quantile")]
        cal = p2[p2.method.str.startswith("cal_")]
        if len(cal):
            c = cal.loc[cal.total_cost.idxmin()]
            steps["C_calibrated_quantile_rule"] = f"{c['method']}|{c['policy']}"
        tr = p2[p2.method == "lgbm_q"]
        if len(tr):
            t = tr.loc[tr.total_cost.idxmin()]
            steps["D_trained_quantile_rule"] = f"{t['method']}|{t['policy']}"
        steps["E_cost_optimal_pairing"] = best_key

        res["steps"] = {}
        for name, key in steps.items():
            if key in L and default_key in L:
                res["steps"][name] = dict(combo=key, **paired_gap(
                    M[:, L[default_key]], M[:, L[key]], args.reps))
        out[block] = res
        print(f"\n=== {block} ({args.run}, econ={args.econ}, reps={args.reps}) ===")
        print(f"default {default_key} vs best {best_key}")
        if "default_vs_best" in res:
            g = res["default_vs_best"]
            print(f"  excess cost {g['point_pct']:+.1f}%  "
                  f"CI95 [{g['ci95_pct'][0]:+.1f}, {g['ci95_pct'][1]:+.1f}]  "
                  f"p={g['p_two_sided']:.4f}")
        for name, g in res["steps"].items():
            print(f"  {name:32s} {g['combo']:42s} "
                  f"A excess {g['point_pct']:+7.1f}% CI95 [{g['ci95_pct'][0]:+7.1f},"
                  f" {g['ci95_pct'][1]:+7.1f}] | step vs A {g['change_b_vs_a_pct']:+7.1f}% "
                  f"CI95 [{g['ci95_change_b_vs_a_pct'][0]:+7.1f}, "
                  f"{g['ci95_change_b_vs_a_pct'][1]:+7.1f}] p={g['p_two_sided']:.4f}")

    d = RESULTS / args.out
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "T12_bootstrap.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {d/'T12_bootstrap.json'}")


if __name__ == "__main__":
    main()
