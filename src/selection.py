"""
selection.py
============
RQ4: how should a practitioner *choose* a forecasting-and-replenishment
strategy, given that per-SKU accuracy metrics do not rank strategies by cost?

We compare eight selection strategies. All of them are fitted using validation
data only; the test block is never used for selection. Each strategy is scored
twice:

  * on the held-out second half of the validation block, and
  * on the test block,

which separates *overfitting to the selection data* from *genuine drift* between
the selection period and the deployment period.

Strategies
----------
  S1 fixed_pooled_best        one pairing for every SKU, chosen on all of val
  S2 fixed_split_half         one pairing chosen on val-A, scored on val-B
  S3 per_series_full          a different pairing per SKU, chosen on all of val
  S4 per_series_split_half    the same, but chosen on val-A and scored on val-B
  S5 menu_policy_family       per-SKU choice restricted to policy families
                              (the forecaster is fixed to the pooled best)
  S6 menu_forecaster          per-SKU choice restricted to forecasters
                              (the policy family is fixed to the pooled best)
  S7 shrunk_per_series        per-SKU choice on a shrinkage blend of the SKU's
                              own validation cost and the pooled cost
  S8 accuracy_driven_default  the incumbent practice: best point forecaster by
                              RMSSE + normal-theory base stock (no cost data used)
  S9 oracle_per_series        per-SKU minimum on the evaluation data itself
                              (unattainable upper bound; reported for headroom)

Usage
-----
  python3 selection.py --run main --econ margin --out main
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
from analysis import load_run, table_accuracy, POLICY_FAMILIES, policy_family  # noqa: E402


def log(m):
    print(m, flush=True)


def load_percost(run: str, block: str):
    """Returns (keys, per-series per-period cost tensor) for one block."""
    z = np.load(RESULTS / run / f"percost_{block}.npz", allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    A = np.stack([z[f"k{i}"] for i in range(len(keys))], axis=0)   # (n_combo, n_per, n)
    return keys, A


def parse_key(k: str):
    econ, method, policy = k.split("|", 2)
    return econ, method, policy


def family_of(policy: str) -> str:
    return policy_family(policy)


def build_index(keys, econ):
    sub = [i for i, k in enumerate(keys) if parse_key(k)[0] == econ]
    methods = [parse_key(keys[i])[1] for i in sub]
    policies = [parse_key(keys[i])[2] for i in sub]
    fams = [family_of(p) for p in policies]
    return (np.array(sub), np.array(methods), np.array(policies), np.array(fams))


def eval_selection(cost_eval: np.ndarray, choice) -> dict:
    """cost_eval: (n_combo, n). choice: int (fixed) or (n,) array of combo indices."""
    n = cost_eval.shape[1]
    if np.isscalar(choice):
        tot = float(cost_eval[choice].sum())
        per = None
    else:
        choice = np.asarray(choice)
        tot = float(cost_eval[choice, np.arange(n)].sum())
        per = choice
    return dict(cost=tot, cost_per_sku=tot / n, choice=per)


def run_strategies(A_val, A_test, idx_val, idx_test, acc_best_method,
                   lam_grid=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99)):
    sub_v, meth_v, pol_v, fam_v = idx_val
    sub_t, meth_t, pol_t, fam_t = idx_test
    Cv = A_val[sub_v]                       # (n_combo, n_per, n)
    Ct = A_test[sub_t]
    assert (meth_v == meth_t).all() and (pol_v == pol_t).all(), \
        "validation and test combo ordering differ"

    nper = Cv.shape[1]
    half = nper // 2
    CvA = Cv[:, :half].sum(1)               # (n_combo, n) selection half
    CvB = Cv[:, half:].sum(1)               # (n_combo, n) held-out half
    CvF = Cv.sum(1)                         # all of validation
    CtF = Ct.sum(1)                         # test

    # map a combo index chosen on val to the same combo on test (identical order)
    n_combo = len(sub_v)
    rows = []

    def add(name, choice_val, note=""):
        if choice_val is None:
            return
        c = np.asarray(choice_val) if not np.isscalar(choice_val) else choice_val
        onB = eval_selection(CvB, c)
        onT = eval_selection(CtF, c)
        rows.append(dict(strategy=name, note=note,
                         valB_cost=onB["cost"], valB_per_sku=onB["cost_per_sku"],
                         test_cost=onT["cost"], test_per_sku=onT["cost_per_sku"],
                         n_distinct=(1 if np.isscalar(c) else len(np.unique(c)))))

    # ---- S1 fixed, pooled best on all of val ------------------------------
    s1 = int(np.argmin(CvF.sum(1)))
    add("S1_fixed_pooled_best", s1, f"{meth_v[s1]}|{pol_v[s1]}")

    # ---- S2 fixed, split-half ---------------------------------------------
    s2 = int(np.argmin(CvA.sum(1)))
    add("S2_fixed_split_half", s2, f"{meth_v[s2]}|{pol_v[s2]}")

    # ---- S3 per-series on all of val ---------------------------------------
    add("S3_per_series_full", np.argmin(CvF, axis=0), "per-SKU argmin on all val")

    # ---- S4 per-series on val-A, scored on val-B ---------------------------
    add("S4_per_series_split_half", np.argmin(CvA, axis=0), "per-SKU argmin on val-A")

    # ---- S5 menu restricted to policy families ------------------------------
    best_method = meth_v[s1]
    mask5 = (meth_v == best_method)
    sub_idx = np.flatnonzero(mask5)
    c5 = sub_idx[np.argmin(CvF[sub_idx], axis=0)] if len(sub_idx) else None
    add("S5_menu_policy_family", c5, f"forecaster fixed to {best_method}")

    # ---- S6 menu restricted to forecasters ----------------------------------
    best_policy = pol_v[s1]
    mask6 = (pol_v == best_policy)
    sub_idx6 = np.flatnonzero(mask6)
    c6 = sub_idx6[np.argmin(CvF[sub_idx6], axis=0)] if mask6.any() else None
    add("S6_menu_forecaster", c6, f"policy fixed to {best_policy}")

    # ---- S7 shrinkage -------------------------------------------------------
    pooled = CvF.mean(1, keepdims=True)
    for lam in lam_grid:
        blended = lam * CvF + (1 - lam) * pooled
        add(f"S7_shrunk_lambda_{lam:g}", np.argmin(blended, axis=0),
            "shrink SKU cost toward pooled mean")

    # ---- S8 accuracy-driven default ----------------------------------------
    cand = np.flatnonzero((meth_v == acc_best_method) &
                          (pol_v == "P1_base_stock_normal"))
    if len(cand):
        add("S8_accuracy_driven_default", int(cand[0]),
            f"{acc_best_method}|P1_base_stock_normal (selected on RMSSE)")

    # ---- S9 oracle ----------------------------------------------------------
    add("S9_oracle_valB", np.argmin(CvB, axis=0), "upper bound on val-B")
    add("S9_oracle_test", np.argmin(CtF, axis=0), "upper bound on test")

    df = pd.DataFrame(rows)
    ref = float(df.loc[df.strategy == "S1_fixed_pooled_best", "test_cost"].iloc[0])
    refB = float(df.loc[df.strategy == "S1_fixed_pooled_best", "valB_cost"].iloc[0])
    df["vs_S1_test_pct"] = 100 * (df.test_cost / ref - 1)
    df["vs_S1_valB_pct"] = 100 * (df.valB_cost / refB - 1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--econ", default="margin")
    ap.add_argument("--out", default="main")
    args = ap.parse_args()

    R = load_run(args.run, block="val")
    acc = table_accuracy(R, args.econ)
    acc_best = acc["rmsse"].idxmin()
    log(f"accuracy-best point forecaster (RMSSE) on val: {acc_best}")

    kv, Av = load_percost(args.run, "val")
    kt, At = load_percost(args.run, "test")
    log(f"per-series cost tensors: val {Av.shape}, test {At.shape}")

    idx_v = build_index(kv, args.econ)
    idx_t = build_index(kt, args.econ)
    df = run_strategies(Av, At, idx_v, idx_t, acc_best)
    outdir = RESULTS / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "T11_selection_strategies.csv", index=False)
    log("\n=== T11 selection strategies ===")
    log(df[["strategy", "note", "valB_cost", "test_cost", "vs_S1_valB_pct",
            "vs_S1_test_pct", "n_distinct"]].round(2).to_string(index=False))

    # headroom decomposition: how much of the oracle gap is capturable?
    s1 = df.loc[df.strategy == "S1_fixed_pooled_best", "test_cost"].iloc[0]
    orc = df.loc[df.strategy == "S9_oracle_test", "test_cost"].iloc[0]
    best_learned = df.loc[df.strategy.str.startswith(("S3", "S4", "S5", "S6", "S7")),
                          "test_cost"].min()
    summary = dict(econ=args.econ,
                   S1_test=float(s1), oracle_test=float(orc),
                   oracle_headroom_pct=float(100 * (s1 - orc) / s1),
                   best_learned_test=float(best_learned),
                   best_learned_vs_S1_pct=float(100 * (best_learned - s1) / s1),
                   capture_rate=float((s1 - best_learned) / max(s1 - orc, 1e-9)))
    with open(outdir / "T11b_headroom.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("\n=== T11b headroom ===\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
