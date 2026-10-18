"""
regime_oos.py
=============
Out-of-sample check of the regime findings.

analysis.py reports, on VALIDATION, which pairing is cheapest inside each demand
regime (level_up, level_down, vol_up, stable, new). That is an in-sample minimum
over 59 pairings and is subject to the same winner's curse the paper documents for
per-SKU selection. Here the regime-specific pairing is chosen on VAL cells only and
then evaluated on the TEST cells of the same regime, against the regime-blind
pairing (the single best pairing on all of VAL). Series are resampled as clusters
for the interval.

Output: results/<run>/T14_regime_oos.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS                                     # noqa: E402
from selection import load_percost, build_index               # noqa: E402


def labels(reg: pd.DataFrame, block: str, n: int):
    r = reg[reg.block == block]
    per = int(r.period.max()) + 1
    lab = np.full((per, n), "unknown", dtype=object)
    lab[r.period.to_numpy(), r.series_ix.to_numpy()] = r.regime.to_numpy()
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--econ", default="margin")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    d = RESULTS / args.run
    kv, Av = load_percost(args.run, "val")
    kt, At = load_percost(args.run, "test")
    iv = build_index(kv, args.econ)
    it = build_index(kt, args.econ)
    Cv, Ct = Av[iv[0]], At[it[0]]                       # (combo, period, n)
    names = [f"{m}|{p}" for m, p in zip(iv[1], iv[2])]
    n = Cv.shape[2]
    reg = pd.read_csv(d / "regimes.csv")
    lv, lt = labels(reg, "val", n), labels(reg, "test", n)

    blind = int(np.argmin(Cv.sum((1, 2))))
    rng = np.random.default_rng(0)
    rows = []
    for r in ["stable", "level_up", "level_down", "vol_up", "new"]:
        mv, mt = lv == r, lt == r
        if mv.sum() == 0 or mt.sum() == 0:
            continue
        cv = np.array([Cv[c][mv].sum() for c in range(len(names))])
        best = int(np.argmin(cv))
        # per-series cost on the TEST cells of this regime
        tb = np.where(mt, At[it[0]][best], 0.0).sum(0)
        tf = np.where(mt, At[it[0]][blind], 0.0).sum(0)
        cell_series = np.flatnonzero(mt.any(0))
        diffs = []
        for _ in range(args.reps):
            idx = rng.choice(cell_series, len(cell_series), replace=True)
            diffs.append((tf[idx].sum() - tb[idx].sum()) / max(tf[idx].sum(), 1e-9))
        rows.append(dict(
            regime=r, val_cells=int(mv.sum()), test_cells=int(mt.sum()),
            val_best=names[best], blind=names[blind],
            val_blind_penalty_pct=100 * (cv[blind] / cv[best] - 1),
            test_cost_regime_choice=float(tb.sum()), test_cost_blind=float(tf.sum()),
            test_gain_of_regime_choice_pct=100 * (tf.sum() - tb.sum()) / max(tf.sum(), 1e-9),
            ci_lo=100 * float(np.percentile(diffs, 2.5)),
            ci_hi=100 * float(np.percentile(diffs, 97.5))))
    df = pd.DataFrame(rows)
    df.to_csv(d / "T14_regime_oos.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
