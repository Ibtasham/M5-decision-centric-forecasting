"""
smoke_test.py
=============
End-to-end plumbing check on a small slice of the real M5 panel: features, every
forecaster, the simulator and every policy. Must pass before any full run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROC
import features as FEAT
import forecasters as FC
import simulation as SIM

N_SUB = 250
H = 28


def load(n_sub=N_SUB, seed=0):
    Y_all = np.load(PROC / "Y.npy", mmap_mode="r")
    meta = pd.read_parquet(PROC / "meta.parquet")
    cal = pd.read_parquet(PROC / "calendar.parquet")
    prices = np.load(PROC / "P.npy", mmap_mode="r")
    pw = np.array(json.load(open(PROC / "price_weeks.json")))
    rng = np.random.default_rng(seed)
    # stratified: take some from each SBC class
    ix = []
    for cls, g in meta.groupby("sbc_class"):
        k = max(2, int(round(n_sub * len(g) / len(meta))))
        ix.extend(rng.choice(g.index.to_numpy(), size=min(k, len(g)), replace=False))
    ix = np.sort(np.array(ix, dtype=np.int64))[:n_sub]
    Y = np.array(Y_all[ix], dtype=np.uint16)
    P = np.array(prices[ix], dtype=np.float32)
    m = meta.iloc[ix].reset_index(drop=True)
    return Y, m, cal, P, pw


def main():
    t0 = time.time()
    Y, meta, cal, prices, pw = load()
    n, T = Y.shape
    print(f"panel {Y.shape} | classes {meta.sbc_class.value_counts().to_dict()}")

    splits = json.load(open(PROC / "splits.json"))
    t0_ix = splits["val"][0] + 4 * 28           # an origin inside the val block
    print(f"origin t0={t0_ix} ({cal.date.iloc[t0_ix].date()})  H={H}")

    groups = pd.Categorical(meta["sbc_class"]).codes.astype(np.int32)
    ctx = dict(cal=cal, prices=prices, price_weeks=pw, groups=groups, meta=meta)

    # ---------------- features ----------------
    st = FEAT.series_static_features(Y, t0_ix)
    print(f"static features: {len(st)} -> {sorted(st)[:4]} ...")
    price_at = FEAT.price_window(prices, pw, cal, t0_ix, H)
    df, fn = FEAT.build_origin_frame(Y, t0_ix, H, cal, price_at, static=st)
    print(f"design matrix: {df.shape} | {len(fn)} features | NaNs {int(df[fn].isna().sum().sum())}")
    assert df.shape == (n * H, len(fn) + 3), df.shape
    assert np.isfinite(df[fn].to_numpy()).all(), "non-finite features"
    # leakage check: no feature may reference day >= t0_ix
    y_probe = Y[:, t0_ix:t0_ix + H].astype(np.float32).ravel()
    for c in fn:
        v = df[c].to_numpy(dtype=np.float64)
        if v.std() > 0:
            r = np.corrcoef(v, y_probe.astype(np.float64))[0, 1]
            if abs(r) > 0.95:
                print(f"  !! suspiciously high corr with future demand: {c} r={r:.3f}")

    # ---------------- forecasters ----------------
    train_origins = [t0_ix - 3 * 28, t0_ix - 2 * 28, t0_ix - 28]
    QS = [0.50, 0.90, 0.95]
    reg = FC.build_registry(H=H, train_origins=train_origins, lgbm_estimators=60,
                            quantiles=QS)
    taus = np.array(QS, dtype=np.float32)
    point, dist = {}, {}
    for name, f in reg.items():
        ts = time.time()
        f.fit(Y, t0_ix, ctx)
        pr = f.predict(Y, t0_ix, H, ctx)
        el = time.time() - ts
        ok_shape = pr.shape == ((len(taus), n, H) if f.kind == "dist" else (n, H))
        assert ok_shape, f"{name}: bad shape {pr.shape}"
        assert np.isfinite(pr).all(), f"{name}: non-finite predictions"
        assert (pr >= 0).all(), f"{name}: negative predictions"
        act = Y[:, t0_ix:t0_ix + H].astype(np.float32)
        pf = pr[0] if pr.ndim == 3 else pr
        mae = float(np.abs(pf - act).mean())
        bias = float((pf - act).mean())
        print(f"  {name:11s} kind={f.kind:5s} {el:6.1f}s  mean={pf.mean():7.3f} "
              f"act={act.mean():7.3f}  MAE={mae:7.3f}  bias={bias:+7.3f}")
        (dist if f.kind == "dist" else point)[name] = pr

    # monotone quantiles
    for name, Q in dist.items():
        assert (np.diff(Q, axis=0) >= -1e-4).all(), f"{name}: quantile crossing"
    print(f"quantile monotonicity OK; taus={taus.tolist()}")

    # ---------------- simulator + policies ----------------
    win0, win1 = t0_ix, t0_ix + 2 * H
    demand = np.ascontiguousarray(Y[:, win0:win1].astype(np.float64).T)   # (T, n)
    Tw = demand.shape[0]
    import common as CM
    wcol = CM.week_index(pw, cal)
    sp = CM.sell_price_at(prices, wcol, t0_ix)
    c, h, pen = SIM.derive_costs(sp, penalty_mode="cr", target_cr=SIM.BASE_CR)
    print(f"costs: unit c median {np.median(c):.2f} | h/day median {np.median(h):.5f} "
          f"| penalty median {np.median(pen):.2f}")

    origins_local = [0, H]
    Tw = origins_local[-1] + H   # window length
    sigma = np.maximum(Y[:, max(t0_ix - 56, 0):t0_ix].astype(np.float64).std(1), 1e-3)
    protection = 7 + SIM.BASE_LEAD

    # P1 normal-theory base stock from each point forecaster
    cache_p, cache_q = {}, {}
    for name, f in reg.items():
        if f.kind == "dist":
            cache_q[name] = [dist[name]] + [f.predict(Y, t0_ix + o, H, ctx)
                                            for o in origins_local[1:]]
        else:
            cache_p[name] = [point[name]] + [f.predict(Y, t0_ix + o, H, ctx)
                                             for o in origins_local[1:]]
    for name, pf in point.items():
        pfc_by_origin = cache_p[name]
        S = SIM.policy_base_stock_normal(pfc_by_origin, sigma, protection, 1.2816,
                                         origins_local, Tw, n)
        r = SIM.simulate(demand, S, np.arange(0, Tw, 7), lead=SIM.BASE_LEAD,
                         h=h, p=pen, K=SIM.BASE_K)
        s = SIM.summarise(r)
        print(f"  P1[{name:10s}] cost/SKU {s['mean_cost_per_series']:9.3f} "
              f"fill {s['fill_rate']:.3f} CSL {s['csl']:.3f} "
              f"onhand {s['avg_on_hand']:8.2f} orders {r['n_orders'].mean():.1f}")

    # P2 quantile base stock
    z = SIM.BASE_CR
    tau_ix = int(np.argmin(np.abs(taus - z)))
    for name, Q in dist.items():
        Q_by_origin = cache_q[name]
        S = SIM.policy_quantile(Q_by_origin, protection, tau_ix, origins_local, Tw, n)
        r = SIM.simulate(demand, S, np.arange(0, Tw, 7), lead=SIM.BASE_LEAD,
                         h=h, p=pen, K=SIM.BASE_K)
        s = SIM.summarise(r)
        print(f"  P2[{name:10s}] cost/SKU {s['mean_cost_per_series']:9.3f} "
              f"fill {s['fill_rate']:.3f} CSL {s['csl']:.3f} onhand {s['avg_on_hand']:8.2f}")

    # P3 (s,S) daily review
    S_surf, s_surf = SIM.policy_sS(Q_by_origin, protection, tau_ix, origins_local,
                                   Tw, n, s_frac=0.5)
    r = SIM.simulate_sS(demand, S_surf, s_surf, SIM.BASE_LEAD, h, pen, SIM.BASE_K)
    print(f"  P3 (s,S)      cost/SKU {SIM.summarise(r)['mean_cost_per_series']:9.3f} "
          f"fill {SIM.summarise(r)['fill_rate']:.3f} orders {np.nanmean(r['n_orders']):.1f}")

    # P4 DDMRP
    pfc_by_origin = cache_p["comb"]
    S4, s4 = SIM.policy_ddmrp(pfc_by_origin, protection, origins_local, Tw, n,
                              lt_days=SIM.BASE_LEAD)
    r = SIM.simulate_sS(demand, S4, s4, SIM.BASE_LEAD, h, pen, SIM.BASE_K)
    print(f"  P4 DDMRP      cost/SKU {SIM.summarise(r)['mean_cost_per_series']:9.3f} "
          f"fill {SIM.summarise(r)['fill_rate']:.3f}")

    # sanity: a zero-inventory policy must lose everything; a huge-S policy must
    # hold a lot and never stock out.
    r0 = SIM.simulate(demand, np.zeros_like(S), np.arange(0, Tw, 7),
                      lead=SIM.BASE_LEAD, h=h, p=pen, K=0.0,
                      init_on_hand=np.zeros(n))
    assert r0["sold"].sum() == 0 and (r0["fill_rate"][r0["demand"] > 0] == 0).all()
    rB = SIM.simulate(demand, np.full_like(S, 1e5), np.arange(0, Tw, 7),
                      lead=SIM.BASE_LEAD, h=h, p=pen, K=SIM.BASE_K)
    assert rB["lost"].sum() == 0 and SIM.summarise(rB)["fill_rate"] == 1.0
    print(f"  boundary checks OK: zero-stock fill=0 | infinite-stock fill=1, "
          f"cost {SIM.summarise(rB)['mean_cost_per_series']:.1f} (holding-dominated)")

    print(f"\nSMOKE TEST PASSED in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
