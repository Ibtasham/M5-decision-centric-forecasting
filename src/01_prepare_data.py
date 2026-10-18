"""
01_prepare_data.py
==================
Streams the raw M5 CSVs into a compact, analysis-ready binary layout that fits
comfortably in a 1 GB RAM sandbox.

Outputs (under <WS>/data/proc/):
  Y.npy            : (30490 x 1969) uint16 matrix of daily unit sales
  P.npy            : (30490 x n_weeks) float32 matrix of weekly sell prices
  meta.parquet     : static attributes + intermittency diagnostics per series
  calendar.parquet : calendar features aligned to the 0-based day index t
  splits.json      : canonical train / validation / test day-index boundaries
  price_weeks.json : the wm_yr_wk codes indexing the columns of P.npy

Notes
-----
* The M5 evaluation panel spans d_1..d_1969 (2011-01-29 .. 2016-06-19).
  `sales_train_evaluation.csv` holds d_1..d_1941; `sales_test_evaluation.csv`
  supplies the final 28 days. They are concatenated to recover the full panel.
* There are 30,490 bottom-level (item x store) demand streams. The often-quoted
  "42,840 series" is the number of *weighted aggregate* series used by the WRMSSE
  metric across the 12 levels of the M5 hierarchy, not the number of SKUs.
* Daily unit sales are stored as uint16 (max observed value is asserted < 65536).
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RAW, PROC, ID_COLS, DAY_COLS, N_DAYS  # noqa: E402

N_SERIES = 30490
CHUNK = 2000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_calendar():
    cal = pd.read_csv(RAW / "calendar.csv")
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal.sort_values("date").reset_index(drop=True)
    cal["t"] = np.arange(len(cal), dtype=np.int32)
    cal["dnum"] = cal["t"] + 1
    for c in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        cal[c] = cal[c].astype("object").where(cal[c].notna(), "None").astype("category")
    cal = cal[["t", "dnum", "date", "wm_yr_wk", "wday", "month", "year",
               "event_name_1", "event_type_1", "event_name_2", "event_type_2",
               "snap_CA", "snap_TX", "snap_WI"]]
    cal.to_parquet(PROC / "calendar.parquet", index=False)
    log(f"calendar: {len(cal)} days  {cal.date.min().date()} .. {cal.date.max().date()}")
    return cal


def build_series_index():
    """Series order is fixed by the (item_id, store_id) order in the train file."""
    head = pd.read_csv(RAW / "sales_train_evaluation.csv", usecols=ID_COLS)
    meta = head.copy()
    meta["series_id"] = (head["item_id"] + "|" + head["store_id"]).to_numpy()
    meta.to_parquet(PROC / "meta_base.parquet", index=False)
    log(f"series index: {len(meta)} rows")
    del head
    gc.collect()
    return meta


def stream_sales(meta):
    """Fill Y (uint16) by reading both sales files in row chunks."""
    Y = np.zeros((N_SERIES, N_DAYS), dtype=np.uint16)
    pos = {s: i for i, s in enumerate(meta["series_id"].to_numpy())}
    gmax = 0
    for fname, cols, off in [("sales_train_evaluation.csv", DAY_COLS[:1941], 0),
                             ("sales_test_evaluation.csv", DAY_COLS[1941:], 1941)]:
        t0 = time.time()
        reader = pd.read_csv(RAW / fname, chunksize=CHUNK)
        n = 0
        for ch in reader:
            rows = np.fromiter((pos[s] for s in
                                (ch["item_id"] + "|" + ch["store_id"]).to_numpy()),
                               dtype=np.int64, count=len(ch))
            vals = ch[cols].to_numpy(dtype=np.uint16)
            Y[rows, off:off + len(cols)] = vals
            gmax = max(gmax, int(vals.max()))
            n += len(ch)
            del ch, vals, rows
        log(f"  {fname}: {n} rows -> cols {off}..{off+len(cols)-1}  ({time.time()-t0:.0f}s)")
        del reader
        gc.collect()
    assert gmax < 65536, f"uint16 overflow: max daily sales {gmax}"
    np.save(PROC / "Y.npy", Y)
    log(f"Y: {Y.shape} uint16 = {Y.nbytes/1e6:.0f} MB | max daily {gmax} | "
        f"nonzero {np.count_nonzero(Y)/Y.size:.4%} | total units {int(Y.sum()):,}")
    return Y


def build_stats(Y, meta):
    """Per-series intermittency / scale diagnostics, computed in row blocks."""
    n = len(meta)
    out = {k: np.zeros(n, dtype=np.float64) for k in
           ["first_sale_t", "n_active_days", "total_units", "mean_daily", "sd_daily",
            "zero_share", "max_daily", "p99_daily", "adi"]}
    B = 4000
    t_all = np.arange(N_DAYS, dtype=np.int32)[None, :]
    for s in range(0, n, B):
        e = min(s + B, n)
        blk = Y[s:e].astype(np.float32)
        nz = blk > 0
        any_s = nz.any(1)
        first = np.where(any_s, nz.argmax(1), N_DAYS).astype(np.float32)
        active = t_all >= first[:, None]
        nact = active.sum(1).astype(np.float32)
        dem = np.where(active, blk, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.nanmean(dem, 1)
            sd = np.nanstd(dem, 1)
            zs = np.nanmean(np.where(active, (~nz).astype(np.float32), np.nan), 1)
            mx = np.nanmax(dem, 1)
            p99 = np.nanpercentile(dem, 99, axis=1)
        adi = np.full(e - s, np.nan, dtype=np.float64)
        for i in range(e - s):
            idx = np.flatnonzero(Y[s + i] > 0)
            if idx.size > 1:
                adi[i] = np.diff(idx).mean()
        for k, v in zip(["first_sale_t", "n_active_days", "total_units", "mean_daily",
                         "sd_daily", "zero_share", "max_daily", "p99_daily", "adi"],
                        [first, nact, blk.sum(1), mu, sd, zs, mx, p99, adi]):
            out[k][s:e] = v
        del blk, dem, nz, active
        gc.collect()

    m = meta.copy().reset_index(drop=True)
    for k, v in out.items():
        m[k] = v.astype(np.float32)
    m["cv2"] = (m["sd_daily"] / m["mean_daily"].replace(0, np.nan)).pow(2).astype(np.float32)

    a = np.nan_to_num(m["adi"].to_numpy(), nan=999.0)
    c = np.nan_to_num(m["cv2"].to_numpy(), nan=0.0)
    m["sbc_class"] = np.select(
        [(a < 1.32) & (c < 0.49), (a < 1.32) & (c >= 0.49),
         (a >= 1.32) & (c < 0.49), (a >= 1.32) & (c >= 0.49)],
        ["smooth", "erratic", "intermittent", "lumpy"], default="unclassified")
    m.to_parquet(PROC / "meta.parquet", index=False)

    log("SBC demand classes:\n" + m["sbc_class"].value_counts().to_string())
    log("cat_id:\n" + m["cat_id"].value_counts().to_string())
    log(f"ADI  median {np.nanmedian(a):.2f}  p90 {np.nanpercentile(a,90):.2f}")
    log(f"CV2  median {np.nanmedian(c):.3f}  p90 {np.nanpercentile(c,90):.3f}")
    log(f"zero_share median {m['zero_share'].median():.3f} | "
        f"mean_daily median {m['mean_daily'].median():.3f}")
    log(f"series with zero history: {(m['total_units']==0).sum()}")
    return m


def build_prices(meta):
    P = np.full((N_SERIES, 1), np.nan, dtype=np.float32)  # resized after week scan
    pos = {s: i for i, s in enumerate(meta["series_id"].to_numpy())}
    weeks = set()
    recs = []
    t0 = time.time()
    reader = pd.read_csv(RAW / "sell_prices.csv", chunksize=1_000_000,
                         dtype={"store_id": "category", "item_id": "category",
                                "wm_yr_wk": np.int32, "sell_price": np.float32})
    for ch in reader:
        key = (ch["item_id"].astype(str) + "|" + ch["store_id"].astype(str)).to_numpy()
        rows = np.fromiter((pos.get(k, -1) for k in key), dtype=np.int64, count=len(ch))
        keep = rows >= 0
        recs.append((rows[keep], ch["wm_yr_wk"].to_numpy()[keep],
                     ch["sell_price"].to_numpy()[keep]))
        weeks.update(np.unique(ch["wm_yr_wk"].to_numpy()).tolist())
        del ch, key, rows
    log(f"  sell_prices streamed in {len(recs)} chunks ({time.time()-t0:.0f}s), "
        f"{len(weeks)} distinct weeks")
    wk = np.array(sorted(weeks), dtype=np.int64)
    wk_ix = {int(w): i for i, w in enumerate(wk)}
    P = np.full((N_SERIES, len(wk)), np.nan, dtype=np.float32)
    for rows, wks, prc in recs:
        cols = np.fromiter((wk_ix[int(w)] for w in wks), dtype=np.int64, count=len(wks))
        P[rows, cols] = prc
    del recs
    gc.collect()
    np.save(PROC / "P.npy", P)
    with open(PROC / "price_weeks.json", "w") as f:
        json.dump([int(w) for w in wk], f)
    log(f"P: {P.shape} float32 = {P.nbytes/1e6:.0f} MB | missing {np.isnan(P).mean():.3%} | "
        f"median {np.nanmedian(P):.2f} | range [{np.nanmin(P):.2f}, {np.nanmax(P):.2f}]")
    return P, wk


def build_splits():
    """
    Blocks are aligned to the 28-day forecast cycle so that every simulated day
    is covered by a freshly produced forecast and every cost window is an exact
    number of periods:

    TRAIN : t in [0, 1465)      forecaster fitting only
    VAL   : t in [1465, 1829)   364 d = 13 x 28-day periods; all tuning and selection
    TEST  : t in [1829, 1969)   140 d =  5 x 28-day periods; held out for reporting
    """
    p = 28
    n_test, n_val = 5, 13
    test0 = N_DAYS - n_test * p
    val0 = test0 - n_val * p
    splits = dict(n_days=N_DAYS, period_len=p,
                  train=[0, val0], val=[val0, test0], test=[test0, N_DAYS])
    splits["val_periods"] = n_val
    splits["test_periods"] = n_test
    with open(PROC / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    log(f"splits: {splits}")
    return splits


if __name__ == "__main__":
    t0 = time.time()
    build_calendar()
    meta = build_series_index()
    Y = stream_sales(meta)
    build_stats(Y, meta)
    del Y
    gc.collect()
    build_prices(meta)
    build_splits()
    log(f"TOTAL {time.time()-t0:.0f}s -> {PROC}")
