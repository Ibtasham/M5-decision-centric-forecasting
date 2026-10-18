"""
fetch_m5.py
===========
Downloads the M5 Forecasting - Accuracy dataset.

The canonical source is Kaggle (https://www.kaggle.com/competitions/m5-forecasting-accuracy/data),
which requires authentication. This script uses the public mirror maintained by
Nixtla, which redistributes the identical competition files:

    https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip   (~50 MB)

If you prefer Kaggle, run instead:
    pip install kaggle && kaggle competitions download -c m5-forecasting-accuracy
and unzip into the same directory.

Expected contents after download (verified by this script):
    calendar.csv                     1,969 days
    sell_prices.csv                  6,841,121 rows
    sales_train_evaluation.csv       30,490 series x d_1..d_1941
    sales_test_evaluation.csv        30,490 series x d_1942..d_1969
    sales_train_validation.csv       the same panel truncated 28 days earlier
    sales_test_validation.csv
    weights_evaluation.csv           42,840 rows (12 hierarchy levels)
    weights_validation.csv
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RAW   # noqa: E402

URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
EXPECTED = {
    "calendar.csv": 1969,
    "sell_prices.csv": 6841122,
    "sales_train_evaluation.csv": 30491,
    "sales_test_evaluation.csv": 30491,
    "sales_train_validation.csv": 30491,
    "sales_test_validation.csv": 30491,
    "weights_evaluation.csv": 42841,
    "weights_validation.csv": 42841,
}


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        n = 0
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            n += len(b)
    print(f"downloaded {n/1e6:.1f} MB in {time.time()-t0:.0f}s -> {dest}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    missing = [f for f in EXPECTED if not (RAW / f).exists()]
    if not missing:
        print(f"all M5 files already present in {RAW}")
    else:
        print(f"missing {len(missing)} files; downloading archive")
        z = RAW / "m5.zip"
        if not z.exists():
            download(URL, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW)
        print("extracted")

    # verify line counts so a truncated download cannot silently propagate
    bad = []
    for name, want in EXPECTED.items():
        p = RAW / name
        if not p.exists():
            bad.append(f"{name}: ABSENT")
            continue
        with open(p, "rb") as f:
            got = sum(1 for _ in f)
        if got != want:
            bad.append(f"{name}: {got} lines, expected {want}")
        else:
            print(f"  ok  {name:32s} {got:>9,} lines")
    if bad:
        raise SystemExit("VERIFICATION FAILED:\n  " + "\n  ".join(bad))
    print(f"\nM5 dataset verified in {RAW}")
    print("next: python3 01_prepare_data.py")


if __name__ == "__main__":
    main()
