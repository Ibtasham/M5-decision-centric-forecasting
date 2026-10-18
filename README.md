# Beyond Forecast Accuracy: Decision-Centric Evaluation of Demand Forecasting for Inventory Replenishment

Code and results for the paper by **Md Ibtashamur Rahman** (Department of Industrial and Production Engineering, Shahjalal University of Science and Technology).

The study simulates a lost-sales, periodic-review inventory system on 3,999 stratified series from the public M5 data. It compares 9 point forecasters, a trained quantile model and 3 calibrated empirical-interval variants under 4 replenishment policies (59 pairings), and measures the cost of choosing forecasters by accuracy instead of by decision cost.

## Layout

| Path | Content |
|---|---|
| `src/` | data preparation, forecasters, simulator, experiment runner, analysis, bootstrap, selection, tuned normal-theory and regime checks, figure and table scripts |
| `results/` | result tables (CSV/JSON) and figures (PNG) used in the paper: `main/`, `seed1/` (independent sample), `sens_*/` (sensitivity), `paper_numbers.json` |
| `paper/` | manuscript source (`manuscript_src.md`), generated tables, build script, submission DOCX/PDF, cover letter, review report |
| `reproduce.sh` | regenerates everything |

Large intermediate files (forecast caches, per-series cost arrays) are not stored in the repository; `reproduce.sh` recreates them.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash reproduce.sh
```

Needs about 30 minutes on 2 CPU cores and 1 GB of RAM. Step 1 downloads M5 (about 50 MB) from the Nixtla mirror into `data/` (set `M5INV_DATA` to use another location). The original source is [Kaggle M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy); the data are not redistributed here and keep their own terms. Building the manuscript files at the end of `reproduce.sh` also needs `pandoc` and Chromium and is skipped if they are missing.

Seeds are fixed (sampling 0, LightGBM 0, bootstrap 0). `python src/smoke_test.py` runs a quick end-to-end plumbing check.

## Key results (test block, margin economics)

- Accuracy-driven default (best RMSSE forecaster + normal-theory base stock): +140.4% cost vs the cost-optimal pairing, 95% CI [128.8, 153.4].
- A tuned normal-theory rule is still +25.8% above the optimum.
- Post-hoc calibrated quantiles: +21.8% cost; trained quantiles: −28.9%.
- Per-SKU selection: +34.9% vs one portfolio-level choice.

## Citation

See `CITATION.cff`. A Zenodo DOI for the release will be added to the paper's data availability statement. Code is MIT-licensed.
