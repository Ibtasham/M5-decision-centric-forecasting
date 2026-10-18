#!/usr/bin/env bash
# reproduce.sh -- regenerate every table and figure in paper/manuscript.md.
#
# Total runtime ~30 minutes on 2 CPU cores within a 1 GB RAM cap.
# Set -e so that a failure stops the pipeline rather than silently producing
# tables from a partial run.
set -euo pipefail
cd "$(dirname "$0")/src"

FILTER='OpenBLAS|RuntimeWarning|nanmean|nanvar|s_size'

echo "############ 1/7  fetch and verify the M5 dataset ############"
python3 -u fetch_m5.py

echo "############ 2/7  prepare the binary panel ############"
python3 -u 01_prepare_data.py 2>&1 | grep -vE "$FILTER"

echo "############ 3/7  end-to-end plumbing check ############"
python3 -u smoke_test.py 2>&1 | grep -vE "$FILTER" | tail -20

echo "############ 4/7  main experiment (val + test, 2 cost models, 59 pairings) ############"
python3 -u run_experiment.py \
    --n_series 4000 --lgbm_estimators 250 \
    --blocks val,test --out main \
    --hw_tune 250 --hw_window 336 \
    --econ margin,cr90 --cache_forecasts --p1_sweep 2>&1 | grep -vE "$FILTER"

echo "############ 5/7  sensitivity suite (reuses the cached forecasts) ############"
bash run_sensitivity.sh

echo "############ 6/7  robustness: second sampling seed ############"
python3 -u run_experiment.py \
    --n_series 4000 --seed 1 --lgbm_estimators 250 \
    --blocks val,test --out seed1 --hw_tune 250 --hw_window 336 \
    --econ margin 2>&1 | grep -vE "$FILTER"
python3 -u analysis.py --runs seed1 --test_run seed1 --econ margin --out seed1 \
    2>&1 | grep -vE "$FILTER" | tail -40

echo "############ 7/7  analysis, selection, inference, collection ############"
python3 -u analysis.py  --runs main --test_run main --econ margin --out main 2>&1 | grep -vE "$FILTER"
python3 -u analysis.py  --runs main --test_run main --econ cr90   --out main_cr90 2>&1 | grep -vE "$FILTER"
python3 -u selection.py --run main --econ margin --out main 2>&1 | grep -vE "$FILTER"
python3 -u bootstrap.py --run main --econ margin --reps 2000 --out main 2>&1 | grep -vE "$FILTER"
python3 -u bootstrap.py --run seed1 --econ margin --reps 2000 --out seed1 2>&1 | grep -vE "$FILTER"
python3 -u selection.py --run seed1 --econ margin --out seed1 2>&1 | grep -vE "$FILTER"
python3 -u collect.py 2>&1 | grep -vE "$FILTER"
python3 -u p1_tuned.py --run main --reps 2000 2>&1 | grep -vE "$FILTER"
python3 -u regime_oos.py --run main 2>&1 | grep -vE "$FILTER"
python3 -u make_figures.py 2>&1 | grep -vE "$FILTER"
python3 -u make_fig_pipeline.py
python3 -u make_tables.py 2>&1 | grep -vE "$FILTER"
(cd ../paper && python3 build.py) || echo 'paper build skipped (needs pandoc, python-docx, playwright+chromium)'
cd ../src

echo
echo "############ DONE ############"
echo "Tables and figures:  results/main/   (TT1..T14, S1, S2, fig*.png)"
echo "Second-seed check:   results/seed1/"
echo "Machine-readable:    results/paper_numbers.json"
