#!/usr/bin/env bash
# run_sensitivity.sh -- policy-side and cost-model sensitivity suite.
#
# All runs reuse the forecast cache produced by the main run, so no model is
# refitted: each configuration only re-executes the inventory simulation. This
# keeps the comparison clean -- the forecasts are held fixed and only the
# decision environment changes, which is exactly what a sensitivity analysis of
# the *decision* layer should do.
set -u
cd "$(dirname "$0")"

BASE="--n_series 4000 --blocks val,test --econ margin --reuse_forecasts main --hw_tune 250"

run () {
  name="$1"; shift
  echo "=============================================================="
  echo "=== SENSITIVITY: $name  ($*)"
  echo "=============================================================="
  python3 -u run_experiment.py $BASE --out "sens_$name" "$@" \
      2>&1 | grep -vE "OpenBLAS|RuntimeWarning|nanmean|nanvar|s_size" | tail -6
}

run lead7      --lead 7
run lead14     --lead 14
run K5         --K 5.0
run review14   --review 14
run holdrate40 --hold_rate 0.40
run costratio75 --cost_ratio 0.75
run penmult3   --penalty_mult 3.0
run sfrac30    --s_frac 0.3
run ddmrp80    --ddmrp_buffer 0.8

echo "=== ALL SENSITIVITY RUNS COMPLETE ==="
