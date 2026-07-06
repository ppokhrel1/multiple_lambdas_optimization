#!/usr/bin/env bash
# Re-run all experiments under the two-time-scale schedule, using TWO GPUs:
#   1D experiments -> GPU 0     2D experiments -> GPU 1   (run in parallel)
#
#   - model theta:  eta_theta  = 1e-5   (fast)
#   - router lambda: eta_lambda = 1e-7  (slow; ~100x separation, Assumption 4)
#   - per-epoch StepLR decay (gamma=0.5 every 150 epochs) => diminishing steps.
#
# Budgets match the manuscript tables:
#   multi_experts / multi_resolution / large_scale : B = 3.0
#   data_assimilation : B = 2.0 (1D), 2.5 (2D)
#
# Output OVERWRITES (>) results/<exp>/<budget>.txt  -- do NOT use >> (append mixes runs).
set -euo pipefail
cd "$(dirname "$0")"

run_one () {  # dim exp budget   (CUDA_VISIBLE_DEVICES set by run_dim)
  local dim=$1 exp=$2 b=$3
  mkdir -p "$dim/results/$exp"
  echo ">>> [gpu ${CUDA_VISIBLE_DEVICES}] $dim/$exp budget=$b"
  ( cd "$dim" && python3 -u "$exp.py" "$b" > "results/$exp/$b.txt" 2>&1 )
  echo "    -> $dim/results/$exp/$b.txt"
}

run_dim () {  # dim gpu da_budget   -- runs this dimension's experiments sequentially on one GPU
  local dim=$1 gpu=$2 da_b=$3
  export CUDA_VISIBLE_DEVICES=$gpu
  run_one "$dim" multi_experts    3.0
  run_one "$dim" multi_resolution 3.0
  run_one "$dim" large_scale      3.0
  run_one "$dim" data_assimilation "$da_b"
}

echo "Launching 1D on GPU 0 and 2D on GPU 1 in parallel..."
run_dim 1d 0 2.0 > run_1d.log 2>&1 &  P1=$!
run_dim 2d 1 2.5 > run_2d.log 2>&1 &  P2=$!

# wait on both; fail loudly if either dimension errors out
fail=0
wait "$P1" || { echo "1D (GPU 0) FAILED -- see run_1d.log"; fail=1; }
wait "$P2" || { echo "2D (GPU 1) FAILED -- see run_2d.log"; fail=1; }
[ "$fail" -eq 0 ] || exit 1

echo "ALL DONE. Progress in run_1d.log / run_2d.log; regenerate figures from ../plot_*.py"
