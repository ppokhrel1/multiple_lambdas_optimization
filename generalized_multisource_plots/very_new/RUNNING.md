# Running the two-time-scale experiments

## What changed vs. the previous runs

The optimizer was split so the experiments genuinely realize the two-time-scale
stochastic-approximation regime the theory (Theorem 2, Assumption 4) assumes:

| group | parameters (matched by name) | learning rate |
|-------|------------------------------|---------------|
| model `theta` | everything except `router.*` and `lam*` (the experts / networks) | `ETA_THETA  = 1e-5` (fast) |
| source weights `lambda` | `router.*` (produces the per-source weights) | `ETA_LAMBDA = 1e-7` (slow) |
| multipliers | `lam_sum`, `lam_budget` | dual ascent, unchanged |

The model and router now live in **separate Adam parameter groups** of the same
optimizer, so `theta` moves ~100x faster than `lambda` (`eta_lambda << eta_theta`).
A per-epoch `StepLR` (`gamma=0.5` every 150 epochs) decays both groups together,
preserving their ratio while giving the diminishing-step behaviour the exact
convergence rate needs. To tighten toward the literal `eta_lambda = eta_theta^2`,
set `ETA_THETA = 1e-3` and `ETA_LAMBDA = 1e-6` at the top of each `train_combiner`.

## How to run

Requires the same environment as before (PyTorch; a GPU is used automatically if
available). Each script takes the **budget** as its only argument and writes a log
to stdout. Run scripts from inside their dimension folder (`1d/` or `2d/`) because
they `import` from `multi_resolution.py` in the same directory.

### Everything at once (recommended, 2 GPUs)

```bash
cd generalized_multisource_plots/very_new
bash run_two_timescale.sh
```

This launches the **1D** experiments on **GPU 0** and the **2D** experiments on
**GPU 1** in parallel (via `CUDA_VISIBLE_DEVICES`); within each GPU the four
experiments run sequentially. Progress goes to `run_1d.log` / `run_2d.log`.
To use a single GPU instead, run the two `run_dim` lines without `&`.

It runs all 8 experiments at the manuscript budgets and writes/overwrites:

```
<dim>/results/multi_experts/3.0.txt
<dim>/results/multi_resolution/3.0.txt
<dim>/results/large_scale/3.0.txt
1d/results/data_assimilation/2.0.txt
2d/results/data_assimilation/2.5.txt
```

### One experiment manually

```bash
cd generalized_multisource_plots/very_new/1d
python -u multi_experts.py 3.0 > results/multi_experts/3.0.txt   # NOTE: '>' overwrite, not '>>'
```

Budgets that match the paper's tables: `3.0` for multi_experts / multi_resolution /
large_scale; `2.0` (1D) and `2.5` (2D) for data_assimilation.

> Use `>` (overwrite). The old `run.sh` used `>>` (append) — re-running with `>>`
> concatenates new output onto the previous run and the plot/table parsers then
> read two mixed runs.

## Regenerate figures and tables

The plotting scripts read `very_new/<dim>/results/...` relative to
`generalized_multisource_plots/`:

```bash
cd generalized_multisource_plots
python plot_multi_experts.py
python plot_multi_resolution.py
python plot_large_scale.py
python plot_da_results.py
```

Then rebuild Table 3 from the refreshed `3.0.txt` logs before recompiling the
manuscript.
