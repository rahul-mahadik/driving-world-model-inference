# PPO replay figures

`analysis.json` is the byte-identical analysis input used for these figures. It
summarizes recorded operator measurements; plotting and replay do not add GPU
timing observations.

From the repository root, with Matplotlib installed:

```sh
python scripts/plot_ppo_supplement.py \
  --kind replay \
  --analysis reports/ppo-supplement-replay/analysis.json \
  --output reports/figures
```

The command writes PNG, SVG and PDF versions of:

- `ppo-regret`: all five PPO training seeds and their mean against Optuna TPE,
  random search and the compiled-query baseline. The untrained initialization
  comparison uses a separate panel and scale.
- `ppo-seed-spread`: every trained seed, the all-seed mean and the comparison
  policies after eight queries.

`reports/figures/ppo-plot-receipt.json` records the input, plotter and output hashes.
The displayed intervals are exploratory bootstrap intervals over six held-out
shape families. They condition on the measured cohort, fixed checkpoints and
aggregated repeats; they do not estimate uncertainty over training seeds or fresh
timing runs. All five PPO seeds are included without selecting the best one.

## Plotting the separate live comparison

The same script accepts a completed JSON summary produced by
`drivebench_tuning.ppo_analysis.live_comparison`:

```sh
python scripts/plot_ppo_supplement.py \
  --kind live \
  --analysis /path/to/completed-live-analysis.json \
  --live-gpu-label 'RTX 4090' \
  --output reports/figures
```

Use the GPU label verified for that input. No live figure is generated without
the completed analysis. Validation requires all 18 tasks and three processes,
including explicit confirmation failures. `ppo-native-speedup` shows the three
within-process paired latency ratios and their task median; `ppo-tuning-cost`
shows recorded wall-clock selection time. Native search uses 24 kernel
configurations plus three reference probes, while PPO uses eight queries plus
an eager reset. These are unequal search budgets. Offline PPO training and final
confirmation time are excluded from the live selection-time comparison. The
plots report neither a live oracle nor live regret.
