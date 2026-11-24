# PPO replay figures

`analysis.json` contains the replay analysis used to generate these figures.
Plotting requires no GPU execution.

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
The displayed bootstrap intervals resample six held-out shape families and
condition on the measured cohort, fixed checkpoints and aggregated repeats.
They exclude uncertainty over training seeds or fresh timing runs. Individual
seed points show the variation across all five trained policies.

## Live comparison figures

The same script accepts a completed JSON summary produced by
`drivebench_tuning.ppo_analysis.live_comparison`:

```sh
python scripts/plot_ppo_supplement.py \
  --kind live \
  --analysis /path/to/completed-live-analysis.json \
  --live-gpu-label 'RTX 4090' \
  --output reports/figures
```

Use the GPU label recorded for that input. Validation requires completed
analysis of all 18 tasks and three processes, including any confirmation
failures. `ppo-native-speedup` shows the three within-process paired latency
ratios and their task median. `ppo-tuning-cost` shows wall-clock selection time
for native search over 24 kernel configurations plus three reference probes,
and PPO with eight queries plus an eager reset.

Selection time includes unequal logging overhead and shared compilation-cache
effects; it excludes offline PPO training and final confirmation. The live
comparison reports latency ratios without a full-grid oracle for regret.
