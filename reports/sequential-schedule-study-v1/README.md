# Live schedule-search comparison

This study compares random walks and greedy local search with an unchanged eager/compiled reference in `DriveBench/KernelSchedule-v0`. Each search action edits the current schedule. Final scoring uses new measurements of the selected schedule and the fixed reference.

The [protocol](protocol.json) and [analysis plan](analysis-plan.json) were fixed before GPU execution. Measurements are pending. The existing configuration-bandit leaderboard is a separate experiment.

## Workloads and methods

The study uses six float16 TEST workloads from `operator-generalization-v2`: three GEGLU shapes and three GroupNorm+SiLU shapes. Each method runs with search and timing-sampling seeds 7101, 7102 and 7103, for 54 episodes. Each workload keeps its fixed input seed. Policy order rotates across task and seed. The tasks and seeds are public; this is a comparison on declared held-out shape families, not a hidden evaluation set.

| Method | Procedure |
| --- | --- |
| Eager/compiled reference | Choose the correct reference with the lower median during setup, then keep that implementation fixed. |
| Random walk | Apply uniformly sampled legal edits and retain the schedule with the lowest observed timing sample. |
| Greedy local search | Explore seeded, shuffled one-edit neighbors of the current best schedule; restore it through legal edits after a nonimproving trial. |

Each episode has a 90-second wall-time ceiling, including reference setup, compilation, policy decisions, edits, failed queries, confirmation and cleanup. TorchInductor and Triton filesystem caches start empty for each episode and persist within it; this does not reset every driver or hardware cache. Search reserves the last 20 seconds for confirmation. A secondary 128-query safety cap bounds exploration; its activation is reported. Methods can finish early, so actual costs accompany the equal ceilings.

The initial fused schedule and all available transformations are defined in the [environment specification](../../extensions/kernel_tuning/docs/sequential-schedules.md). Returning to an earlier schedule requires legal, charged edits. A repeated measurement collects fresh samples. The reference comparator uses the same setup and confirmation machinery, and its score is reported separately from environment return.

## Measurement and analysis

Each successful query records seven CUDA-event samples after two warmups and a numerical check against eager PyTorch. Search sees a sampled timing. Final scoring uses the ratio of the selected schedule's median to a fresh median of the fixed reference, measured immediately beforehand. The reference-only method also makes two fresh confirmation probes. Confirmation observations do not change the chosen schedule.

The primary score is `clip(1 - candidate_median / reference_median, -1, 1)`. An unsuccessful or unconfirmed episode receives -1. The report includes every attempted episode, conditional latency ratios, elapsed time, failed queries and invalid actions. The environment's sampled reward and cumulative return remain separate from this median-based study score.

Means give each task equal weight. The registered analysis uses 10,000 paired hierarchical bootstrap draws: six task families sampled with replacement, then three seeds within each sampled family. All methods share the same resampling indices. The report gives percentile 95% intervals for method scores and paired differences. Six families provide limited information about variation across workloads; these intervals do not establish transfer to other operators or devices.

A hardware/runtime change stops the campaign and prevents pooled scoring. Interrupted attempts remain in the evidence. This comparison evaluates two simple search baselines on one cohort; it trains no RL policy and measures no exhaustive oracle or end-to-end video latency.

## Reproduction

Install the extension's GPU and test dependencies on a compatible Linux/NVIDIA machine, then run the CUDA checks and frozen study:

```sh
python -m pip install './extensions/kernel_tuning[gpu,test]'
export TORCHINDUCTOR_COMPILE_THREADS=1
python -m pytest -q extensions/kernel_tuning/tests/test_schedule_kernels.py \
  extensions/kernel_tuning/tests/test_schedule_measurement.py -m gpu
# Continue only if all 28 CUDA cases pass, with no skips.
python -m drivebench_tuning.schedule_study \
  --output ./schedule-study \
  --cache-root ./schedule-cache \
  --max-seconds 5700 \
  --protocol reports/sequential-schedule-study-v1/protocol.json \
  --expected-protocol-sha256 428153e29b8644c12d26c578bd8b8f06dda6e1280a476fb09411168a4929e803
```

Use new output and cache directories and an external process deadline. The runner checks the protocol against the installed scientific source. `episodes.jsonl` and the per-episode journals retain the trajectories; `summary.json` and `completed.json` or `partial.json` record accounting and completion. A completed campaign can include failed episodes.
