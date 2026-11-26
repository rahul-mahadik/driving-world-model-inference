# Live sequential schedule comparison

This registered study completed 54 live RTX 4090 episodes comparing a fixed eager/compiled reference, random walks and greedy local search on six float16 TEST shape families across two known operators. Each method received three search/timing seeds and a 90-second wall-clock ceiling, including setup, compilation, search, failures and fresh final measurements. Each workload used a fixed input seed. Forty-eight submissions were confirmed; six random-walk queries timed out. See the [released results, source and trajectories](../../results/2026-11-26-sequential-schedules/README.md).

The runner admits an exploration query only when its full 15-second timeout and the 20-second final-measurement reserve fit, checked before and after schedule edits. It stops at 128 exploration queries if that safety cap is reached first. Actual time and cap hits are reported.

The primary score is `clip(1 - candidate latency / fresh reference latency, -1, 1)`; failed or unconfirmed episodes score −1. The analysis uses 10,000 paired hierarchical bootstrap resamples over six task families and three nested seeds. This is a descriptive operator comparison on one GPU.

The fixed reference had the highest observed mean score (−0.0306), followed by greedy local search (−0.1640) and random walks (−0.4346). The paired reference-minus-greedy interval [−0.0660, 0.3512] includes zero. The study does not evaluate a trained sequential RL policy or establish transfer to unseen operators.

See [protocol](protocol.json), [registered analysis plan](analysis-plan.json), and [environment specification](../../extensions/kernel_tuning/docs/sequential-schedules.md).

## Reproduce

Recompute the released scores and intervals on a CPU:

```sh
python scripts/recompute_schedule_results.py --summary results/2026-11-26-sequential-schedules/summary.json --episodes results/2026-11-26-sequential-schedules/episodes.json
```

The command checks numerical consistency of the public arrays and statistics using the published submission outcomes. Repeating the GPU study requires the frozen source and its protocol:

```sh
python -m drivebench_tuning.schedule_study --write-protocol ./protocol.json
python -m drivebench_tuning.schedule_study --output ./study --cache-root ./scratch-caches --protocol ./protocol.json --expected-protocol-sha256 <printed-hash>
python scripts/analyze_schedule_study_v2.py --study ./study --source-manifest ./frozen/source-manifest.json --frozen-source ./frozen/source-payload --analysis-plan reports/sequential-schedule-study-v2/analysis-plan.json --output ./analysis
```

The current registered protocol binds the collected source. A changed source needs a new prospective protocol and separate results.

## Revision history

The original v1 comparison was interrupted after 13 attempts when its query-admission rule was found to shrink timeouts below normal measurement cost, terminating workers before final confirmation. All 13 attempts and the passing 28-case CUDA validation are preserved separately. V2 changes only that admission rule and its source/protocol binding; tasks, seeds, policies, budgets, scoring and bootstrap remain unchanged. V1 and v2 are not pooled.
