# Live sequential schedule comparison

This registered study compares a fixed eager/compiled reference, random walks and greedy local search on six float16 TEST workloads. Each method gets three seeds and a 90-second wall-clock ceiling, including setup, compilation, search, failures and fresh final measurements. Measurements are pending.

The runner admits an exploration query only when its full 15-second timeout and the 20-second final-measurement reserve fit, checked before and after schedule edits. It stops at 128 exploration queries if that safety cap is reached first. Actual time and cap hits are reported.

The primary score is `clip(1 - candidate latency / fresh reference latency, -1, 1)`; failed or unconfirmed episodes score −1. The analysis uses 10,000 paired hierarchical bootstrap resamples over six task families and three nested seeds. This is a descriptive operator comparison on one GPU.

See [protocol](protocol.json), [registered analysis plan](analysis-plan.json), and [environment specification](../../extensions/kernel_tuning/docs/sequential-schedules.md).

## Reproduce

```sh
python -m drivebench_tuning.schedule_study --write-protocol ./protocol.json
python -m drivebench_tuning.schedule_study --output ./study --cache-root ./scratch-caches --protocol ./protocol.json --expected-protocol-sha256 <printed-hash>
python scripts/analyze_schedule_study_v2.py --study ./study --source-manifest ./frozen/source-manifest.json --frozen-source ./frozen/source-payload --analysis-plan reports/sequential-schedule-study-v2/analysis-plan.json --output ./analysis
```

The current registered protocol binds the collected source. A changed source needs a new prospective protocol and separate results.

## Revision history

The original v1 comparison was interrupted after 13 attempts when its query-admission rule was found to shrink timeouts below normal measurement cost, terminating workers before final confirmation. All 13 attempts and the passing 28-case CUDA validation are preserved separately. V2 changes only that admission rule and its source/protocol binding; tasks, seeds, policies, budgets, scoring and bootstrap remain unchanged. V1 and v2 are not pooled.
