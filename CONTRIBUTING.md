# Contributing benchmark components

Contributions can add workloads, evaluators, search policies, examples or checks. First-party code is [MIT licensed](LICENSE); preserve [third-party notices](NOTICE.md) and contribute material you have the right to share.

## Submit a policy result

Evaluate the `operator-generalization-v2` test split with eight candidate trials
and twenty episodes per workload. Use the published RTX PRO 4000 table,
latency-only cost, and evaluation seed 2101. Choose policy settings on the
training/validation splits before running this evaluation.

Open a pull request with the complete trajectory JSON, policy source or pinned
model version, training data and cost, and a short reproduction command. Put
new results under `results/submissions/<policy-name>/`; keep the published
reference files unchanged. Include every episode, repeated action and failed
query. API-driven policies should also include their prompt and usage totals,
with credentials removed.

We recompute regret from the table, check the episode budget and selected
configurations, and add verified results to the appropriate leaderboard panel.
The main panel uses twenty episodes per workload; smaller evaluations belong in
a separate comparison. Recorded timings are public, so submitted policies must
disclose access to test cells beyond their queried observations.

The [policy guide](docs/operator-generalization.md) describes callbacks and
recorded-table evaluation. A submission uses `run_replay_policy` to keep the
same recorded-repeat schedule as the published baselines.

## Submit a sequential schedule policy

Use `DriveBench/KernelSchedule-v0` to edit the current schedule, measure it, and
submit a final candidate. Policies may use the observation, legal-action mask
and their own history. Keep the evaluator and numerical checks unchanged.

For comparison with the [live schedule study](reports/sequential-schedule-study-v2/README.md),
use its six workloads, three seeds and 90-second episode ceilings. Include
reference setup, compilation, policy execution, failed queries and worker
cleanup in elapsed time. Reserve time for a fresh reference measurement and
final candidate confirmation; do not admit a query whose full timeout cannot
fit before that reserve.

Submit the prospective protocol, policy code or pinned model, runtime and cache
settings, every action and measurement, and all failed or unconfirmed episodes.
Use a new result directory and retain the original journals privately when
they contain machine identifiers; publish an allowlisted export. The study's
primary score includes failed submissions. It is not oracle regret, and results
from different GPUs or software cohorts belong in separate comparisons.

## Development

```sh
python -m pip install -e '.[test,plots]' -e 'extensions/kernel_tuning[test]'
python -m pytest -q tests
python -m pytest -q extensions/kernel_tuning/tests
```

CUDA tests require NVIDIA hardware and skip when it is unavailable. Use them to validate GPU execution separately from the CPU interface tests. For the search, evaluation, suite, adapter and CLI contracts, install `extensions/kernel_tuning[interface-test]` in an environment without Torch. Install the root package too for the inference-bridge tests and example.

## Workloads and implementations

- Use the [external evaluator contract](extensions/kernel_tuning/docs/extending.md). Keep the baseline, quality checks, numerical tolerances and timing/memory boundaries explicit.
- Give a workload suite a version and record its exact hash. Split by workload groups before measuring; shape-generalization tests need held-out shapes.
- Record accepted and rejected configurations. Return explicit failures for unsupported configurations and identify the implementation actually executed.
- Report compilation, training and evaluation costs separately. Preserve partial attempts and raw samples. Independently confirm selected candidates before claiming improvement.
- Keep historical reports and sealed artifacts unchanged. New experiments use new output paths and new cohort identities.

## Policies

Configuration-bandit policies consume observations and choose candidate indices. Compare them with equal candidate budgets and matched task/runtime/objective identities. Include search without replacement and a fixed-order baseline, with additional baselines appropriate to the method. Schedule policies instead choose transformations and use elapsed-time budgets. Report training cost and choose policy settings on validation data before testing.

The [agent interface](docs/agent-environment.md) also accepts structured `inspect`, `propose`, `profile`, `trial` and `submit` actions. Keep policy-visible observations explicit and small, excluding evaluator paths, raw exceptions, credentials and final inputs. State whether the policy uses pretrained inference or additional RL training. For a new executor, provide a synthetic integration that exercises its evidence reader and scoring functions. Claims of GPU execution require a separately recorded real-runtime validation.

Update the [readiness table](docs/benchmark-readiness.md) with the validation performed and links to its evidence. Publish unsuccessful candidates and inconclusive scores with the same provenance as successes.

## Packaging and review

Build both wheels and run their console demos from outside the checkout in an environment without Torch. Keep resource files in package data. Verify redistribution rights before including datasets, model weights or generated media. Exclude secrets, provider identities and private run logs. Include public contract tests and exact reproduction instructions for reported results.
