# Contributing benchmark components

Contributions can add workloads, evaluators, search policies, examples or checks. First-party code is [MIT licensed](LICENSE); preserve [third-party notices](NOTICE.md) and contribute material you have the right to share.

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

Policies consume observations and choose candidate indices. Compare them with equal candidate budgets and matched task/runtime/objective identities. Include search without replacement and a fixed-order baseline, with additional baselines appropriate to the method. Report training cost and choose policy settings on validation data before testing.

The [agent interface](docs/agent-environment.md) also accepts structured `inspect`, `propose`, `profile`, `trial` and `submit` actions. Keep policy-visible observations explicit and small, excluding evaluator paths, raw exceptions, credentials and final inputs. State whether the policy uses pretrained inference or additional RL training. For a new executor, provide a synthetic integration that exercises its evidence reader and scoring functions. Claims of GPU execution require a separately recorded real-runtime validation.

Update the [readiness table](docs/benchmark-readiness.md) with the validation performed and links to its evidence. Publish unsuccessful candidates and inconclusive scores with the same provenance as successes.

## Packaging and review

Build both wheels and run their console demos from outside the checkout in an environment without Torch. Keep resource files in package data. Verify redistribution rights before including datasets, model weights or generated media. Exclude secrets, provider identities and private run logs. Include public contract tests and exact reproduction instructions for reported results.
