# Contributing benchmark components

This project prioritizes reusable measurement and search infrastructure. Contributions can add workloads, adapters, policies, examples or checks. You do not need to write a custom GPU kernel to participate. First-party code is [MIT licensed](LICENSE); preserve [third-party notices](NOTICE.md) and only contribute material you have the right to share.

## Development

```sh
python -m pip install -e '.[test,plots]' -e 'extensions/kernel_tuning[test]'
python -m pytest -q tests
python -m pytest -q extensions/kernel_tuning/tests
```

CUDA tests skip explicitly without NVIDIA hardware. CPU success does not validate GPU kernels, new workload performance or video quality. A separate interface-only environment can install `extensions/kernel_tuning[interface-test]` without Torch and exercise the search, evaluation, suite, adapter and CLI contract tests. Install the root package too for the inference-bridge tests and example.

## Workloads and implementations

- Use the [external evaluator contract](extensions/kernel_tuning/docs/extending.md). Keep the baseline, quality checks, numerical tolerances and timing/memory boundaries explicit.
- Give a workload suite a version and record its exact hash. Split by real workload groups before measuring; changing seeds alone is not shape generalization.
- Benchmark accepted and rejected results honestly. Unsupported configurations should fail explicitly. Never silently substitute a different implementation.
- Report compilation, training and evaluation costs separately. Preserve partial attempts and raw samples. Independently confirm selected candidates before claiming improvement.
- Keep historical reports and sealed artifacts unchanged. New experiments use new output paths and new cohort identities.

## Policies

Policies consume observations and choose candidate indices. Compare them with equal candidate budgets and matched task/runtime/objective identities. Include search without replacement and a fixed-order baseline; stronger baselines may be needed for scientific claims. Report training cost and use validation for choices before test. No particular RL algorithm is required to use the environment.

## Packaging and review

Build both wheels and run their console demos from outside the checkout in an environment without Torch. Keep resource files in package data. Check licenses/notices and omit datasets, model weights, generated media, secrets, provider identities and private run logs from releases. Include public contract tests and exact reproduction instructions for any result claimed.
