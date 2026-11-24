# DriveBench

DriveBench is a public GPU benchmark and Gymnasium environment for choosing operator configurations under a trial budget. Download measured GPU results, compare search policies on CPU, or collect the same workloads on another NVIDIA GPU.

[Dataset and results](results/2026-11-23-operator-generalization-pro-4000/README.md) · [Environment and collection](docs/operator-generalization.md) · [Website](https://drivebench-research.rahul-a-mahadik.chatgpt.site) · [Citation](CITATION.cff)

## Measured benchmark

The current dataset contains **60 operator workloads**, **24 Triton launch configurations**, compiled PyTorch and an eager reference. GEGLU and GroupNorm+SiLU each contribute ten shapes in float16, bfloat16 and float32. Training, validation and test sets contain 30, 12 and 18 workloads; all dtypes of a shape stay in the same split.

Three fresh processes on an **NVIDIA RTX PRO 4000 Blackwell** produced **4,680 measurements** across 1,560 workload/configuration cells. Each successful measurement retains 30 CUDA-event samples, allocator memory, preparation time and numerical checks against eager. All collected configurations passed those checks.

The six evaluated selectors are a compiled-query baseline, random search, Optuna TPE, a ridge cost model trained on the training split, and two pretrained LLM selectors. The compiled-query baseline had the lowest mean test regret among the four methods evaluated over 20 episodes per task. The baseline starts with eager, queries compiled and retains its observed incumbent. The LLM comparison uses a separate, matched one-episode-per-task panel. See the [results](RESULTS.md) for the measurements and definitions.

Configuration rankings changed on a second GPU: across the 18 test workloads, mean rank correlation between RTX PRO 4000 and RTX 4090 was 0.301, and only 6 source-optimal choices remained target-optimal. The [transfer dataset and plots](results/2026-11-23-operator-generalization-pro-4000/README.md#transfer-to-rtx-4090) retain all 1,404 target measurements.

## Run a comparison without a GPU

```sh
git clone https://github.com/rahul-mahadik/driving-world-model-inference.git
cd driving-world-model-inference
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e . -e './extensions/kernel_tuning[search]'

tar -xzf results/2026-11-23-operator-generalization-pro-4000/data.tar.gz -C results

python -m drivebench_tuning.generalization_cli run \
  --suite operator-generalization-v2 \
  --table results/2026-11-23-operator-generalization-pro-4000/table.json \
  --protocol protocols/operator-generalization-v2-compile-isolation-v1.json \
  --output ./policy-comparison
```

The archive includes the measured table, original sample values, policy trajectories and trained selector. Its manifest records every extracted file's size and SHA-256. Replay uses recorded GPU measurements and identifies the source of each observation; running more policy episodes adds no independent GPU samples.

## Use the environment

`BudgetedTuningEnv` implements Gymnasium's `reset`/`step` interface. The published task gives a policy **eight trials** to choose among 25 configurations; the environment accepts other trial budgets. Observations expose the workload, remaining budget and queried outcomes. Reward measures improvement in the best numerically valid configuration found. Repeated, rejected and failed actions consume trials.

Use the [Python example](docs/operator-generalization.md#use-the-gymnasium-interface) to connect a policy. The same interface supports measured-table replay and live CUDA evaluation. It supports RL training; the published learned selector was trained by supervised regression, not RL. The supplied configuration actions do not execute arbitrary agent-written code.

To collect fresh measurements, follow the [GPU collection instructions](docs/operator-generalization.md#measure-the-workloads). Set `TORCHINDUCTOR_COMPILE_THREADS=1` before launching each GPU process. The runner checks that setting, records runtime provenance and separates preparation from steady-state timing. Compute provisioning and job deadlines remain the caller's responsibility.

## Video-model coverage

The repository also provides a [Vista inference adapter](docs/modern-vista.md), a [full-model measurement protocol](docs/modern-baseline-measurement.md), and [agent tools](docs/agent-environment.md). These are separate from the operator dataset. Operator numerical correctness does not establish video quality preservation or an end-to-end model speedup. Earlier Vista measurements remain in the [evidence archive](research/archive/README.md).

First-party code is [MIT licensed](LICENSE). Model weights, datasets and third-party software retain their [separate licenses](NOTICE.md). See [CONTRIBUTING.md](CONTRIBUTING.md) for reproducible submissions.
