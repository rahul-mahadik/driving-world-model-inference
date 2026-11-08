# GPU inference and RL kernel tuning

The [fresh GPU study](../../results/2026-11-07-kernel-tuning/README.md) now supplies operator correctness, allocation measurements and GPU-backed RL search evidence. The implementation-stage descriptions below are preserved as historical context; the dated result report states current limits. No whole-model or RL-superiority claim is made.

| Current evidence | Result |
| --- | --- |
| Revised correctness suite | 188 passed, including 75 GPU tests; no skips. Original constant-input eager contract remains failed. |
| Operator allocation | 6 of 6 workloads confirmed lower incremental peak allocation than eager in both orders; compiled comparisons and timing regressions are also reported. |
| RL configuration search | GPU-backed Q-learning and fresh equal-budget evaluation completed; no superiority or generalization claim. |
| Whole-model custom-kernel benefit | Not evaluated. Historical H100 compiler timing is separate. |

This extension adds custom Triton operators and a Gymnasium tuning environment to the driving-world-model benchmark. It is separate from the frozen `drivebench` implementation and the sealed H100 timing cohort. It never provisions a GPU, changes funding, or restarts an old job.

## Historical implementation-stage evidence — superseded by the table above

| Component | Implementation | Evidence |
| --- | --- | --- |
| Driving-world-model inference benchmark | Existing Vista benchmark: latency, throughput, peak allocated memory, failures, runtime and input identities | Completed H100 comparison: 200 successful requests per role; 57.34 → 43.66 s median, 23.85% lower latency; **7.68% more** peak allocated memory. This is a compiler result. |
| Custom Triton kernels | Fused GEGLU and GroupNorm+SiLU, plus scoped Vista adapter | CPU references/contracts and graph-capture tests pass. NVIDIA CUDA compilation, numerical correctness, performance and memory benefit are **not yet validated**. |
| RL environment | Gymnasium reset/step, configuration actions, measured latency/memory rewards, correctness constraints, trial budgets | CPU interface tests pass, including `check_env`; a real CUDA evaluator and reference Q learner are implemented. GPU-backed training and RL improvement are **not yet demonstrated**. |

Do not attribute the historical compiler improvement to these kernels or to RL. Do not claim quality preservation. Operator-level improvements would not, by themselves, establish a full-video speedup or a reduction in peak model memory.

## Fused operations

- **GEGLU:** leave Vista's linear projection unchanged; fuse exact erf-GELU and multiplication. The kernel retains the intermediate dtype rounding and avoids writing the full GELU activation to GPU memory.
- **GroupNorm+SiLU:** compute per-chunk mean/M2, merge moments, then fuse normalization, affine transformation and SiLU. This uses three launches. It preserves Vista's FP32 normalization → input dtype cast → SiLU boundary. Compact statistics replace full-size normalization intermediates. Different reduction order still needs numerical validation.

These are inference-only kernels; no custom backward is supplied. Supported inputs are contiguous float16, bfloat16 or float32; GEGLU accepts `[..., 2D]`, GroupNorm accepts NCHW with affine parameters. Unsupported layouts, devices and configurations raise errors. There is no silent eager fallback. Launch choices are block sizes 64–4096 and 1/2/4/8 warps; the default search uses eight block-size/warp combinations. GroupNorm permits at most 4096 chunks per group.

The code structurally removes intermediate tensors; **the resulting allocation reduction has not been measured on a GPU**. The benchmark tests that claim against eager and compiled controls rather than assuming a benefit.

## Install and verify locally

Use a separate environment from the historical Vista runtime:

```sh
python -m pip install -e 'extensions/kernel_tuning[test]'
python -m pytest -q extensions/kernel_tuning/tests
```

CPU tests deliberately use references and explicitly marked synthetic evaluator fixtures. Skipped CUDA tests are not passing CUDA evidence. `torch.compile(..., backend="eager", fullgraph=True)` tests graph capture only.

The prospective NVIDIA runtime is **Python ≥3.10, PyTorch 2.10.0, Triton 3.6.0, Gymnasium 1.2.3**, with CUDA/driver identity recorded from the actual host. Install the `[gpu]` extra in that separate runtime. This is a new operator cohort, not the historical Torch 2.0.1 / Triton 2.x Vista cohort. Its provisioned image, package inventory and hardware need fresh verification before results are admitted.

## Measure operators on an admitted GPU

After CUDA correctness tests pass, run in an already authorized, bounded allocation. Apply the existing host timeout and provider guard; the CLI's inner budget cannot interrupt a stuck CUDA/compiler call. Do not run these commands against any retired transport or reuse an old deadline.

```sh
python -m pytest -q extensions/kernel_tuning/tests -m gpu

drivebench-tune benchmark \
  --workloads extensions/kernel_tuning/configs/smoke.json \
  --output outputs/kernel-tuning/smoke-NEW-ID \
  --rounds 3 --repetitions 30 --max-seconds 900

drivebench-tune benchmark \
  --workloads extensions/kernel_tuning/configs/vista_profile_shapes.json \
  --output outputs/kernel-tuning/profile-NEW-ID \
  --rounds 3 --repetitions 30 --max-seconds 1800
```

Output directories must be new. The existing bounded host launcher must bind the extension, workload configuration, independently verified runtime and entrypoint hashes; a new runtime must not be mislabeled with the old environment proof. Do not mutate the old image or launcher to bypass that check.

`smoke.json` contains small engineering shapes. `vista_profile_shapes.json` derives selected shapes from the preserved historical eager trace, with source hashes and derivation notes. Inputs are seeded random tensors, **not recorded model activations**. The historical profile identifies candidate operations, not their current compiled cost. The Vista capture helper can record actual shapes for a later separately admitted model run.

For every configuration the study alternates eager/custom/compiled control order across rounds. It selects among candidates passing every round, then performs fresh confirmation measurements. Incorrect candidates cannot win. Records include:

- Raw CUDA-event samples, synchronized host wall time, p50/p95 and operator invocations/second.
- Peak live allocator increase above resident inputs, output bytes and an estimated temporary allocation overhead. Reserved memory is recorded separately.
- Correctness tolerances, maximum absolute error/RMSE, immutable input hashes, source hashes and physical GPU/runtime binding.
- Started, individual-sample, completed and failed records. Partial attempts are preserved if the run is interrupted.

Compilation, warmups, input generation, correctness checks, host transfers and evidence writes are excluded from latency and consume the job's total time allowance. CUDA events bracket one invocation and can include stream gaps; they are not a sum of hardware kernel durations. Memory figures describe the isolated operator, not the whole model. Allocator caching is not memory savings.

## RL task

`KernelTuningEnv` is a finite-budget configuration search task, not a driving simulator. One action selects `(block_size, num_warps)`. Every step invokes the evaluator again; repeated actions consume trials and get fresh samples. Reset measures an eager baseline for the selected workload.

The observation contains operator/dtype/shape/group features, remaining budget, incumbent metrics, and per-action counts/outcomes. With baseline latency `L0`, baseline peak allocation `M0` and memory weight `w`:

```text
cost = (L/L0 + w * M/M0) / (1+w)
reward = clip(1 - cost, -1, 1)
incorrect/failed configuration reward = -2
```

Correctness is a hard constraint. The baseline stays the incumbent if every configuration is worse. The episode truncates at its trial budget. CUDA feedback must carry a matching workload/configuration, physical-runtime cohort, input hashes and fresh trial ID. Synthetic feedback is disabled by default.

```sh
drivebench-tune tune \
  --workloads extensions/kernel_tuning/configs/smoke.json \
  --output outputs/kernel-tuning/rl-NEW-ID \
  --episodes 12 --trials 8 --memory-weight 0.25 --max-seconds 1800
```

The reference tabular Q learner uses workload identity and trials remaining. Learned, random and grid policies receive equal fresh evaluation trial budgets, with rotated order. Training cost is reported separately. Evaluation uses the same workload keys with new measurements; it does not establish generalization to unseen shapes or models. Best-observed search outcomes are subject to selection bias, so the policy report explicitly leaves `rl_improvement_validated:false` until a separate confirmation study supports that claim.

## Optional Vista integration

`vista_adapter.py` supplies reversible, instance-scoped forward overrides. It checks pinned source hashes, keeps learned weights/projections and state-dict keys, and restores original forwards on normal exit or exceptions. Use an exclusively owned **eager** model in eval mode and under `torch.no_grad()`; do not mix compilation or concurrent use with an active override.

```python
from drivebench_tuning.contract import KernelConfig
from drivebench_tuning.vista_adapter import vista_kernel_overrides

with torch.no_grad():
    with vista_kernel_overrides(eager_model, geglu_config=KernelConfig(256, 4)):
        result = existing_generation_call()
```

This example is an integration seam, not a claimed completed model run. `capture_vista_workloads` records shapes/dtypes/module paths during an ordinary eager pass without saving tensors. Capture and override contexts are separate because fusion bypasses intermediate hooks. Enabling GroupNorm overrides selects recognized adjacent pairs in the passed subtree; use a captured NCHW subtree because Vista also contains unsupported 5D normalization. There is no implicit fallback for those inputs.

## Evidence needed to finish the proposed claims

1. Restore provider access and verify actual resources, remaining funds, current prices and full cleanup reserves. No new allocation/funding is admitted by this document.
2. Run CUDA tests and the bounded operator study in a fresh cohort; review numerical failures and both eager/compiled comparisons.
3. Run the measured tuning environment, preserve its training/evaluation costs, and independently confirm any selected configuration before claiming an RL improvement.
4. A full-model benefit additionally requires a separately labeled integration/correctness/performance study. It cannot reuse the old speed ratio or reverse the old quality rejection.
5. Publish a new reviewed revision when repository access permits. All sealed historical revisions remain immutable.

Source contracts: [Vista GEGLU](https://github.com/OpenDriveLab/Vista/blob/cc9821b4253ca7987c32757613d2fc2448fa9f5d/vwm/modules/attention.py), [Vista GroupNorm32](https://github.com/OpenDriveLab/Vista/blob/cc9821b4253ca7987c32757613d2fc2448fa9f5d/vwm/modules/diffusionmodules/util.py), [Triton fusion tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html), [Gymnasium environment API](https://gymnasium.farama.org/introduction/create_custom_env/), [PyTorch CUDA events](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.Event.html).
