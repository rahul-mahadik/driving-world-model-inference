# Benchmark coverage

The current release measures operator configuration search and sequential kernel-schedule editing through two Gymnasium tracks. The configuration-bandit track supports replay of published measurements; the schedule track uses a live GPU. Model-level inference uses separate adapters and quality criteria.

## Released evidence

| Component | Measured coverage | Interpretation |
| --- | --- | --- |
| [Operator dataset](../results/2026-11-23-operator-generalization-pro-4000/README.md) | 60 GEGLU and GroupNorm+SiLU workloads, 24 Triton configurations, compiled PyTorch and eager; 4,680 measurements from three fresh processes on RTX PRO 4000 | Numerical agreement and isolated-operator costs on seeded tensors |
| [Policy comparison](../results/2026-11-23-operator-generalization-pro-4000/README.md#twenty-episode-policy-comparison) | Compiled-query, random, TPE and ridge selectors over 20 replay episodes per test workload; two pretrained LLM selectors in a matched one-episode panel | Fixed-budget selection on a finite measured table |
| [PPO training](ppo-training.md) | Five seeds, 512,000 training steps on training workloads, followed by held-out evaluation of every checkpoint and an untrained initialization | PPO training with recorded GPU feedback; evaluation episodes reuse the table's measurements |
| [Live PPO/native-autotuning comparison](../results/2026-11-24-ppo-native-autotune/README.md#live-gpu-comparison) | 18 test workloads in three fresh RTX 4090 processes; 1,296 native configuration trials, 432 PPO queries and 216 independent confirmations | One frozen PPO checkpoint with fresh feedback; native search and PPO have different trial budgets |
| [Hardware transfer](../results/2026-11-23-operator-generalization-pro-4000/README.md#transfer-to-rtx-4090) | 1,404 RTX 4090 measurements of the same 18 test workloads and configurations | Ranking and frozen-choice transfer across one pair of GPU/runtime cohorts |
| [Sequential schedule search](../results/2026-11-26-sequential-schedules/README.md) | 54 live RTX 4090 episodes on six float16 TEST shape families; fixed reference, random walks and greedy local search; 48 confirmed submissions and six query timeouts | Stateful schedule edits under a shared 90-second ceiling, with three search/timing seeds and fixed workload input seeds; exploratory comparison without a trained sequential RL policy |
| [Vista inference](modern-vista.md) | Historical full-model timing cohorts and one modern A100 eager request after three warmups | Full-model execution is established for the recorded workloads; a joint performance/held-out-quality result remains incomplete |
| [Configuration-agent tools](agent-environment.md) | CPU integration through the synthetic inference backend, tool calls, evidence reader and scoring | The `AgentSession` model-level workflow still needs real Vista quality calibration and end-to-end evaluation |

The operator benchmark's training, validation and test splits contain 30, 12 and 18 workloads. Each exact shape and its three dtypes stay in one split; both operators occur in every split. This evaluates transfer to held-out shapes within the supplied operators.

The sequential study uses the six TEST shape families in float16 from the same two known operators. Its 28-case CUDA validation covers representative kernel paths and the environment lifecycle. The fixed reference had the highest observed mean score; the paired interval against greedy search included zero. The study does not establish whole-operator generalization or exhaustively validate the schedule space.

Operator checks cover numerical agreement, finite outputs and unchanged inputs. Video quality, trajectory fidelity and end-to-end Vista speedups require model-level evaluation. The [results index](../RESULTS.md) links the measured findings and their uncertainty estimates; older experiments remain in the [archive](../research/archive/README.md).

## Using and extending the benchmark

Use the [collection and replay guide](operator-generalization.md) to reproduce the current dataset or connect a policy. `BudgetedTuningEnv` exposes task features, query history, an incumbent and a remaining trial budget through Gymnasium's `reset`/`step` interface. The published comparison uses eight trials. Other budgets and objectives define new comparisons and belong in their protocols.

`ScheduleTuningEnv` exposes the current schedule, legal edits, operator features, hardware context and remaining wall time. Follow the [schedule specification](../extensions/kernel_tuning/docs/sequential-schedules.md) for a live episode. Its published [trajectories and reproduction command](../results/2026-11-26-sequential-schedules/README.md) retain intermediate queries and failures; the final score uses independent confirmation against a paired reference.

Each result identifies the workload, inputs, reference, candidate set, objective, code and runtime. Keep measurements from different physical GPUs or runtimes in separate cohorts. Report the appropriate latency and memory boundaries: operator CUDA-event timing, cold inference requests and warm full-model requests measure different work.

Selection uses noisy observations. Fix the selected configuration before collecting independent confirmation, and report selection, training and confirmation costs separately. Retain unsupported settings, incorrect outputs, OOMs, timeouts and partial runs alongside successful measurements. The [measurement contract](public-benchmark-protocol.md) describes these requirements.

For a custom model evaluator, specify the quality metrics and margins using development data before selection, and keep final evaluation inputs outside policy observations. Reference seed variability, numerical tolerances and perceptual quality measure different properties. The supplied interfaces require the evaluator to implement these checks; the library cannot verify an arbitrary external implementation from its returned metadata.

## Execution boundary

The supplied policies choose configurations or apply predefined schedule transformations. The configuration-agent interface exposes allowlisted tools and a public observation, while the operator owns source code, metric assets, final inputs and process termination. The local Python API assumes a trusted evaluator and does not isolate malicious policy code.

Running agent-written Python or kernels would require an operating-system isolation boundary and adversarial testing. Cloud provisioning, funding and job deadlines also remain outside the library.
