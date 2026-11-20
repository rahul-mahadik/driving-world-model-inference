# Measured operator search

The [RTX PRO 4500 dataset](../results/2026-11-20-operator-search-rtx-pro-4500/README.md)
contains 360 GPU measurements and policy-evaluation records. To use existing
measurements, start with the replay command below and set `--table` to
`results/2026-11-20-operator-search-rtx-pro-4500/table.json`.

The operator suite has 12 workloads and nine candidate configurations. A table
collection measures every candidate and an eager reference in each workload,
using three separate processes on one physical GPU. Each successful measurement records
30 CUDA-event samples after five warmups. Incorrect outputs and execution
failures remain in the table.

The table is an optional CPU backend for the existing budgeted search
environment. A policy receives the workload, candidate descriptions, remaining
budget and outcomes of its own queries. Each episode allows four policy turns.
The policy does not receive unqueried measurements or the table optimum.

## Collect a table

Use the pinned operator GPU dependencies. These commands do not allocate a GPU.
Run them on the same device and software environment, with an external job
deadline. Give each process a new output directory:

```sh
python -m drivebench_tuning.table_collection collect --repeat-index 0 --output repeat-0
python -m drivebench_tuning.table_collection collect --repeat-index 1 --output repeat-1
python -m drivebench_tuning.table_collection collect --repeat-index 2 --output repeat-2
python -m drivebench_tuning.table_collection merge --runs repeat-0 repeat-1 repeat-2 --output table.json
```

The merge validates complete coverage, source and runtime identities, input
hashes, numerical checks and the underlying samples. Each cell summarizes the
three process repeats. Variability across event samples within one process and
variability across process medians are reported separately. A complete table
does not imply every candidate succeeded.

## Run policies on recorded measurements

```sh
python -m drivebench_tuning.replay_cli \
  --table table.json --split test --trials 4 --repeats 20 \
  --seed 1901 --output policy-comparison
```

This runs a fixed compiled configuration, grid search and random search on the
four test workloads. Grid and random visit candidates without replacement; the
fixed policy spends the same four turns on its one configuration. The output
contains each trajectory, a regret-versus-budget summary and selected
configurations for live confirmation.

The backend is labeled `measured_replay`. It performs zero new GPU measurements.
The comparison runner chooses its first recorded repeat as
`(seed + 100003 * task_index + episode_repeat) % recorded_repeats`.
The eager reference and each candidate's first query use that repeat; repeated
queries cycle through the remaining recorded repeats. Matching task/repeat
pairs use the same source records even when two policies run different numbers
of episodes. Every replay event retains the table hash and original measurement
ID. Twenty policy episodes therefore do not constitute twenty independent GPU runs.

Regret is calculated after policy execution against the lowest median paired
cost among configurations that passed every recorded repeat, including the
eager reference. A selected configuration with any recorded failure receives
reference cost for this analysis. The table's per-cell failures remain visible
in the accompanying data.

## Evaluate an LLM selector

`ResponsesIndexPolicy` accepts an explicitly configured
`BudgetedResponsesClient`, a suite and a model snapshot. It sends workload and
candidate descriptions plus queried outcomes through a strict action-index
schema. Provider tools, automatic retries and response storage are disabled.
The existing API cost ledger reserves each call before transmission and retains
charges for failures. Credentials are supplied by the caller outside the
repository.

```python
from drivebench_tuning.replay import MeasuredTable
from drivebench_tuning.replay_policy import ResponsesIndexPolicy, run_replay_policy

table = MeasuredTable.load("table.json")
# bounded_client is a BudgetedResponsesClient with an explicit model/pricing
# policy, durable dollar cap and privately configured official SDK client.
selector = ResponsesIndexPolicy(
    bounded_client, suite=table.suite, model=model_snapshot,
    max_calls=48, max_output_tokens=1024,
)
result = run_replay_policy(
    table, selector, name=model_snapshot, split="test",
    repeats=3, seed=1901, trials=4,
)
```

The runnable [API comparison example](../extensions/kernel_tuning/examples/compare_llm_replay.py)
accepts a model snapshot, current input/output token prices, a dollar cap, and a
private key file and cost ledger outside the checkout. Run it with `--help` for
its required arguments. It saves each policy turn durably, applies an absolute
wall-time limit (600 seconds by default), and reports API usage separately from
the GPU measurement budget. No API call is made while
collecting or replaying the fixed, grid and random baselines.

This evaluates a pretrained policy; it does not train the model with RL. Each
invalid action, refusal or failed API call consumes a policy turn, earns zero
reward and produces no candidate measurement. The result preserves the public
observations, accepted actions, failure categories and API usage.

## Confirm selections

`replay_cli.selection_manifest(table, reports)` combines completed episodes
into a deduplicated list of selected task/configuration pairs. It retains every
episode-to-selection mapping, the original input hashes and the physical
GPU/runtime cohort. Fresh confirmation regenerates inputs and reruns numerical
checks, with eager and compiled references measured alongside the selection.
Confirmation data never feeds back into the already completed policy episodes.
Live confirmation requires selections from the original, unsanitized table on
its original GPU/runtime. The published table uses pseudonymous hardware
identities and supports CPU replay; its selection manifests cannot be used for
live confirmation. On another GPU, collect a new table, run the policies on that
table, and confirm those selections as a separate cohort.
Run each confirmation repeat in a fresh process on the original GPU/runtime:

```sh
python -m drivebench_tuning.confirmation --selections policy-comparison/confirmation-request.json --repeat-index 0 --output confirmation-0
python -m drivebench_tuning.confirmation --selections policy-comparison/confirmation-request.json --repeat-index 1 --output confirmation-1
python -m drivebench_tuning.confirmation --selections policy-comparison/confirmation-request.json --repeat-index 2 --output confirmation-2
```

The local Python interface separates selector observations from evaluator data;
it does not isolate arbitrary untrusted code. Anyone downloading the public
table can inspect all cells. Evaluations should disclose this access policy and
freeze policy settings before scoring a test split. Full-model Vista timing and
video-quality evaluation are separate from this operator task.

## Export figures

After collection and policy evaluation, the optional plotting script exports
SVG and PDF figures: a task/configuration latency matrix with failed cells,
regret versus trial budget, and fresh confirmation ratios against compiled
PyTorch. Install the plotting extra from the repository root, then run:

```sh
python -m pip install '.[plots]'
python scripts/plot_operator_search.py --table table.json \
  --policy-reports policy-comparison/fixed-compiled.json \
    policy-comparison/grid.json policy-comparison/random.json \
  --confirmation-dirs confirmation-0 confirmation-1 confirmation-2 \
  --output figures
```

The policy and confirmation arguments are optional. Figures identify the number
of recorded GPU processes separately from policy episode counts, retain failed
outcomes and show raw confirmation-repeat dots. They include no confidence
intervals. The analysis receipt binds source and figure hashes and stores the
plotted values; source files remain unchanged. CPU test fixtures are rejected
by default, and an explicitly requested test rendering is watermarked.

When policies have different repeat counts, use `--comparison-repeats 3`, for
example, to plot the matching first three episodes per task for every policy.
The figure labels plotted and available counts. A separate baseline-only plot
can retain all 20 baseline repeats; it uses the same finite GPU measurements.

## Prepare a public copy

The collection retains raw hardware identity locally. To publish a table with
consistent GPU pseudonyms and redacted execution-error messages, use
`scripts/export_operator_table.py --table table.json --runs repeat-0 repeat-1
repeat-2 --output public-table`. Then use
`scripts/export_related_operator_results.py` to export matching policy reports
and confirmation directories against that public table. Both scripts expose
all required paths through `--help`.

These exports preserve measurements and trial IDs, regenerate the affected
metadata hashes, and record the original and derived artifact digests. They do
not rerun measurements or policy calls. Keep the original files unchanged;
plot and distribute the mutually bound public copies together.
