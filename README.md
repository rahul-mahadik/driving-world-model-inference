# Driving World Model Inference Optimization

## A100 custom kernels and measured RL search — 2026-11-07

The [separate operator study](results/2026-11-07-kernel-tuning/README.md) passed 75 GPU tests with no skipped tests. Two-order confirmation found lower incremental peak allocation than eager for 6 of 6 tested operator workloads; per-workload eager and compiled comparisons retain measured regressions; the original failed numerical test contract is documented separately. A tabular Q learner completed GPU-backed configuration search and equal-budget learned/random/grid evaluation. RL superiority, unseen-shape generalization and whole-video benefits remain unvalidated. Historical compiler timing and quality claims are unchanged. GitHub publication remains unverified.

## Historical implementation-only snapshot — before the November 7 GPU study

This dated snapshot records the CPU-only state of V14 and is superseded by the GPU results above.

The [standalone extension](extensions/kernel_tuning/README.md) implements custom Triton GEGLU and GroupNorm+SiLU kernels, a reversible Vista adapter, and a Gymnasium environment that tunes launch configurations using measured latency/memory feedback. At that implementation stage, local verification had passed 113 CPU tests and skipped 75 CUDA tests. GPU correctness, allocation and GPU-backed RL evidence had not yet been collected. The new operator study supersedes those evidence gaps; full-model integration benefits remain unvalidated. The historical compiler result below is not attributed to these new components. This revision is prepared locally; GitHub publication remains unverified.

## H100 performance and memory measured; output quality pending — 2026-10-30

The [H100 performance and memory comparison](results/2026-10-30-h100-tradeoff/README.md)
measures the time to generate one **24-future-frame, 576×1024 video** from a
conditioning image and trajectory using pretrained Vista at fixed 50 sampling
steps with unchanged model weights. The candidate uses `torch.compile`/Inductor
on the denoiser; the reference runs upstream eager inference. Request latency
includes input loading/preprocessing through host-ready generated frames, excluding
model loading, compilation, warmups and output export.

The comparison completed 200 measured requests per role across eight interleaved
blocks on the same physical GPU/runtime. Median latency was 57.33839 s upstream
and 43.66332 s compiled: 23.85% lower.
Measured throughput increased 31.34%, with 7.68% higher peak allocated GPU memory.
The 50 independent scenes were each repeated four times per role; confidence
intervals and block variability are reported. No historical H200 timings are pooled.
This is a performance/memory result with output quality still pending, not a
completed joint speed–quality result. The old qualification remains false and no
quality preservation is claimed. The 16-pair held-out check has not run. The full 179-file local
evidence archive, including all 14 raw proof arrays, was independently hash
verified on October 30. GitHub publication was unverified as of preparation. Earlier sections retain their dated cohort scope.

## Preliminary H200 trade-off snapshot — 2026-10-29

The [recovered H200 comparison](results/2026-10-29-h200-descriptive/README.md) contains one completed
50-request block per role, against the planned 200 requests per role. Median
latency was 53.76568 s upstream and 40.84276 s compiled; observed throughput was 31.65%
higher with 7.68% higher peak allocated memory. These preliminary observations
do not establish the planned headline comparison. The remaining timing blocks,
between-block variability and 16-pair held-out quality check are incomplete.
The partial 35-row compiler block is preserved and excluded. The old qualification
remains false, and no quality-preserving gain is claimed. Its sealed review
rejection and subsequent preference for compiled remain visible with their caveat.
This snapshot concerns only recovered historical H200 evidence; later allocations
or runs are not reported here. Prior sections below retain their dated scope.

## Historical compact H100 update — 2026-10-28

The [compact performance report](results/2026-10-28-h100-performance/README.md) records completed 32-scene development comparisons, an exact eager anchor and a complete compiler semantic diagnostic. The default compiler stays within the declared development metric limits but fails strict pixel equality. Seven blinded static cases are sealed: six accepted ties and one localized geometry difference. After sealing, attribution showed that the reviewer preferred the compiled display in the rejected case. This does not show compiler-induced degradation; the unchanged conservative review gate still rejects the difference. The completed norm qualifier returned qualified:false; its numerical and semantic checks passed, but localized review did not. Held-out quality and repeated headline timing are unfinished; no quality-preserving speedup or GitHub publication is claimed. Compute has been released; the retained 300 GB disk continues to incur storage charges. The earlier reports below retain their dated scope.

## A100 47-step screening snapshot

The [four-scene 47-step A100 report](results/2026-10-28-a100-step47/README.md) records completed
generation, CPU LPIPS/RGB/temporal comparison against the earlier four-scene
math-SDPA eager50 reference, and two independent blind static agent reviews as
of **2026-10-28T04:46:09.012565+00:00**. Both complete evidence archives are locally hash-verified.
Mean LPIPS was 0.000990859 and mean RGB RMSE was 0.003132802. Each reviewer
reported four ties, with the same two cases visually unacceptable on both sides
because of vehicle deformation. Review covered all 24 ordered future frames per
case in downscaled atlases and 13 selected native pairs per reviewer; it did not
include playback or human review. Ties do not establish equivalence or quality.

The 47-step descriptive median request duration was 96.2168 seconds. This short
development screen establishes no matched latency, speedup, calibrated quality,
action, safety, H100, or full-study acceptance. All such gates remain false.
Zero additional thresholded global pixel-change transitions relative to eager50
does not mean no low-change transitions or correct motion. Local specification
builder failure and the separately reviewed alias-preserving repair are retained
in the report and exact historical source archive.

The [32-scene A100 companion](results/2026-10-28-a100-development/README.md) and
[earlier A100 snapshot](results/2026-10-27-a100-pilot/README.md) retain their own
dated evidence and limits. The original A100 default runtime failed exact checks;
the separate math-SDPA exact pass covers one selected development request only.
The [historical helper archive](research/a100-pilot-v1/README.md) is for inspection
and is not runnable after relocation. Frozen d8 source, historical H100 evidence,
and the 18-variant study plan are unchanged. V5's 1,095 CPU tests and five synthetic
CLI checks retain their historical attribution; the six new helper/config copies
received hash checks and syntax parsing only. GitHub publication has not completed.

**Historical H100 development status (2026-10-27 UTC):** all 32 selected development scenes have upstream and two compiled results. The [development-only approved decision](results/2026-10-27-development-qualification/development-decision.json) advances compiled conditioning reuse to an 18-variant study after explicit metric criteria and agent-assisted blinded reviews. Historical strict numerical failures remain failures. Main held-out quality and headline timing are unfinished. This is a locally staged publication; GitHub publication has not completed.

A reproducible inference and measurement harness for generating numeric-trajectory-conditioned futures with [Vista](https://github.com/OpenDriveLab/Vista) and nuScenes.

**Status:** the compiler has a verified 200-request-per-role H100 performance/memory comparison; output quality is still pending. The 16-pair held-out quality check is unrun; the full new-cohort local evidence archive is independently hash verified. Historical qualification remains false. The original 18-variant study is preserved and deferred. This is a locally prepared evidence snapshot; GitHub publication was unverified as of preparation. [Results and limitations](RESULTS.md) retain historical source and scope.

The initial session demonstrated exact upstream agreement, two-candidate serial isolation, and exact conditioning-cache reuse on one development clip. Profiling identified memory movement, pointwise work, normalization, and attention as targets. Compiler trials failed the unchanged numerical requirement. Three exploratory unprofiled timings were approximately 57.5 seconds each; the benchmark's combined numerical-evidence gate remained false.

## Try the CPU demonstration

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m drivebench.prepare --config configs/data.yaml --synthetic --output data/synthetic
python -m drivebench.benchmark --config configs/demo.yaml --output outputs/demo-finalists/baseline
python -m drivebench.benchmark --config configs/demo_cached.yaml --output outputs/demo-finalists/cached
python -m drivebench.evaluate --runs outputs/demo-finalists --reference outputs/demo-finalists/baseline
python -m drivebench.report --runs outputs/demo-finalists
```

Open the generated `outputs/demo-finalists/report.html`. Synthetic data exercises software contracts; its timings are not GPU results. Use fresh output directories for reruns.

## Interface

```python
from drivebench.engine import InferenceEngine
from drivebench.types import GenerationConfig, OptimizationConfig

engine = InferenceEngine(model=model_config, generation=GenerationConfig(),
                         optimizations=OptimizationConfig(max_batch_size=1))
context = engine.prepare_context(observed_frames)
result = engine.generate(context, trajectories, seeds=[23, 24],
                         candidate_ids=["recorded", "alternative"])
frames = result.frames["recorded"]
```

Configure `model_config` using `configs/baseline.yaml`; trajectories have shape `[8]` or `[4,2]` per candidate in Vista's verified initial-LiDAR coordinate convention. Defaults are 25 total frames, 1 observed and 24 generated, 576×1024, 50 Euler EDM steps, guidance 2.5, FP16 autocast, and decode chunk 14. Host frames are float32 `[25,576,1024,3]`.

**Experimental joint denoising permits requested batches up to 16; the default and upstream batch size remain one.** This is a software cap, not a measured GPU capacity. GPU batching, compiler probes and quality evaluation require their own source-matched evidence. The published initial isolation result used serial `[1,1]` execution. No current batching or optimization win is established by that historical result. Requested candidate counts, actual GPU batch sizes and OOM fallback are reported separately.

## Reproduce and inspect

- [Resumable full-study workflow](docs/full-study.md)
- [GPU setup and measurement runbook](docs/gpu-runbook.md)
- [Environment definitions](environments/README.md)
- [Vista integration and action semantics](docs/vista-integration.md)
- [Experiment and validation protocol](docs/experiment-protocol.md)
- [Published result evidence](results/2026-10-26-h100/README.md)
- [Third-party notices and excluded assets](NOTICE.md)

The tested H100 compatibility baseline uses Torch 2.0.1/CUDA 11.8 and source-built xFormers 0.0.20 for native sm90. The original CUDA 11.7 runtime and packaged CUDA 11.8 attention wheel failed on this hardware. This is an explicit compatibility change, not proof of equivalence to an executed CUDA 11.7 baseline. Modern dependencies are a separate unvalidated option.

Datasets, annotations, checkpoints, raw outputs, credentials, and cloud account details are excluded. Obtain the external assets separately under their terms and configure the relative paths. `results/` contains compact derived evidence and original-source hashes; transformed JSON has separate published checksums.

## Current study tools

The package fingerprint for this revision is `d8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb`. The initial measured package was `2a8fed43b4f2d13e94f9310e9078cfada379f19d8f2f74107fe1066991b7ff00`. See [revision provenance](results/study-source-2026-10-26/source-provenance.json). The earlier V3 publication passed 723 CPU tests. Historical revisions added development qualification and standalone diagnostic tools; their validation keeps its original attribution. The October 30 performance/memory result is reported above; paired held-out output quality remains pending.

`drivebench.study` plans immutable disjoint development/held-out manifests, streams paired quality metrics before retiring unselected arrays, records controlled same-seed action-response diagnostics, and schedules balanced timing blocks. `drivebench.calibrate_pilot` separately freezes a development-derived variability tolerance before held-out timing, with confirmatory reruns for apparent regressions. Existing strict performance calibration keeps its original requirements.

`scripts/probe_vista_optimization.py` preserves bounded compiler/decoder/shape-policy failures, `scripts/audit_scene_coverage.py` audits actual coverage, and `scripts/prepare_execution_screen.py` creates explicit discovery workloads. `scripts/prepare_perceptual_evaluation.py` stages pinned LPIPS assets outside the inference environment and verifies CPU execution before isolated evaluation. No third-party package binaries or model weights are included.

Follow the [full-study instructions](docs/full-study.md) for commands and limitations. The main quality matrix does not substitute for independent batching, decoder, I/O or multi-candidate scaling trials. All templates require licensed data, a verified GPU environment and fresh matching numerical proofs.

The latest standalone workflow includes replacement-allocation timing companions with immutable proof/lock bindings and a report that joins timing only to verified quality and review evidence. The [registered study draft](docs/registered-study-draft.md) remains a proposal; the [remaining diagnostic queue](docs/remaining-diagnostics-queue.md) is a procedure, not an execution claim. [Current-source discovery evidence](results/2026-10-27-d8-discovery/README.md) preserves failed and completed attempts. The original qualification-dependent study remains deferred. The separate October 30 performance/memory comparison is complete; paired held-out output quality remains pending.

The earlier V3 publication passed **723 CPU tests** and its full synthetic prepare → baseline/cached benchmark → evaluate → report workflow. The bootstrap unit fixture is portable and uses no licensed records or private GPU proofs. [Validation receipt](results/study-source-2026-10-26/validation.json).

## Development qualification update

[New development evidence](results/2026-10-27-development-qualification/README.md) includes 32-scene compiled/upstream metrics, actual-input compiler and attention-operator diagnostics, and retained memory timeouts. Strict-zero differences remain explicit. The current code adds exact semantic sidecars, bounded quality runs, sealed review assembly, and recorded-future diagnostics. That historical development bundle alone establishes no speedup. The separate October 30 comparison reports measured performance; held-out output quality remains pending.

The sealed V4 publication passed **883 CPU tests** and the complete synthetic prepare → baseline/cached benchmark → evaluate → report workflow. [V4 validation receipt](results/2026-10-27-development-qualification/validation.json). Synthetic results exercise software contracts only.

## Numerical qualification and input diagnostics

The new [compiler norm qualification](docs/compiler-norm-qualification.md),
[bounded timing workflow](docs/norm-qualified-timing.md) and
[supplementary report](docs/norm-timing-report.md) implement a separate contract
using development-selected numerical limits. They preserve historical failed
bitwise-equality checks and require exact candidate-order/action-isolation
semantics, frozen quality criteria and completed review evidence. These tools
completed the historical rejected compact compiler qualification; they have not qualified an output-preserving GPU finalist. The separate October 30 performance/memory comparison is complete under the later trade-off scope; its output-quality check is still pending.

The [input pipeline screen](results/2026-10-27-input-pipeline-screen/README.md)
contains completed diagnostic observations from a partial execution queue.
Small samples, sequential trials and queue-inclusive request latency limit their
interpretation. Those historical input diagnostics do not establish a performance gain. The original full held-out/action-response/finalist study remains deferred; the separate October 30 performance/memory comparison is complete, with its narrower held-out output-quality check still pending.

The V5 publication copy passed **1,095 CPU tests** and all five synthetic CLI stages. [V5 validation receipt](results/2026-10-27-norm-tooling-validation.json). These checks validate software contracts; they do not complete the real GPU experiment.
