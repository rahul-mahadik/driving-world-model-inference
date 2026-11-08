# Experiment protocol

## Experimental units and splits

The request unit is one observed scene prefix plus one or more numeric trajectories and independent candidate seeds. Scene IDs, not overlapping windows, are the unit of split disjointness and bootstrap resampling. Smoke is a development subset. Final evaluation and calibration use different scenes. A second-seed manifest preserves a fixed evaluation subset's frames/actions while changing every candidate seed.

Use primary 25-frame, 576×1024, 50-step settings for engineering comparisons. Record 1 conditioned versus 24 new frames. Precision is already mixed in upstream; enabling autocast is not a new optimization. Decoder chunk 14 is upstream's default. Alternative chunks can change temporal boundary behavior and require quality evaluation.

## Pipeline boundaries

`prepare_context` hashes preprocessed image content and model/preprocessing identity and prepares action-independent tensors. The cache key includes checkpoint, shape, dtype, conditioning count and augmentation settings. Nonzero stochastic augmentation disables cross-request reuse. Cached deterministic posterior parameters are sampled separately per candidate with upstream RNG draw shapes/order. Action conditioning and approximate features never cross candidate boundaries.

Measured latency includes request resolution, file I/O, RGB conversion, center crop, Lanczos resize, normalization, transfers, conditioning, denoising, decoding, and decoded host frames. Cold loading, warmup/compilation, and frame export are recorded separately. Prefetch latency includes queue residence; throughput uses elapsed sequence time rather than adding overlapping request times. Stage CUDA event durations are diagnostic and can overlap.

Warmups continue until a minimum count and stable compiler graph count are reached; recompilation during measurement removes headline eligibility. Profiler runs are in separate directories. Do not use their timings as speed results. For continuous throughput use `save_outputs: 0`; quality runs declare the saved subset in the snapshot.

## Workloads

| Workload | Candidate counts | Report |
|---|---:|---|
| Single future |1| Warm median and p95 request latency |
| Candidate set |4,8,16| Complete set latency, actual batch sizes |
| Independent scenes |1 or declared fixed count| Completed host futures per elapsed second |
| Memory scaling |16 with requested batches1,2,4,8,16| Actual batch/fallback, allocated/reserved peaks, OOM attempts |
| Second seed | Fixed 16-clip evaluation subset | Stability of conclusions across seeds |

The current Vista adapter executes actual batch1. Memory-scaling configs above1 are recorded as unsupported serial fallbacks, not successful batched memory experiments.

## Measurement and uncertainty

The benchmark emits p95 only with at least 20 successful requests. Headline eligibility additionally requires200 successful requests, at least 2 scenes, passed hardware identity, stable compilation, no OOM attempts and an unprofiled real backend. Scene-cluster bootstrap uses 2000 seeded resamples; repeated windows/seeds within scenes remain grouped. Rented-GPU drift also needs repeated interleaved comparisons; bootstrap alone does not remove it.

Performance regressions must exceed both the declared relative tolerance and measurement uncertainty, with a separate confirmatory baseline/candidate rerun on matching hardware/model/workload. Tolerances start unset. `drivebench.regression --help` documents the comparison CLI. Synthetic results cannot pass a GPU regression gate.

Before comparing a change, collect at least four independent repetitions of the same baseline, each with at least 200 successful matched requests from at least two scenes. The baseline model, generation, optimization settings, Python/packages/toolkit, GPU, and workload must match. Every calibration run must be headline-eligible, unprofiled, and free of failures, OOM attempts, and measured recompilation. Repeated copies of the same timings are rejected.

Choose a relative latency tolerance explicitly, then freeze a new policy:

```bash
python -m drivebench.calibrate_performance \
  --runs outputs/variability/baseline-1 outputs/variability/baseline-2 \
         outputs/variability/baseline-3 outputs/variability/baseline-4 \
  --relative-latency-tolerance 0.05 \
  --output configs/performance-v1.yaml
python -m drivebench.regression \
  --baseline outputs/regression/baseline --candidate outputs/regression/candidate \
  --tolerances configs/performance-v1.yaml
```

`0.05` above is an illustrative user choice, not a recommended or pre-calibrated limit. Calibration computes paired scene-cluster bootstrap intervals with 4,000 fixed-seed resamples for p50 and p95 for every baseline-run pair. The selected tolerance must cover the largest upper 95% bound across both comparison directions, including the point estimates. This is a conservative observed-variability check, not a simultaneous confidence guarantee or proof that future machine drift is bounded. The command rejects a tolerance that is too small; it never chooses or relaxes it automatically.

The new policy records the freeze time, exact identities, source artifact hashes, paired drift estimates, uncertainty, and verification rule. Existing policy files are never overwritten. Regression rechecks those source artifacts and statistics, requires baseline identity to match the calibration, and rejects a policy frozen after a comparison starts when timestamps are available. Keep calibration directories alongside the policy; video arrays are not read. Setting `calibrated: true` manually is insufficient. Collect comparison runs after freezing, and supply independent `--confirm-baseline` and `--confirm-candidate` artifacts for any apparent regression before rejecting a change. Use a separately calibrated policy for each fixed workload or changed baseline configuration.

## Optimization selection

Generate the independent experiment matrix with `drivebench.experiments`. Select targets using measured stage/operator/kernel costs and peak allocation evidence. If an operator occupies fraction f and can improve by s, ideal end-to-end speedup is at most `1 / ((1-f) + f/s)` before overhead. Validate actual integrated behavior; never multiply isolated speedups.

Candidate flags cover compilation, graph breaks/fullgraph audits, fixed versus dynamic shapes, compiler CUDA graphs, decoder compilation/chunking, conditioning reuse, data staging/prefetch, precision, steps and fixed-interval middle-block reuse. Approximate reuse and compiled denoiser are currently rejected together pending a graph-break audit. The middle block is an interception experiment, not a claim that profiling established it as the dominant bottleneck.

The key question is whether feature reuse retains more perceptual/temporal/action-response behavior than reduced steps at the same **measured** latency. Steps 50/40/30/20 use the same upstream EDM schedule implementation with the changed discretization count. Preserve out-of-tolerance observations on the curve. Quantization and Triton kernels are conditional extensions; wait for measured supported operations and implement fallbacks/correctness tests if justified.

## Numerical and quality gates

Use `drivebench.validate` for declared-tolerance upstream/eager agreement, candidate reorder/isolation, and context cache parity. It sequentially loads models to avoid holding two full Vista instances on one GPU. This is separate from generation-quality evaluation.

Run LPIPS optionally with `--lpips` in the GPU environment; unavailable model packages/weights are explicit missing metrics. Record the installed package/version and evaluator weight provenance for final comparisons. Generated outputs are compared only after excluding conditioning frames. Ground truth uses the same center crop/resize and is evaluated only for recorded actions. Temporal change, derivative, and freeze metrics are diagnostic image proxies, not trajectory-adherence measures.

Freeze tolerances selected on development scenes, including numerical, LPIPS and temporal limits. Maintain complete output pairs and scene-disjoint evaluation evidence. Review deterministic blinded A/B motion previews for disappearance, duplication, freezing, geometry changes and action response. Keep the private review key separate from reviewers. Only passing frozen metrics plus accepted review can label behavior quality-preserving. Such a label is relative to the chosen constraints and reference; it does not certify driving feasibility.
