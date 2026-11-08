# Portable GPU runbook

Use a dedicated full H100 80 GB Linux machine with persistent storage. Acquire nuScenes and the published Vista checkpoint under their applicable terms. Credentials and provider lifecycle records belong outside the repository.

1. Read the [environment recipes](../environments/README.md). Build the explicit `upstream-cu118-sm90` baseline; keep the pinned Vista checkout pristine. The container mounts the project at `/workspace`, a portable container path rather than a host-specific location.
2. Execute `drivebench.verify_environment` on the actual machine. Real CUDA execution, Nsight Systems kernels, and positive Nsight Compute SM hardware counters are required; installation alone is insufficient.
3. Prepare verified data with `drivebench.prepare` and `configs/data.yaml`. Configure paths before running. Keep development, calibration, and held-out scenes disjoint. Inference consumes observed frames; future frames are evaluation-only.
4. Run strict upstream/refactored, candidate-isolation, and cache checks with the same model/source/runtime/configuration as subsequent measurements. Specify numerical tolerances before observing results. For an exact diagnostic, use `--atol 0 --rtol 0`.
5. Profile independently using `drivebench.profile --tool torch`, `--tool nsys`, and a small selected-kernel `--tool ncu` experiment. CUDA-event durations and summed kernel times use different boundaries; do not add overlapping times.
6. Use `drivebench.benchmark` for unprofiled timing. Separate model load/warmups, retain failures, and use `save_outputs: 0` for independent throughput runs. `uncached` disables decoded/conditioning reuse; OS filesystem cache remains uncontrolled.
7. Apply the [experiment protocol](experiment-protocol.md) for paired/interleaved repetitions, frozen development tolerances, held-out metrics, and blinded review. A failed numerical gate is retained and blocks headline eligibility.

```bash
python -m drivebench.verify_environment --dedicated --write-lock outputs/environment/reference-lock.json
python -m drivebench.validate --config configs/baseline.yaml --mode all \
  --atol 0 --rtol 0 --candidates 2 --output outputs/validation/numerical.json
python -m drivebench.profile --config configs/baseline.yaml --tool nsys --output outputs/profile-nsys
python -m drivebench.benchmark --config configs/baseline.yaml
```

Point `numerical_validation` at the matching successful proof before a benchmark. The validation command checks one selected request and records that scope; it does not establish all-scene quality. Default templates may need environment/data path changes and are not a promise that a new machine is ready.

The initial measured baseline preserves upstream loader parameter flags and uses `no_grad`; freezing every parameter or using `inference_mode` changed autocast cache eligibility. Initial uncorrected timing artifacts are not accepted upstream references. See [the published results](../RESULTS.md).

The initial session contains exact one-clip strict/cache/isolation results, two failed compiler trials, profiler evidence, and three exploratory timings. The expanded study is still pending. Experimental joint batching and study tooling are now implemented, while their full GPU validation and held-out conclusions remain unfinished. Use the [current study workflow](full-study.md); the initial results never substitute for fresh source-matched proofs.
