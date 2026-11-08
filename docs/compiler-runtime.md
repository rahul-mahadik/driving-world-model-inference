# Scoped registered compiler runtime

The integrated runtime is an explicit model opt-in; existing optimization fields
and historical probe launchers retain their original serialization and behavior.

```yaml
model:
  compiler_runtime: registered_einops_attention_v1
optimizations:
  compile_denoiser: true
  compile_mode: default
  dynamic_shapes: false
  fullgraph: false
```

It requires the pinned Torch2.0.1+cu118, einops0.6.1 and xFormers0.0.20 sources.
It calls the official [einops registration API](https://github.com/arogozhnikov/einops/blob/v0.6.1/einops/_torch_specific.py),
sets Dynamo's cache limit to512, and uses one stable
[Dynamo-disabled callable](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/_dynamo/eval_frame.py)
around the unchanged native attention operator. Construction and generation run
under Vista's existing process execution lock. The six registration entries,
native attention symbol and compiler settings restore before leaving each scope,
including exceptions and OOM retries. Upstream numerical validation explicitly
removes the opt-in selectors and uses the original runtime.

Compiler errors remain unsuppressed. An unimplemented eager frame, disabled
compiler, mutated settings, or successful generation without positive compiled
graph and Inductor-kernel evidence causes failure. Per-request counters and
restoration evidence are retained in cache statistics; they do not establish CUDA
graph replay. Source hashes, exact runtime declarations and a runtime implementation
digest are part of model identity and must match numerical proofs. Repeated requests
reuse the same wrapper object and compiled callable. An explicit
`dynamic_shapes: true` selects a separate identity for the bounded shape experiment.
Decoder compilation, fullgraph capture and other compile modes are outside this
runtime policy.

A separately selected full-block boundary supports the compiled feature-cache
experiment:

```yaml
model:
  compiler_runtime: registered_einops_attention_v1
  compiler_feature_boundary: output_blocks.9
optimizations:
  compile_denoiser: true
  feature_cache_module: output_blocks.9
  feature_cache_interval: 1
  max_batch_size: 1
```

Interval1 is the required boundary-only control: it executes the original full
block without cache cloning. Only after that control passes compilation/stability
checks should intervals2/3/4 be evaluated. The stable disabled full-block wrapper
keeps Python cache state outside the compiled region and restores the original
forward method after each scope. Joint candidate batching with this boundary is
rejected. Both selectors are bound in numerical proofs, adopted-run validation,
timing-block identity and development variability calibration.

CPU lifecycle tests establish restoration and identity checks only. The integrated
path requires fresh GPU validation, compilation stability and development quality
calibration before held-out selection. A faster preliminary probe that failed the
strict zero-error comparison remains a failed strict diagnostic; this integration
does not rewrite that result or inherit a quality-preserving claim.
