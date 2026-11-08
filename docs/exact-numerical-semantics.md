# Exact semantics for tolerance-qualified compiled timing

The frozen d8 `drivebench.validate` command applies its explicit `atol`/`rtol` to
upstream agreement, candidate isolation and cache comparisons. A development
policy may permit compiled upstream rounding differences, but it must not relax
candidate isolation or conditioning reuse. The standalone timing companion now
requires a separately recomputed exact-zero sidecar for every proof containing a
nonzero tolerance, including a nonzero tolerance inside an individual check.

`scripts/verify_numerical_semantics.py` reads the retained NPZ arrays with pickle
disabled and checks candidate order, unchanged candidates after another action
changes, and conditioning-cache equality at `atol=rtol=0`. It also requires the
cache key and exercised-cache evidence when cache outputs are present or selected.
It hashes every retained output, the original proof, configuration, full runtime,
input provenance, frozen package source and verifier. Checking the sidecar
recomputes the comparisons; a declared `passed` value alone is insufficient.
Failed comparisons are saved as failed sidecars and cannot authorize timing.

The original proof and historical strict failures remain unchanged. This sidecar
proves only the selected request's exact semantics. It neither accepts upstream
agreement nor establishes broad generation quality. The development-frozen
numerical policy hash must also be bound in the external launch specification;
the original validation CLI does not take a policy-file argument.

After a fresh replacement GPU passes environment verification, use a fresh
configuration with its proof and lock. Keep primary generation settings and the
selected optimizations unchanged. The supported compiled-conditioning50 finalist
has no feature boundary, feature interval1, FP16, decode14 and `cond_aug=0`.
`all` does not support approximate feature reuse; its existing boundary/cache
configuration conflict must not be bypassed for this proof.

```bash
# Frozen development values only; do not choose them from held-out results.
python -m drivebench.validate --config <finalist-new-gpu.yaml> --mode all \
  --request-index 28 --candidates 2 --atol <frozen-atol> --rtol <frozen-rtol> \
  --cache-bytes <selected-cache-budget> --output <fresh-numerical-proof.json>

# CPU only; retain the original proof and every referenced NPZ.
python scripts/verify_numerical_semantics.py create \
  --proof <fresh-numerical-proof.json> --sidecar <fresh-exact-semantics.json>
python scripts/verify_numerical_semantics.py check \
  --proof <fresh-numerical-proof.json> --sidecar <fresh-exact-semantics.json>
```

Use the guarded standalone pilot commands for this workflow. Direct core
`calibrate_pilot` commands do not know about the standalone sidecar; a nonzero
pilot policy without the guarded receipts is rejected by the timing companion.
The new runtime YAML may contain only `environment_verification` and
`environment_lock`. Compatibility with prior quality permits only GPU UUID/PCI
replacement; numerical proofs, pilots and every timing block must share the full
new runtime identity. Never combine timing blocks across allocations.

```bash
python scripts/run_timing_companion.py prepare-pilot \
  --study <frozen-study> --output <fresh-pilot-config.json> \
  --baseline <baseline> --candidate-count 1 \
  --numerical-proof <baseline-proof.json> --semantic-sidecar <baseline-semantics.json> \
  --runtime-config <new-runtime.yaml>
python scripts/run_timing_companion.py pilot-plan \
  --config <fresh-pilot-config.json> --output <fresh-pilot> --relative-tolerance-floor 0
python scripts/run_timing_companion.py pilot-run --pilot <fresh-pilot>
python scripts/run_timing_companion.py pilot-freeze \
  --pilot <fresh-pilot> --output <fresh-pilot>/policy.json
python scripts/run_timing_companion.py plan --study <frozen-study> \
  --output <fresh-timing-companion> --baseline <baseline> --candidate <finalist> \
  --candidate-count 1 --pilot-policy <fresh-pilot>/policy.json \
  --runtime-config <new-runtime.yaml> \
  --proof <baseline>=<baseline-proof.json> --proof <finalist>=<finalist-proof.json> \
  --semantic-sidecar <baseline>=<baseline-semantics.json> \
  --semantic-sidecar <finalist>=<finalist-semantics.json>
python scripts/run_timing_companion.py run --companion <fresh-timing-companion>
```

A single all-mode proof can supply both backend branches only if their exact
effective settings match; otherwise produce distinct proofs. Zero-tolerance
proofs may omit sidecars. If a sidecar is supplied, it is always verified. Repeat
the workload-specific pilot and companion with candidate-count4 when that
workload is required; B1 quality does not qualify joint B2 batching.

The companion rechecks semantics before/after timing blocks, before aggregation
and before comparison. `report_full_study.py` calls its authoritative `_load`,
so missing, failed or stale sidecars prevent report verification too. Include
both scripts, the sidecar, proof, every retained output and numerical-policy file
in the bounded launcher's input hashes. All guarded phases must use immutable
fresh paths and the allocation's unchanged absolute deadline. No provider-native
deadline or acceptance guarantee is implied.

Previous helper bytes are preserved locally at
`outputs/proposals/timing-exact-semantics-v1/run_timing_companion.py`, SHA256
`db013d2b0516fa07936eb2497032af27fd3f376afe20996b7f8c43c253a2b992`.
