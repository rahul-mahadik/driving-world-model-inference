# A100 diagnostic, generation and evaluation source archive

These files are exact, text-only copies of the helpers and configurations used
for the October 27 A100 pilot. They are published for source inspection and
provenance. The [numeric report](../../results/2026-10-27-a100-pilot/README.md)
records what passed, what failed and what was pending at the snapshot.

The files have been relocated under `historical-source/`. Their `ROOT`
calculations, fixed allocation deadline, original `/workspace` paths, required
GPU environment, retained proof artifacts and original artifact layout are
unchanged. **Do not run these relocated copies as a reproduction.** Dataset
files, checkpoints, binary profiler traces and private host records are absent.
A future reproduction must define fresh paths, runtime proofs and resource
bounds and review a separately versioned runner. A historical helper's deadline
must not be edited and presented as the original experiment's source.

| Source | Role |
|---|---|
| [A100 admission](historical-source/outputs/host/verify-a100-pilot-v1.py) | Separate sm80 environment and original-runtime checks |
| [Bounded container launcher](historical-source/outputs/host/run-bounded-a100-job-v1.py) | Fixed-input admission, timeout, exact container cleanup |
| [Original baseline configuration](historical-source/outputs/host/a100-pilot-01/configs/baseline.yaml) | Original eager A100 settings |
| [First-step probe](historical-source/outputs/host/a100-repeatability-v1/probe.py) | Early input capture and same-input denoiser replay |
| [Encoder localization](historical-source/outputs/host/a100-encoder-localization-v1/probe.py) | Encoder, posterior and RNG observation before denoising |
| [Encoder A/B](historical-source/outputs/host/a100-encoder-ab-v1/probe.py) | Encoding-scoped original, cuDNN, math-SDPA and restored controls |
| [Process-wide math-SDPA wrapper](historical-source/outputs/host/a100-math-sdpa-v1/runtime.py) | Separate policy proof/lock and same-process strict validation |
| [Math-SDPA configuration](historical-source/outputs/host/a100-math-sdpa-v1/baseline.json) | Explicit new policy with unchanged model/generation settings |
| [Per-frame comparison](historical-source/outputs/host/compare-a100-first-input-v1.py) | CPU comparison of transferred early captures |

The helper chain also uses unchanged
[compiler comparison utilities](../../scripts/probe_compiler_operator.py) and
[candidate memory utilities](../../scripts/probe_candidate_memory.py).
[Source provenance](source-provenance.json) binds every exact copy and those
retained dependencies. The three diagnostic JSON configurations are archived
beside their corresponding producers.

The policy wrapper forces math-only torch SDPA through upstream, refactored,
cached-posterior and decoder execution in one process, checking active flags
on every call. xFormers policy is unchanged. It adds a distinct algorithm-policy
identity so the original numerical consumer rejects this evidence as an
original-runtime pass. Raw validation pass/fail values are preserved.

The additional historical generation and evaluation sources are archived here:

| Source | Role |
|---|---|
| [Four-scene generation](historical-source/outputs/host/a100-math-quality-v1/quality.py) | Bounded generation and provenance under the separate math-SDPA policy |
| [Eager50 config](historical-source/outputs/host/a100-math-quality-v1/configs/eager50.json) | 50-step reference generation |
| [Eager40 config](historical-source/outputs/host/a100-math-quality-v1/configs/eager40.json) | Approximate 40-step generation |
| [Feature2 config](historical-source/outputs/host/a100-math-quality-v1/configs/feature2.json) | Approximate interval-2 feature reuse at 50 steps |
| [Descriptive evaluator](historical-source/outputs/host/a100-math-quality-eval-v1/evaluate.py) | Metric/native-export generation with runtime and input checks |
| [Artifact recovery verifier](historical-source/outputs/host/a100-math-quality-eval-recovery-v1/verify.py) | Existing eager40 artifacts verified separately from its failed job |

The retained dependencies also include the unchanged
[bounded native review exporter](../../scripts/run_quality_bounded.py),
[perceptual setup helper](../../scripts/prepare_perceptual_evaluation.py) and
[generic tolerance configuration](../../configs/tolerances.yaml). Their exact
hashes are bound in [source provenance](source-provenance.json).

The configurations preserve their original `evaluation_pending` fields; these
are historical generation inputs. The dated report records later observations.
The source's fixed rate constant is historical resource-budget provenance,
not a current provider quote or a billing record. Opaque artifact filename
suffixes identify historical evidence files, not provider accounts or resources.
Private host specs, cleanup/proof records, data, weights and raw outputs are
excluded. The CPU-LPIPS invocation wrapper is also excluded because it contains
a private host literal; the report's source hash records its separate identity.
Consequently these copies alone do not reproduce that invocation.

All six newly added files were hash-checked; the three Python helpers were
AST-parsed without importing or executing them. None was tested from this
relocated subtree, and V5's earlier 1,095 CPU tests and five synthetic CLI checks
are not attributed to them. The original strict failure remains failed; the
new-policy strict pass covers one selected development request. Three
four-scene generations and their backups completed. The eager40 evaluation
remains failed even though its artifacts were separately verified; feature2's
CPU evaluation completed. Both comparisons received two agent-assisted static
reviews with four ties each and two shared unacceptable cases. No quality,
action, safety, runtime-speedup, full-study or individual-kernel cause is established.


## Separate 32-scene development companion

The following exact historical copies support inspection of the separate
[32-scene development report](../../results/2026-10-28-a100-development/README.md):

| Source | Role |
|---|---|
| [Development generator](historical-source/outputs/host/a100-math-development-v1/development.py) | Fresh 32-scene eager50 generation with three warmups and retained arrays |
| [Development configuration](historical-source/outputs/host/a100-math-development-v1/configs/eager50.json) | Fixed development manifest, recorded actions and 50-step generation |
| [Recorded-future evaluator](historical-source/outputs/host/a100-math-development-eval-v1/evaluate.py) | CPU NumPy/Pillow future-only diagnostics; no LPIPS or review exporter |

All 32 generation requests, the recorded-future diagnostics and both complete
archives are verified in the dated report. The generic numerical, quality and
headline gates remain false. Earlier four-scene evidence and historical H100
evidence retain their separate scope. The original selected-request strict pass
does not establish exact agreement for all 32 scenes.

The two new Python files were AST-parsed and all three new copies hash-checked;
they were not imported or executed from this relocated archive. The evaluator
requires private generation admission/specification/host records, plus data and
proof artifacts omitted from this repository. Its completed-generation wording
states a prerequisite; execution claims come from the dated evidence report.
Historical `evaluation_pending` config fields remain exact generation inputs.
The fixed allocation deadlines and original paths are preserved. V5's 1,095 CPU
tests and five synthetic CLI checks remain prior validation only.

## Four-scene 47-step screening helpers

The [47-step report](../../results/2026-10-28-a100-step47/README.md) binds completed
generation, CPU comparison, two limited static agent reviews and both complete
archives. Six exact historical files are added for inspection:

| Source | Role |
|---|---|
| [Generator](historical-source/outputs/host/a100-math-step47-v1/step47.py) | Four-scene 47-step eager math-SDPA generation |
| [Configuration](historical-source/outputs/host/a100-math-step47-v1/configs/eager47.json) | Original fixed generation input |
| [Original generation builder](historical-source/outputs/host/a100-math-step47-v1/prepare_specs.py) | Preserved failed local admission-alias validation |
| [Generation builder v2](historical-source/outputs/host/a100-math-step47-v1/prepare_specs_v2.py) | Separate reviewed repair; exact normalized provenance and gates retained |
| [Comparison evaluator](historical-source/outputs/host/a100-math-step47-eval-v1/evaluate.py) | CPU LPIPS and RGB/temporal comparison; native review exports |
| [Comparison builder v2](historical-source/outputs/host/a100-math-step47-eval-v1/prepare_spec_v2.py) | Actual completed comparison specification preparation |

[Builder history](../../results/2026-10-28-a100-step47/builder-history.json)
distinguishes local specification failure from generation failure. The omitted
original comparison builder remains hash-bound operational provenance because
the private archiver imports its inventory utilities; it was not never-used.
These relocated sources retain original deadlines, path semantics and references
to omitted host specifications, operator proofs, data, arrays and LPIPS assets.
They are not portable or runnable from this archive. Prepared-only docstrings
and generation evaluation-pending fields are exact historical inputs; actual
execution status comes from the dated report. Generic historical price constants
are provenance, not current quotes. Actual account/billing/host data are excluded.

All six copies are hash-checked; five Python sources AST-parse and one JSON
configuration parses. No copied helper was imported or executed. V5's 1,095 CPU
tests and five synthetic CLI checks remain historical validation only. All
quality, matched-performance, H100 and full-study acceptance gates remain false.
