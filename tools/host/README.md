# Bounded Vista worker launchers

These are trusted operator tools. The policy can request an allowlisted trial; it cannot edit these files, their specifications, the scoring worker or the quality policy. The controller invokes a worker through one launcher at a time and verifies terminal metadata and cleanup before it accepts an output.

## Portable adapter for a new study

Use [run-bounded-portable-v1.py](run-bounded-portable-v1.py) at its checked-in host-relative path `tools/host/run-bounded-portable-v1.py`. It is a small revision of the same lifecycle that accepts the operator's pinned runtime instead of requiring a historical local image. Its `runtime` object has exactly three fields:

| Field | Required value |
| --- | --- |
| `image_id` | The full `sha256:...` ID of an image already installed on the host; mutable tags and automatic pulls are rejected |
| `python_executable` | An exact absolute interpreter path inside that image/workspace |
| `environment_proof_kind` | The declared identifier of the actual environment verification contract |

Build a suitable image and interpreter before admission; the [preserved container/bootstrap recipes](../../environments/containers/README.md) provide a starting point for the H100 Vista environment. A build does not verify model correctness or compatibility. The fresh proof must report a complete physical GPU/runtime/tool identity, match `environment_proof_kind`, bind the same `runtime` object as `launch_runtime`, and bind the new `allocation_deadline_epoch`. The separate environment lock must equal the proof's runtime identity. Preserve the original verification results alongside that binding; setting `passed` in a hand-written JSON file is not a verification procedure.

The controller pins the launcher file itself with `launcher_sha256`. The portable container has networking disabled, refuses image pulls, and explicitly uses the image's `timeout` executable as its entrypoint. Pre-stage model weights, OpenCLIP assets, LPIPS assets and all dependencies before the worker starts. No model or policy may choose a launcher, image, interpreter, environment contract or source hash. Adaptations to the launcher/proof contract are evaluator changes and require fresh review and a new cohort. This release has CPU lifecycle/contract tests; successful real Vista execution through this portable adapter has not yet been established.

## Preserved historical adapters

The two historical launchers are byte-for-byte copies of the reviewed sources; [historical-launchers.json](historical-launchers.json) records their hashes and required execution paths. The A100 source was also preserved in the [historical pilot](../../research/a100-pilot-v1/historical-source/outputs/host/run-bounded-a100-job-v1.py). Their original docstrings say “Private” because these began as operator-side tools, not agent tools. They contain no provider credentials or instance bindings.

| Historical adapter | Required host-relative path | Pinned environment |
| --- | --- | --- |
| [H100 v2](run-bounded-job-v2.py) | `outputs/host/run-bounded-job-v2.py` | Fixed Docker image and `/workspace/outputs/venvs/vista-upstream-cu118-sm90/bin/python` |
| [A100 v1](run-bounded-a100-job-v1.py) | `outputs/host/run-bounded-a100-job-v1.py` | Fixed Docker image and `/workspace/outputs/venvs/vista-a100-cu118/bin/python`, with the A100 proof contract |

The historical image digest refers to an existing local Docker image, not a downloadable container published by this repository. Copying the launcher alone does not reproduce that image, virtual environment, model weights or inputs. Historical proof files and allocation deadlines are evidence, not reusable authorization for another machine. Build and verify a fresh runtime and use a new immutable deadline before running any worker.

## Host and trust boundary

Run the trusted controller on the Linux GPU host, **outside** the worker container. The controller shares the existing project root with the launcher; the launcher mounts it at `/workspace`. Worker specs therefore name `/workspace/...` paths, while the controller transport names the corresponding host paths. The launcher serializes jobs with the existing single-GPU lock, binds source and input hashes, enforces process timeouts, and removes only the exact container bearing its own labels.

These controls are not an isolation boundary for arbitrary hostile code: the historical worker has a writable project mount. This configuration-agent track permits only validated configuration changes and executes trusted model/evaluation code. A code-editing competition needs a separate OS-level isolation design and is outside this release.

An operator still owns GPU allocation, an independent absolute resource deadline, prepaid/cleanup reserves, persistent evidence storage and verified backup. These files never allocate a resource or replenish a balance. Follow the [study reproduction recipe](../../docs/agent-study-reproduction.md) before admitting real jobs.
