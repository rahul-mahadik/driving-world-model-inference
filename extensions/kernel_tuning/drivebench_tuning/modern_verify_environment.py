"""Fresh declared-device preflight; never reclassifies a historical H100 proof.

Reuses the existing environment collector, bounded command primitive, CUDA
smoke kernel and Nsight parsers. The historical verifier remains unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from drivebench.benchmarks.environment import collect_environment, command, reference_identity
from drivebench.verify_environment import CUDA_SMOKE, has_hardware_counters, has_cuda_kernel_trace

POLICY_KIND = "modern_declared_gpu_v1"


def validate_policy(policy):
    expected = {"schema_version", "gpu_name", "minimum_memory_mib", "minimum_host_ram_bytes", "mig_supported"}
    if (not isinstance(policy, dict) or set(policy) != expected
            or type(policy["schema_version"]) is not int or policy["schema_version"] != 1):
        raise ValueError("Exact declared modern hardware policy required")
    if not isinstance(policy["gpu_name"], str) or not policy["gpu_name"].strip():
        raise ValueError("An exact GPU name is required")
    for name in ("minimum_memory_mib", "minimum_host_ram_bytes"):
        if type(policy[name]) is not int or policy[name] <= 0:
            raise ValueError("Positive integral hardware capacity required")
    if type(policy["mig_supported"]) is not bool:
        raise ValueError("Declare MIG support explicitly")


def idle_failures(environment):
    result = environment.get("gpu_processes", {})
    if result.get("returncode") != 0:
        return ["cannot verify active GPU processes"]
    others = [row for row in csv.reader(result.get("stdout", "").splitlines())
              if row and row[0].strip() != str(os.getpid())]
    return ["other GPU compute processes are active"] if others else []


def hardware_failures(environment, policy, dedicated):
    validate_policy(policy)
    failures = []
    if environment.get("system") != "Linux":
        failures.append("modern GPU verification requires Linux")
    ram = environment.get("host_ram_bytes")
    if type(ram) is not int or ram < policy["minimum_host_ram_bytes"]:
        failures.append("host RAM is below the declared minimum or unknown")
    devices = environment.get("gpus", [])
    if environment.get("nvidia_smi", {}).get("returncode") != 0 or len(devices) != 1:
        failures.append("exactly one independently visible full GPU is required")
    else:
        gpu = devices[0]
        if gpu.get("name") != policy["gpu_name"] or not gpu.get("uuid"):
            failures.append("GPU name/identity differs from the declared device")
        try:
            memory = float(gpu.get("memory_total_mib", "nan"))
        except (TypeError, ValueError):
            memory = float("nan")
        if not math.isfinite(memory) or memory < policy["minimum_memory_mib"]:
            failures.append("GPU memory is below the declared minimum or unknown")
        mode = gpu.get("mig_mode")
        if policy["mig_supported"]:
            if mode != "Disabled":
                failures.append("MIG-capable device must report Disabled")
        elif mode not in ("N/A", "[N/A]"):
            failures.append("Declared MIG-unsupported device must explicitly report N/A")
    if not dedicated:
        failures.append("dedicated allocation requires explicit operator attestation")
    failures.extend(idle_failures(environment))
    if not (environment.get("torch") or {}).get("cuda_available"):
        failures.append("working CUDA PyTorch is required")
    if "triton" not in {name.lower() for name in environment.get("packages", {})}:
        failures.append("Triton is unavailable")
    tools = environment.get("tools", {})
    for name in ("nvcc", "nsys", "ncu"):
        if tools.get(name, {}).get("returncode") != 0:
            failures.append(name + " is unavailable")
    return failures


def verify(output, policy, dedicated=False, collect=True):
    validate_policy(policy)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    environment = collect_environment()
    failures = hardware_failures(environment, policy, dedicated)
    proof = {"hardware_counters_collected": False, "nsys_trace_collected": False,
             "torch_cuda_execution_passed": False, "modern_runtime_passed": False}
    runtime = None
    if not failures and collect:
        proof["modern_runtime"] = command(
            [sys.executable, "-m", "drivebench_tuning.modern_vista", "--check-runtime"], timeout=120)
        if proof["modern_runtime"]["returncode"] == 0:
            try:
                from .modern_vista import RUNTIME
                diagnostic = json.loads(proof["modern_runtime"]["stdout"])
                runtime = diagnostic["actual_runtime"]
                capability = runtime["device_capability"]
                proof["modern_runtime_passed"] = (
                    diagnostic["modern_runtime"] == RUNTIME
                    and runtime["torch"] == RUNTIME["torch_version"]
                    and runtime["cuda"] == RUNTIME["cuda_version"]
                    and runtime["device_name"] == policy["gpu_name"]
                    and isinstance(capability, list) and len(capability) == 2
                    and all(type(x) is int and x >= 0 for x in capability) and capability[0] >= 8)
            except (ValueError, KeyError, TypeError):
                pass
        if proof["modern_runtime_passed"]:
            proof["torch_cuda"] = command([
                sys.executable, "-c", "import torch; x=torch.ones(32, device='cuda'); x.mul_(3); "
                "torch.cuda.synchronize(); assert x.sum().item()==96.0; print('CUDA tensor execution passed')",
            ], timeout=120)
            proof["torch_cuda_execution_passed"] = proof["torch_cuda"]["returncode"] == 0
            if proof["torch_cuda_execution_passed"]:
                source, binary = output / "cuda_smoke.cu", output / "cuda_smoke"
                source.write_text(CUDA_SMOKE)
                architecture = "sm_" + "".join(str(x) for x in runtime["device_capability"])
                proof["compile"] = command(["nvcc", "-O2", "-arch=" + architecture,
                                             str(source.resolve()), "-o", str(binary.resolve())], timeout=120)
                if proof["compile"]["returncode"] == 0:
                    proof["nsys"] = command(["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none",
                        "--force-overwrite=true", "-o", str((output / "smoke-systems").resolve()),
                        str(binary.resolve())], timeout=120)
                    proof["ncu"] = command(["ncu", "--metrics", "sm__cycles_elapsed.avg,gpu__time_duration.sum",
                        "--csv", "--page", "raw", "--force-overwrite", "-o",
                        str((output / "smoke-compute").resolve()), str(binary.resolve())], timeout=120)
                    proof["hardware_counters_collected"] = (proof["ncu"]["returncode"] == 0
                        and has_hardware_counters(proof["ncu"]["stdout"]))
                    if proof["nsys"]["returncode"] == 0 and (output / "smoke-systems.nsys-rep").is_file():
                        proof["nsys_stats"] = command(["nsys", "stats", "--report", "cuda_gpu_kern_sum",
                            "--format", "csv", str((output / "smoke-systems.nsys-rep").resolve())], timeout=120)
                        proof["nsys_trace_collected"] = (proof["nsys_stats"]["returncode"] == 0
                            and has_cuda_kernel_trace(proof["nsys_stats"]["stdout"], kernel="drivebench_smoke"))
    for key, label in (("modern_runtime_passed", "actual modern runtime compatibility"),
                       ("torch_cuda_execution_passed", "actual CUDA tensor execution"),
                       ("hardware_counters_collected", "actual Nsight Compute SM counter collection"),
                       ("nsys_trace_collected", "actual Nsight Systems CUDA trace")):
        if not proof[key]:
            failures.append(label + " has not passed")
    after = collect_environment()
    failures.extend("postcheck: " + message for message in hardware_failures(after, policy, dedicated))
    if reference_identity(after) != reference_identity(environment):
        failures.append("GPU or full runtime changed during preflight")
    result = {"schema_version": 1, "proof_kind": POLICY_KIND, "hardware_policy": dict(policy),
              "passed": not failures, "failures": failures, "dedicated_attestation": dedicated,
              "environment": environment, "environment_after": after, "profiling_proof": proof,
              "actual_modern_runtime": runtime, "legacy_h100_proof": False,
              "model_loaded": False, "model_memory_fit_proven": False, "quality_validated": False,
              "external_process_exit_and_cleanup_still_required": True,
              "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--minimum-memory-mib", required=True, type=int)
    parser.add_argument("--minimum-host-ram-gib", required=True, type=int)
    parser.add_argument("--mig-unsupported", action="store_true")
    parser.add_argument("--dedicated", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--write-lock", type=Path)
    args = parser.parse_args()
    policy = {"schema_version": 1, "gpu_name": args.gpu_name,
              "minimum_memory_mib": args.minimum_memory_mib,
              "minimum_host_ram_bytes": args.minimum_host_ram_gib * 2**30,
              "mig_supported": not args.mig_unsupported}
    result = verify(args.output, policy, dedicated=args.dedicated)
    if args.write_lock:
        if not result["passed"]:
            parser.error("Cannot freeze a runtime before actual modern verification passes")
        args.write_lock.parent.mkdir(parents=True, exist_ok=True)
        with args.write_lock.open("x") as stream:
            json.dump(reference_identity(result["environment"]), stream, indent=2)
    print(json.dumps({"passed": result["passed"], "failures": result["failures"], "output": args.output}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
