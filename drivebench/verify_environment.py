"""Fail closed until a full H100 and actual hardware-counter collection pass."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from pathlib import Path

from .benchmarks.environment import collect_environment, command, reference_identity

CUDA_SMOKE = r'''
#include <cuda_runtime.h>
#include <cstdio>
__global__ void drivebench_smoke(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float a = x[i]; for (int k=0;k<100;k++) a=a*1.00001f+0.001f; x[i]=a; }
}
int main() {
  float *x; int n=1<<20;
  if(cudaMalloc(&x,n*sizeof(float))!=cudaSuccess) return 1;
  if(cudaMemset(x,0,n*sizeof(float))!=cudaSuccess) return 2;
  drivebench_smoke<<<(n+255)/256,256>>>(x,n);
  if(cudaDeviceSynchronize()!=cudaSuccess) return 3;
  float first; cudaMemcpy(&first,x,sizeof(float),cudaMemcpyDeviceToHost);
  cudaFree(x); return (first>0.09f && first<0.11f) ? 0 : 4;
}
'''


def has_hardware_counters(text):
    """Accept NCU long/wide CSV only with a positive SM hardware counter."""
    def positive(value):
        try:
            number = float((value or "").replace(",", ""))
            return math.isfinite(number) and number > 0
        except (TypeError, ValueError):
            return False

    lines = text.splitlines()
    for start, line in enumerate(lines):
        header = next(csv.reader([line]), [])
        if {"Metric Name", "Metric Value"}.issubset(header):
            for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
                if row.get("Metric Name") == "sm__cycles_elapsed.avg" and positive(row.get("Metric Value")):
                    return True
        elif {"ID", "Kernel Name", "sm__cycles_elapsed.avg"}.issubset(header):
            # NCU 2025 raw output places metrics in columns and a units row
            # before kernel rows. Require a kernel identity, not a header/unit.
            for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
                if ((row.get("ID") or "").isdigit() and (row.get("Kernel Name") or "").strip()
                        and positive(row.get("sm__cycles_elapsed.avg"))):
                    return True
    return False


def has_cuda_kernel_trace(text, kernel=None):
    """Require an actual positive CUDA kernel entry in nsys's CSV statistics."""
    lines = text.splitlines()
    for start, line in enumerate(lines):
        if "Total Time (ns)" not in line or "Instances" not in line or "Name" not in line:
            continue
        for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
            if kernel and kernel not in (row.get("Name") or ""):
                continue
            try:
                duration = float((row.get("Total Time (ns)") or "").replace(",", ""))
                instances = float((row.get("Instances") or "").replace(",", ""))
                if math.isfinite(duration) and duration > 0 and math.isfinite(instances) and instances >= 1:
                    return True
            except ValueError:
                pass
    return False


def verify(output, dedicated=False, collect=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = collect_environment()
    failures = []
    if env["system"] != "Linux":
        failures.append("reference host must run Linux")
    if not env.get("host_ram_bytes") or env["host_ram_bytes"] < 64_000_000_000:
        failures.append("reference host requires at least 64 GB RAM")
    if len(env["gpus"]) != 1:
        failures.append("exactly one full GPU must be visible")
    else:
        gpu = env["gpus"][0]
        if "H100" not in gpu["name"] or not 79000 <= float(gpu["memory_total_mib"]) <= 83000:
            failures.append("reference GPU must be an H100 80 GB; record SXM or PCIe variant")
        if gpu["mig_mode"] != "Disabled":
            failures.append("MIG must be disabled")
    if not dedicated:
        failures.append("dedicated allocation requires operator attestation: --dedicated")
    processes = env["gpu_processes"]
    if processes["returncode"] != 0:
        failures.append("cannot verify active GPU processes")
    if processes["returncode"] == 0 and processes["stdout"]:
        other = [row for row in csv.reader(io.StringIO(processes["stdout"])) if row and row[0].strip() != str(os.getpid())]
        if other:
            failures.append("other GPU compute processes are active; reference allocation is not idle/dedicated")
    if not env.get("torch") or not env["torch"]["cuda_available"]:
        failures.append("working CUDA PyTorch is required")
    if "triton" not in {p.lower() for p in env["packages"]}:
        failures.append("Triton is unavailable")
    for name, result in env["tools"].items():
        if result["returncode"] != 0:
            failures.append(f"{name} unavailable")
    proof = {"hardware_counters_collected": False, "nsys_trace_collected": False,
             "torch_cuda_execution_passed": False}
    if env.get("torch") and env["torch"]["cuda_available"]:
        # Device discovery can succeed even when the wheel has no executable
        # kernels for this architecture. A subprocess also releases its context
        # before the independent Nsight counter collection starts.
        proof["torch_cuda"] = command([
            sys.executable, "-c",
            "import torch; x=torch.ones(32, device='cuda'); x.mul_(3); "
            "torch.cuda.synchronize(); value=x.sum().item(); assert value == 96.0; "
            "print('CUDA tensor execution passed: sum=96.0')",
        ], timeout=120)
        proof["torch_cuda_execution_passed"] = proof["torch_cuda"]["returncode"] == 0
        if not proof["torch_cuda_execution_passed"]:
            failures.append("PyTorch CUDA tensor execution failed")
    if collect and env["gpus"] and all(r["returncode"] == 0 for r in env["tools"].values()):
        source, binary = output / "cuda_smoke.cu", output / "cuda_smoke"
        source.write_text(CUDA_SMOKE)
        proof["compile"] = command(["nvcc", "-O2", str(source.resolve()), "-o", str(binary.resolve())], timeout=120)
        if proof["compile"]["returncode"] == 0:
            proof["nsys"] = command(["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--force-overwrite=true",
                                     "-o", str((output / "smoke-systems").resolve()), str(binary.resolve())], timeout=120)
            proof["ncu"] = command(["ncu", "--metrics", "sm__cycles_elapsed.avg,gpu__time_duration.sum", "--csv", "--page", "raw",
                                    "--force-overwrite", "-o", str((output / "smoke-compute").resolve()), str(binary.resolve())], timeout=120)
            proof["hardware_counters_collected"] = proof["ncu"]["returncode"] == 0 and has_hardware_counters(proof["ncu"]["stdout"])
            if proof["nsys"]["returncode"] == 0 and (output / "smoke-systems.nsys-rep").is_file():
                proof["nsys_stats"] = command(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                                              str((output / "smoke-systems.nsys-rep").resolve())], timeout=120)
                proof["nsys_trace_collected"] = (proof["nsys_stats"]["returncode"] == 0
                    and has_cuda_kernel_trace(proof["nsys_stats"]["stdout"], kernel="drivebench_smoke"))
    if not proof["hardware_counters_collected"]:
        failures.append("actual Nsight Compute SM hardware-counter collection has not passed")
    if not proof["nsys_trace_collected"]:
        failures.append("Nsight Systems CUDA trace has not passed")
    result = {"passed": not failures, "failures": failures, "dedicated_attestation": dedicated,
              "environment": env, "profiling_proof": proof}
    (output / "verification.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/environment")
    parser.add_argument("--dedicated", action="store_true", help="attest that this is a dedicated full GPU allocation")
    parser.add_argument("--allow-cpu", action="store_true", help="record local limitations without a failing exit code; never passes GPU verification")
    parser.add_argument("--write-lock", type=Path)
    args = parser.parse_args()
    result = verify(args.output, dedicated=args.dedicated)
    if args.write_lock:
        if not result["passed"]:
            parser.error("cannot freeze an environment before reference verification passes")
        args.write_lock.parent.mkdir(parents=True, exist_ok=True)
        args.write_lock.write_text(json.dumps(reference_identity(result["environment"]), indent=2))
    print(json.dumps({"passed": result["passed"], "failures": result["failures"], "output": args.output}, indent=2))
    return 0 if result["passed"] or args.allow_cpu else 1


if __name__ == "__main__":
    raise SystemExit(main())
