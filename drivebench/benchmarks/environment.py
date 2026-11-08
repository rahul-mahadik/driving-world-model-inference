from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone


def command(argv, timeout=30):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"command": argv, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": argv, "returncode": -1, "stdout": "", "stderr": str(error)}


def xformers_build_identity():
    """Distinguish official and rebuilt wheels that share a package version."""
    try:
        distribution = importlib.metadata.distribution("xformers")
    except importlib.metadata.PackageNotFoundError:
        return None
    metadata_path = Path(distribution.locate_file("xformers/cpp_lib.json"))
    metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest() if metadata_path.is_file() else None
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    archive = direct.get("archive_info", {})
    wheel_hash = archive.get("hashes", {}).get("sha256")
    if wheel_hash is None and archive.get("hash", "").startswith("sha256="):
        wheel_hash = archive["hash"].split("=", 1)[1]
    if wheel_hash is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", wheel_hash):
        raise ValueError("Installed xFormers wheel provenance has an invalid SHA256")
    return {"version": distribution.version, "cpp_lib_sha256": metadata_hash,
            "installed_wheel_sha256": wheel_hash.lower() if wheel_hash else None}


def collect_environment():
    gpu = command(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,pci.bus_id,mig.mode.current,power.limit,clocks.max.sm,clocks.max.memory",
                   "--format=csv,noheader,nounits"])
    devices = []
    if gpu["returncode"] == 0:
        keys = ["name", "uuid", "memory_total_mib", "driver_version", "pci_bus_id", "mig_mode", "power_limit_w", "max_sm_clock_mhz", "max_memory_clock_mhz"]
        devices = [dict(zip(keys, [x.strip() for x in row])) for row in csv.reader(io.StringIO(gpu["stdout"]))]
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        ram = None
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    out = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
           "system": platform.system(), "python": sys.version, "executable": sys.executable,
           "cuda_runtime_environment": {name: os.environ.get(name) for name in (
               "CUDA_MODULE_LOADING", "CUDA_CACHE_PATH", "CUDA_CACHE_MAXSIZE", "CUDA_CACHE_DISABLE",
               "CUDA_FORCE_PTX_JIT", "CUDA_DISABLE_PTX_JIT", "CUBLAS_WORKSPACE_CONFIG")},
           "host_ram_bytes": ram, "packages": packages, "gpus": devices, "nvidia_smi": gpu,
           "xformers_build": xformers_build_identity(),
           "gpu": devices[0] if len(devices) == 1 else None,
           "gpu_processes": command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"]),
           "tools": {name: command([name, "--version"]) for name in ("nvcc", "nsys", "ncu")},
           "git": {"commit": command(["git", "rev-parse", "HEAD"])["stdout"],
                   "status": command(["git", "status", "--porcelain"])["stdout"]}}
    try:
        import torch
        out["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                        "cuda_available": torch.cuda.is_available(),
                        "cudnn": torch.backends.cudnn.version(),
                        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    except ImportError:
        out["torch"] = None
    return out


def reference_identity(environment):
    return {"gpu": environment.get("gpu"), "torch": environment.get("torch"),
            "cuda_runtime_environment": environment.get("cuda_runtime_environment"),
            "xformers_build": environment.get("xformers_build"),
            "python": environment.get("python"), "packages": environment.get("packages"),
            "tools": {k: v.get("stdout") for k, v in environment.get("tools", {}).items()}}


def source_identity():
    """Fingerprint the executing package, including uncommitted/untracked Python."""
    from drivebench.config import digest, file_sha256
    root = Path(__file__).resolve().parents[1]
    return digest({str(path.relative_to(root)): file_sha256(path) for path in sorted(root.rglob("*.py"))})
