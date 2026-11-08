#!/usr/bin/env python3
"""Download pinned Vista only after same-GPU hardware-counter verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request


REVISION = "a92d28c7b3da931ec959d587153dc45fffe570ef"
SHA256 = "1b267bb753aa5c5e4d363f6e4c864dd4e343996dedc18b5ab20af6d20be34b98"
SIZE_BYTES = 10053461572
URL = f"https://huggingface.co/OpenDriveLab/Vista/resolve/{REVISION}/vista.safetensors"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_gpu_identity() -> tuple[str, str]:
    if platform.system() != "Linux":
        raise RuntimeError("Checkpoint download is gated to the verified Linux reference host")
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    rows = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 2:
        raise RuntimeError("Expected exactly one reference GPU")
    return tuple(item.strip() for item in rows[0])


def require_hardware_proof(path: Path) -> dict:
    # Avoid import/install dependence when invoking this script directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from drivebench.verify_environment import has_cuda_kernel_trace, has_hardware_counters

    result = json.loads(path.read_text())
    proof = result.get("profiling_proof", {})
    if result.get("passed") is not True or result.get("dedicated_attestation") is not True or result.get("failures"):
        raise RuntimeError("Run drivebench.verify_environment --dedicated successfully before downloading")
    if proof.get("hardware_counters_collected") is not True or proof.get("nsys_trace_collected") is not True:
        raise RuntimeError("Both actual Nsight hardware counters and CUDA trace are required")
    if proof.get("ncu", {}).get("returncode") != 0 or not has_hardware_counters(proof.get("ncu", {}).get("stdout", "")):
        raise RuntimeError("Verification contains no positive SM hardware counter")
    if proof.get("nsys", {}).get("returncode") != 0:
        raise RuntimeError("Nsight Systems collection did not succeed")
    if (proof.get("nsys_stats", {}).get("returncode") != 0
            or not has_cuda_kernel_trace(proof.get("nsys_stats", {}).get("stdout", ""), kernel="drivebench_smoke")):
        raise RuntimeError("Verification contains no positive CUDA smoke-kernel trace")
    for filename in ("smoke-systems.nsys-rep", "smoke-compute.ncu-rep"):
        artifact = path.parent / filename
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"Missing nonempty profiler artifact: {artifact}")
    recorded = result.get("environment", {}).get("gpus", [])
    if len(recorded) != 1 or not recorded[0].get("uuid") or not recorded[0].get("driver_version"):
        raise RuntimeError("Verification does not identify the GPU and driver")
    current_uuid, current_driver = current_gpu_identity()
    if (current_uuid, current_driver) != (recorded[0]["uuid"], recorded[0]["driver_version"]):
        raise RuntimeError("Verification belongs to a different GPU or driver; rerun on this host")
    return result


def download(destination: Path, verification: Path) -> dict:
    require_hardware_proof(verification)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == SIZE_BYTES and checksum(destination) == SHA256:
            return {"path": str(destination.resolve()), "sha256": SHA256, "status": "already_verified"}
        raise RuntimeError("Destination exists with different content; preserve it and choose a new path")
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise RuntimeError(f"Partial download exists: {partial}; preserve/move it before retrying")
    digest, size = hashlib.sha256(), 0
    request = urllib.request.Request(URL, headers={"User-Agent": "drivebench-checkpoint/1"})
    # Exclusive create prevents collisions. Failed/incomplete downloads remain .part.
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as stream:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != SIZE_BYTES:
            raise RuntimeError(f"Unexpected download size: {content_length}")
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > SIZE_BYTES:
                raise RuntimeError("Response exceeds pinned checkpoint length")
            stream.write(block)
            digest.update(block)
        stream.flush()
        os.fsync(stream.fileno())
    if size != SIZE_BYTES or digest.hexdigest() != SHA256:
        raise RuntimeError("Checkpoint byte count or SHA256 mismatch; .part retained for diagnosis")
    # Hard-link publication refuses to overwrite a file created concurrently.
    # Source and target are in the same directory/filesystem.
    os.link(partial, destination)
    partial.unlink()
    receipt = {"path": str(destination.resolve()), "sha256": SHA256, "size_bytes": SIZE_BYTES,
               "revision": REVISION, "url": URL, "verification_sha256": checksum(verification), "status": "downloaded_verified"}
    destination.with_suffix(destination.suffix + ".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/vista.safetensors"))
    parser.add_argument("--verification", type=Path, default=Path("outputs/environment/verification.json"))
    args = parser.parse_args()
    print(json.dumps(download(args.output, args.verification), indent=2))


if __name__ == "__main__":
    main()
