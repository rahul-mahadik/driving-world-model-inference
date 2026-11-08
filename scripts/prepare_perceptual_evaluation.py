#!/usr/bin/env python3
"""Stage pinned LPIPS and AlexNet assets outside the inference environment.

No pip invocation or dependency installation occurs. The default final check uses
the current interpreter's existing Torch/torchvision/SciPy, CPU only, with network
access blocked during LPIPS construction and forward execution.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import sys
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlsplit
from urllib.request import urlopen
import zipfile

PYPI_URL = "https://pypi.org/pypi/lpips/0.1.4/json"
WHEEL_NAME = "lpips-0.1.4-py3-none-any.whl"
WHEEL_URL = "https://files.pythonhosted.org/packages/9b/13/1df50c7925d9d2746702719f40e864f51ed66f307b20ad32392f1ad2bb87/lpips-0.1.4-py3-none-any.whl"
WHEEL_SHA256 = "fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e"
WHEEL_BYTES = 53763
ALEXNET_NAME = "alexnet-owt-7be5be79.pth"
ALEXNET_URL = "https://download.pytorch.org/models/" + ALEXNET_NAME
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
ALEXNET_BYTES = 244408911
LEARNED_RELATIVE = "lpips/weights/v0.1/alex.pth"
LEARNED_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
ALLOWED_HOSTS = {"pypi.org", "files.pythonhosted.org", "download.pytorch.org"}


class PreparationError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def check_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.password or parsed.query:
        raise PreparationError("download_url_not_allowlisted")


def validate_file(path: Path, expected_sha256: str, expected_bytes: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise PreparationError("artifact_missing_or_symlink")
    if path.stat().st_size != expected_bytes:
        raise PreparationError("artifact_size_mismatch")
    if file_sha256(path) != expected_sha256:
        raise PreparationError("artifact_sha256_mismatch")


def download_pinned(url: str, destination: Path, expected_sha256: str, expected_bytes: int,
                    *, offline: bool = False, opener=urlopen) -> dict:
    """Validate cache hits fully; publish fresh downloads only after exact checks."""
    check_url(url)
    if destination.exists():
        validate_file(destination, expected_sha256, expected_bytes)
        return {**file_record(destination), "url": url, "cache_hit": True}
    if offline:
        raise PreparationError("offline_artifact_missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".partial", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, opener(url, timeout=60) as response:
            check_url(response.geturl())
            size = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                size += len(chunk)
                if size > expected_bytes:
                    raise PreparationError("download_exceeds_pinned_size")
                output.write(chunk)
        validate_file(temporary, expected_sha256, expected_bytes)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {**file_record(destination), "url": url, "cache_hit": False}


def verify_pypi_metadata(cache: Path, offline: bool) -> dict:
    path = cache / "lpips-0.1.4-pypi.json"
    if not path.exists():
        if offline:
            raise PreparationError("offline_pypi_metadata_missing")
        with urlopen(PYPI_URL, timeout=60) as response:
            check_url(response.geturl())
            data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise PreparationError("pypi_metadata_too_large")
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    payload = json.loads(path.read_text())
    wheels = [x for x in payload["urls"] if x["filename"] == WHEEL_NAME]
    if payload["info"]["version"] != "0.1.4" or len(wheels) != 1:
        raise PreparationError("pypi_version_or_wheel_mismatch")
    row = wheels[0]
    if row["digests"]["sha256"] != WHEEL_SHA256 or row["url"] != WHEEL_URL or row["size"] != WHEEL_BYTES:
        raise PreparationError("pypi_wheel_pin_mismatch")
    return {**file_record(path), "url": PYPI_URL, "wheel_sha256_matches_hard_pin": True}


def wheel_payloads(wheel: Path) -> dict[str, bytes]:
    validate_file(wheel, WHEEL_SHA256, WHEEL_BYTES)
    with zipfile.ZipFile(wheel) as archive:
        payloads = {}
        for item in archive.infolist():
            if item.is_dir():
                continue
            name = PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in item.filename or name.parts[0] not in {"lpips", "lpips-0.1.4.dist-info"}:
                raise PreparationError("unsafe_wheel_member")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise PreparationError("wheel_symlink_not_allowed")
            if item.filename in payloads:
                raise PreparationError("duplicate_wheel_member")
            payloads[item.filename] = archive.read(item)
    record_name = "lpips-0.1.4.dist-info/RECORD"
    records = list(csv.reader(io.StringIO(payloads[record_name].decode())))
    if {row[0] for row in records} != set(payloads):
        raise PreparationError("wheel_record_members_mismatch")
    for name, digest, size in records:
        if name == record_name:
            continue
        expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(payloads[name]).digest()).rstrip(b"=").decode()
        if digest != expected or int(size) != len(payloads[name]):
            raise PreparationError("wheel_record_integrity_mismatch")
    if hashlib.sha256(payloads[LEARNED_RELATIVE]).hexdigest() != LEARNED_SHA256:
        raise PreparationError("lpips_learned_weight_pin_mismatch")
    return payloads


def stage_wheel(wheel: Path, target: Path) -> dict:
    if target.resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise PreparationError("target_must_be_outside_current_environment")
    payloads = wheel_payloads(wheel)
    if target.exists():
        if target.is_symlink():
            raise PreparationError("target_symlink_not_allowed")
        for name, data in payloads.items():
            path = target / name
            validate_file(path, hashlib.sha256(data).hexdigest(), len(data))
        extras = [p for p in target.rglob("*") if p.is_file() and "__pycache__" not in p.parts and str(p.relative_to(target)) not in payloads]
        if extras:
            raise PreparationError("unexpected_file_in_package_target")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=target.name + ".", dir=target.parent))
        try:
            for name, data in payloads.items():
                path = temporary / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return {"target": str(target.resolve()), "version": "0.1.4", "installation_method": "verified wheel extraction; no dependency installation",
            "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in sorted(payloads.items())}}


@contextlib.contextmanager
def network_blocked(torch):
    def denied(*args, **kwargs):
        raise PreparationError("network_forbidden_during_cpu_verification")
    old = (socket.socket.connect, socket.create_connection, torch.hub.download_url_to_file)
    socket.socket.connect = denied
    socket.create_connection = denied
    torch.hub.download_url_to_file = denied
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection, torch.hub.download_url_to_file = old


def verify_cpu(target: Path, torch_home: Path, threads: int) -> dict:
    if "torch" in sys.modules or "lpips" in sys.modules:
        raise PreparationError("verification_requires_fresh_process")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_HOME"] = str(torch_home.resolve())
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(target.resolve()))
    import torch
    import torchvision
    import numpy as np
    import scipy
    import lpips
    if not Path(lpips.__file__).resolve().is_relative_to(target.resolve()):
        raise PreparationError("lpips_imported_outside_target")
    if torch.__version__.split("+")[0] != "2.0.1" or torchvision.__version__.split("+")[0] != "0.15.2":
        raise PreparationError("evaluation_dependencies_differ_from_frozen_inference_versions")
    if importlib.metadata.version("lpips") != "0.1.4":
        raise PreparationError("lpips_import_version_mismatch")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    before = torch.cuda.is_initialized()
    with network_blocked(torch), torch.no_grad():
        model = lpips.LPIPS(net="alex", version="0.1", pretrained=True,
                            pnet_rand=False, model_path=str(target / LEARNED_RELATIVE), verbose=False).cpu().eval()
        left = torch.linspace(-1, 1, 3 * 64 * 64, dtype=torch.float32).reshape(1, 3, 64, 64)
        right = torch.flip(left, dims=[3])
        same = float(model(left, left).item())
        different = float(model(left, right).item())
        reverse = float(model(right, left).item())
    after = torch.cuda.is_initialized()
    if before or after or any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise PreparationError("verification_was_not_cpu_only")
    if not np.isfinite([same, different, reverse]).all() or same != 0 or different <= 0 or abs(different - reverse) > 1e-6:
        raise PreparationError("lpips_smoke_invariants_failed")
    weights = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        weights.update(f"{name}:{array.dtype}:{array.shape}".encode())
        weights.update(array.tobytes())
    alex_module = importlib.import_module("torchvision.models.alexnet")
    return {"passed": True, "device": "cpu", "CUDA_VISIBLE_DEVICES": "", "cuda_initialized_before": before, "cuda_initialized_after": after,
            "network_blocked_during_model_construction_and_forward": True, "input_shape": [1, 3, 64, 64], "input_range": [-1, 1],
            "identical_lpips": same, "different_lpips": different, "reverse_lpips": reverse,
            "model_state_dict_sha256": weights.hexdigest(), "state_hash_method": "sorted name:dtype:shape plus contiguous tensor bytes; matches PerceptualMetric convention",
            "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "lpips": importlib.metadata.version("lpips")},
            "source_files": {"torchvision_alexnet": file_record(Path(alex_module.__file__)), "lpips": file_record(Path(lpips.__file__)), "lpips_model": file_record(target / "lpips/lpips.py"), "lpips_trunk": file_record(target / "lpips/pretrained_networks.py")},
            "scope": "Dependency/weight integrity and CPU execution smoke only; not calibration, model quality, or a perceptual acceptance threshold."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("outputs/evaluation-packages/lpips"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/evaluation-packages/downloads"))
    parser.add_argument("--torch-home", type=Path, default=Path("outputs/evaluation-packages/torch-home"))
    parser.add_argument("--receipt", type=Path, default=Path("outputs/evaluation-packages/setup-receipt.json"))
    parser.add_argument("--prepare-only", action="store_true", help="Stage verified assets without importing Torch; receipt marks CPU verification pending")
    parser.add_argument("--offline", action="store_true", help="Require previously downloaded, fully verified assets; no network acquisition")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.receipt.exists():
        parser.error("Receipt exists; select a fresh path to preserve prior evidence")
    for path in (args.target, args.cache, args.torch_home, args.receipt):
        if path.resolve().is_relative_to(Path(sys.prefix).resolve()):
            parser.error("Every output path must be outside the active Python environment")
    receipt = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(), "passed": False,
               "no_dependency_installation": True, "frozen_inference_environment_mutated": False,
               "script": file_record(Path(__file__)), "python_executable": sys.executable,
               "pin_provenance": {"lpips": "Hard-pinned wheel hash and size from official PyPI 0.1.4 JSON; metadata rechecked against pin.",
                   "alexnet": "Full SHA256 computed from official HTTPS object and checked against official torchvision filename hash prefix 7be5be79, then hard-pinned here; torchvision publishes that prefix, not a separate full-hash manifest.",
                   "torchvision_source": "https://github.com/pytorch/vision/blob/v0.15.2/torchvision/models/alexnet.py", "learned_weights": "LPIPS v0.1 AlexNet learned weights contained in fully verified 0.1.4 wheel; independent hard pin and wheel RECORD checked."}}
    try:
        receipt["pypi_metadata"] = verify_pypi_metadata(args.cache, args.offline)
        wheel = args.cache / WHEEL_NAME
        receipt["wheel"] = download_pinned(WHEEL_URL, wheel, WHEEL_SHA256, WHEEL_BYTES, offline=args.offline)
        receipt["package"] = stage_wheel(wheel, args.target)
        cached_weights = args.cache / ALEXNET_NAME
        receipt["alexnet_download"] = download_pinned(ALEXNET_URL, cached_weights, ALEXNET_SHA256, ALEXNET_BYTES, offline=args.offline)
        destination = args.torch_home / "hub/checkpoints" / ALEXNET_NAME
        if destination.exists():
            validate_file(destination, ALEXNET_SHA256, ALEXNET_BYTES)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached_weights, destination)
            validate_file(destination, ALEXNET_SHA256, ALEXNET_BYTES)
        receipt["alexnet_cached_weight"] = file_record(destination)
        receipt["lpips_learned_weight"] = file_record(args.target / LEARNED_RELATIVE)
        receipt["evaluation_environment"] = {"PYTHONPATH_prefix": str(args.target.resolve()), "TORCH_HOME": str(args.torch_home.resolve()), "CUDA_VISIBLE_DEVICES": ""}
        if args.prepare_only:
            receipt["status"] = "assets_verified_cpu_execution_pending"
        else:
            receipt["cpu_verification"] = verify_cpu(args.target.resolve(), args.torch_home.resolve(), args.threads)
            receipt["status"] = "cpu_verified"
            receipt["passed"] = True
        receipt["assets_verified"] = True
        code = 0
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        code = 1
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "passed": receipt["passed"], "receipt": str(args.receipt)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
