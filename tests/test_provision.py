"""Provisioning gates and streaming integrity are testable without a GPU/network."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_checkpoint.py"
spec = importlib.util.spec_from_file_location("drivebench_download_checkpoint", SCRIPT)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


@pytest.mark.parametrize("returncode", [0, 1])
def test_preflight_executes_tensors_even_when_cuda_discovery_succeeds(tmp_path, monkeypatch, returncode):
    from drivebench import verify_environment as verifier
    environment = {"system": "Linux", "host_ram_bytes": 128_000_000_000,
                   "gpus": [{"name": "NVIDIA H100 80GB HBM3", "memory_total_mib": "81559", "mig_mode": "Disabled"}],
                   "gpu_processes": {"returncode": 0, "stdout": ""},
                   "torch": {"cuda_available": True}, "packages": {"triton": "2.0.0"}, "tools": {}}
    monkeypatch.setattr(verifier, "collect_environment", lambda: environment)
    commands = []
    def run(argv, timeout):
        commands.append(argv)
        return {"returncode": returncode, "stdout": "", "stderr": "no kernel image" if returncode else ""}
    monkeypatch.setattr(verifier, "command", run)
    result = verifier.verify(tmp_path, dedicated=True, collect=False)
    assert len(commands) == 1 and "torch.ones" in commands[0][2] and "torch.cuda.synchronize" in commands[0][2]
    assert result["profiling_proof"]["torch_cuda_execution_passed"] is (returncode == 0)
    assert ("PyTorch CUDA tensor execution failed" in result["failures"]) is (returncode != 0)
    assert not result["passed"]  # Tensor success alone never substitutes for profiler proof.


def proof(tmp_path):
    path = tmp_path / "verification.json"
    record = {"passed": True, "failures": [], "dedicated_attestation": True,
              "environment": {"gpus": [{"uuid": "GPU-test", "driver_version": "550.1"}]},
              "profiling_proof": {"hardware_counters_collected": True, "nsys_trace_collected": True,
                "ncu": {"returncode": 0, "stdout": '"Metric Name","Metric Value"\n"sm__cycles_elapsed.avg","1024"'},
                "nsys": {"returncode": 0},
                "nsys_stats": {"returncode": 0,
                               "stdout": '"Total Time (ns)","Instances","Name"\n1000,1,drivebench_smoke'}}}
    path.write_text(json.dumps(record))
    (tmp_path / "smoke-systems.nsys-rep").write_bytes(b"test trace")
    (tmp_path / "smoke-compute.ncu-rep").write_bytes(b"test counters")
    return path, record


def test_actual_counter_and_artifact_evidence_required(tmp_path, monkeypatch):
    path, record = proof(tmp_path)
    monkeypatch.setattr(downloader, "current_gpu_identity", lambda: ("GPU-test", "550.1"))
    assert downloader.require_hardware_proof(path)["passed"]
    record["profiling_proof"]["ncu"]["stdout"] = '"Metric Name","Metric Value"\n"gpu__time_duration.sum","1024"'
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="positive SM"):
        downloader.require_hardware_proof(path)
    path, record = proof(tmp_path)
    (tmp_path / "smoke-compute.ncu-rep").write_bytes(b"")
    with pytest.raises(RuntimeError, match="Missing nonempty"):
        downloader.require_hardware_proof(path)


def test_reject_proof_from_other_machine_or_failed_verification(tmp_path, monkeypatch):
    path, record = proof(tmp_path)
    monkeypatch.setattr(downloader, "current_gpu_identity", lambda: ("GPU-other", "550.1"))
    with pytest.raises(RuntimeError, match="different GPU"):
        downloader.require_hardware_proof(path)
    record["passed"] = False
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="successfully"):
        downloader.require_hardware_proof(path)


def test_reject_empty_cuda_trace_even_when_report_exists(tmp_path, monkeypatch):
    path, record = proof(tmp_path)
    monkeypatch.setattr(downloader, "current_gpu_identity", lambda: ("GPU-test", "550.1"))
    record["profiling_proof"]["nsys_stats"]["stdout"] = "No CUDA kernels were collected"
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="positive CUDA"):
        downloader.require_hardware_proof(path)


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}


def test_streaming_hash_and_publish(tmp_path, monkeypatch):
    payload = b"small fake checkpoint for integrity test"
    expected = hashlib.sha256(payload).hexdigest()
    path, record = proof(tmp_path)
    monkeypatch.setattr(downloader, "require_hardware_proof", lambda path: record)
    monkeypatch.setattr(downloader, "SHA256", expected)
    monkeypatch.setattr(downloader, "SIZE_BYTES", len(payload))
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *args, **kwargs: Response(payload))
    output = tmp_path / "ckpts" / "vista.safetensors"
    receipt = downloader.download(output, path)
    assert output.read_bytes() == payload
    assert receipt["sha256"] == expected and receipt["status"] == "downloaded_verified"
    assert not output.with_suffix(".safetensors.part").exists()
    assert output.with_suffix(".safetensors.receipt.json").exists()
    assert downloader.download(output, path)["status"] == "already_verified"


def test_corruption_keeps_partial_and_does_not_publish(tmp_path, monkeypatch):
    payload = b"corrupted checkpoint"
    path, record = proof(tmp_path)
    monkeypatch.setattr(downloader, "require_hardware_proof", lambda path: record)
    monkeypatch.setattr(downloader, "SHA256", "0" * 64)
    monkeypatch.setattr(downloader, "SIZE_BYTES", len(payload))
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *args, **kwargs: Response(payload))
    output = tmp_path / "vista.safetensors"
    with pytest.raises(RuntimeError, match="mismatch"):
        downloader.download(output, path)
    assert not output.exists()
    assert output.with_suffix(".safetensors.part").read_bytes() == payload
    with pytest.raises(RuntimeError, match="Partial download exists"):
        downloader.download(output, path)


def test_gate_runs_before_network_or_directory_creation(tmp_path, monkeypatch):
    def denied(path):
        raise RuntimeError("GPU proof missing")
    def unexpected(*args, **kwargs):
        raise AssertionError("network must not run before verification")
    monkeypatch.setattr(downloader, "require_hardware_proof", denied)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", unexpected)
    output = tmp_path / "ckpts" / "vista.safetensors"
    with pytest.raises(RuntimeError, match="GPU proof missing"):
        downloader.download(output, tmp_path / "missing.json")
    assert not output.parent.exists()
