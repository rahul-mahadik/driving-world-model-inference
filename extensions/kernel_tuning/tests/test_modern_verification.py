"""Offline verification orchestration checks; all hardware observations mocked."""
import copy
import json
from pathlib import Path

import pytest

from drivebench_tuning import modern_verify_environment as verifier
from drivebench_tuning.modern_vista import RUNTIME


def policy():
    return {"schema_version": 1, "gpu_name": "NVIDIA RTX A6000",
            "minimum_memory_mib": 45000, "minimum_host_ram_bytes": 32 * 2**30,
            "mig_supported": False}


def environment():
    gpu = {"name": "NVIDIA RTX A6000", "uuid": "CPU-TEST-NOT-A-GPU", "memory_total_mib": "49140",
           "mig_mode": "[N/A]"}
    return {"system": "Linux", "host_ram_bytes": 64 * 2**30,
            "gpus": [gpu], "gpu": gpu, "nvidia_smi": {"returncode": 0},
            "gpu_processes": {"returncode": 0, "stdout": ""},
            "torch": {"cuda_available": True, "version": RUNTIME["torch_version"]},
            "packages": {"torch": RUNTIME["torch_version"], "triton": "test"},
            "tools": {name: {"returncode": 0, "stdout": "fixture"} for name in ("nvcc", "nsys", "ncu")}}


@pytest.mark.parametrize("mode,supported,passes", [
    ("Disabled", True, True), ("N/A", False, True), ("[N/A]", False, True),
    ("Enabled", True, False), ("unknown", False, False), (None, False, False),
    ("N/A", True, False), ("Disabled", False, False),
])
def test_mig_is_explicit_not_a_blanket_exception(mode, supported, passes):
    env, declared = environment(), policy()
    env["gpu"]["mig_mode"] = mode
    declared["mig_supported"] = supported
    assert (not verifier.hardware_failures(env, declared, True)) is passes


@pytest.mark.parametrize("mutation", [
    lambda e: e["gpu"].update(name="NVIDIA H100"),
    lambda e: e["gpu"].update(uuid=""),
    lambda e: e["gpu"].update(memory_total_mib="nan"),
    lambda e: e["gpu"].update(memory_total_mib="44999"),
    lambda e: e.update(host_ram_bytes=31 * 2**30),
    lambda e: e.update(gpus=e["gpus"] * 2),
    lambda e: e["gpu_processes"].update(stdout="876543, unrelated-job, 4000"),
    lambda e: e["gpu_processes"].update(returncode=1),
    lambda e: e.update(torch=None),
    lambda e: e["tools"]["ncu"].update(returncode=1),
])
def test_hardware_changes_fail_closed(mutation):
    env = environment()
    mutation(env)
    assert verifier.hardware_failures(env, policy(), True)


def install_fake_host(monkeypatch, output, *, missing_counter=False, missing_trace=False,
                      runtime_failure=False, cuda_failure=False, after_mutation=None):
    calls = []
    before, after = environment(), environment()
    if after_mutation:
        after_mutation(after)
    states = iter([before, after])
    monkeypatch.setattr(verifier, "collect_environment", lambda: copy.deepcopy(next(states)))
    def command(argv, timeout=30):
        calls.append(argv)
        result = {"command": argv, "returncode": 0, "stdout": "", "stderr": ""}
        if "--check-runtime" in argv:
            result.update(returncode=int(runtime_failure), stdout=json.dumps({
                "modern_runtime": RUNTIME, "actual_runtime": {"torch": RUNTIME["torch_version"],
                "cuda": RUNTIME["cuda_version"], "device_name": policy()["gpu_name"],
                "device_capability": [8, 6]}}))
        elif "-c" in argv:
            result["returncode"] = int(cuda_failure)
        elif argv[:2] == ["nsys", "profile"]:
            (output / "smoke-systems.nsys-rep").write_text("MOCK NOT REAL TRACE")
        elif argv[:2] == ["nsys", "stats"]:
            result["stdout"] = "Total Time (ns),Instances,Name\n100,1,drivebench_smoke\n" if not missing_trace else ""
        elif argv[0] == "ncu":
            result["stdout"] = "Metric Name,Metric Value\nsm__cycles_elapsed.avg,123\n" if not missing_counter else ""
        return result
    monkeypatch.setattr(verifier, "command", command)
    return calls


def test_complete_mock_proof_keeps_model_validation_false(monkeypatch, tmp_path):
    output = tmp_path / "proof"
    calls = install_fake_host(monkeypatch, output)
    result = verifier.verify(output, policy(), dedicated=True)
    assert result["passed"] is True
    assert result["proof_kind"] == "modern_declared_gpu_v1"
    assert result["hardware_policy"] == policy()
    assert result["legacy_h100_proof"] is False
    assert result["model_loaded"] is False and result["model_memory_fit_proven"] is False
    assert result["quality_validated"] is False
    assert result["external_process_exit_and_cleanup_still_required"] is True
    assert any("-arch=sm_86" in argv for argv in calls)
    assert (output / "verification.json").is_file()
    with pytest.raises(FileExistsError):
        verifier.verify(output, policy(), dedicated=True)


@pytest.mark.parametrize("kwargs", [
    {"missing_counter": True}, {"missing_trace": True}, {"runtime_failure": True},
    {"cuda_failure": True},
    {"after_mutation": lambda e: e["gpu"].update(uuid="CHANGED")},
    {"after_mutation": lambda e: e["gpu_processes"].update(stdout="876543, other, 500")},
])
def test_incomplete_or_changed_proof_cannot_pass(monkeypatch, tmp_path, kwargs):
    output = tmp_path / "proof"
    install_fake_host(monkeypatch, output, **kwargs)
    result = verifier.verify(output, policy(), dedicated=True)
    assert result["passed"] is False and result["failures"]
    assert json.loads((output / "verification.json").read_text())["passed"] is False


def test_no_collection_and_no_attestation_cannot_be_proof(monkeypatch, tmp_path):
    output = tmp_path / "proof"
    calls = install_fake_host(monkeypatch, output)
    result = verifier.verify(output, policy(), dedicated=False, collect=False)
    assert not result["passed"] and not calls


@pytest.mark.parametrize("changes", [
    {"schema_version": True}, {"mig_supported": "false"}, {"minimum_memory_mib": True},
    {"minimum_host_ram_bytes": 0}, {"gpu_name": ""}, {"unknown": True},
])
def test_invalid_policy_is_rejected_before_probes(tmp_path, changes):
    declared = {**policy(), **changes}
    with pytest.raises(ValueError):
        verifier.verify(tmp_path / "proof", declared, dedicated=True)
    assert not (tmp_path / "proof").exists()
