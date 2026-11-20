"""Offline verification orchestration checks; all hardware observations mocked."""
import copy
import json
from pathlib import Path

import pytest

from drivebench_tuning import modern_verify_environment as verifier
from drivebench_tuning.modern_vista import RUNTIME


def policy():
    return {"schema_version": 1, "gpu_name": "NVIDIA A40",
            "minimum_memory_mib": 45000, "minimum_host_ram_bytes": 32 * 2**30,
            "mig_supported": False}


def environment():
    gpu = {"name": "NVIDIA A40", "uuid": "CPU-TEST-NOT-A-GPU", "memory_total_mib": "49140",
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
    lambda e: e["tools"]["nsys"].update(returncode=1),
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
    assert result["proof_kind"] == "modern_declared_gpu_counter_free_v2"
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
    {"missing_trace": True}, {"runtime_failure": True},
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


def test_counter_unavailability_does_not_change_counter_free_proof(monkeypatch,tmp_path):
    output=tmp_path/'proof'
    calls=install_fake_host(monkeypatch,output,missing_counter=True)
    result=verifier.verify(output,policy(),dedicated=True)
    assert result['passed']
    assert not any(argv[0]=='ncu' for argv in calls)
    assert result['profiling_proof']['hardware_counters_collected'] is False
    assert result['profiling_proof']['hardware_counters_required'] is False
    assert result['profiling_proof']['hardware_counter_status']=='not_requested_counter_free_protocol'
    assert all('--cpuctxsw=none' in argv for argv in calls if argv[:2]==['nsys','profile'])


def test_absent_ncu_tool_is_allowed_but_nsys_is_required():
    env=environment();env['tools'].pop('ncu')
    assert verifier.hardware_failures(env,policy(),True)==[]
    env['tools'].pop('nsys')
    assert verifier.hardware_failures(env,policy(),True)


def test_measurement_gate_requires_new_complete_counter_free_proof(monkeypatch,tmp_path):
    from drivebench_tuning.restart_measurement import _check_modern_proof
    output=tmp_path/'proof';install_fake_host(monkeypatch,output)
    good=verifier.verify(output,policy(),dedicated=True)
    hardware={k:v for k,v in policy().items() if k!='schema_version'}
    _check_modern_proof(good,hardware)
    for mutation in (
        lambda p:p.update(proof_kind='modern_declared_gpu_v1'),
        lambda p:p.update(passed=False),
        lambda p:p['profiling_proof'].update(nsys_trace_collected=False),
        lambda p:p['profiling_proof'].update(hardware_counters_required=True),
        lambda p:p['profiling_proof'].update(hardware_counters_collected=True),
        lambda p:p['profiling_proof'].update(hardware_counter_status='failed'),
        lambda p:p['profiling_proof'].update(torch_cuda_execution_passed=False),
        lambda p:p['hardware_policy'].update(gpu_name='OTHER'),
    ):
        bad=copy.deepcopy(good);mutation(bad)
        with pytest.raises(ValueError):_check_modern_proof(bad,hardware)


def test_new_reference_gate_checks_lock_source_and_runtime(monkeypatch,tmp_path):
    output=tmp_path/'proof';install_fake_host(monkeypatch,output)
    result=verifier.verify(output,policy(),dedicated=True)
    from drivebench.benchmarks.environment import reference_identity
    lock=tmp_path/'lock.json';lock.write_text(json.dumps(reference_identity(result['environment'])))
    cfg={'model':{'backend':'modern_vista'},'environment_verification':str(output/'verification.json'),'environment_lock':str(lock)}
    assert verifier.validate_reference_environment(cfg,result['environment'])['passed']
    changed=copy.deepcopy(result['environment']);changed['gpu']['uuid']='OTHER'
    with pytest.raises(ValueError):verifier.validate_reference_environment(cfg,changed)
    lock.write_text('{}')
    with pytest.raises(ValueError):verifier.validate_reference_environment(cfg,result['environment'])
    lock.write_text(json.dumps(reference_identity(result['environment'])))
    result['verifier_source_sha256']='0'*64;(output/'verification.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):verifier.validate_reference_environment(cfg,result['environment'])


def test_modern_worker_scopes_new_gate_and_restores_legacy_gate(monkeypatch,tmp_path):
    import sys
    from drivebench_tuning import restart_measurement as worker
    from drivebench import benchmark
    from drivebench.benchmarks.environment import reference_identity
    from drivebench.config import file_sha256
    output=tmp_path/'proof';install_fake_host(monkeypatch,output)
    result=verifier.verify(output,policy(),dedicated=True)
    lock=tmp_path/'lock.json';lock.write_text(json.dumps(reference_identity(result['environment'])))
    hardware={k:v for k,v in policy().items() if k!='schema_version'}
    config={'model':{'backend':'modern_vista'},'environment_verification':str(output/'verification.json'),'environment_lock':str(lock)}
    plan={'schedule':[{'run_id':'test-only','role':'modern_eager','block':0,'position':0}],
          'configs':{'modern_eager':config},'hardware':hardware,'cache_condition':'test-only'}
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    monkeypatch.setattr(worker,'_validate_plan',lambda x:None)
    monkeypatch.setattr(worker,'_USED_PROCESS',False)
    # This fixture stops at the gate; it never imports or executes a model.
    monkeypatch.delitem(sys.modules,'torch',raising=False)
    old_gate=benchmark._validate_reference_environment;old_engine=benchmark.InferenceEngine
    def fake_benchmark(cfg,dest):
        assert benchmark._validate_reference_environment is verifier.validate_reference_environment
        assert benchmark._validate_reference_environment(cfg,result['environment'])['passed']
        raise RuntimeError('TEST STOP BEFORE MODEL')
    monkeypatch.setattr(benchmark,'run_benchmark',fake_benchmark)
    with pytest.raises(RuntimeError,match='TEST STOP BEFORE MODEL'):
        worker.collect_restart(path,file_sha256(path),'test-only',tmp_path/'worker',deadline_unix=__import__('time').time()+30)
    assert benchmark._validate_reference_environment is old_gate
    assert benchmark.InferenceEngine is old_engine
    assert (tmp_path/'worker/restart-failure.json').is_file()
