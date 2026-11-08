"""Guarded compiler orchestration fixtures; no Torch, model, or GPU execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import signal
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from drivebench.benchmarks.environment import reference_identity
from drivebench.config import digest, file_sha256, load_config
from scripts import validate_compiled_preflight as launcher


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def declared(tmp_path):
    base = load_config(ROOT / "configs/preflight_scene0103_corrected.yaml")
    base["output"] = str(tmp_path / "eager")
    base["preflight"]["validation_output"] = str(tmp_path / "eager.json")
    base["data"]["manifest"] = str(tmp_path / "clip.jsonl")
    base_path = tmp_path / "preflight_scene0103_corrected.yaml"
    base_path.write_text(yaml.safe_dump(base))
    config = copy.deepcopy(base)
    config.update(name="CPU orchestration fixture", output=str(tmp_path / "compiled"))
    config["optimizations"]["compile_denoiser"] = True
    config["benchmark"].update(requests=1, warmup_requests=2, max_warmup_requests=3)
    config["preflight"]["validation_output"] = str(tmp_path / "compiled/strict-agreement.json")
    config_path = tmp_path / "compiled.yaml"
    config_path.write_text(yaml.safe_dump(config))
    record = {"request_id": base["preflight"]["request_id"],
              "scene_id": base["preflight"]["scene_id"], "preflight_only": True}
    manifest = Path(base["data"]["manifest"])
    manifest.write_text(json.dumps(record) + "\n")
    metadata = manifest.with_suffix(".jsonl.meta.json")
    metadata.write_text(json.dumps({"preflight_config_sha256": file_sha256(base_path),
                                    "manifest_sha256": file_sha256(manifest)}))
    provenance = {**record, "synthetic": False, "candidate_ids": ["candidate-fixture"],
                  "record_sha256": digest(record), "manifest_sha256": file_sha256(manifest),
                  "manifest_metadata_sha256": file_sha256(metadata)}
    identity = {"parameter_grad_policy": "preserve_upstream_loader_flags", "execution_grad_mode": "no_grad",
                "autocast_cache_enabled": True,
                "parameter_dtypes": {"torch.float32": {"tensors": 2, "elements": 10,
                    "requires_grad_tensors": 1, "requires_grad_elements": 6}}}
    proof = {"schema_version": 2, "passed": True, "mode": "agreement", "atol": 0, "rtol": 0,
             "config": base, "request_id": record["request_id"], "generation": base["generation"],
             "optimizations": base["optimizations"], "validated_request_count": 1,
             "checks": {"upstream_agreement": {"passed": True}},
             "input_provenance": provenance, "input_provenance_sha256": digest(provenance),
             "runtime_identity": reference_identity({}), "implementation_sha256": "fixture-source",
             "runs": {label: {"model_identity": {**copy.deepcopy(identity), "backend": backend},
                 "candidate_count": 1, "denoiser_evaluations": 50}
                 for label, backend in (("upstream", "upstream"), ("refactored", "vista"))}}
    proof_path = Path(base["preflight"]["validation_output"])
    proof_path.write_text(json.dumps(proof))
    return SimpleNamespace(base=base, base_path=base_path, config=config, config_path=config_path,
                           proof=proof, proof_path=proof_path, manifest=manifest, metadata=metadata)


def test_default_resolves_corrected_base_and_validates_its_evidence(declared):
    config, base_path, output, report = launcher.inspect_config(declared.config_path)
    assert base_path == declared.base_path
    assert report == output / "strict-agreement.json"
    evidence = launcher.baseline_input_evidence(config, base_path)
    assert evidence["sha256"] == file_sha256(declared.proof_path)
    assert evidence["corrected_model_identities"]["upstream"]["execution_grad_mode"] == "no_grad"


def test_explicit_base_location_is_bound_in_attempt(declared, monkeypatch):
    moved = declared.base_path.with_name("explicit-base.yaml")
    moved.write_bytes(declared.base_path.read_bytes())
    declared.base_path.unlink()
    monkeypatch.setattr(launcher, "baseline_input_evidence", lambda *args: (_ for _ in ()).throw(ValueError("stop before Torch")))
    assert launcher.main(["--config", str(declared.config_path), "--base-config", str(moved)]) == 1
    attempt = json.loads((Path(declared.config["output"]) / "attempt.json").read_text())
    assert attempt["base_config"] == str(moved)
    assert attempt["base_config_sha256"] == file_sha256(moved)
    assert attempt["status"] == "failed"


@pytest.mark.parametrize("section,key,value", [
    ("generation", "steps", 49), ("optimizations", "compile_mode", "reduce-overhead"),
    ("optimizations", "dynamic_shapes", True), ("optimizations", "compile_decoder", True),
    ("benchmark", "requests", 2), ("benchmark", "headline", True),
    ("preflight", "atol", 0.01), ("data", "manifest", "old.jsonl"),
])
def test_config_changes_are_rejected_before_execution(declared, section, key, value):
    declared.config[section][key] = value
    declared.config_path.write_text(yaml.safe_dump(declared.config))
    with pytest.raises(ValueError):
        launcher.inspect_config(declared.config_path)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(passed=False),
    lambda p: p["runs"]["upstream"]["model_identity"].pop("parameter_grad_policy"),
    lambda p: p["runs"]["refactored"]["model_identity"].update(execution_grad_mode="inference_mode"),
    lambda p: p["runs"]["refactored"]["model_identity"].update(autocast_cache_enabled=False),
    lambda p: p["runs"]["upstream"]["model_identity"]["parameter_dtypes"]["torch.float32"].update(requires_grad_tensors=0),
    lambda p: p["runs"]["refactored"]["model_identity"]["parameter_dtypes"]["torch.float32"].update(requires_grad_elements=7),
    lambda p: p["runs"]["upstream"].update(denoiser_evaluations=49),
    lambda p: p["checks"]["upstream_agreement"].update(passed=False),
    lambda p: p.update(input_provenance_sha256="changed"),
])
def test_old_or_incomplete_baseline_proofs_are_rejected(declared, mutation):
    mutation(declared.proof)
    declared.proof_path.write_text(json.dumps(declared.proof))
    with pytest.raises(ValueError):
        launcher.baseline_input_evidence(declared.config, declared.base_path)


@pytest.mark.parametrize("which", ["manifest", "metadata", "base_path"])
def test_changed_baseline_inputs_are_rejected(declared, which):
    path = getattr(declared, which)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        launcher.baseline_input_evidence(declared.config, declared.base_path)


@pytest.mark.parametrize("symlink", [False, True])
def test_attempt_destination_never_overwrites(declared, symlink):
    output = Path(declared.config["output"])
    if symlink:
        output.symlink_to(output.with_name("missing"), target_is_directory=True)
    else:
        output.mkdir()
        (output / "preserved.txt").write_text("old attempt")
    with pytest.raises(FileExistsError):
        launcher.main(["--config", str(declared.config_path)])
    if not symlink:
        assert (output / "preserved.txt").read_text() == "old attempt"


def test_termination_preserves_failed_attempt_and_restores_handler(declared, monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(launcher, "baseline_input_evidence", lambda *args: signal.raise_signal(signal.SIGTERM))
    assert launcher.main(["--config", str(declared.config_path)]) == 143
    attempt = json.loads((Path(declared.config["output"]) / "attempt.json").read_text())
    assert attempt["status"] == "failed" and attempt["termination_signal"] == signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.parametrize("graphs,kernel,numerics,expected", [
    (1, True, True, 0), (0, True, True, 1), (1, False, True, 1), (1, True, False, 1),
])
def test_success_requires_strict_agreement_and_positive_codegen(declared, monkeypatch, graphs, kernel, numerics, expected):
    """Mock only the GPU/compiler boundary; exercise the real attempt gates."""
    from drivebench import benchmark, validate
    from drivebench.benchmarks import environment
    torch = ModuleType("torch")
    torch.__version__ = "2.0.1+cu118"
    dynamo = ModuleType("torch._dynamo")
    dynamo.config = SimpleNamespace(disable=False, suppress_errors=True)
    inductor = ModuleType("torch._inductor")
    inductor_config = ModuleType("torch._inductor.config")
    inductor_config.trace = SimpleNamespace(enabled=False)
    inductor.config = inductor_config
    torch._dynamo, torch._inductor = dynamo, inductor
    for name, module in (("torch", torch), ("torch._dynamo", dynamo),
                         ("torch._inductor", inductor), ("torch._inductor.config", inductor_config)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(environment, "collect_environment", lambda: {})
    monkeypatch.setattr(environment, "source_identity", lambda: "fixture-source")
    monkeypatch.setattr(benchmark, "_validate_reference_environment", lambda *args: {"passed": True})
    state = {"graphs": 0}
    monkeypatch.setattr(launcher, "compiler_state", lambda: {"stats": {"unique_graphs": state["graphs"]}})
    def fake_validate(argv):
        assert argv[argv.index("--atol") + 1] == "0" and argv[argv.index("--rtol") + 1] == "0"
        assert dynamo.config.suppress_errors is False and inductor_config.trace.enabled
        assert dynamo.config.debug_dir_root == str(Path(declared.config["output"]) / "compiler-debug")
        state["graphs"] = graphs
        generated = Path(dynamo.config.debug_dir_root) / "fixture/output_code.py"
        generated.parent.mkdir(parents=True)
        generated.write_text("async_compile.triton('fixture')" if kernel else "# CPU-only output fixture")
        report = {"passed": numerics, "atol": 0, "rtol": 0,
                  "request_id": declared.proof["request_id"], "input_provenance": declared.proof["input_provenance"],
                  "runtime_identity": declared.proof["runtime_identity"], "implementation_sha256": "fixture-source"}
        Path(argv[argv.index("--output") + 1]).write_text(json.dumps(report))
        return 0 if numerics else 1
    monkeypatch.setattr(validate, "main", fake_validate)
    assert launcher.main(["--config", str(declared.config_path)]) == expected
    attempt = json.loads((Path(declared.config["output"]) / "attempt.json").read_text())
    assert (attempt["status"] == "passed") is (expected == 0)
    assert attempt["positive_cuda_codegen_evidence"] is (graphs > 0 and kernel)


def test_old_compiled_config_is_not_accepted_with_corrected_default():
    with pytest.raises(ValueError, match="inherited data"):
        launcher.inspect_config(ROOT / "configs/preflight_scene0103_compiled.yaml")
