"""Runtime-boundary orchestration fixtures. No real Torch, compilation, or GPU."""
from __future__ import annotations

import functools
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from drivebench.config import file_sha256
from scripts import validate_compiled_attention_boundary as boundary
from test_compiled_preflight import declared


def test_codegen_requires_new_source_and_cuda_artifact(tmp_path):
    source = tmp_path / "cache/module.py.debug/output_code.py"
    source.parent.mkdir(parents=True)
    source.write_text("async_compile.triton('fixture')")
    records, before, binaries = boundary.generated_records([tmp_path], {})
    assert not binaries and not records[0]["contains_triton_kernel"]
    (tmp_path / "kernel.ptx").write_text("fixture PTX bytes, not executable")
    records, _, binaries = boundary.generated_records([tmp_path], {})
    assert binaries and records[0]["contains_triton_kernel"]
    records, _, _ = boundary.generated_records([tmp_path], boundary.cache_snapshot([tmp_path]))
    assert records == []


def test_cache_symlinks_rejected(tmp_path):
    (tmp_path / "alias").symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="symlinks"):
        boundary.cache_snapshot([tmp_path])


def test_prior_archive_must_match_published_hash(tmp_path):
    p = tmp_path / "prior.npz"
    p.write_bytes(b"old outputs")
    prior = {"runs": {"upstream": {"outputs": {"normal": str(p)},
                                 "outputs_sha256": {"normal": file_sha256(p)}}}}
    assert boundary.verify_prior_outputs(prior)[0] == p
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        boundary.verify_prior_outputs(prior)


@pytest.mark.parametrize("changed_upstream,fail_compiler", [(False, False), (True, False), (False, True)])
def test_boundary_binding_reference_prerequisite_and_restoration(declared, monkeypatch, changed_upstream, fail_compiler):
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
    def disable(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            return fn(*args, **kwargs)
        wrapped._torchdynamo_orig_callable = fn
        return wrapped
    dynamo.disable = disable
    xformers, ops = ModuleType("xformers"), ModuleType("xformers.ops")
    xformers.ops = ops
    def memory_efficient_attention():
        return "unchanged native fixture"
    memory_efficient_attention.__module__ = "xformers.ops.fmha"
    ops.memory_efficient_attention = memory_efficient_attention
    for name, module in (("torch", torch), ("torch._dynamo", dynamo), ("torch._inductor", inductor),
                         ("torch._inductor.config", inductor_config), ("xformers", xformers), ("xformers.ops", ops)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(boundary.inspect, "getsourcefile", lambda fn: __file__)
    monkeypatch.setattr(boundary, "NATIVE_SOURCE_SHA256", file_sha256(__file__))
    monkeypatch.setattr(environment, "collect_environment", lambda: {})
    monkeypatch.setattr(environment, "source_identity", lambda: "fixture-source")
    monkeypatch.setattr(benchmark, "_validate_reference_environment", lambda *args: {"passed": True})
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "prior-cache-must-be-restored")
    monkeypatch.setenv("TORCH_COMPILE_DEBUG", "0")
    declared.config["runtime_patch"] = boundary.patch_declaration()
    declared.config_path.write_text(yaml.safe_dump(declared.config))
    prior_array = declared.proof_path.with_suffix(".npz")
    np.savez(prior_array, candidate=np.ones(4, np.float32))
    declared.proof["runs"]["upstream"].update(outputs={"normal": str(prior_array)},
        outputs_sha256={"normal": file_sha256(prior_array)})
    declared.proof_path.write_text(json.dumps(declared.proof))
    state = {"graphs": 0, "compiled_reached": False}
    monkeypatch.setattr(boundary.base, "compiler_state", lambda: {"stats": {"unique_graphs": state["graphs"]}})
    def fake_run(*args, backend, **kwargs):
        assert ops.memory_efficient_attention is not memory_efficient_attention
        assert ops.memory_efficient_attention() == "unchanged native fixture"
        if backend != "upstream":
            state["compiled_reached"] = True
            if fail_compiler:
                raise RuntimeError("deliberate compiler failure")
            state["graphs"] = 2
            import os
            cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
            cache.mkdir(parents=True)
            (cache / "generated.py").write_text("async_compile.triton('fixture')")
            triton = Path(os.environ["TRITON_CACHE_DIR"])
            triton.mkdir(parents=True)
            (triton / "kernel.ptx").write_text("fixture PTX, no GPU execution")
        values = np.zeros(4, np.float32) if backend == "upstream" and changed_upstream else np.ones(4, np.float32)
        return {"normal": {"candidate": values}}, {"model_identity": {"backend": backend}}
    monkeypatch.setattr(validate, "_run", fake_run)
    def fake_validation(argv):
        assert dynamo.config.suppress_errors is False
        attempt = json.loads((Path(declared.config["output"]) / "attempt.json").read_text())
        assert attempt["runtime_patch"] == declared.config["runtime_patch"]
        _, upstream = validate._run(backend="upstream")
        _, refactored = validate._run(backend="vista")
        report = {"passed": True, "atol": 0, "rtol": 0, "request_id": declared.proof["request_id"],
                  "input_provenance": declared.proof["input_provenance"], "implementation_sha256": "fixture-source",
                  "runtime_identity": declared.proof["runtime_identity"],
                  "runs": {"upstream": upstream, "refactored": refactored}}
        Path(argv[argv.index("--output") + 1]).write_text(json.dumps(report))
        return 0
    monkeypatch.setattr(validate, "main", fake_validation)
    code = boundary.main(["--config", str(declared.config_path), "--base-config", str(declared.base_path)])
    assert (code == 0) is (not changed_upstream and not fail_compiler)
    assert state["compiled_reached"] is (not changed_upstream)
    assert ops.memory_efficient_attention is memory_efficient_attention
    assert validate._run is fake_run and validate.main is fake_validation
    import os
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "prior-cache-must-be-restored"
    attempt = json.loads((Path(declared.config["output"]) / "attempt.json").read_text())
    assert attempt["runtime_patch_receipt_sha256"] == file_sha256(Path(declared.config["output"]) / "runtime-patch.json")
    if code == 0:
        report = json.loads(Path(declared.config["preflight"]["validation_output"]).read_text())
        assert report["runtime_patch"] == declared.config["runtime_patch"]
        assert report["patched_upstream_matches_corrected_prior"]["passed"]
        assert report["runtime_implementation_sha256"] == report["runs"]["refactored"]["model_identity"]["runtime_implementation_sha256"]


def test_changed_launcher_declaration_rejected_before_attempt(declared):
    declared.config["runtime_patch"] = {**boundary.patch_declaration(), "launcher_sha256": "stale"}
    declared.config_path.write_text(yaml.safe_dump(declared.config))
    with pytest.raises(ValueError, match="declaration"):
        boundary.main(["--config", str(declared.config_path), "--base-config", str(declared.base_path)])
    assert not Path(declared.config["output"]).exists()
