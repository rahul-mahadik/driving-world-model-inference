"""CPU stand-ins verify lifecycle and proof binding, never compiled GPU results."""
import copy
from dataclasses import replace
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from drivebench.config import validate_run_config
from drivebench.runtime import compiler
from drivebench.types import OptimizationConfig


def dependencies():
    allowed, disallowed = set(), set()
    functions = {name: (lambda: None) for name in compiler.FUNCTIONS}
    def native(*args):
        return args
    wrappers = []
    def disable(fn):
        def wrapper(*args):
            return fn(*args)
        wrapper._torchdynamo_orig_callable = fn
        wrappers.append(wrapper)
        return wrapper
    def registration():
        for fn in functions.values():
            allowed.add(id(fn))
            disallowed.discard(id(fn))
    dynamo = SimpleNamespace(disable=disable,
        config=SimpleNamespace(cache_size_limit=64, disable=False, suppress_errors=False),
        allowed_functions=SimpleNamespace(_allowed_function_ids=allowed, _disallowed_function_ids=disallowed))
    return SimpleNamespace(dynamo=dynamo, native=native, ops=SimpleNamespace(memory_efficient_attention=native),
        registration=registration, functions=functions, counters={"stats": {"unique_graphs": 0}},
        metrics=SimpleNamespace(generated_kernel_count=0), source_hashes={"fixture": "a" * 64}, wrappers=wrappers)


def assert_restored(d, runtime):
    assert d.ops.memory_efficient_attention is d.native
    assert d.dynamo.config.cache_size_limit == 64
    assert d.dynamo.config.suppress_errors is False and d.dynamo.config.disable is False
    assert not d.dynamo.allowed_functions._allowed_function_ids
    assert not d.dynamo.allowed_functions._disallowed_function_ids
    assert runtime.last_execution["restoration_complete"] is True


def test_disabled_runtime_is_lazy_and_preserves_serialized_optimization_fields(monkeypatch):
    options = OptimizationConfig()
    before = options.to_dict()
    monkeypatch.setattr(compiler, "_load_runtime", lambda *_: pytest.fail("Disabled runtime imported GPU dependencies"))
    assert compiler.create_compiler_runtime({}, options, None) is None
    assert compiler.validate_compiler_runtime({"backend": "upstream"}, options) is None
    assert options.to_dict() == before and "compiler_runtime" not in before
    subprocess.run([sys.executable, "-c", "import drivebench.runtime.compiler; import sys; assert 'torch' not in sys.modules"], check=True)


@pytest.mark.parametrize("model,options", [
    ({"compiler_runtime": "typo"}, OptimizationConfig(compile_denoiser=True)),
    ({"compiler_runtime": compiler.RUNTIME_NAME, "backend": "upstream"}, OptimizationConfig(compile_denoiser=True)),
    ({"compiler_runtime": compiler.RUNTIME_NAME, "backend": "synthetic"}, OptimizationConfig(compile_denoiser=True)),
    ({"compiler_runtime": compiler.RUNTIME_NAME}, OptimizationConfig()),
    ({"compiler_runtime": compiler.RUNTIME_NAME}, OptimizationConfig(compile_denoiser=True, fullgraph=True)),
    ({"compiler_runtime": compiler.RUNTIME_NAME}, OptimizationConfig(compile_denoiser=True, compile_decoder=True)),
    ({"compiler_runtime": compiler.RUNTIME_NAME}, OptimizationConfig(compile_denoiser=True, compile_mode="reduce-overhead")),
    ({"compiler_runtime": compiler.RUNTIME_NAME}, OptimizationConfig(compile_denoiser=True, feature_cache_interval=2, feature_cache_module="output_blocks.9")),
])
def test_unsupported_runtime_declarations_fail_before_gpu_import(model, options):
    with pytest.raises(ValueError):
        validate_run_config({"model": model, "optimizations": options.to_dict()})


def test_output9_boundary_requires_explicit_single_candidate_selector_and_changes_identity():
    model = {"compiler_runtime": compiler.RUNTIME_NAME, "compiler_feature_boundary": "output_blocks.9"}
    options = OptimizationConfig(compile_denoiser=True, feature_cache_module="output_blocks.9")
    for interval in (1, 2, 3, 4):
        assert compiler.validate_compiler_runtime(model, replace(options, feature_cache_interval=interval)) == compiler.RUNTIME_NAME
    for invalid_model, invalid_options in (({**model, "compiler_runtime": None}, options),
            ({**model, "compiler_feature_boundary": "middle_block"}, options),
            (model, replace(options, max_batch_size=2)), (model, replace(options, feature_cache_module=None))):
        with pytest.raises(ValueError):
            compiler.validate_compiler_runtime(invalid_model, invalid_options)
    normal = compiler.RegisteredCompilerRuntime(dependencies()).identity
    boundary = compiler.RegisteredCompilerRuntime(dependencies(), feature_boundary="output_blocks.9").identity
    assert normal["implementation_sha256"] != boundary["implementation_sha256"]
    dynamic = compiler.RegisteredCompilerRuntime(dependencies(), dynamic_shapes=True).identity
    assert dynamic["implementation_sha256"] != normal["implementation_sha256"]
    assert compiler.validate_compiler_runtime({"compiler_runtime": compiler.RUNTIME_NAME},
        OptimizationConfig(compile_denoiser=True, dynamic_shapes=True)) == compiler.RUNTIME_NAME


def test_stable_wrapper_repeated_scopes_and_no_upstream_leak():
    d = dependencies(); runtime = compiler.RegisteredCompilerRuntime(d)
    identity = runtime.identity
    with runtime.activate():
        first = d.ops.memory_efficient_attention
        assert first is not d.native and first(1, 2) == (1, 2)
        assert d.dynamo.config.cache_size_limit == 512
    assert_restored(d, runtime)
    with runtime.activate(require_graphs=True):
        assert d.ops.memory_efficient_attention is first
        d.counters["stats"]["unique_graphs"] = 3
        d.metrics.generated_kernel_count = 4
    assert_restored(d, runtime)
    with runtime.activate(require_graphs=True):
        assert d.ops.memory_efficient_attention is first
    assert_restored(d, runtime)
    assert len(d.wrappers) == 1 and runtime.identity == identity
    identity["source_sha256"]["fixture"] = "mutated"
    assert runtime.identity["source_sha256"]["fixture"] == "a" * 64
    assert runtime.last_execution["graphs_created_by_runtime"] == 3


def test_oom_restores_globals_and_retains_compilation_evidence_for_retry():
    d = dependencies(); runtime = compiler.RegisteredCompilerRuntime(d)
    with pytest.raises(MemoryError):
        with runtime.activate(require_graphs=True):
            d.counters["stats"]["unique_graphs"] = 1
            d.metrics.generated_kernel_count = 1
            raise MemoryError("CPU stand-in for OOM")
    assert_restored(d, runtime)
    assert runtime.last_execution["passed"] is False
    with runtime.activate(require_graphs=True):
        pass
    assert runtime.last_execution["passed"] is True


@pytest.mark.parametrize("failure", ["partial_registration", "silent_fallback", "settings_change", "no_codegen", "nested_scope"])
def test_failure_paths_restore_state_without_silent_fallback(failure):
    d = dependencies(); runtime = compiler.RegisteredCompilerRuntime(d)
    if failure == "partial_registration":
        def fail():
            d.dynamo.allowed_functions._allowed_function_ids.add(id(next(iter(d.functions.values()))))
            raise RuntimeError("registration failed")
        d.registration = fail
    with pytest.raises(RuntimeError):
        with runtime.activate(require_graphs=True):
            if failure == "silent_fallback":
                d.counters["unimplemented"] = {"unsupported frame": 1}
            elif failure == "settings_change":
                d.dynamo.config.suppress_errors = True
            elif failure == "nested_scope":
                with runtime.activate():
                    pass
    assert_restored(d, runtime)
    assert runtime.last_execution["passed"] is False


def test_rejects_preexisting_registration_and_disabled_compiler():
    for change in (lambda d: setattr(d.dynamo.config, "disable", True),
                   lambda d: setattr(d.dynamo.config, "cache_size_limit", 512),
                   lambda d: d.dynamo.allowed_functions._allowed_function_ids.add(id(next(iter(d.functions.values()))))):
        d = dependencies(); runtime = compiler.RegisteredCompilerRuntime(d)
        change(d)
        with pytest.raises(RuntimeError, match="clean pinned"):
            with runtime.activate():
                pytest.fail("Leaked runtime state was accepted")


def test_counter_collection_error_cannot_prevent_global_restoration():
    d = dependencies(); runtime = compiler.RegisteredCompilerRuntime(d)
    with pytest.raises(RuntimeError, match="counter evidence"):
        with runtime.activate(require_graphs=True):
            del d.metrics.generated_kernel_count
    assert_restored(d, runtime)
    assert runtime.last_execution["passed"] is False


def test_numerical_proof_binds_compiler_runtime_and_reference_stays_clean(tmp_path):
    from test_benchmark import _numerical_fixture
    from drivebench.benchmarks.agreement import numerical_evidence
    config, generation, options, env, identity, records, proof, path = _numerical_fixture(tmp_path)
    options = replace(options, compile_denoiser=True)
    runtime_identity = compiler.RegisteredCompilerRuntime(dependencies()).identity
    config["model"]["compiler_runtime"] = compiler.RUNTIME_NAME
    identity = {**identity, "compiler_runtime": runtime_identity}
    for name in ("refactored", "isolation"):
        proof["runs"][name] = copy.deepcopy(proof["runs"][name])
        proof["runs"][name]["effective_config"]["optimizations"] = options.to_dict()
        proof["runs"][name]["model_identity"]["compiler_runtime"] = runtime_identity
    path.write_text(json.dumps(proof))
    assert numerical_evidence(config, generation, options, env, identity, records)["passed"]
    proof["runs"]["isolation"]["model_identity"].pop("compiler_runtime")
    path.write_text(json.dumps(proof))
    gate = numerical_evidence(config, generation, options, env, identity, records)
    assert not gate["passed"] and any("compiler runtime" in r for r in gate["reasons"])
    proof["runs"]["isolation"]["model_identity"]["compiler_runtime"] = runtime_identity
    proof["runs"]["upstream"]["model_identity"]["compiler_runtime"] = runtime_identity
    path.write_text(json.dumps(proof))
    gate = numerical_evidence(config, generation, options, env, identity, records)
    assert not gate["passed"] and any("Upstream numerical reference" in r for r in gate["reasons"])


def test_validator_strips_runtime_only_from_upstream_reference(monkeypatch):
    import drivebench.validate as validation
    seen = []
    class Stop(Exception):
        pass
    def engine(model, *_):
        seen.append(model)
        raise Stop
    monkeypatch.setattr(validation, "InferenceEngine", engine)
    config = {"model": {"compiler_runtime": compiler.RUNTIME_NAME, "compiler_feature_boundary": "output_blocks.9", "checkpoint": "fixture"}}
    for backend in ("upstream", "vista"):
        with pytest.raises(Stop):
            validation._run(config, None, None, None, None, backend=backend)
    assert "compiler_runtime" not in seen[0]
    assert "compiler_feature_boundary" not in seen[0]
    assert seen[1]["compiler_runtime"] == compiler.RUNTIME_NAME
    assert seen[1]["compiler_feature_boundary"] == "output_blocks.9"
    assert config["model"]["compiler_runtime"] == compiler.RUNTIME_NAME


@pytest.mark.parametrize("selector", ["compiler_runtime", "compiler_feature_boundary"])
def test_timing_blocks_cannot_omit_runtime_selector(selector, tmp_path):
    from test_study import timing_fixtures
    from drivebench.study import aggregate_timing_blocks
    paths = timing_fixtures(tmp_path, "runtime", 1.)
    config_path = paths[1] / "config.json"
    config = json.loads(config_path.read_text())
    config["model"][selector] = compiler.RUNTIME_NAME if selector == "compiler_runtime" else "output_blocks.9"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="identical workload"):
        aggregate_timing_blocks(paths, tmp_path / "aggregate")


def test_legacy_baseline_calibration_binds_runtime_without_forbidding_variant_comparison(tmp_path):
    from test_study import timing_fixtures
    from drivebench.calibrate_performance import calibration_identity
    from drivebench.evaluation.io import load_run
    from drivebench.regression import _comparison_identity
    baseline = load_run(timing_fixtures(tmp_path, "runtime", 1.)[0])
    candidate = copy.deepcopy(baseline)
    candidate.config["model"]["compiler_runtime"] = compiler.RUNTIME_NAME
    assert _comparison_identity(candidate) == _comparison_identity(baseline)
    assert calibration_identity(candidate) != calibration_identity(baseline)
