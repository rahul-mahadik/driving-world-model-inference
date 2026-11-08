"""Validate the pre-plan development configuration contract without GPU execution."""
from dataclasses import replace

from drivebench.config import load_config, validate_run_config
from drivebench.runtime.compiler import RUNTIME_NAME
from drivebench.types import GenerationConfig


def test_compiled_bootstrap_configs_preserve_upstream_generation_and_all_development_outputs():
    upstream = load_config("configs/study_upstream_development.yaml")
    reference_generation, _ = validate_run_config(upstream)
    configs = [load_config("configs/study_bootstrap_compiled_default.yaml"),
               load_config("configs/study_bootstrap_compiled_conditioning.yaml")]
    options = []
    for config in configs:
        generation, optimization = validate_run_config(config)
        options.append(optimization)
        assert generation == reference_generation == GenerationConfig()
        assert config["data"] == upstream["data"]
        assert config["data"]["verify_checksums"] is True
        assert config["model"]["backend"] == "vista"
        assert config["model"]["compiler_runtime"] == RUNTIME_NAME
        assert config["model"]["compiler_feature_boundary"] is None
        assert config["benchmark"] == {"workload": "single_future", "candidate_count": 1, "requests": 32,
            "warmup_requests": 3, "max_warmup_requests": 5,
            "input_condition": "reused_conditioning" if optimization.conditioning_cache_bytes else "uncached",
            "save_outputs": 32, "headline": False}
        assert config["numerical_validation"] is None
        assert config["environment_lock"] == upstream["environment_lock"]
        assert config["environment_verification"] == upstream["environment_verification"]
        assert optimization.compile_denoiser and not optimization.compile_decoder
        assert not optimization.dynamic_shapes and not optimization.fullgraph
        assert optimization.compile_mode == "default" and optimization.max_batch_size == 1
    assert options[1] == replace(options[0], conditioning_cache_bytes=2147483648)
    assert configs[0]["output"] != configs[1]["output"]
    tolerances = load_config("configs/study_bootstrap_unfrozen_tolerances.yaml")
    assert tolerances["metrics"] == {} and not tolerances["calibrated"] and not tolerances["frozen"]
