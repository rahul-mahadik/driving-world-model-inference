"""Check an unmaterialized study proposal without running generation or planning."""
from pathlib import Path
import json

import yaml

from drivebench.config import file_sha256, load_config, merge, validate_run_config
from drivebench.study import _variant_config


def test_registered_draft_preserves_bootstrap_adoption_and_ablation_contracts():
    specification = yaml.safe_load(Path("configs/full_study_registered_draft.yaml").read_text())
    assert specification["draft_status"]["materialized"] is False
    assert specification["optimized_reference"] == "compiled-conditioning"
    assert specification["draft_status"]["fallback_optimized_reference"] == "conditioning-reuse"
    assert len(specification["variants"]) == 17
    base = load_config(specification["base_config"])
    plan = {**specification, "base_config": base}
    for name, filename in (
        ("compiled-default", "configs/study_bootstrap_compiled_default.yaml"),
        ("compiled-conditioning", "configs/study_bootstrap_compiled_conditioning.yaml"),
    ):
        generated = _variant_config(plan, name, base["data"]["manifest"], 32, 1)
        bootstrap = load_config(filename)
        for key in ("data", "model", "generation", "benchmark"):
            assert generated[key] == bootstrap[key]
        assert validate_run_config(generated) == validate_run_config(bootstrap)
    for name, changes in specification["variants"].items():
        config = merge(base, changes)
        generation, options = validate_run_config(config)
        assert options.max_batch_size == 1
        assert generation.total_frames == 25 and generation.height == 576 and generation.width == 1024
        if name.startswith("steps-"):
            assert generation.steps == int(name.removeprefix("steps-"))
            assert options.compile_denoiser and options.conditioning_cache_bytes == 2147483648
        if name.startswith("feature-") or name == "compiled-boundary-1":
            assert options.feature_cache_module == "output_blocks.9"
            assert config["model"]["compiler_feature_boundary"] == "output_blocks.9"
            assert options.compile_denoiser
        if name.startswith("eager-"):
            assert not options.compile_denoiser
            assert options.conditioning_cache_bytes == 2147483648
    assert specification["quality"]["shard_clips"] == 32
    assert specification["quality"]["candidate_count"] == 1
    assert specification["quality"]["action_subset_clips"] == 16
    assert specification["quality"]["action_candidate_count"] == 2
    assert specification["timing"] == {"blocks": 4, "requests_per_block": 50, "candidate_count": 4, "warmups": 3}


def test_budget_receipt_is_bound_and_counts_both_workloads_and_pilots():
    report = json.loads(Path("reports/registered-study-budget-draft.json").read_text())
    assert report["status"] == "planning_only_not_execution_or_a_cost_guarantee"
    assert report["specification_sha256"] == file_sha256(report["specification"])
    phases = report["phase_estimates"]
    assert phases["development_all_17"]["futures"] == 32 * 17
    assert phases["single_timing_2x200"]["futures"] == 400
    assert phases["candidate4_timing_2x200"]["futures"] == 1600
    assert phases["single_pilot_4x50"]["futures"] == 200
    assert phases["candidate4_pilot_4x50"]["futures"] == 800
    assert sum(row["generation_s"] for row in phases.values()) / 3600 == report["accounting"]["generation_hours"]
    assert report["candidate4_same_gpu_window_hours_before_evaluation_or_failures"] > 16
