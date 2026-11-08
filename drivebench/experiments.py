"""Materialize independent experiment flags and balanced finalist run order."""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import yaml

from .config import load_config, merge, validate_run_config


def experiment_matrix(base):
    experiments = {}
    def add(name, changes):
        cfg = merge(base, changes)
        cfg["name"] = name
        cfg["output"] = f"outputs/experiments/{name}"
        validate_run_config(cfg)
        experiments[name] = cfg
    add("eager", {})
    add("compile-denoiser", {"optimizations": {"compile_denoiser": True}})
    add("compile-decoder", {"optimizations": {"compile_decoder": True}})
    add("compile-cudagraphs", {"optimizations": {"compile_denoiser": True, "compile_mode": "reduce-overhead"}})
    add("compile-fullgraph-audit", {"optimizations": {"compile_denoiser": True, "fullgraph": True}})
    add("compile-dynamic", {"optimizations": {"compile_denoiser": True, "dynamic_shapes": True}})
    add("bf16", {"generation": {"dtype": "bf16"}})
    for size in (1, 7, 14, 25):
        add(f"decode-chunk-{size}", {"generation": {"decode_chunk_size": size}})
    for steps in (50, 40, 30, 20):
        add(f"steps-{steps}", {"generation": {"steps": steps}})
    for interval in (2, 3, 4):
        add(f"feature-reuse-{interval}", {"optimizations": {"compile_denoiser": False, "feature_cache_interval": interval,
                                                           "feature_cache_module": "middle_block"}})
    for candidates in (1, 4, 8, 16):
        add(f"candidates-{candidates}", {"benchmark": {"workload": "candidate_set", "candidate_count": candidates}})
    for batch in (1, 2, 4, 8, 16):
        add(f"memory-batch-{batch}", {"benchmark": {"workload": "memory_scaling", "candidate_count": 16},
                                         "optimizations": {"max_batch_size": batch}})
    for condition in ("uncached", "warm_filesystem", "reused_conditioning"):
        add(f"input-{condition}", {"benchmark": {"input_condition": condition}, "optimizations": {
            "conditioning_cache_bytes": 2147483648 if condition == "reused_conditioning" else 0}})
    add("independent-prefetch", {"benchmark": {"workload": "independent_scenes", "save_outputs": 0},
                                  "optimizations": {"prefetch_depth": 2, "pinned_memory": True, "async_transfer": True}})
    return experiments


def balanced_order(names, repeats=4, seed=23):
    """Pair rounds in reverse order (AB/BA) with seeded permutation between pairs."""
    if repeats < 2 or repeats % 2:
        raise ValueError("use an even repeat count >=2 for balanced interleaving")
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("provide at least two distinct finalist configurations")
    rng = random.Random(seed)
    order = []
    for _ in range(repeats // 2):
        current = list(names)
        rng.shuffle(current)
        order.extend(current)
        order.extend(reversed(current))
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output", default="outputs/experiment-plan")
    parser.add_argument("--finalists", nargs="+", help="config files for balanced interleaved comparisons")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="execute each finalist in an isolated process")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    if args.finalists:
        order = balanced_order(args.finalists, args.repeats)
        plan = [{"index": i, "config": name, "output": str(destination / f"{i:03d}-{Path(name).stem}")}
                for i, name in enumerate(order)]
        (destination / "schedule.json").write_text(json.dumps(plan, indent=2))
        if args.execute:
            for row in plan:
                subprocess.run([sys.executable, "-m", "drivebench.benchmark", "--config", row["config"], "--output", row["output"]], check=True)
    else:
        for name, cfg in experiment_matrix(load_config(args.config)).items():
            (destination / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        if args.execute:
            parser.error("select finalists explicitly before executing; the matrix can be expensive")
    print(destination)


if __name__ == "__main__":
    main()
