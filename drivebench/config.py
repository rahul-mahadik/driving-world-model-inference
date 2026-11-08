from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from .types import GenerationConfig, OptimizationConfig


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path) -> dict:
    """Load YAML, with bounded recursive inheritance; paths resolve from project root."""
    def read(p, stack):
        p = Path(p).resolve()
        if p in stack:
            raise ValueError(f"configuration inheritance cycle at {p}")
        with p.open() as f:
            value = yaml.safe_load(f)
        if not isinstance(value, dict):
            raise ValueError("configuration must be a mapping")
        base = value.pop("extends", None)
        return merge(read(p.parent / base, stack + [p]), value) if base else value
    return read(path, [])


def merge(base, update):
    out = dict(base)
    for key, value in update.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def validate_run_config(config):
    generation = GenerationConfig(**config.get("generation", {}))
    optimizations = OptimizationConfig(**config.get("optimizations", {}))
    model = config.get("model", {})
    if model.get("backend", "vista") not in {"vista", "upstream", "synthetic"}:
        raise ValueError("unknown model backend")
    from .runtime.compiler import validate_compiler_runtime
    validate_compiler_runtime(model, optimizations)
    benchmark = config.get("benchmark", {})
    if benchmark.get("candidate_count", 1) < 1:
        raise ValueError("candidate_count must be positive")
    if benchmark.get("requests", 200) < 1 or benchmark.get("warmup_requests", 3) < 0:
        raise ValueError("invalid measurement or warmup count")
    if benchmark.get("input_condition", "uncached") not in {"uncached", "warm_filesystem", "reused_conditioning"}:
        raise ValueError("unknown input_condition")
    if benchmark.get("headline", False) and benchmark.get("requests", 200) < 200:
        raise ValueError("headline results require at least 200 measured requests")
    if benchmark.get("headline", False) and model.get("backend") == "synthetic":
        raise ValueError("synthetic outputs cannot be headline results")
    if benchmark.get("save_outputs", 0) < 0:
        raise ValueError("save_outputs must be nonnegative")
    if benchmark.get("warmup_requests", 3) > benchmark.get("max_warmup_requests", max(10, benchmark.get("warmup_requests", 3))):
        raise ValueError("max_warmup_requests must accommodate warmup_requests")
    return generation, optimizations
