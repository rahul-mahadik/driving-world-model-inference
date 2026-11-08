"""Shared benchmark artifact reader; outputs must stay inside their run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from ..data.manifests import sha256_file


@dataclass
class Run:
    path: Path
    config: dict[str, Any]
    environment: dict[str, Any]
    measurements: list[dict[str, Any]]
    summary: dict[str, Any]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def synthetic(self) -> bool:
        return self.summary.get("backend", get_nested(self.config, "model.backend", "backend")) not in {"vista", "upstream"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_run(path: str | Path) -> Run:
    path = Path(path).resolve()
    measurements = []
    with (path / "measurements.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected measurement object in {path}")
                measurements.append(row)
    return Run(path, read_json(path / "config.json"), read_json(path / "environment.json"),
               measurements, read_json(path / "summary.json"))


def artifact_sha256(run: Run) -> dict[str, str]:
    """Bind evaluation and threshold evidence to exact benchmark artifacts."""
    return {name: sha256_file(run.path / name) for name in
            ("config.json", "environment.json", "measurements.jsonl", "summary.json")}


def find_runs(path: str | Path) -> list[Run]:
    root = Path(path).resolve()
    paths = [root] if (root / "measurements.jsonl").exists() else sorted(
        p.parent for p in root.glob("*/measurements.jsonl"))
    if not paths:
        raise ValueError(f"No benchmark runs found under {root}")
    return [load_run(p) for p in paths]


def output_path(run: Run, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Planned generated output was not saved")
    path = (run.path / value).resolve()
    if not path.is_relative_to(run.path):
        raise ValueError(f"Output path escapes run directory: {value}")
    if not path.is_file():
        raise ValueError(f"Missing generated output: {path}")
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def quality_rows(run: Run) -> list[dict]:
    saves = get_nested(run.config, "benchmark.save_outputs")
    if saves is None:
        return run.measurements
    return [row for index, row in enumerate(run.measurements) if row.get("request_index", index) < int(saves)]


def candidates(run: Run) -> dict[tuple[str, int, str, int], tuple[dict, dict]]:
    indexed = {}
    for row in quality_rows(run):
        if row.get("status", "ok") not in {"ok", "success"}:
            continue
        for candidate in row.get("candidates", []):
            key = (str(row["request_id"]), int(row.get("repeat", 0)),
                   str(candidate["candidate_id"]), int(candidate["seed"]))
            if key in indexed:
                raise ValueError(f"Duplicate candidate measurement {key} in {run.path}")
            indexed[key] = (row, candidate)
    return indexed


def get_nested(value: dict, *paths: str) -> Any:
    for path in paths:
        node: Any = value
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            return node
    return None


def experiment_identity(run: Run) -> dict[str, Any]:
    """Comparison identity excludes optimization flags but includes workload semantics."""
    cfg, env = run.config, run.environment
    gpu = get_nested(env, "gpu", "cuda.device", "gpus")
    if isinstance(gpu, list):
        gpu = gpu[0] if len(gpu) == 1 else gpu
    if isinstance(gpu, dict):
        gpu = {key: gpu.get(key) for key in ("name", "total_memory", "total_memory_bytes", "memory_total_mib",
                                            "mig_mode", "uuid", "driver_version", "power_limit_w",
                                            "max_sm_clock_mhz", "max_memory_clock_mhz")
               if gpu.get(key) is not None}
    return {
        "backend": run.summary.get("backend", get_nested(cfg, "model.backend", "backend")),
        "gpu": gpu,
        "model_commit": get_nested(cfg, "model.upstream_commit", "model.vista_commit", "upstream_commit")
        or get_nested(env, "upstream_commit", "vista_commit"),
        "checkpoint_sha256": get_nested(cfg, "model.checkpoint_sha256", "checkpoint_sha256")
        or get_nested(env, "checkpoint_sha256"),
        "manifest_sha256": get_nested(cfg, "manifest_sha256", "data.manifest_sha256")
        or get_nested(env, "manifest_sha256"),
        "workload": get_nested(cfg, "workload") or {
            key: get_nested(cfg, f"benchmark.{key}") for key in
            ("workload", "candidate_count", "input_condition", "seed")
            if get_nested(cfg, f"benchmark.{key}") is not None},
        "generation": {key: get_nested(cfg, f"generation.{key}") for key in
                       ("total_frames", "frames", "num_frames", "height", "width", "conditioning_frames", "guidance_scale", "cond_aug")
                       if get_nested(cfg, f"generation.{key}") is not None},
    }
