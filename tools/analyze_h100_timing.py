#!/usr/bin/env python3
"""Reconstruct the historical H100 timing statistics using only Python's stdlib.

This analyzes recorded observations; it performs no inference, GPU work or network
access. The released bootstrap schedule reproduces the original seed-0 draws.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "results/2026-10-30-h100-tradeoff/numerical-audit"
SCHEDULE = ("upstream", "compiled-default", "compiled-default", "upstream",
            "compiled-default", "upstream", "upstream", "compiled-default")
ROW_FIELDS = {
    "block_index", "role", "role_block_index", "request_index", "scene_index",
    "repeat", "seed", "latency_s", "peak_allocated_bytes", "peak_reserved_bytes",
    "denoiser_evaluations", "status", "failure_count", "fallback_count",
    "oom_attempt_count", "source_line_sha256",
}


def quantile(values: list[float], q: float) -> float:
    """NumPy's default linear percentile convention, without importing NumPy."""
    if not values or not 0 <= q <= 1:
        raise ValueError("A nonempty sample and quantile in [0, 1] are required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    difference = ordered[upper] - ordered[lower]
    # Match the two-sided interpolation used by the historical NumPy analysis.
    return (ordered[lower] + difference * weight if weight < .5
            else ordered[upper] - difference * (1 - weight))


def checked_rows(rows: list[dict]) -> dict[str, list[list[dict]]]:
    """Require all eight completed blocks and exact paired scene/seed coverage."""
    if len(rows) != 400:
        raise ValueError("Exactly 400 observed requests are required")
    by_role = {role: [] for role in set(SCHEDULE)}
    reference_cells = None
    for block_index, role in enumerate(SCHEDULE):
        block = rows[block_index * 50:(block_index + 1) * 50]
        role_block = len(by_role[role])
        for request_index, row in enumerate(block):
            if set(row) != ROW_FIELDS:
                raise ValueError("Unexpected or missing observation fields")
            for name in ROW_FIELDS - {"role", "latency_s", "status", "source_line_sha256"}:
                if type(row[name]) is not int:
                    raise ValueError("Observation counters and identities must be integers")
            if (row["block_index"] != block_index or row["role"] != role
                    or row["role_block_index"] != role_block
                    or row["request_index"] != request_index):
                raise ValueError("Original block and within-block order must be preserved")
            if (row["status"] != "ok" or row["repeat"] != 0
                    or row["denoiser_evaluations"] != 50
                    or any(row[name] != 0 for name in
                           ("failure_count", "fallback_count", "oom_attempt_count"))):
                raise ValueError("This complete historical cohort has no admitted failures")
            value = row["latency_s"]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Latency must be finite and positive")
            if (row["peak_allocated_bytes"] <= 0
                    or row["peak_reserved_bytes"] < row["peak_allocated_bytes"]
                    or row["seed"] < 0):
                raise ValueError("Invalid memory or seed measurement")
            digest = row["source_line_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(
                    c not in "0123456789abcdef" for c in digest):
                raise ValueError("Missing original-line identity")
        if {row["scene_index"] for row in block} != set(range(50)):
            raise ValueError("Every block must contain all 50 independent scene groups")
        cells = [(row["scene_index"], row["seed"]) for row in block]
        if reference_cells is not None and cells != reference_cells:
            raise ValueError("Every role and repeated block must use the same scenes and seeds")
        reference_cells = cells
        by_role[role].append(block)
    return by_role


def checked_draws(text: str) -> list[list[int]]:
    draws = [[int(x) for x in line.split(",")] for line in text.splitlines()]
    if len(draws) != 4000 or any(len(d) != 50 or any(x not in range(50) for x in d)
                                 for d in draws):
        raise ValueError("The historical bootstrap requires 4000 draws of 50 scene groups")
    return draws


def reconstruct(rows: list[dict], draws: list[list[int]]) -> dict:
    by_role = checked_rows(rows)
    blocks = {}
    latencies = {}
    throughput = {}
    memory = {}
    clustered = {}
    for role in ("upstream", "compiled-default"):
        role_blocks = by_role[role]
        flat = [row for block in role_blocks for row in block]
        values = [row["latency_s"] for row in flat]
        blocks[role] = [quantile([r["latency_s"] for r in b], .5) for b in role_blocks]
        latencies[role] = {"p50": quantile(values, .5), "p95": quantile(values, .95)}
        # Historical aggregation sums each block first, then the four block sums.
        wall = sum(sum(row["latency_s"] for row in block) for block in role_blocks)
        throughput[role] = {
            "measurement_wall_s": wall, "completed_futures": len(flat),
            "futures_per_s": len(flat) / wall,
            "generated_frames_per_s": len(flat) * 24 / wall,
            "warmup_initialization_and_compilation_excluded": True,
        }
        memory[role] = {name: max(row[name] for row in flat)
                        for name in ("peak_allocated_bytes", "peak_reserved_bytes")}
        clustered[role] = [[row["latency_s"] for row in flat if row["scene_index"] == i]
                           for i in range(50)]
    metrics = {}
    for label, q in (("p50", .5), ("p95", .95)):
        change = latencies["compiled-default"][label] / latencies["upstream"][label] - 1
        estimates = []
        for selected in draws:
            a = [x for i in selected for x in clustered["upstream"][i]]
            b = [x for i in selected for x in clustered["compiled-default"][i]]
            estimates.append(quantile(b, q) / quantile(a, q) - 1)
        interval = [quantile(estimates, .025), quantile(estimates, .975)]
        metrics[label] = {
            "relative_latency_change": change, "ci95": interval,
            "speed_ratio_reference_over_compiled": 1 / (1 + change),
            "speed_ratio_ci95": [1 / (1 + interval[1]), 1 / (1 + interval[0])],
        }
    return {
        "paired_requests": 200, "independent_scenes": 50,
        "metrics": metrics, "latency_s": latencies, "block_medians_s": blocks,
        "observed_block_median_range_s": {k: [min(v), max(v)] for k, v in blocks.items()},
        "throughput": throughput, "measured_cuda_memory": memory,
        "qualified": False, "quality_preserving": False,
        "combined_headline_claim": False, "performance_regression_policy_calibrated": False,
    }


def compare_recorded(actual, expected, path="comparison") -> int:
    """Verify every reconstructed field; allow only floating-point roundoff."""
    if isinstance(actual, dict):
        if not isinstance(expected, dict):
            raise ValueError(f"Wrong recorded type: {path}")
        return sum(compare_recorded(v, expected[k], path + "." + k)
                   for k, v in actual.items())
    if isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError(f"Wrong recorded length: {path}")
        return sum(compare_recorded(a, b, path + f"[{i}]")
                   for i, (a, b) in enumerate(zip(actual, expected)))
    if type(actual) is float:
        if type(expected) not in (int, float) or not math.isfinite(expected) or not math.isclose(
                actual, expected, rel_tol=1e-13, abs_tol=1e-13):
            raise ValueError(f"Recorded statistic differs: {path}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Recorded value differs: {path}")
    return 1


def analyze(directory: Path = DEFAULT_DATA) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    required_files = {"measurements.jsonl", "bootstrap-scene-indices.csv", "provenance.json"}
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != required_files:
        raise ValueError("Audit manifest must bind every numerical and provenance input")
    for name, record in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Audit manifest entries must be immediate files")
        value = (directory / name).read_bytes()
        if len(value) != record["bytes"] or hashlib.sha256(value).hexdigest() != record["sha256"]:
            raise ValueError("Audit file checksum differs: " + name)
    summary_bytes = (directory.parent / "summary.json").read_bytes()
    if hashlib.sha256(summary_bytes).hexdigest() != manifest["historical_summary_sha256"]:
        raise ValueError("Historical summary identity differs")
    rows = [json.loads(line) for line in (directory / "measurements.jsonl").read_text().splitlines()]
    draws = checked_draws((directory / "bootstrap-scene-indices.csv").read_text())
    result = reconstruct(rows, draws)
    checked = compare_recorded(result, json.loads(summary_bytes)["comparison"])
    return {
        "schema_version": 1,
        "status": "historical_numerical_statistics_reproduced",
        "observations": len(rows), "bootstrap_resamples": len(draws),
        "recorded_fields_checked": checked,
        "comparison": result,
        "scope": "Public numerical reconstruction only; no GPU rerun, raw-output verification or quality validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, help="Write JSON to a new file; existing files are refused")
    args = parser.parse_args()
    encoded = json.dumps(analyze(args.data), indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
