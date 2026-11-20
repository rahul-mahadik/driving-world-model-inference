#!/usr/bin/env python3
"""Check released numerical consistency only; no GPU or withheld-data validation."""
import json
import math
from pathlib import Path


def verify(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    rows = [json.loads(line) for line in (directory / "measurements.jsonl").read_text().splitlines() if line.strip()]
    cold = json.loads((directory / "cold-start.json").read_text())
    warmups = cold["warmups"]
    if len(rows) != 1 or len(warmups) != 3:
        raise ValueError("Exactly one measured row and three distinct warmup rows are required")
    row = rows[0]
    expected = {
        "measured_requests": len(rows), "warmup_requests": len(warmups),
        "latency_s": row["latency_s"], "initialization_s": cold["model_load_s"],
        "warmup_latencies_s": [item["latency_s"] for item in warmups],
        "peak_allocated_bytes": row["peak_allocated_bytes"],
        "peak_reserved_bytes": row["peak_reserved_bytes"],
        "warmup_peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in warmups),
        "warmup_peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in warmups),
        "stage_timings_s": row["stage_timings_s"],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError("Published summary differs from rows: " + key)
    cold_prefix = cold["model_load_s"] + sum(item["latency_s"] for item in warmups)
    if not math.isclose(summary["cold_prefix_s"], cold_prefix, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Cold prefix must be initialization plus excluded warmups")
    for item in rows + warmups:
        if (item["status"] != "ok" or item["denoiser_evaluations"] != 50
                or item["cache_stats"]["denoiser_evaluations_complete"] is not True
                or item["failures"] or item["fallbacks"] or item["cache_stats"]["oom_attempts"]
                or not math.isfinite(item["latency_s"]) or item["latency_s"] <= 0):
            raise ValueError("Unexpected request status or incomplete work counters")
    for key in ("quality_validated", "quality_preserving", "comparison_complete", "headline_eligible", "compiled_role_executed"):
        if summary[key] is not False:
            raise ValueError("Unsupported claim: " + key)
    for key in ("speedup", "noise_estimate", "confidence_interval", "p95_s", "compile_only_s", "compile_break_even_requests"):
        if summary[key] is not None:
            raise ValueError("Unsupported estimate: " + key)
    return {"published_numeric_consistency": "passed", "measured_rows": 1,
            "warmups": 3, "gpu_execution_performed": False,
            "withheld_raw_output_verified": False, "quality_evaluated": False}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), sort_keys=True))
