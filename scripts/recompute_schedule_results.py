#!/usr/bin/env python3
"""Reproduce v2 public scores and registered bootstrap statistics on a CPU.

Uses summary.json, episodes.json and export-manifest.json from one public export.
No GPU, private journals, account access, or third-party packages are required.
Submission outcomes are the published flags: their original journal bindings
are checked by the exporter, not reconstructed here. This command verifies the
published sample arrays, scores, and aggregate statistics for internal consistency.

Prints the verified aggregate summary. --receipt writes one optional new receipt
outside the input export; inputs are never modified. Synthetic software fixtures
require --allow-test-fixtures and remain explicitly labeled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_schedule_study_v2 import aggregate
from plot_schedule_study import finite, load_export


BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20261125
VERIFICATION_SCOPE = (
    "Recomputed published sample medians, latency ratios, failure-adjusted scores, "
    "and registered aggregate statistics. Submission outcomes use the published "
    "success flags; this does not independently verify their private-journal bindings."
)


def compare(actual, expected, path):
    """Compare every aggregate field, including paired intervals and accounting."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(path + ": fields differ")
        for key, value in expected.items():
            compare(actual[key], value, path + "." + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(path + ": list differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(finite(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(path + ": numeric value differs")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(path + ": value differs")


def sample_median(measurement, path):
    if measurement is None:
        return None
    if not isinstance(measurement, dict):
        raise ValueError(path + ": measurement must be an object or null")
    status = measurement.get("status")
    if status not in ("ok", "incorrect", "error", "timeout") or type(measurement.get("correct")) is not bool:
        raise ValueError(path + ": invalid measurement outcome")
    if measurement["correct"] is not (status == "ok"):
        raise ValueError(path + ": status/correctness disagreement")
    samples = measurement.get("raw_samples_ms")
    if not isinstance(samples, list) or any(finite(value) <= 0 for value in samples):
        raise ValueError(path + ": invalid timing samples")
    if status == "ok":
        if len(samples) != 7:
            raise ValueError(path + ": successful measurement requires seven samples")
        median = float(statistics.median(samples))
    else:
        median = None
    compare(measurement.get("median_ms"), median, path + ".median_ms")
    return median


def verify_episode(row):
    path = "episode " + str(row["episode_id"])
    measurements = row.get("measurements")
    roles = ("baseline", "paired_final_reference", "confirmation")
    if not isinstance(measurements, dict) or set(measurements) != set(roles):
        raise ValueError(path + ": all measurement roles are required")
    medians = {role: sample_median(measurements[role], path + "." + role) for role in roles}
    if row["successful"]:
        if row.get("status") != "finished" or row["elapsed_seconds"] >= 90 or any(value is None for value in medians.values()):
            raise ValueError(path + ": published success lacks valid measurements or time accounting")
        ratio = medians["confirmation"] / medians["paired_final_reference"]
        score = max(-1., min(1., 1 - ratio))
        speedup = 1 / ratio
    else:
        ratio, speedup, score = None, None, -1.
    compare(row.get("confirmed_latency_ratio"), ratio, path + ".confirmed_latency_ratio")
    compare(row.get("confirmed_speedup"), speedup, path + ".confirmed_speedup")
    compare(row["primary_score"], score, path + ".primary_score")


def recompute(summary_path, episodes_path, *, allow_test_fixtures=False):
    summary, lookup, fixture, hashes = load_export(summary_path, episodes_path,
                                                  allow_test_fixtures=allow_test_fixtures)
    if summary["study_id"] != "sequential-schedules-live-v2":
        raise ValueError("This command reproduces only the registered v2 study")
    rows = list(lookup.values())
    for row in rows:
        verify_episode(row)
    tasks = {task["task_id"]: task for task in summary["tasks"]}
    totals = aggregate(rows, tasks, summary["scope"]["seeds"],
                       replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED)
    for field, recomputed in totals.items():
        if field not in summary:
            raise ValueError("Published summary lacks " + field)
        compare(summary[field], recomputed, "summary." + field)
    input_directory = Path(summary_path).resolve().parent
    for name, expected in hashes.items():
        if hashlib.sha256((input_directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Public input changed during reproduction")
    return dict(schema_version=1, study_id=summary["study_id"], verified=True,
                evidence_kind=summary["evidence_kind"], scientific_results=not fixture,
                verification_scope=VERIFICATION_SCOPE,
                source_sha256=hashes, episodes_verified=len(rows),
                failed_or_unconfirmed_retained=sum(not row["successful"] for row in rows),
                new_measurements=0, source_files_modified=False,
                numerical_comparison_tolerance=dict(relative=1e-12, absolute=1e-12),
                aggregates=totals)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--allow-test-fixtures", action="store_true")
    args = parser.parse_args(argv)
    if args.receipt is not None:
        if (args.receipt.exists() or args.receipt.is_symlink()
                or args.receipt.resolve().is_relative_to(args.summary.resolve().parent)):
            raise ValueError("Receipt must be a new file outside the input export")
        if not args.receipt.parent.is_dir():
            raise ValueError("Receipt parent directory must already exist")
    result = recompute(args.summary, args.episodes, allow_test_fixtures=args.allow_test_fixtures)
    result["reproduction_code_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("recompute_schedule_results.py", "analyze_schedule_study_v2.py", "plot_schedule_study.py")
    }
    if args.receipt is not None:
        with args.receipt.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
