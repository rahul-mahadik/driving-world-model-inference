#!/usr/bin/env python3
"""Generate SVG/PDF figures from measured operator tables and policy reports.

Requires the installed kernel-tuning extension and matplotlib. No provider/API
access or inference is performed. Each output directory is new; source records
remain unchanged. CPU test fixtures are rejected unless explicitly admitted,
and any resulting test figure is watermarked and marked in its receipt.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from drivebench_tuning.replay import MeasuredTable, add_replay_analysis
from drivebench_tuning.search import TrialResult
from drivebench_tuning.suite import _reject_constant, _strict_object


def read_json(path):
    raw = Path(path).read_bytes()
    if len(raw) > 100_000_000:
        raise ValueError("input exceeds 100 MB bound")
    value = json.loads(raw, object_pairs_hook=_strict_object, parse_constant=_reject_constant)
    json.dumps(value, allow_nan=False)
    return value, hashlib.sha256(raw).hexdigest()


def is_fixture(table):
    runtime = table._document["runtime"]
    return ("FIXTURE" in str(runtime.get("gpu_uuid", "")).upper()
            or any(row["measurement"]["evidence"].get("fixture_only") is True
                   for row in table._document["rows"]))


def latency_matrix(table):
    """Paired process-median ratios; any failure masks a candidate cell."""
    result = []
    for task in table.measured_tasks:
        row = []
        for candidate in table.suite.candidates:
            trials = [table.row(task.task_id, candidate.candidate_id, r) for r in table.repeat_indices]
            failures = sum(m["status"] != "ok" for m in trials)
            ratios = [m["latency_ms"] / table.row(task.task_id, None, r)["latency_ms"]
                      for m, r in zip(trials, table.repeat_indices) if m["status"] == "ok"]
            row.append({"task_id": task.task_id, "candidate_id": candidate.candidate_id,
                        "process_repeats": len(trials), "failed_repeats": failures,
                        "median_paired_latency_ratio": median(ratios) if not failures else None})
        result.append(row)
    return result


def _check_replayed_measurement(measurement, table):
    trial = TrialResult(**measurement)
    evidence = measurement["evidence"]
    if (trial.backend != "measured_replay" or trial.cohort_id != table.cohort_id
            or evidence.get("table_sha256") != table.sha256
            or evidence.get("fresh_gpu_measurement") is not False
            or evidence.get("source_backend") != "cuda"):
        raise ValueError("policy trial is not replay from this table")
    source = table.row(trial.task_id, trial.candidate_id, evidence["source_repeat_index"])
    if source["trial_id"] != evidence.get("source_trial_id"):
        raise ValueError("replay source trial identity differs")
    for key in ("input_digest", "status", "correct", "latency_ms", "peak_allocated_bytes"):
        if measurement[key] != source[key]:
            raise ValueError("replayed outcome differs from its recorded source")


def policy_curves(paths, table, *, comparison_repeats=None):
    if comparison_repeats is not None and (type(comparison_repeats) is not int or comparison_repeats < 1):
        raise ValueError("comparison_repeats must be a positive integer")
    results, receipts, names = [], [], set()
    common_split = None
    for path in paths:
        report, sha = read_json(path)
        if (report.get("status") != "completed" or report.get("backend") != "measured_replay"
                or report.get("table_sha256") != table.sha256
                or report.get("suite_sha256") != table.suite.sha256
                or report.get("fresh_gpu_measurements") != 0):
            raise ValueError("a completed replay report from this exact table is required")
        name = report.get("policy_name")
        if type(name) is not str or not name or name in names:
            raise ValueError("policy names must be nonempty and unique")
        names.add(name)
        split = report.get("split")
        if split not in table.suite.splits or common_split not in (None, split):
            raise ValueError("policy comparisons must use the same declared task split")
        common_split = split
        task_ids = {table.suite.tasks[i].task_id for i in table.suite.splits[split]}
        repeats = report.get("repeats")
        if type(repeats) is not int or repeats < 1 or len(report["episodes"]) != len(task_ids) * repeats:
            raise ValueError("policy episode count differs from its declared repeats")
        if comparison_repeats is not None and comparison_repeats > repeats:
            raise ValueError("comparison repeats exceed a supplied report's available repeats")
        plotted_repeats = repeats if comparison_repeats is None else comparison_repeats
        seen = set()
        # Validate every source episode, including any later excluded from the
        # matched comparison. Filtering cannot hide corrupted or partial data.
        for episode in report["episodes"]:
            key = (episode["task_id"], episode["repeat"])
            if (episode["status"] != "completed" or key in seen or key[0] not in task_ids
                    or type(key[1]) is not int or not 0 <= key[1] < repeats):
                raise ValueError("incomplete, duplicate or wrong-split policy episode")
            seen.add(key)
            _check_replayed_measurement(episode["initial_info"]["baseline"], table)
            budget = episode["initial_info"]["trial_budget"]
            if len(episode["steps"]) != budget:
                raise ValueError("policy trajectory does not contain the full turn budget")
            for step in episode["steps"]:
                measurement = step["info"]["measurement"]
                if measurement is not None:
                    _check_replayed_measurement(measurement, table)
                elif not step.get("policy_rejection"):
                    raise ValueError("missing replay measurement without a rejected policy turn")
        weight = report["protocol"]["memory_weight"]
        checked = add_replay_analysis(report, table, memory_weight=weight)
        plotted_episodes = [episode for episode in checked["episodes"] if episode["repeat"] < plotted_repeats]
        task_curves = defaultdict(lambda: defaultdict(list))
        for episode in plotted_episodes:
            for point in episode["table_analysis"]["curve"]:
                task_curves[episode["task_id"]][point["trials_used"]].append(point["table_regret"])
        budgets = sorted(next(iter(task_curves.values())))
        curve = []
        for budget in budgets:
            means = [sum(task_curves[task_id][budget]) / len(task_curves[task_id][budget])
                     for task_id in sorted(task_ids)]
            curve.append({"trial_budget": budget, "mean_task_regret": sum(means) / len(means)})
        results.append({"policy": name, "episodes_per_task": plotted_repeats, "task_count": len(task_ids),
                        "available_episodes_per_task": repeats,
                        "available_total_policy_episodes": len(checked["episodes"]),
                        "comparison_repeats": comparison_repeats,
                        "total_policy_episodes": len(plotted_episodes), "curve": curve,
                        "failed_or_incorrect_trials": sum(e["failed_or_incorrect_trials"] for e in plotted_episodes),
                        "rejected_policy_turns": sum(e["rejected_policy_actions"] for e in plotted_episodes),
                        "api_cost": checked.get("api_cost"), "api_cost_scope": "complete source report",
                        "memory_weight": weight})
        receipts.append({"kind": "policy_report", "sha256": sha})
    if len({r["memory_weight"] for r in results}) > 1:
        raise ValueError("policy objectives differ")
    return results, receipts


def confirmation_points(directories, table):
    """Read full completed repeats, verifying original source/input bindings."""
    points, receipts = [], []
    seen_repeats, seen_trials = set(), {row["measurement"]["trial_id"] for row in table._document["rows"]}
    expected_pairs = None
    expected_selection = None
    for directory in directories:
        directory = Path(directory)
        complete, complete_sha = read_json(directory / "completed.json")
        report, result_sha = read_json(directory / "result.json")
        plan, plan_sha = read_json(directory / "plan.json")
        runtime, runtime_sha = read_json(directory / "runtime.json")
        if (complete.get("completed") is not True or (directory / "failed.json").exists()
                or (directory / "cleanup-failed.json").exists()
                or report.get("backend") != "cuda" or report.get("cohort_id") != table.cohort_id
                or report.get("suite_sha256") != table.suite.sha256
                or runtime != table._document["runtime"]):
            raise ValueError("confirmation is incomplete or from a different cohort")
        repeat = report.get("repeat_index")
        if type(repeat) is not int or repeat < 0 or repeat in seen_repeats or plan.get("repeat_index") != repeat:
            raise ValueError("confirmation repeat index is invalid or duplicated")
        seen_repeats.add(repeat)
        selection_sha = report.get("selection_sha256")
        if (selection_sha != plan.get("selection_sha256") or expected_selection not in (None, selection_sha)
                or plan.get("independent_of_policy_selection") is not True
                or plan.get("selection_updates_permitted") is not False):
            raise ValueError("confirmation selection binding changed")
        expected_selection = selection_sha
        selections = plan["selection_document"]
        if selections.get("cohort_id") != table.cohort_id or selections.get("table_sha256") != table.sha256:
            raise ValueError("confirmation selection refers to another table")
        current_pairs = {(r["task_id"], r["candidate_id"]) for r in report["results"]}
        selected_pairs = {(r["task_id"], r["candidate_id"]) for r in selections["selections"]}
        if (current_pairs != selected_pairs or len(current_pairs) != len(report["results"])
                or expected_pairs not in (None, current_pairs)
                or complete.get("selected_pairs") != len(current_pairs)
                or complete.get("cuda_trials") != len(current_pairs) * 3):
            raise ValueError("confirmation selection coverage differs")
        expected_pairs = current_pairs
        for row in report["results"]:
            task_id, candidate_id = row["task_id"], row["candidate_id"]
            if set(row["measurements"]) != {"eager", "compiled", "selected"}:
                raise ValueError("confirmation reference role missing")
            for role, measurement in row["measurements"].items():
                trial = TrialResult(**measurement)
                expected_candidate = "torch-compiled" if role == "compiled" else candidate_id if role == "selected" else None
                source_input = table.row(task_id, None, table.repeat_indices[0])["input_digest"]
                if (trial.backend != "cuda" or trial.cohort_id != table.cohort_id or trial.task_id != task_id
                        or trial.candidate_id != expected_candidate or trial.input_digest != source_input
                        or trial.trial_id in seen_trials):
                    raise ValueError("confirmation role, inputs, source cohort or trial freshness differs")
                seen_trials.add(trial.trial_id)
                evidence = measurement["evidence"]
                if trial.correct:
                    samples = evidence.get("cuda_event_ms", [])
                    if (len(samples) != plan["repetitions"] or evidence.get("input_integrity_verified") is not True
                            or evidence.get("correctness", {}).get("passed") is not True
                            or any(type(s) not in (float, int) or not math.isfinite(s) or s <= 0 for s in samples)
                            or median(samples) != trial.latency_ms):
                        raise ValueError("confirmation samples or correctness evidence differ")
            compiled, selected = (row["measurements"][role] for role in ("compiled", "selected"))
            ratio = compiled["latency_ms"] / selected["latency_ms"] if compiled["correct"] and selected["correct"] else None
            if ratio != row["descriptive_speedup_vs"]["compiled"]:
                raise ValueError("confirmation speed ratio differs from raw measurements")
            points.append({"task_id": task_id, "candidate_id": candidate_id, "process_repeat": repeat,
                           "speedup_vs_compiled": ratio, "selected_correct": selected["correct"],
                           "compiled_correct": compiled["correct"]})
        receipts.append({"kind": "confirmation_repeat", "repeat_index": repeat,
                         "artifact_sha256": {"completed.json": complete_sha, "result.json": result_sha,
                                             "plan.json": plan_sha, "runtime.json": runtime_sha}})
    return points, receipts


def _finish(figure, output, name, *, fixture):
    if fixture:
        figure.text(.5, .5, "CPU TEST FIXTURE — NOT BENCHMARK RESULTS", ha="center", va="center",
                    rotation=25, color="#a00000", fontsize=16, alpha=.38, weight="bold")
    for extension in ("svg", "pdf"):
        figure.savefig(output / f"{name}.{extension}", bbox_inches="tight")


def render(table, matrix, curves, points, output, *, fixture=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to export figures") from exc
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    values = np.array([[c["median_paired_latency_ratio"] if c["median_paired_latency_ratio"] is not None else np.nan
                        for c in row] for row in matrix])
    finite = values[np.isfinite(values)]
    low = min(.8, float(finite.min())) if finite.size else .8
    high = max(1.2, float(finite.max())) if finite.size else 1.2
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(max(11, .42 * len(table.suite.candidates) + 3),
                                    max(6.5, .22 * len(matrix) + 1.8)))
    norm = TwoSlopeNorm(vmin=low, vcenter=1, vmax=high)
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(range(len(matrix)), [t.task_id for t in table.measured_tasks], fontsize=8)
    ax.set_xticks(range(len(table.suite.candidates)), [c.candidate_id.replace("triton-", "") for c in table.suite.candidates],
                  rotation=35, ha="right")
    ax.set_title("Operator latency and numerical correctness", loc="left", pad=12)
    ax.set_xlabel("Configuration (Triton: block size and warps)")
    for y, row in enumerate(matrix):
        for x, cell in enumerate(row):
            ratio = cell["median_paired_latency_ratio"]
            label = f"{ratio:.2f}×" if ratio is not None else f"{cell['failed_repeats']}/{cell['process_repeats']} fail"
            if len(matrix) * len(row) <= 300:
                ax.text(x, y, label, ha="center", va="center", fontsize=8,
                        color="white" if ratio is not None and (norm(ratio) < .18 or norm(ratio) > .82) else "#111111")
    task_splits = {table.suite.tasks[index].task_id: split
                   for split, indices in table.suite.splits.items() for index in indices}
    measured_splits = [task_splits[task.task_id] for task in table.measured_tasks]
    for index in range(1, len(measured_splits)):
        if measured_splits[index] != measured_splits[index - 1]:
            ax.axhline(index - .5, color="white", linewidth=2)
    bar = fig.colorbar(image, ax=ax, fraction=.025, pad=.025)
    bar.set_label("Median paired latency / eager latency (lower is faster)")
    repeats, samples = table._document["planned_repeats"], table._document["protocol"]["repetitions"]
    fig.text(.12, -.025, f"{repeats} fresh-process repeats/cell; {samples} CUDA-event samples/measurement. "
             "Cells with any incorrect output or execution failure are masked.", fontsize=8)
    fig.tight_layout()
    _finish(fig, output, "operator-latency-correctness", fixture=fixture)
    plt.close(fig)
    if curves:
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        for curve in curves:
            count_label = (f"{curve['episodes_per_task']} episodes/task" if curve["comparison_repeats"] is None else
                           f"{curve['episodes_per_task']} plotted / {curve['available_episodes_per_task']} available per task")
            ax.plot([p["trial_budget"] for p in curve["curve"]],
                    [p["mean_task_regret"] for p in curve["curve"]], marker="o", markersize=4,
                    label=f"{curve['policy']} ({count_label})")
        ax.set(title="Configuration-search regret", xlabel="Policy turns used",
               ylabel="Mean task regret (normalized latency/memory cost)")
        ax.set_xticks(sorted({p["trial_budget"] for c in curves for p in c["curve"]}))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), frameon=False, fontsize=8)
        fig.text(.12, -.03, f"Each curve averages {curves[0]['task_count']} tasks equally. "
                 f"Replay reuses {repeats} measured process repeats; policy episodes add no GPU samples.\n"
                 "No confidence intervals or policy-superiority test are implied.", fontsize=8)
        fig.tight_layout()
        _finish(fig, output, "policy-regret-vs-budget", fixture=fixture)
        plt.close(fig)
    if points:
        groups = defaultdict(list)
        for point in points:
            groups[(point["task_id"], point["candidate_id"])].append(point)
        pairs = sorted(groups, key=lambda p: (p[0], p[1] or ""))
        fig, ax = plt.subplots(figsize=(9.5, max(3, .42 * len(pairs) + 1.8)))
        all_x = []
        for y, pair in enumerate(pairs):
            records = sorted(groups[pair], key=lambda p: p["process_repeat"])
            valid = [p["speedup_vs_compiled"] for p in records if p["speedup_vs_compiled"] is not None]
            for index, record in enumerate(records):
                ratio = record["speedup_vs_compiled"]
                if ratio is not None:
                    offset = (index - (len(records) - 1) / 2) * .12
                    ax.scatter(ratio, y + offset, s=22, color="#35608d", alpha=.8)
                    all_x.append(ratio)
            if valid:
                ax.scatter(median(valid), y, marker="D", s=22, color="#111111", zorder=3)
            failed = len(records) - len(valid)
            count_label = f"{len(records)} repeat" + ("s" if len(records) != 1 else "")
            ax.text(1.01, y, count_label + (f", {failed} failed" if failed else ""),
                    transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        ax.axvline(1, color="#777777", linewidth=1, linestyle="--")
        ax.set_yticks(range(len(pairs)), [f"{t} / {c or 'eager'}" for t, c in pairs], fontsize=8)
        ax.invert_yaxis()
        ax.set_title("Fresh selected-configuration confirmation", loc="left", pad=12)
        ax.set_xlabel("Compiled reference latency / selected latency (greater than 1 is faster)")
        ax.grid(axis="x", color="#dddddd", linewidth=.5)
        if not all_x:
            ax.set_xlim(.8, 1.2)
        fig.text(.12, -.02, "Dots: raw process repeats. Diamonds: medians of valid ratios. "
                 "Failures remain counted; no ratio is plotted when either output fails correctness.\n"
                 "The same GPU/runtime and regenerated inputs are required. No confidence intervals are shown.", fontsize=8)
        fig.tight_layout()
        _finish(fig, output, "confirmation-vs-compiled", fixture=fixture)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--policy-reports", type=Path, nargs="*", default=[])
    parser.add_argument("--confirmation-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--comparison-repeats", type=int,
                        help="Plot the first N episodes per task for every policy; validate all original records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-test-fixtures", action="store_true", help="Watermark all output as CPU test fixtures")
    args = parser.parse_args(argv)
    table = MeasuredTable.load(args.table)
    fixture = is_fixture(table) or args.allow_test_fixtures
    if fixture and not args.allow_test_fixtures:
        raise ValueError("CPU fixture evidence cannot produce benchmark result figures")
    matrix = latency_matrix(table)
    curves, policy_receipts = policy_curves(args.policy_reports, table, comparison_repeats=args.comparison_repeats)
    points, confirmation_receipts = confirmation_points(args.confirmation_dirs, table)
    args.output.mkdir(parents=True, exist_ok=False)
    render(table, matrix, curves, points, args.output, fixture=fixture)
    receipt = {"schema_version": 1, "test_fixture": fixture, "table_sha256": table.sha256,
               "table_process_repeats": len(table.repeat_indices), "new_gpu_measurements": 0,
               "independent_samples_added_by_plotting": 0, "comparison_repeats": args.comparison_repeats,
               "inputs": policy_receipts + confirmation_receipts,
               "matrix": matrix, "policy_curves": curves, "confirmation_points": points,
               "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())}}
    with (args.output / "analysis.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "test_fixture": fixture,
                      "figures": sorted(receipt["figure_sha256"]), "new_gpu_measurements": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
