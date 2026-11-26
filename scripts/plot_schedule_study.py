#!/usr/bin/env python3
"""Plot validated public sequential-study exports as SVG and PDF.

Requires matplotlib. This performs no inference or significance testing. Inputs
must be summary.json and episodes.json from one analysis export, with their
export-manifest.json. Synthetic fixtures require explicit opt-in and are visibly
labeled. Output must be a new directory outside the source export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


POLICIES = ("compiled_eager_no_edit", "random_walk", "greedy_local")
LABELS = ("Eager / compiled reference", "Random walk", "Greedy local search")
COLORS = ("#687684", "#2058aa", "#5a8ea8")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024**2:
        raise ValueError("Expected a bounded regular JSON file")
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=_object)
    json.dumps(value, allow_nan=False)
    return value, hashlib.sha256(raw).hexdigest(), len(raw)


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return value


def load_export(summary_path, episodes_path, *, allow_test_fixtures=False):
    summary_path, episodes_path = Path(summary_path).resolve(), Path(episodes_path).resolve()
    if summary_path.name != "summary.json" or episodes_path.name != "episodes.json" or summary_path.parent != episodes_path.parent:
        raise ValueError("Use summary.json and episodes.json from the same public export")
    summary, summary_sha, summary_size = read_json(summary_path)
    episodes, episodes_sha, episodes_size = read_json(episodes_path)
    manifest, manifest_sha, _ = read_json(summary_path.parent / "export-manifest.json")
    entries = {row["file"]: row for row in manifest["files"]}
    if len(entries) != len(manifest["files"]):
        raise ValueError("Duplicate export manifest entries")
    for name, sha, size in (("summary.json", summary_sha, summary_size), ("episodes.json", episodes_sha, episodes_size)):
        if entries.get(name, {}).get("sha256") != sha or entries[name].get("bytes") != size:
            raise ValueError("Input differs from its public export manifest")
    kind = summary.get("evidence_kind")
    fixture = kind == "synthetic-software-test-fixture"
    if kind not in ("measured-cuda-study", "synthetic-software-test-fixture"):
        raise ValueError("Unknown evidence kind")
    if fixture and not allow_test_fixtures:
        raise ValueError("Synthetic data require --allow-test-fixtures")
    if (summary.get("scientific_results") is not (not fixture)
            or episodes.get("scientific_results") is not (not fixture)
            or episodes.get("evidence_kind") != kind or manifest.get("evidence_kind") != kind):
        raise ValueError("Evidence labels disagree")
    if summary.get("schema_version") != 1 or summary.get("study_id") not in ("sequential-schedules-live-v1", "sequential-schedules-live-v2"):
        raise ValueError("Unknown public study schema")
    scope = summary["scope"]
    if (scope.get("policies") != list(POLICIES) or scope.get("task_families") != 6
            or scope.get("operators") != 2 or scope.get("attempts") != 54
            or scope.get("episode_ceiling_seconds") != 90):
        raise ValueError("Expected the six-task, three-method, 90-second study")
    seeds = scope["seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(seed) is not int for seed in seeds):
        raise ValueError("Expected three distinct seeds")
    tasks = {task["task_id"]: task for task in summary["tasks"]}
    if len(tasks) != 6 or len(summary["tasks"]) != 6:
        raise ValueError("Expected six unique tasks")
    for task in tasks.values():
        if task["operator"] not in ("geglu", "groupnorm_silu") or task["dtype"] != "float16":
            raise ValueError("Unexpected operator or dtype")
        if not task["shape"] or any(type(size) is not int or size <= 0 for size in task["shape"]):
            raise ValueError("Invalid workload shape")
    if any(sum(task["operator"] == operator for task in tasks.values()) != 3
           for operator in ("geglu", "groupnorm_silu")):
        raise ValueError("Expected three tasks per operator")
    rows = episodes["episodes"]
    expected = {(task, seed, policy) for task in tasks for seed in seeds for policy in POLICIES}
    lookup = {(row["task_id"], row["seed"], row["policy"]): row for row in rows}
    if len(rows) != 54 or len(lookup) != 54 or set(lookup) != expected:
        raise ValueError("All 54 paired attempts, including failures, are required")
    for row in rows:
        if type(row["successful"]) is not bool:
            raise ValueError("Invalid outcome flag")
        score, elapsed = finite(row["primary_score"]), finite(row["elapsed_seconds"])
        if not -1 <= score <= 1 or elapsed < 0 or row["episode_budget_seconds"] != 90:
            raise ValueError("Invalid score or wall-clock accounting")
        if not row["successful"] and score != -1:
            raise ValueError("Failed or unconfirmed attempts must retain score -1")
    means = {(row["task_id"], row["policy"]): row for row in summary["task_policy_means"]}
    if len(means) != 18 or len(summary["task_policy_means"]) != 18 or set(means) != {(task, policy) for task in tasks for policy in POLICIES}:
        raise ValueError("Incomplete task/method means")
    for (task, policy), row in means.items():
        values = [lookup[(task, seed, policy)] for seed in seeds]
        if (row["attempts"] != 3 or row["confirmed"] != sum(value["successful"] for value in values)
                or not math.isclose(finite(row["mean_primary_score"]), statistics.mean(value["primary_score"] for value in values), abs_tol=1e-12, rel_tol=1e-12)):
            raise ValueError("Task means differ from the individual attempts")
    hashes = {"summary.json": summary_sha, "episodes.json": episodes_sha, "export-manifest.json": manifest_sha}
    return summary, lookup, fixture, hashes


def task_label(task):
    operator = "GEGLU" if task["operator"] == "geglu" else "GroupNorm + SiLU"
    family = re.search(r"(?:^|-)f(\d+)(?:-|$)", task["task_id"])
    suffix = " / f" + family.group(1) if family else ""
    return operator + suffix + "\n" + " × ".join(str(size) for size in task["shape"])


def render(summary, lookup, output, *, fixture):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    tasks = sorted(summary["tasks"], key=lambda task: (task["operator"], task["task_id"]))
    seeds = sorted(summary["scope"]["seeds"])
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.labelsize": 15, "axes.titlesize": 17,
                         "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "text.color": "#182632", "axes.labelcolor": "#182632",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9ba8b3", "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(17, 9.5), sharey=True)
        fig.subplots_adjust(left=.215, right=.98, bottom=.25, top=.80, wspace=.15)
        for index, task in enumerate(tasks):
            for method, (policy, color) in enumerate(zip(POLICIES, COLORS)):
                values = [lookup[(task["task_id"], seed, policy)] for seed in seeds]
                y = index + (method - 1) * .27
                for ax, field in zip(axes, ("primary_score", "elapsed_seconds")):
                    for offset, row in zip((-.055, 0., .055), values):
                        ax.scatter(row[field], y + offset, s=38, c=color,
                                   marker="o" if row["successful"] else "x", linewidths=1.6,
                                   alpha=.85, zorder=4)
                    # Slight vertical separation keeps all three points visible,
                    # including cells with identical outcomes or three failures.
                    ax.scatter(statistics.mean(row[field] for row in values), y + .105,
                               marker="D", s=42, color=color, edgecolor="white", linewidth=.6, zorder=5)
        for ax in axes:
            ax.set_ylim(5.65, -.65)
            ax.set_yticks(range(6))
            ax.grid(axis="x", color="#d9e0e6", linewidth=.7, zorder=0)
            ax.tick_params(axis="y", length=0, pad=12)
            ax.axhline(2.5, color="#d9e0e6", linewidth=1)
        axes[0].set_yticklabels([task_label(task) for task in tasks])
        axes[1].tick_params(axis="y", labelleft=False)
        axes[0].set_xlim(-1.10, 1.05)
        axes[0].set_xticks([-1, -.5, 0, .5, 1])
        axes[0].axvline(0, color="#52616d", linewidth=1.1, linestyle="--")
        axes[0].set(title="A   Failure-adjusted speed score", xlabel="Primary score (higher is better)")
        time_max = max(100., max(row["elapsed_seconds"] for row in lookup.values()) * 1.08)
        axes[1].set_xlim(0, time_max)
        step = max(30, 30 * math.ceil(time_max / 150))
        axes[1].set_xticks(sorted(set(range(0, math.floor(time_max / step) * step + 1, step)) | {90}))
        axes[1].axvline(90, color="#52616d", linewidth=1.1, linestyle="--")
        axes[1].set(title="B   Actual episode wall time", xlabel="Elapsed time (seconds)")
        title = "Sequential schedule search"
        if fixture:
            title = "SYNTHETIC FIXTURE — layout check, not study results"
        fig.suptitle(title, fontsize=20, x=.215, ha="left", y=.985)
        context = "6 tasks · 2 operators · 3 seeds per method · 90-second episode ceiling"
        if not fixture:
            context = summary["runtime"]["gpu_name"] + " · " + context
        fig.text(.215, .926, context, fontsize=14, color="#52616d")
        handles = [Line2D([], [], color=color, marker="o", linestyle="none", markersize=7, label=label)
                   for label, color in zip(LABELS, COLORS)]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.208, .905), ncol=3,
                   frameon=False, fontsize=14, columnspacing=2.0, handletextpad=.5)
        fig.text(.215, .085,
                 "Circles: seed observations; crosses: failed or unconfirmed episodes; diamonds: arithmetic means.\n"
                 "Failed episodes retain score −1. Score = clipped (1 − selected latency / paired reference latency).",
                 fontsize=14, linespacing=1.55)
        fig.text(.215, .025,
                 "Wall time includes setup, compilation, search, confirmation, and cleanup. Dashed lines mark score 0 and 90 seconds.",
                 fontsize=14, color="#52616d")
        metadata = {"Title": title, "Creator": "DriveBench plot_schedule_study.py"}
        for extension in ("svg", "pdf"):
            fig.savefig(output / ("sequential-schedule-study." + extension), bbox_inches="tight", pad_inches=.22, metadata=metadata)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-test-fixtures", action="store_true")
    args = parser.parse_args(argv)
    summary, lookup, fixture, hashes = load_export(args.summary, args.episodes,
                                                  allow_test_fixtures=args.allow_test_fixtures)
    output = args.output.resolve()
    if args.output.is_symlink() or output.exists() or output.is_relative_to(args.summary.resolve().parent):
        raise ValueError("Output must be a new directory outside the source export")
    output.mkdir(parents=True, exist_ok=False)
    render(summary, lookup, output, fixture=fixture)
    for name, expected in hashes.items():
        if hashlib.sha256((args.summary.resolve().parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Source export changed during plotting")
    receipt = {"schema_version": 1, "study_id": summary["study_id"], "test_fixture": fixture,
               "scientific_results": not fixture, "source_sha256": hashes,
               "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "plotted_attempts": 54, "omitted_attempts": 0,
               "point_estimate": "arithmetic mean of all three seeds per task and method",
               "new_measurements": 0, "significance_test_performed": False,
               "figures": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())}}
    with (output / "plot-receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "test_fixture": fixture, "figures": list(receipt["figures"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
