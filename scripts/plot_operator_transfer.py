#!/usr/bin/env python3
"""Plot cross-GPU ranks and frozen-selection regret from verified measured tables.

All plotted values are recomputed from both source tables and the frozen primary
selection record. The target table never changes the selected configurations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from drivebench_tuning.generalization_analysis import transfer_analysis
from drivebench_tuning.replay import MeasuredTable
from plot_operator_search import is_fixture, read_json, _finish


def render(analysis, output, *, fixture=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    records = analysis["records"]
    fig, ax = plt.subplots(figsize=(10, max(4, .3 * len(records) + 1.5)))
    for y, row in enumerate(records):
        value = row["rank_correlation"]
        if value is None:
            ax.text(0, y, "undefined", va="center", color="#777777", fontsize=8)
        else:
            ax.barh(y, value, color="#416b87", height=.65)
        ax.text(1.02, y, f"{row['ranked_candidates_passing_on_both']} candidates",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.set_yticks(range(len(records)), [r["task_id"] for r in records], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="#888888", linewidth=.7)
    ax.set(title="Configuration ranking across GPU cohorts", xlabel="Spearman rank correlation")
    ax.grid(axis="x", color="#dddddd", linewidth=.5)
    fig.text(.12, -.025, f"{analysis['source_gpu']} → {analysis['target_gpu']}\n"
             "Ranks include only configurations passing every repeat on both GPUs. "
             "Each row is a workload, not an independent hardware sample.", fontsize=8)
    fig.tight_layout()
    _finish(fig, output, "cross-gpu-rank-correlation", fixture=fixture)
    plt.close(fig)

    groups = defaultdict(list)
    for row in records:
        groups["Source-table optimum"].append(row["source_optimal_transfer"])
        for selected in row["frozen_policy_selections"]:
            groups[selected["policy"]].append(selected)
    fig, ax = plt.subplots(figsize=(9, max(3.8, .48 * len(groups) + 1.5)))
    summaries = []
    for y, (name, values) in enumerate(groups.items()):
        regrets = [v["target_regret"] for v in values]
        mean = float(np.mean(regrets))
        failures = sum(v["target_failure_rate"] > 0 for v in values)
        ax.barh(y, mean, color="#c7d6df", height=.65)
        offsets = np.linspace(-.19, .19, len(values))
        ax.scatter(regrets, y + offsets, color="#345b74", s=17, alpha=.8, zorder=3)
        ax.text(1.02, y, f"{failures}/{len(values)} failed cells",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        summaries.append({"policy": name, "mean_task_transfer_regret": mean,
                          "task_count": len(values), "selected_cells_with_failure": failures})
    ax.set_yticks(range(len(groups)), list(groups))
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set(title="Frozen choices on the second GPU", xlabel="Regret relative to the target-table optimum")
    ax.grid(axis="x", color="#dddddd", linewidth=.5)
    fig.text(.12, -.03, "Bars: equally weighted task means. Dots: individual tasks. "
             "Cost is latency normalized to eager; failed selections receive reference cost.\n"
             "Primary choices are frozen before target collection. The target optimum is descriptive.", fontsize=8)
    fig.tight_layout()
    _finish(fig, output, "cross-gpu-selection-regret", fixture=fixture)
    plt.close(fig)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument("--target-table", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-test-fixtures", action="store_true")
    args = parser.parse_args(argv)
    source, target = MeasuredTable.load(args.source_table), MeasuredTable.load(args.target_table)
    frozen, selection_file_sha = read_json(args.selections)
    fixture = is_fixture(source) or is_fixture(target) or args.allow_test_fixtures
    if fixture and not args.allow_test_fixtures:
        raise ValueError("CPU test fixtures cannot produce benchmark result figures")
    analysis = transfer_analysis(source, target, frozen)
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = render(analysis, args.output, fixture=fixture)
    receipt = {"schema_version": 1, "test_fixture": fixture, "analysis": analysis,
               "selection_file_sha256": selection_file_sha, "policy_summaries": summaries,
               "new_gpu_measurements": 0, "independent_samples_added_by_plotting": 0,
               "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(args.output.iterdir())}}
    with (args.output / "analysis.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "test_fixture": fixture,
                      "figures": sorted(receipt["figure_sha256"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
