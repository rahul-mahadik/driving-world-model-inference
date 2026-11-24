#!/usr/bin/env python3
"""Derive website summaries from published tables and recorded trajectories.

Python standard library only. No policies, training, inference, or GPU calls run.
Usage: python3 reports/site-leaderboard-20261124/derive_leaderboard.py
Use --output PATH for a second verification output; existing files are preserved.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import tarfile


PRIMARY = "results/2026-11-23-operator-generalization-pro-4000/data.tar.gz"
PPO = "results/2026-11-24-ppo-native-autotune/ppo-data.tar.gz"
PREFIX = "2026-11-23-operator-generalization-pro-4000/"
PPO_PREFIX = "ppo-training-2026-11-23/"
LABELS = {
    "fixed-compiled": "Compile with eager fallback",
    "random": "Random search",
    "optuna-tpe": "TPE",
    "ridge-selector": "Ridge selector",
    "ppo-five-seeds": "PPO (all five training seeds)",
    "gpt-5.4-mini-2026-03-17": "GPT-5.4 mini",
    "gpt-5.4-nano-2026-03-17": "GPT-5.4 nano",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def percentile(sorted_values, p):
    position = (len(sorted_values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def interval(task_values, families):
    """Enumerate the complete six-family percentile bootstrap distribution."""
    require(set(task_values) == set(families), "All eighteen TEST workloads required")
    groups = defaultdict(list)
    for task, family in families.items():
        groups[family].append(task_values[task])
    require(len(groups) == 6 and {len(v) for v in groups.values()} == {3},
            "Bootstrap requires six equal-size families with three dtypes each")
    means = [statistics.mean(groups[key]) for key in sorted(groups)]
    draws = sorted(sum(means[i] for i in sample) / 6
                   for sample in itertools.product(range(6), repeat=6))
    return {"mean": statistics.mean(task_values.values()),
            "lower_95": percentile(draws, .025), "upper_95": percentile(draws, .975)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("leaderboard.json"))
    args = parser.parse_args()
    sources = []

    def member(archive, archive_path, name):
        raw = archive.extractfile(name).read()
        sources.append({"archive": archive_path, "member": name,
                        "bytes": len(raw), "sha256": sha(raw)})
        return json.loads(raw), sha(raw)

    with tarfile.open(args.root / PRIMARY) as primary, tarfile.open(args.root / PPO) as ppo:
        table, table_hash = member(primary, PRIMARY, PREFIX + "table.json")
        bindings, _ = member(ppo, PPO, PPO_PREFIX + "source-bindings.json")
        require(bindings["published_table_sha256"] == table_hash
                and bindings["suite_sha256"] == table["suite_sha256"], "PPO/public table binding mismatch")
        tasks = {t["id"]: t for t in table["suite"]["tasks"]}
        families = {task: (t["workload"]["operator"], tuple(t["workload"]["shape"]),
                          t["workload"].get("num_groups"), t["workload"].get("eps"))
                    for task, t in tasks.items() if t["split"] == "test"}
        candidates = [None] + [c["id"] for c in table["suite"]["candidates"]]
        rows = {}
        for row in table["rows"]:
            m = row["measurement"]
            key = (m["task_id"], m["candidate_id"], row["repeat_index"])
            require(key not in rows, "Duplicate table row")
            require(m["status"] == "ok" and m["correct"] is True
                    and m["evidence"]["correctness"]["passed"] is True,
                    "This published table is expected to contain only passing rows")
            require(close(m["latency_ms"], statistics.median(m["evidence"]["cuda_event_ms"])),
                    "Row latency disagrees with raw CUDA-event samples")
            rows[key] = m
        require(set(rows) == {(t, c, r) for t in tasks for c in candidates for r in range(3)},
                "Incomplete measured table")
        costs = {(task, candidate): statistics.median(
                    rows[(task, candidate, r)]["latency_ms"] / rows[(task, None, r)]["latency_ms"]
                    for r in range(3)) for task in families for candidate in candidates}
        oracles = {task: min(costs[(task, c)] for c in candidates) for task in families}
        reports = {}

        def consume(report, expected_table_hash):
            require(report["status"] == "completed" and report["split"] == "test"
                    and report["backend"] == "measured_replay"
                    and report["fresh_gpu_measurements"] == 0
                    and report["table_sha256"] == expected_table_hash
                    and report["suite_sha256"] == table["suite_sha256"]
                    and report["protocol"]["trial_budget"] == 8
                    and report["protocol"]["memory_weight"] == 0,
                    "Unexpected replay report protocol or source")
            name = report["policy_name"]
            require(name not in reports, "Duplicate policy report")
            compact, seen = [], set()
            for e in report["episodes"]:
                task, repeat = e["task_id"], e["repeat"]
                require((task, repeat) not in seen and task in families
                        and e["status"] == "completed" and len(e["steps"]) == 8,
                        "Duplicate or incomplete TEST episode")
                seen.add((task, repeat))
                curve, actions, repeated = [], set(), 0
                for index, step in enumerate(e["steps"]):
                    info = step["info"]
                    selected = info["incumbent_candidate_id"]
                    value = max(0., costs[(task, selected)] - oracles[task])
                    stored = e["table_analysis"]["curve"][index]
                    require(stored["selected_candidate_id"] == selected
                            and close(stored["table_selection_cost"], costs[(task, selected)])
                            and close(stored["table_regret"], value)
                            and info["trials_used"] == index + 1,
                            "Stored curve differs from source-row reconstruction")
                    curve.append(value)
                    action = step["action"]
                    repeated += int(action in actions)
                    actions.add(action)
                    if name == "fixed-compiled":
                        require(info["measurement"]["candidate_id"] == "torch-compiled"
                                and selected in {None, "torch-compiled"},
                                "Compile baseline is not compiled queries with eager fallback")
                compact.append({"task": task, "repeat": repeat, "curve": curve,
                                "repeated_action_turns": repeated,
                                "final_candidate": e["final_info"]["incumbent_candidate_id"]})
            require(seen == {(t, r) for t in families for r in range(report["repeats"])},
                    "Missing episode repeats")
            reports[name] = {"repeats": report["repeats"], "episodes": compact}

        for index in range(6):
            report, _ = member(primary, PRIMARY, PREFIX + f"related/policy-{index:03d}.json")
            consume(report, table_hash)
        for seed in range(3101, 3106):
            report, _ = member(ppo, PPO, PPO_PREFIX + f"recorded/evaluation/ppo-seed-{seed}.json")
            consume(report, bindings["recorded_table_sha256"])

    def task_values(names, repeats, turn):
        values = defaultdict(list)
        for name in names:
            for e in reports[name]["episodes"]:
                if e["repeat"] < repeats:
                    values[e["task"]].append(e["curve"][turn - 1])
        require({len(v) for v in values.values()} == {len(names) * repeats}, "Unequal task weights")
        return {task: statistics.mean(values[task]) for task in sorted(values)}

    ppo_names = [f"ppo-seed-{seed}" for seed in range(3101, 3106)]
    panels = []
    for count in (20, 1):
        names = [name for name in LABELS if not name.startswith("gpt-") or count == 1]
        panel_rows = []
        baseline = task_values(["fixed-compiled"], count, 8)
        for name in names:
            policies = ppo_names if name == "ppo-five-seeds" else [name]
            final = task_values(policies, count, 8)
            fourth = task_values(policies, count, 4)
            panel_rows.append({"policy": name, "label": LABELS[name],
                "replays_per_task_per_policy_seed": count, "policy_seeds": len(policies),
                "episodes": 18 * count * len(policies),
                "final_regret": interval(final, families),
                "query4_regret": interval(fourth, families),
                "final_minus_compile_with_eager_fallback": interval(
                    {t: final[t] - baseline[t] for t in final}, families),
                "per_task_final_regret": final, "per_task_query4_regret": fourth,
                "curve": [{"queries": turn, "mean_regret": statistics.mean(
                    task_values(policies, count, turn).values())} for turn in range(1, 9)]})
        panel_rows.sort(key=lambda row: (row["final_regret"]["mean"], row["policy"]))
        panels.append({"id": f"replays-{count}",
            "label": "20 replay episodes per workload" if count == 20 else "First replay episode per workload",
            "ppo_aggregation": "Equal mean over all five frozen training seeds; no best-seed selection",
            "rows": panel_rows})

    live_path = "results/2026-11-24-ppo-native-autotune/live-analysis.json"
    live_raw = (args.root / live_path).read_bytes()
    live = json.loads(live_raw)
    sources.append({"path": live_path, "sha256": sha(live_raw), "bytes": len(live_raw)})
    transfer_path = "results/2026-11-23-operator-generalization-pro-4000/figures/transfer/analysis.json"
    transfer_raw = (args.root / transfer_path).read_bytes()
    transfer = json.loads(transfer_raw)["analysis"]
    sources.append({"path": transfer_path, "sha256": sha(transfer_raw), "bytes": len(transfer_raw)})
    transfer_groups = defaultdict(dict)
    for row in transfer["records"]:
        transfer_groups["Source-table optimum"][row["task_id"]] = row["source_optimal_transfer"]["target_regret"]
        for selection in row["frozen_policy_selections"]:
            transfer_groups[selection["policy"]][row["task_id"]] = selection["target_regret"]
    require(all(set(values) == set(families) for values in transfer_groups.values()),
            "Transfer results must cover the same eighteen workloads")
    transfer_summary = {row["policy"]: row["mean_target_regret"]
                        for row in transfer["summary"]["policies"]}
    transfer_summary["Source-table optimum"] = transfer["summary"]["mean_source_optimal_transfer_regret"]
    require(all(close(statistics.mean(values.values()), transfer_summary[name])
                for name, values in transfer_groups.items()), "Transfer mean mismatch")
    live_percentages = {}
    for name, summary in live["ppo_speedup_ratios"].items():
        ratio = summary["estimate"]
        live_percentages[name] = {
            "comparator_latency_over_ppo_latency": ratio,
            "ppo_latency_reduction_percent": 100 * (1 - 1 / ratio),
            "ppo_throughput_increase_percent": 100 * (ratio - 1),
            "ppo_latency_reduction_lower_95": 100 * (1 - 1 / summary["lower_95"]),
            "ppo_latency_reduction_upper_95": 100 * (1 - 1 / summary["upper_95"]),
            "interval_source": "Published paired-latency family bootstrap; monotone transform, not a new experiment"}
    duplicate_turns = {name: {"repeated": sum(e["repeated_action_turns"] for e in report["episodes"]),
                              "total": 8 * len(report["episodes"])}
                       for name, report in reports.items() if name.startswith("gpt-")}
    pure_compiled = {t: costs[(t, "torch-compiled")] - oracles[t] for t in families}
    ideal_reference_fallback = {t: min(1., costs[(t, "torch-compiled")]) - oracles[t] for t in families}
    result = {
        "kind": "published_operator_leaderboard_derivation", "schema_version": 1,
        "script_sha256": sha(Path(__file__).read_bytes()), "suite_sha256": table["suite_sha256"],
        "table_sha256": table_hash, "sources": sources,
        "fresh_gpu_measurements": 0, "policy_runs": 0, "training_runs": 0,
        "regret_definition": "Selected median paired normalized latency minus the minimum over eager and all 25 candidates. Each candidate cost is the median of three same-process candidate/eager latency ratios. Zero is the measured finite-table oracle.",
        "bootstrap": {"method": "Exact percentile shape-family cluster bootstrap",
            "confidence_level": .95, "ordered_resamples": 6 ** 6,
            "shape_families": 6, "workloads": 18, "dtypes_per_family": 3,
            "quantile_interpolation": "linear, index (n - 1) * p",
            "aggregation": "Average replay episodes and, for PPO, all five fixed training seeds within each workload; resample six shape families with all three dtypes together.",
            "scope": "Conditional on this measured table, recorded trajectories and fixed checkpoints. Replays are reused measurements, not new GPU samples. Six clusters give exploratory descriptive intervals, not a population policy/GPU ranking or an interval over new training seeds."},
        "panels": panels,
        "compile_baseline": {"queries": "torch-compiled on every turn",
            "incumbent_rule": "Retain eager unless a queried measurement improves the incumbent",
            "final_eager_episodes_of_360": sum(e["final_candidate"] is None for e in reports["fixed-compiled"]["episodes"]),
            "pure_compiled_regret": interval(pure_compiled, families),
            "perfect_table_based_eager_or_compiled_regret": interval(ideal_reference_fallback, families),
            "reference_values_are_diagnostics_not_policy_results": True},
        "llm_repeated_action_turns": duplicate_turns,
        "transfer": {"source_gpu": transfer["source_gpu"], "target_gpu": transfer["target_gpu"],
            "source_optima_remaining_optimal": sum(close(row["source_optimal_transfer"]["target_regret"], 0.)
                                                   for row in transfer["records"]),
            "mean_rank_correlation": statistics.mean(row["rank_correlation"] for row in transfer["records"]),
            "rows": [{"policy": name, "label": LABELS.get(name, name),
                      "mean_regret": statistics.mean(values.values()),
                      "per_task_regret": values} for name, values in transfer_groups.items()],
            "scope": "Frozen source choices evaluated on the RTX 4090 table; equal workload weights. PPO was not in this earlier transfer experiment."},
        "live_comparison_percentages": live_percentages,
        "live_limit": "Separate RTX 4090 allocation; one prospectively selected PPO seed; native 24 configurations plus three fallback probes versus eight PPO queries. No fresh live oracle or equal-search-budget regret comparison."}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {args.output}: 2 panels, 12 policy rows, 18 workloads, 6 families")


if __name__ == "__main__":
    main()
