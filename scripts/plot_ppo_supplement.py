#!/usr/bin/env python3
"""Plot recorded PPO analyses; this command never executes policies or GPU work."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

COLORS = {"ppo": "#345b74", "seed": "#a8c3d6", "optuna-tpe": "#b27437",
          "random": "#3d806f", "fixed-compiled": "#41474c", "initial": "#9b5553"}
TRAINED = [f"ppo-seed-{seed}" for seed in range(3101, 3106)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(document):
    if (document.get("kind") != "ppo_measured_replay_supplement"
            or document.get("fresh_gpu_measurements") != 0
            or document.get("ppo_training_seeds") != 5):
        raise ValueError("completed five-seed measured-replay analysis required")
    policies = {p["policy"]: p for p in document["policies"]}
    required = TRAINED + ["ppo-untrained-seed-3101", "optuna-tpe", "random", "fixed-compiled"]
    if any(name not in policies for name in required):
        raise ValueError("missing requested comparison or training seed")
    for name in required:
        p = policies[name]
        if p["episodes"] != 360 or len(p["per_task_final_regret"]) != 18:
            raise ValueError("expected eighteen workloads and twenty replay episodes per workload")
        if [r["trials"] for r in p["curve"]] != list(range(1, 9)):
            raise ValueError("complete eight-turn curve required")
        values = [r["mean_table_regret"] for r in p["curve"]]
        if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("finite nonnegative measured regret required")
        if not math.isclose(values[-1], p["mean_final_regret"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("curve and final summary differ")
    aggregate = document["ppo_mean_across_seeds"]
    if (aggregate["shape_families"] != 6 or aggregate["tasks"] != 18
            or not math.isclose(aggregate["estimate"], mean(policies[n]["mean_final_regret"] for n in TRAINED),
                                rel_tol=0, abs_tol=1e-12)):
        raise ValueError("five-seed aggregate or family-cluster interval differs")
    return policies


def finish(fig, output, name):
    paths = []
    for suffix in ("png", "svg", "pdf"):
        path = output / f"{name}.{suffix}"
        metadata = {"Title": name.replace("-", " "), "Creator": "DriveBench plot_ppo_supplement.py"}
        if suffix == "pdf": metadata.update(CreationDate=None, ModDate=None)
        if suffix == "svg": metadata["Date"] = None
        fig.savefig(path, dpi=190, facecolor="white", metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_curves(document, policies, output):
    turns = list(range(1, 9))
    get = lambda name: [p["mean_table_regret"] for p in policies[name]["curve"]]
    average = [mean(get(name)[i] for name in TRAINED) for i in range(8)]
    fig, (main, initial) = plt.subplots(1, 2, figsize=(11.4, 4.9), gridspec_kw={"width_ratios": [2.1, 1]})
    fig.subplots_adjust(left=.075, right=.985, bottom=.25, top=.80, wspace=.26)
    for i, name in enumerate(TRAINED):
        main.plot(turns, get(name), color=COLORS["seed"], linewidth=1.0,
                  label="Individual PPO seeds" if i == 0 else None, zorder=1)
    main.plot(turns, average, color=COLORS["ppo"], linewidth=2.2, marker="o", markersize=3.7,
              label="PPO · mean of 5 seeds", zorder=5)
    for name, label, style in [("optuna-tpe", "Optuna TPE", "-"), ("random", "Random", "-"),
                               ("fixed-compiled", "Compiled-query baseline", "--")]:
        main.plot(turns, get(name), color=COLORS[name], linewidth=1.6, linestyle=style, label=label)
    main.set_title("Policy comparison", loc="left", pad=10)
    main.set_ylabel("Mean table regret (lower is better)")
    main.set_ylim(0, .105)
    main.yaxis.set_major_locator(MultipleLocator(.02))
    main.legend(frameon=False, loc="upper right", fontsize=8.2)

    initial.plot(turns, get("ppo-untrained-seed-3101"), color=COLORS["initial"], linewidth=1.7,
                 linestyle="--", label="Untrained initialization")
    initial.plot(turns, get("ppo-seed-3101"), color=COLORS["ppo"], linewidth=2.0,
                 marker="o", markersize=3.3, label="Trained PPO · seed 3101")
    initial.set_title("Effect of training", loc="left", pad=10)
    initial.set_ylim(0, .25)
    initial.yaxis.set_major_locator(MultipleLocator(.05))
    initial.legend(frameon=False, loc="center right", fontsize=8.0)
    for ax in (main, initial):
        ax.set_xlabel("Configuration queries used")
        ax.set_xticks(turns); ax.set_xlim(.8, 8.2)
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
        ax.set_axisbelow(True)
    fig.suptitle("Held-out search on measured operator timings", x=.075, ha="left", y=.985, fontsize=13)
    fig.text(.075, .90, "18 workloads · 6 shape families · 20 recorded-measurement episodes per workload", fontsize=9, color="#555555")
    fig.text(.075, .13,
             "Thin lines show all five PPO training seeds; the thick line averages them. Seed 3101 was fixed in advance for live evaluation.\n"
             "Regret uses median paired latency cost across three recorded GPU processes; more queries can expose noisy incumbents.\n"
             "Replay adds no GPU timing samples. Panel axes differ; the initialization comparison has its own scale.",
             fontsize=8.1, va="top", linespacing=1.5)
    return finish(fig, output, "ppo-regret")


def plot_seed_spread(document, policies, output):
    labels = [f"PPO · seed {seed}" for seed in range(3101, 3106)]
    rows = [policies[name]["family_bootstrap"] for name in TRAINED]
    rows += [document["ppo_mean_across_seeds"], policies["optuna-tpe"]["family_bootstrap"],
             policies["fixed-compiled"]["family_bootstrap"], policies["random"]["family_bootstrap"]]
    labels += ["PPO · all-seed mean", "Optuna TPE", "Compiled-query baseline", "Random"]
    colors = [COLORS["ppo"]] * 6 + [COLORS["optuna-tpe"], COLORS["fixed-compiled"], COLORS["random"]]
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    fig.subplots_adjust(left=.27, right=.96, bottom=.25, top=.82)
    for i, (row, color) in enumerate(zip(rows, colors)):
        value, low, high = row["estimate"], row["lower_95"], row["upper_95"]
        if row["shape_families"] != 6 or row["tasks"] != 18:
            raise ValueError("expected six-family cluster bootstrap")
        ax.plot([low, high], [i, i], color=color, linewidth=1.2, alpha=.62)
        ax.plot([low, low], [i-.1, i+.1], color=color, linewidth=.8, alpha=.62)
        ax.plot([high, high], [i-.1, i+.1], color=color, linewidth=.8, alpha=.62)
        ax.scatter(value, i, marker="D" if i == 5 else "o", s=42 if i == 5 else 28,
                   color=color, edgecolors="white", linewidth=.6, zorder=3)
        ax.text(.138, i, f"{value:.4f}", ha="right", va="center", fontsize=8.5, color="#444444")
    ax.axhline(5.5, color="#d0d0d0", linewidth=.7)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis(); ax.set_ylim(8.6, -.6)
    ax.set_xlim(0, .14); ax.xaxis.set_major_locator(MultipleLocator(.02))
    ax.set_xlabel("Final mean table regret after eight queries (lower is better)")
    ax.grid(axis="x", color="#dddddd", linewidth=.5); ax.set_axisbelow(True)
    fig.suptitle("PPO varies across training seeds", x=.08, ha="left", y=.975, fontsize=13)
    fig.text(.08, .895, "All five trained checkpoints are reported; no best-seed selection.", fontsize=9, color="#555555")
    fig.text(.08, .135,
             "Dots show each training seed; the diamond averages all five. Whiskers are exploratory 95% shape-family bootstrap intervals.\n"
             "Only six held-out shape families are resampled. Intervals condition on these checkpoints, the measured GPU cohort and\n"
             "aggregated repeats; they are not uncertainty over training seeds or fresh timing runs.",
             fontsize=8.1, va="top", linespacing=1.5)
    return finish(fig, output, "ppo-seed-spread")


def validate_live(document):
    """Require the completed three-process summary, including failed confirmations."""
    if (document.get("kind") != "fresh_ppo_native_autotune_confirmation"
            or document.get("live_regret_reported") is not False
            or document.get("equal_search_budget") is not False
            or document.get("offline_ppo_training_cost_excluded_from_search_time") is not True
            or document.get("search_budgets") != {
                "native_configs": 24, "native_fallback_probes": 3, "ppo_queries": 8}):
        raise ValueError("completed live analysis with declared unequal search budgets required")
    tasks = document["tasks"]
    if len(tasks) != 18 or len({t["task_id"] for t in tasks}) != 18:
        raise ValueError("eighteen distinct held-out tasks required")
    for task in tasks:
        repeats = task["repeats"]
        if len(repeats) != 3 or {r["repeat_index"] for r in repeats} != {0, 1, 2}:
            raise ValueError("three distinct process repeats per task required")
        for repeat in repeats:
            for key in ("native_tuning_seconds", "native_fallback_seconds", "ppo_search_seconds"):
                value = repeat[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("positive measured search times required")
        for comparator in ("native", "compiled", "native_with_fallback"):
            values = [r["ratios"][comparator] for r in repeats]
            if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v <= 0)
                   for v in values):
                raise ValueError("positive finite speedup ratios or explicit failures required")
            aggregate = task["median_speedup_ratios"][comparator]
            if any(v is None for v in values):
                if aggregate is not None:
                    raise ValueError("incomplete confirmation must not have a task aggregate")
            elif aggregate is None or not math.isclose(aggregate, median(values), rel_tol=1e-12):
                raise ValueError("task median differs from paired process ratios")
    counts = document["counts"]
    if (counts.get("native_config_trials") != 1296 or counts.get("ppo_candidate_queries") != 432
            or counts.get("confirmation_passed", 0) + counts.get("confirmation_failed", 0) != 216):
        raise ValueError("live trial totals differ from the declared cohort")
    return sorted(tasks, key=lambda t: t["task_id"])


def task_label(task):
    return (task["task_id"].removeprefix("test-")
            .replace("groupnorm_silu", "GN+SiLU").replace("geglu", "GEGLU")
            .replace("bfloat16", "BF16").replace("float16", "FP16").replace("float32", "FP32")
            .replace("-", " · "))


def live_axes(title, subtitle):
    fig, ax = plt.subplots(figsize=(10.2, 8.2))
    fig.subplots_adjust(left=.265, right=.96, bottom=.20, top=.83)
    fig.suptitle(title, x=.075, ha="left", y=.98, fontsize=13)
    fig.text(.075, .925, subtitle, fontsize=9, color="#555555")
    return fig, ax


def plot_live_speedup(document, tasks, output, live_gpu_label):
    fig, ax = live_axes("PPO versus native Triton autotuning",
                        f"{live_gpu_label} · 18 held-out workloads · 3 fresh processes · frozen PPO seed 3101")
    finite = []
    for index, task in enumerate(tasks):
        for repeat in sorted(task["repeats"], key=lambda r: r["repeat_index"]):
            value = repeat["ratios"]["native_with_fallback"]
            if value is not None:
                finite.append(value)
                ax.scatter(value, index + (repeat["repeat_index"] - 1) * .14, s=17,
                           color=COLORS["seed"], alpha=.9, zorder=2,
                           label="Within-process paired ratio" if index == 0 and repeat["repeat_index"] == 0 else None)
        value = task["median_speedup_ratios"]["native_with_fallback"]
        if value is not None:
            ax.scatter(value, index, s=40, marker="D", color=COLORS["ppo"], edgecolors="white",
                       linewidth=.5, zorder=3, label="Median of 3 process ratios" if index == 0 else None)
        else:
            ax.text(.99, index, "incomplete confirmation", transform=ax.get_yaxis_transform(),
                    fontsize=7, ha="right", va="center", color=COLORS["initial"])
    ax.axvline(1, color=COLORS["fixed-compiled"], linestyle="--", linewidth=1)
    ax.set_xscale("log")
    low, high = min([1] + finite) / 1.15, max([1] + finite) * 1.15
    ax.set_xlim(low, high)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.3g}×"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, pos: f"{x:.3g}×"))
    ax.set_yticks(range(len(tasks)), [task_label(t) for t in tasks], fontsize=8)
    ax.set_ylim(len(tasks) - .4, -.6)
    ax.set_xlabel("Native-with-fallback latency / PPO latency (log scale; >1 favors PPO)")
    ax.grid(axis="x", which="both", color="#dddddd", linewidth=.5); ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=8)
    fig.text(.075, .115,
             "Native search tries 24 kernel configurations and 3 reference probes; PPO uses 8 queries plus an eager reset.\n"
             "Fallback choices are fixed before independent confirmation. Dots are process repeats, not confidence intervals.\n"
             "PPO was trained on the PRO 4000 table. This is a separate live GPU cohort; no live oracle or regret is reported.",
             fontsize=8.1, va="top", linespacing=1.5)
    return finish(fig, output, "ppo-native-speedup")


def plot_live_cost(document, tasks, output, live_gpu_label):
    fig, ax = live_axes("Live tuning cost under unequal search budgets",
                        f"{live_gpu_label} · wall-clock selection time · 18 workloads × 3 fresh processes")
    all_values = []
    for index, task in enumerate(tasks):
        ppo_values = [r["ppo_search_seconds"] for r in task["repeats"]]
        native_values = [r["native_tuning_seconds"] + r["native_fallback_seconds"] for r in task["repeats"]]
        all_values += ppo_values + native_values
        ax.plot([median(ppo_values), median(native_values)], [index, index], color="#dddddd", linewidth=1.2, zorder=1)
        for values, color, offset, label in ((native_values, COLORS["optuna-tpe"], -.12, "Native: 24 configs + 3 probes"),
                                            (ppo_values, COLORS["ppo"], .12, "PPO: 8 queries + eager reset")):
            ax.scatter(values, [index + offset] * 3, s=15, color=color, alpha=.28, zorder=2)
            ax.scatter(median(values), index + offset, s=35, marker="D", color=color,
                       edgecolors="white", linewidth=.5, label=label if index == 0 else None, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(min(all_values) / 1.3, max(all_values) * 1.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    ax.set_yticks(range(len(tasks)), [task_label(t) for t in tasks], fontsize=8)
    ax.set_ylim(len(tasks) - .4, -.6)
    ax.set_xlabel("Live selection wall time in seconds (log scale; lower is better)")
    ax.grid(axis="x", which="both", color="#dddddd", linewidth=.5); ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=8)
    fig.text(.075, .115,
             "Faint dots are individual processes; diamonds show each workload's three-process median. Search budgets differ.\n"
             "Native time includes autotuning and reference fallback probes. PPO time includes its eager reset and eight queries.\n"
             "Times include unequal logging overhead and shared compilation caches; they measure this instrumented harness.\n"
             "Offline PPO training and independent final confirmations are excluded.",
             fontsize=8.1, va="top", linespacing=1.5)
    return finish(fig, output, "ppo-tuning-cost")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("replay", "live"), default="replay")
    p.add_argument("--analysis", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--source-copy", type=Path, help="Optional byte-identical copy of the supplied analysis")
    p.add_argument("--live-gpu-label", default="Live GPU cohort", help="Display label verified against the measured cohort")
    args = p.parse_args(argv)
    raw = args.analysis.read_bytes(); document = json.loads(raw)
    rows = validate(document) if args.kind == "replay" else validate_live(document)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.source_copy:
        args.source_copy.parent.mkdir(parents=True, exist_ok=True)
        if args.source_copy.exists() and args.source_copy.read_bytes() != raw:
            raise ValueError("refuse to replace another analysis")
        args.source_copy.write_bytes(raw)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "path", "pdf.fonttype": 42, "svg.hashsalt": "drivebench-ppo-v1"})
    paths = (plot_curves(document, rows, args.output) + plot_seed_spread(document, rows, args.output)
             if args.kind == "replay" else
             plot_live_speedup(document, rows, args.output, args.live_gpu_label)
             + plot_live_cost(document, rows, args.output, args.live_gpu_label))
    receipt = {"kind": f"ppo_{args.kind}_figure_export", "source_analysis_sha256": hashlib.sha256(raw).hexdigest(),
               "plotter_sha256": sha(__file__), "new_gpu_measurements_by_plotter": 0, "policy_execution_performed": False,
               "matplotlib_version": matplotlib.__version__,
               "outputs": [{"file": f.name, "bytes": f.stat().st_size, "sha256": sha(f)} for f in paths],
               "curves": "all five PPO seeds, arithmetic mean at each query; existing baseline curves unchanged" if args.kind == "replay"
                         else "within-process paired confirmation ratios and their task medians; separate search wall-time comparison",
               "intervals": "copied from six-shape-family conditional exploratory bootstrap in source analysis" if args.kind == "replay"
                            else "no intervals; all three process values and medians are shown"}
    receipt_name = "ppo-plot-receipt.json" if args.kind == "replay" else "ppo-live-plot-receipt.json"
    (args.output / receipt_name).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
