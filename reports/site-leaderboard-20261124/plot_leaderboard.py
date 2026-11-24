#!/usr/bin/env python3
"""Render two compact figures from leaderboard.json; requires Matplotlib."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("leaderboard.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("figures"))
    args = parser.parse_args()
    raw = args.data.read_bytes()
    data = json.loads(raw)
    if data["fresh_gpu_measurements"] != 0 or data["bootstrap"]["shape_families"] != 6:
        raise ValueError("Unexpected source summary")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.spines.bottom": False,
                         "axes.labelcolor": "#354052", "text.color": "#243044",
                         "xtick.color": "#586375", "ytick.color": "#586375",
                         "svg.fonttype": "none", "svg.hashsalt": "drivebench-site-20261124",
                         "savefig.facecolor": "white"})
    args.output.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for suffix in ("svg", "png"):
            path = args.output / f"{name}.{suffix}"
            if path.exists():
                raise FileExistsError(path)
            metadata = {"Date": None} if suffix == "svg" else {}
            fig.savefig(path, dpi=180, bbox_inches="tight", metadata=metadata)
            if suffix == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)

    rows = {r["policy"]: r for r in next(p for p in data["panels"] if p["id"] == "replays-20")["rows"]}
    styles = [("fixed-compiled", "Compile + eager fallback", "#243a56", "--"),
              ("ppo-five-seeds", "PPO (five-seed mean)", "#25828a", "-"),
              ("optuna-tpe", "TPE", "#bb7049", "-"),
              ("random", "Random", "#8a91a0", "-")]
    fig, ax = plt.subplots(figsize=(7.4, 3.45))
    for policy, label, color, linestyle in styles:
        curve = rows[policy]["curve"]
        ax.plot([r["queries"] for r in curve], [r["mean_regret"] for r in curve],
                label=label, color=color, linestyle=linestyle, linewidth=2,
                marker="o" if policy != "fixed-compiled" else None, markersize=3)
    ax.set(xlabel="Queries", ylabel="Mean regret", xlim=(.85, 8.15), ylim=(0, None))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color="#e5e8eb", linewidth=.7)
    ax.tick_params(length=0)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout(pad=1.0)
    save(fig, "search-regret")

    transfer = data["transfer"]["rows"]
    transfer = sorted(transfer, key=lambda row: row["mean_regret"])
    labels = {"fixed-compiled": "Compile + eager fallback", "random": "Random",
              "optuna-tpe": "TPE", "ridge-selector": "Ridge selector",
              "gpt-5.4-mini-2026-03-17": "GPT-5.4 mini", "gpt-5.4-nano-2026-03-17": "GPT-5.4 nano",
              "Source-table optimum": "Source-table optimum"}
    fig, ax = plt.subplots(figsize=(7.4, 3.45))
    values = [row["mean_regret"] for row in transfer]
    ax.barh(range(len(transfer)), values, height=.58,
            color=["#25828a" if row["policy"] == "Source-table optimum" else "#b6c5d2" for row in transfer])
    for y, value in enumerate(values):
        ax.text(value + .003, y, f"{value:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(transfer)), [labels[row["policy"]] for row in transfer])
    ax.invert_yaxis()
    ax.set(xlabel="Mean regret on RTX 4090", xlim=(0, max(values) * 1.17))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="x", color="#e5e8eb", linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    fig.tight_layout(pad=1.0)
    save(fig, "transfer-regret")
    receipt = {"kind": "derived_site_figures", "new_gpu_measurements": 0,
               "data_sha256": hashlib.sha256(raw).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(args.output.iterdir()) if p.suffix in {".svg", ".png"}}}
    with (args.output / "manifest.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
