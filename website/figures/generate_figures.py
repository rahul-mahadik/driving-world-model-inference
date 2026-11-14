#!/usr/bin/env python3
"""Reproduce descriptive H100 figures from hash-pinned public V22 observations.

No inference, GPU, model API, provider operation, download, or data modification.
Requires matplotlib; the sealed numerical analyzer itself uses only stdlib.
The output directory must be new. Source and data pins are checked before import.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile


AUDIT = "results/2026-10-30-h100-tradeoff/numerical-audit"
INPUT_PINS = {
    "tools/analyze_h100_timing.py": "83f0561744010e14ccc6ed28563bf41060071c0280e9cd5c95bed8761e4d8130",
    AUDIT + "/manifest.json": "32a1f719be1c08b8db77328eacbcbdf1bff4a45aae9047673e4f7ad1518a8180",
    AUDIT + "/measurements.jsonl": "143426c18acfec990c5994402a10395e86cfae4f852f6f9049af6ad9d4395631",
    AUDIT + "/provenance.json": "19dd45113647abf627f223a5c04960e92c834bad3267a5028ea553ef8163459e",
    AUDIT + "/bootstrap-scene-indices.csv": "12f9adffde1b813534f879ee5cbd141e5ff50c6603d70c03f671d23ae6fca6e8",
    "results/2026-10-30-h100-tradeoff/summary.json": "20257029c1843100d73763ed79eb96fbc29125456385f8a311471f6f9bc7ccc8",
}
ROLES = ("upstream", "compiled-default")
LABELS = {"upstream": "Upstream eager", "compiled-default": "Compiled denoiser"}
COLORS = {"upstream": "#0072B2", "compiled-default": "#D55E00"}
MARKERS = {"upstream": "o", "compiled-default": "s"}
COMMON_CAPTION = (
    "Historical October 30, 2026 cohort: one NVIDIA H100 80 GB GPU and one fixed runtime. "
    "Each role has 200 measured requests from the same 50 scene groups, repeated four times; "
    "the 400 requests are not 400 independent scenes. Each request generates 24 future frames "
    "at 576 × 1024 with pretrained Vista and 50 denoising steps. Request latency includes "
    "input loading/preprocessing through host-ready frames and excludes initialization, "
    "compilation, warmups and output export. These figures describe performance and allocator "
    "memory only: held-out output-quality validation was not completed, and quality "
    "preservation is not established. No hardware cohorts are pooled."
)
CAPTIONS = {
    "h100-latency-ecdf": {
        "title": "Distribution of measured H100 request latency",
        "caption": (
            "Role-wise empirical cumulative distributions use every actual request, without "
            "smoothing, binning or resampling. Panel a uses a common 0–65 second axis; panels b "
            "and c explicitly zoom each role using different x-axis ranges. Dashed vertical "
            "lines mark medians computed with the historical linear-quantile convention. "
            "The curves are descriptive distributions, not confidence intervals. " + COMMON_CAPTION
        ),
        "alt": "Two empirical latency distributions: upstream median 57.338 seconds and compiled "
               "median 43.663 seconds. A common-axis overview is followed by separately labeled zooms. "
               "200 requests per role arise from 50 scenes repeated four times; output quality remains unvalidated.",
    },
    "h100-block-chronology": {
        "title": "Balanced block order and observed block medians",
        "caption": (
            "Each marker is the median of the 50 actual requests in its original chronological "
            "process block. The fixed role sequence is U,C,C,U,C,U,U,C, with U = upstream eager "
            "and C = compiled denoiser. The top panel uses a zero-based latency axis; the bottom "
            "panel shows each block median minus its own role's overall 200-request median, in "
            "milliseconds. The lower panel is a descriptive change of origin, not a speedup, "
            "noise bound or confidence interval. Block index denotes launch order, not elapsed "
            "wall-clock time. Four repeats of the same 50 scenes per role do not establish "
            "stationarity or cross-hardware generalization. " + COMMON_CAPTION
        ),
        "alt": "Eight chronological 50-request blocks follow U,C,C,U,C,U,U,C. Upstream block "
               "medians are about 57.3 seconds and compiled medians about 43.7 seconds. A separate "
               "panel displays millisecond deviations from each role's overall median; no uncertainty bars are shown.",
    },
    "h100-latency-memory": {
        "title": "Measured latency and allocated-memory tradeoff",
        "caption": (
            "Zero-based bars compare role-wise median latency and the maximum recorded CUDA "
            "allocated-memory peak across the 200 requests in each role. Allocated bytes are "
            "converted to GiB using 2^30 bytes per GiB; this is neither reserved allocator "
            "memory nor total GPU/system memory. Median latency is 23.85% lower while peak "
            "allocated memory is 7.68% higher for the compiled denoiser. No variance or error "
            "bars are inferred for memory. The separately recorded 31.34% sequential-throughput "
            "increase uses the sum of measured request latencies, not the reciprocal of the "
            "median, and is not a concurrent-serving claim. " + COMMON_CAPTION
        ),
        "alt": "Median latency falls from 57.338 to 43.663 seconds, while peak CUDA allocated "
               "memory rises from 52.21 to 56.22 GiB. Both bar axes start at zero. This is a "
               "performance–memory tradeoff with output quality unvalidated.",
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_inputs(repository: Path):
    for relative, expected in INPUT_PINS.items():
        path = repository / relative
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("Pinned public input differs: " + relative)
    spec = importlib.util.spec_from_file_location("sealed_h100_analysis", repository / "tools/analyze_h100_timing.py")
    analyzer = importlib.util.module_from_spec(spec)
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True  # The sealed source tree must stay read-only.
        spec.loader.exec_module(analyzer)
    finally:
        sys.dont_write_bytecode = previous_bytecode
    analysis = analyzer.analyze(repository / AUDIT)
    if analysis["observations"] != 400 or analysis["recorded_fields_checked"] != 48:
        raise ValueError("Unexpected historical analysis coverage")
    rows = [json.loads(line) for line in (repository / AUDIT / "measurements.jsonl").read_text().splitlines()]
    return analyzer, analysis, rows


def plot_data(analyzer, analysis, rows):
    comparison = analysis["comparison"]
    aggregates = []
    for role in ROLES:
        memory = comparison["measured_cuda_memory"][role]
        aggregates.append({
            "role": role, "requests": 200, "scene_groups": 50,
            "median_latency_s": comparison["latency_s"][role]["p50"],
            "p95_latency_s": comparison["latency_s"][role]["p95"],
            "peak_allocated_bytes": memory["peak_allocated_bytes"],
            "peak_allocated_gib": memory["peak_allocated_bytes"] / 2**30,
            "sequential_futures_per_s": comparison["throughput"][role]["futures_per_s"],
        })
    blocks = []
    for index, role in enumerate(analyzer.SCHEDULE):
        values = [row["latency_s"] for row in rows if row["block_index"] == index]
        median = analyzer.quantile(values, .5)
        blocks.append({"block_index": index, "role": role, "requests": len(values),
                       "median_latency_s": median,
                       "deviation_from_role_median_ms": 1000 * (median - comparison["latency_s"][role]["p50"])})
    ecdf = {}
    for role in ROLES:
        ordered = sorted((row for row in rows if row["role"] == role),
                         key=lambda row: (row["latency_s"], row["block_index"], row["request_index"]))
        ecdf[role] = [{"rank": index, "cumulative_fraction": index / 200,
                       **{name: row[name] for name in ("latency_s", "block_index", "request_index", "scene_index")}}
                      for index, row in enumerate(ordered, 1)]
    return {"schema_version": 1, "kind": "derived_plot_coordinates_from_actual_public_observations",
            "input_sha256": INPUT_PINS, "source_observations": 400, "scene_groups": 50,
            "repetitions_per_scene_per_role": 4, "physical_gpus": 1, "quality_preserving": False,
            "aggregates": aggregates, "blocks": blocks, "ecdf": ecdf,
            "relative_changes": {
                "median_latency": comparison["metrics"]["p50"]["relative_latency_change"],
                "allocated_memory": aggregates[1]["peak_allocated_bytes"] / aggregates[0]["peak_allocated_bytes"] - 1,
                "sequential_throughput": aggregates[1]["sequential_futures_per_s"] / aggregates[0]["sequential_futures_per_s"] - 1},
            "definitions": {"quantile": "Historical NumPy linear interpolation, reconstructed by pinned stdlib analyzer.",
                            "ecdf": "Right-continuous step function, cumulative count / 200 separately for each role.",
                            "memory": "Maximum CUDA allocated-memory peak over the role's observed requests; bytes / 2^30.",
                            "uncertainty": "No new uncertainty estimates or uncertainty bands are plotted."}}


def configure(plt):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                        "axes.titlesize": 11, "axes.labelsize": 10,
                        "axes.edgecolor": "#9CA3AF", "axes.linewidth": .7,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": .6,
                        "axes.axisbelow": True, "xtick.color": "#374151", "ytick.color": "#374151",
                        "text.color": "#17212B", "axes.labelcolor": "#17212B",
                        "figure.facecolor": "white", "savefig.facecolor": "white",
                        "svg.hashsalt": "drivebench-h100-v23-figures", "pdf.fonttype": 42,
                        "svg.fonttype": "path", "path.simplify": False})


def heading(fig, title, subtitle):
    fig.text(.075, .965, title, ha="left", va="top", fontsize=16, weight="semibold")
    fig.text(.075, .911, subtitle, ha="left", va="top", fontsize=9.5, color="#58616D")


def footnote(fig, text):
    fig.text(.075, .035, text, ha="left", va="bottom", fontsize=8.5, color="#58616D")


def save(fig, output, stem, plt):
    for extension in ("svg", "png", "pdf"):
        meta = {"Creator": "DriveBench historical H100 figure generator", "Date": None,
                "Description": CAPTIONS[stem]["caption"]}
        if extension == "png":
            meta = {"Software": "DriveBench historical H100 figure generator", "Description": CAPTIONS[stem]["caption"]}
        if extension == "pdf":
            meta = {"Creator": "DriveBench historical H100 figure generator", "CreationDate": None, "ModDate": None,
                    "Subject": CAPTIONS[stem]["caption"]}
        fig.savefig(output / (stem + "." + extension), dpi=220, metadata=meta)
    plt.close(fig)


def ecdf_fig(data, output, plt):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 5.0), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    fig.subplots_adjust(left=.075, right=.98, bottom=.23, top=.73, wspace=.34)
    heading(fig, "H100 request-latency distribution", "200 requests per role · 50 scene groups × 4 repeats · one GPU/runtime cohort")
    overview = axes[0]
    for role in ROLES:
        points = data["ecdf"][role];x = [p["latency_s"] for p in points];y = [p["cumulative_fraction"] for p in points]
        overview.step([0] + x + [65], [0] + y + [1], where="post", color=COLORS[role], linewidth=2, label=LABELS[role])
    overview.set(xlim=(0, 65), ylim=(0, 1.035), xlabel="Request latency (s)", ylabel="Cumulative fraction of requests")
    overview.set_title("a  Common-axis comparison", loc="left", pad=13)
    overview.legend(loc="upper left", frameon=False, fontsize=9)
    for ax, role, letter, aggregate in zip(axes[1:], ROLES, ("b", "c"), data["aggregates"]):
        points=data["ecdf"][role];x=[p["latency_s"] for p in points];y=[p["cumulative_fraction"] for p in points]
        lower=math.floor(min(x)*100)/100-.01;upper=math.ceil(max(x)*100)/100+.01
        ax.step([lower]+x+[upper],[0]+y+[1],where="post",color=COLORS[role],linewidth=2)
        ax.axvline(aggregate["median_latency_s"],color=COLORS[role],linestyle="--",linewidth=1)
        ax.set(xlim=(lower,upper),ylim=(0,1.035),xlabel="Request latency (s)")
        ax.set_title(letter+"  "+LABELS[role]+" (zoom)",loc="left",pad=13)
        from matplotlib.ticker import MaxNLocator, FormatStrFormatter
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.text(.04,.95,f"Median {aggregate['median_latency_s']:.3f} s\np95 {aggregate['p95_latency_s']:.3f} s",transform=ax.transAxes,ha="left",va="top",fontsize=8.8,bbox={"facecolor":"white","edgecolor":"none","alpha":.93})
    fig.text(.54,.125,"Zoom panels use different x-axis ranges; no smoothing or confidence bands.",ha="center",fontsize=9,color="#58616D")
    footnote(fig,"Historical performance only; output quality unvalidated. Latency excludes initialization, compilation, warmups and export.")
    save(fig,output,"h100-latency-ecdf",plt)


def chronology_fig(data, output, plt):
    fig, axes=plt.subplots(2,1,figsize=(10.4,6.2),sharex=True,gridspec_kw={"height_ratios":[1.4,1]})
    fig.subplots_adjust(left=.105,right=.975,bottom=.19,top=.78,hspace=.29)
    heading(fig,"Balanced block order and observed medians","Eight chronological blocks · 50 requests per block · same 50 scenes in each block")
    for role in ROLES:
        blocks=[b for b in data["blocks"] if b["role"]==role];x=[b["block_index"] for b in blocks]
        y=[b["median_latency_s"] for b in blocks];delta=[b["deviation_from_role_median_ms"] for b in blocks]
        axes[0].scatter(x,y,color=COLORS[role],marker=MARKERS[role],s=48,label=LABELS[role],zorder=3)
        axes[1].scatter(x,delta,color=COLORS[role],marker=MARKERS[role],s=42,zorder=3)
        for index,value in zip(x,y):axes[0].annotate(f"{value:.3f}",(index,value),xytext=(0,8),textcoords="offset points",ha="center",fontsize=8.5,color=COLORS[role])
    axes[0].set(ylim=(0,65),ylabel="Block median latency (s)")
    axes[0].set_title("a  Absolute latency",loc="left",fontsize=10,pad=10)
    axes[0].legend(loc="lower left",frameon=False,fontsize=9,ncol=2)
    axes[1].axhline(0,color="#7D8590",linewidth=.8)
    bound=max(10,math.ceil(max(abs(b["deviation_from_role_median_ms"]) for b in data["blocks"])/10)*10+10)
    axes[1].set(ylim=(-bound,bound),ylabel="Deviation from own\nrole median (ms)",xlabel="Chronological process-block index and role")
    axes[1].set_title("b  Descriptive within-role deviations; not a confidence interval",loc="left",fontsize=10,pad=10)
    axes[1].set_xticks(range(8),[str(b["block_index"])+"\n"+("U" if b["role"]=="upstream" else "C") for b in data["blocks"]])
    axes[1].set_xlim(-.5,7.5)
    footnote(fig,"U: upstream eager; C: compiled denoiser. Block index is launch order, not elapsed time. No stationarity or quality-preservation claim.")
    save(fig,output,"h100-block-chronology",plt)


def tradeoff_fig(data, output, plt):
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.7))
    fig.subplots_adjust(left=.18,right=.975,bottom=.25,top=.72,wspace=.28)
    heading(fig,"Latency–memory tradeoff on one H100","200 requests per role · 50 scene groups × 4 repeats · unchanged weights and 50 steps")
    for ax,field,title,xlabel,change in [
        (axes[0],"median_latency_s","a  Median request latency","Seconds",f"{data['relative_changes']['median_latency']*100:.2f}% median latency"),
        (axes[1],"peak_allocated_gib","b  Peak CUDA allocated memory","GiB (bytes / 2³⁰)",f"+{data['relative_changes']['allocated_memory']*100:.2f}% allocated memory")]:
        for index,row in enumerate(data["aggregates"]):
            value=row[field];ax.barh(index,value,height=.48,color=COLORS[row['role']],alpha=.92)
            label=f"{value:.3f} s" if field=="median_latency_s" else f"{value:.2f} GiB"
            ax.text(value+.8,index,label,va="center",fontsize=10)
        ax.set(xlim=(0,70),ylim=(1.7,-.7),xlabel=xlabel)
        ax.set_yticks([0,1],[LABELS[r] for r in ROLES] if ax is axes[0] else ["",""])
        ax.set_xticks([0,20,40,60]);ax.grid(axis="y",visible=False)
        ax.set_title(title,loc="left",pad=15)
        ax.text(0,-.28,change,transform=ax.transAxes,fontsize=11,weight="semibold")
    footnote(fig,"Allocated memory is not total device memory. Latency excludes initialization, compilation, warmups and export. Output quality unvalidated.")
    save(fig,output,"h100-latency-memory",plt)


def run(repository: Path, output: Path):
    analyzer,analysis,rows=load_inputs(repository)
    if output.exists() or output.is_symlink():raise FileExistsError("Use a new output directory; earlier artifacts are preserved")
    output.mkdir(parents=True)
    data=plot_data(analyzer,analysis,rows)
    write_json(output/"plot-data.json",data)
    write_json(output/"captions.json",CAPTIONS)
    with (output/"plot-data.csv").open("x",newline="") as stream:
        fields=["record_kind","role","block_index","requests","median_latency_s","p95_latency_s","deviation_from_role_median_ms","peak_allocated_bytes","peak_allocated_gib","sequential_futures_per_s"]
        writer=csv.DictWriter(stream,fields,extrasaction="ignore");writer.writeheader()
        for row in data["aggregates"]:writer.writerow({"record_kind":"role_aggregate",**row})
        for row in data["blocks"]:writer.writerow({"record_kind":"chronological_block",**row})
    with tempfile.TemporaryDirectory(prefix=".matplotlib-cache-",dir=output.parent) as cache:
        os.environ["MPLCONFIGDIR"]=cache
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        configure(plt)
        ecdf_fig(data,output,plt);chronology_fig(data,output,plt);tradeoff_fig(data,output,plt)
    for relative,expected in INPUT_PINS.items():
        if sha(repository/relative)!=expected:raise ValueError("Source input changed during plotting")
    report={"schema_version":1,"status":"historical_figures_reproduced","source_commit":"da39fd8e40a0d337e392fff1006920c7f6608676",
            "generator_sha256":sha(Path(__file__)),"input_sha256":INPUT_PINS,
            "recorded_fields_reconstructed":analysis["recorded_fields_checked"],"source_rows":400,
            "runtime":{"python":platform.python_version(),"matplotlib":metadata.version("matplotlib"),"numpy":metadata.version("numpy"),"pillow":metadata.version("pillow")},
            "plotting_rules":{"ecdf_no_smoothing":True,"bar_axes_start_at_zero":True,"new_uncertainty_bands":False,"pooled_hardware":False,"quality_preservation_claim":False},
            "files":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output/"figure-manifest.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository",required=True,type=Path,help="V22 public repository tree containing the pinned historical inputs")
    parser.add_argument("--output",required=True,type=Path,help="Fresh artifact directory")
    args=parser.parse_args();report=run(args.repository.resolve(),args.output.resolve())
    print(json.dumps({"status":report["status"],"source_rows":400,"recorded_fields_reconstructed":report["recorded_fields_reconstructed"],"figures":3,"formats":["svg","png","pdf"]}))


if __name__=="__main__":main()
