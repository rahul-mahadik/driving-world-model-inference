"""Generate an offline HTML report from actual benchmark artifacts only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any
import html
import json
import os

import numpy as np

from .evaluation.io import Run, artifact_sha256, find_runs, get_nested, load_run, read_json, write_json


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}" if np.isfinite(value) else "unavailable"
    return html.escape(str(value))


def measured_summary(run: Run) -> dict[str, Any]:
    good = [r for r in run.measurements if r.get("status", "ok") in {"ok", "success"}
            and isinstance(r.get("latency_s"), (float, int)) and np.isfinite(r["latency_s"]) and r["latency_s"] > 0]
    latencies = [r["latency_s"] for r in good]
    stages: dict[str, list[float]] = defaultdict(list)
    for row in good:
        for name, value in row.get("stage_timings_s", {}).items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                stages[name].append(value)
    futures = sum(len(r.get("candidates", [])) or int(r.get("candidate_count", 0)) for r in good)
    cold_file = run.path / "cold_start.json"
    cold = read_json(cold_file) if cold_file.is_file() else {}
    return {"run": run.name, "synthetic": run.synthetic, "requests": len(good),
            "failures": len(run.measurements) - len(good),
            "p50_s": float(np.median(latencies)) if latencies else None,
            "p95_s": float(np.quantile(latencies, 0.95)) if len(latencies) >= 20 else None,
            "p50_ci95_s": run.summary.get("latency_p50_ci95_s"),
            "p95_ci95_s": run.summary.get("latency_p95_ci95_s") if len(latencies) >= 20 else None,
            "throughput_futures_s": run.summary.get("throughput_futures_s"),
            "throughput_includes_export": run.summary.get("throughput_includes_export"),
            "futures_per_request_second": futures / sum(latencies) if latencies else None,
            "peak_allocated_gib": max((r.get("peak_allocated_bytes", 0) for r in good), default=0) / 2**30,
            "peak_reserved_gib": max((r.get("peak_reserved_bytes", 0) for r in good), default=0) / 2**30,
            "denoiser_evaluations": sum(r.get("denoiser_evaluations", 0) for r in good),
            "candidate_counts": sorted({r.get("candidate_count", 1) for r in good}),
            "batch_sizes": sorted({r.get("batch_size", 1) for r in good}),
            "headline_eligible": bool(run.summary.get("headline_eligible")) and len(good) >= 200
            and not run.synthetic and len(good) == len(run.measurements),
            "stage_means_s": {name: float(np.mean(values)) for name, values in stages.items()},
            "model_load_s": cold.get("model_load_s"),
            "warmup_request_latencies_s": [row["latency_s"] for row in cold.get("warmups", []) if "latency_s" in row],
            "compilation_s": run.summary.get("compilation_s")}


def _chart(points: list[dict], metric: str, title: str) -> str:
    present = [p for p in points if p.get("p50_s") is not None and p.get(metric) is not None]
    if not present:
        return f"<h3>{html.escape(title)}</h3><p>No measured paired values are available.</p>"
    max_x = max(p["p50_s"] for p in present) * 1.12 or 1
    max_y = max(p[metric] for p in present) * 1.2 or 0.001
    body = ['<svg viewBox="0 0 900 370" role="img" aria-label="' + html.escape(title, quote=True) + '">',
            '<path d="M75 25V315H865" fill="none" stroke="#94a3b8"/>']
    for tick in range(6):
        x, y = 75 + 790 * tick / 5, 315 - 290 * tick / 5
        body.append(f'<path d="M{x:.1f} 315v5 M70 {y:.1f}h5" stroke="#94a3b8"/>')
        body.append(f'<text x="{x:.1f}" y="337" text-anchor="middle">{max_x*tick/5:.2f}</text>')
        body.append(f'<text x="65" y="{y+4:.1f}" text-anchor="end">{max_y*tick/5:.4f}</text>')
    for point in present:
        x, y = 75 + point["p50_s"] / max_x * 790, 315 - point[metric] / max_y * 290
        color = "#f59e0b" if point["synthetic"] else "#34d399" if point.get("quality_preserving") else "#60a5fa"
        name = html.escape(point["run"])
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"><title>{name}: '
                    f'{point["p50_s"]:.3f}s, {point[metric]:.5f}</title></circle>')
        body.append(f'<text x="{min(x+10, 760):.1f}" y="{max(y-10, 17):.1f}">{name}</text>')
    body.append('<text x="470" y="365" text-anchor="middle">Median request latency (seconds; lower is faster)</text></svg>')
    return f'<h3>{html.escape(title)}</h3>' + "".join(body)


def build_report(runs_path: str | Path, *, evaluation_path: str | Path | None = None,
                 output: str | Path | None = None) -> Path:
    runs = find_runs(runs_path)
    target = Path(output) if output else Path(runs_path) / "report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    evaluation_file = Path(evaluation_path) if evaluation_path else Path(runs_path) / "evaluation.json"
    evaluation = read_json(evaluation_file) if evaluation_file.is_file() else {"runs": [], "notes": []}
    evaluated = {entry["run"]: entry for entry in evaluation["runs"]}
    invalid_evaluations = []
    reference_valid = False
    if evaluation.get("reference"):
        try:
            reference_valid = artifact_sha256(load_run(evaluation["reference"])) == evaluation.get("reference_artifact_sha256")
        except (OSError, ValueError):
            pass
    for run in runs:
        entry = evaluated.get(run.name)
        if entry and (not reference_valid or entry.get("run_path") != str(run.path)
                      or entry.get("artifact_sha256") != artifact_sha256(run)):
            invalid_evaluations.append(run.name)
            del evaluated[run.name]
    summaries = [measured_summary(run) for run in runs]
    warnings = []
    if invalid_evaluations:
        warnings.append('<p class="warning">Quality results are stale or lack artifact fingerprints for: '
                        + html.escape(", ".join(invalid_evaluations)) + '. Run evaluation again before using these results.</p>')
    if any(run.synthetic for run in runs):
        warnings.append('<div class="warning"><strong>SYNTHETIC FUNCTIONAL TEST DATA PRESENT.</strong> '
                        'Synthetic timings and images do not measure Vista, NVIDIA GPU performance, or driving quality.</div>')
    if not all(s["headline_eligible"] for s in summaries):
        warnings.append('<p class="warning">Runs without at least 200 successful measured requests and a verified real backend '
                        'are exploratory. No validated performance improvement is claimed here.</p>')
    table_rows, quality_rows, points, stage_rows, cold_rows = [], [], [], [], []
    for run, summary in zip(runs, summaries):
        entry = evaluated.get(run.name, {})
        gate = entry.get("gate", {})
        badge = '<span class="badge">SYNTHETIC</span>' if run.synthetic else '<span class="badge real">VISTA</span>'
        ci = summary["p50_ci95_s"]
        ci_text = f" [{_fmt(ci[0])}, {_fmt(ci[1])}]" if isinstance(ci, list) and len(ci) == 2 else " [CI unavailable]"
        table_rows.append('<tr><td>' + html.escape(run.name) + ' ' + badge + '</td>' + ''.join('<td>' + val + '</td>' for val in [
            str(summary["requests"]), str(summary["failures"]), _fmt(summary["p50_s"]) + ci_text,
            _fmt(summary["p95_s"]), _fmt(summary["throughput_futures_s"]),
            "yes" if summary["throughput_includes_export"] is True else "no" if summary["throughput_includes_export"] is False else "unknown",
            _fmt(summary["peak_allocated_gib"]), _fmt(summary["peak_reserved_gib"]),
            html.escape(str(summary["candidate_counts"])), html.escape(str(summary["batch_sizes"])),
            str(summary["denoiser_evaluations"]), "yes" if summary["headline_eligible"] else "no"]) + '</tr>')
        aggregates = entry.get("aggregates", {})
        rmse, lpips, frozen = [aggregates.get(metric, {}).get("mean") for metric in ("rmse", "lpips", "freeze_fraction_increase")]
        reason = "; ".join(gate.get("reasons", [])) if entry else "Reference run or evaluation not performed"
        quality_rows.append('<tr>' + ''.join('<td>' + text + '</td>' for text in [html.escape(run.name),
            str(entry.get("matched_pairs", 0)), _fmt(rmse, 6), _fmt(lpips, 6), _fmt(frozen, 6),
            "yes" if gate.get("quality_preserving") else "not established", html.escape(reason)]) + '</tr>')
        points.append({**summary, "rmse": rmse, "lpips": lpips, "quality_preserving": gate.get("quality_preserving", False)})
        for stage, value in summary["stage_means_s"].items():
            stage_rows.append(f'<tr><td>{html.escape(run.name)}</td><td>{html.escape(stage)}</td><td>{value:.6f}</td></tr>')
        cold_rows.append('<tr>' + ''.join('<td>' + value + '</td>' for value in [html.escape(run.name),
            _fmt(summary["model_load_s"]), ", ".join(_fmt(x) for x in summary["warmup_request_latencies_s"]) or "unavailable",
            _fmt(summary["compilation_s"])]) + '</tr>')
    review_link = ""
    review = evaluation.get("review", {})
    if review.get("index"):
        relative = os.path.relpath(review["index"], target.parent)
        review_link = '<p><a href="' + html.escape(relative, quote=True) + '">Open blinded visual comparisons and review CSV</a></p>'
    notes = evaluation.get("notes", [])
    html_text = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Driving World Model Inference Optimization</title><style>
body{font:15px/1.6 system-ui;background:#0b1220;color:#e2e8f0;margin:32px auto;max-width:1500px;padding:0 24px}
h1{font-size:32px}h2{margin-top:42px}h3{font-size:18px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #334155}th{color:#a5b4fc}
.scroll{overflow:auto}.warning{background:#452b0c;padding:16px;border-left:4px solid #f59e0b;margin:18px 0}
.badge{background:#92400e;padding:3px 6px;border-radius:4px;font-size:11px;white-space:nowrap}.real{background:#065f46}
svg{max-width:100%;background:#111827}svg text{fill:#cbd5e1;font:12px system-ui}a{color:#93c5fd}code{color:#c4b5fd}
</style></head><body><h1>Driving World Model Inference Optimization</h1>
<p>Measured execution, reference-output agreement, and diagnostic quality comparisons. No physical driving accuracy is established.</p>'''
    html_text += "".join(warnings)
    html_text += '<h2>Performance</h2><div class="scroll"><table><thead><tr>' + ''.join(f'<th>{x}</th>' for x in [
        "Run", "Requests", "Failures", "p50 s [95% CI]", "p95 s", "Futures/second", "Export included?", "Allocated GiB", "Reserved GiB",
        "Candidates", "Batch sizes", "Denoiser calls", "Headline eligible"]) + '</tr></thead><tbody>' + ''.join(table_rows) + '</tbody></table></div>'
    html_text += '<p>Throughput is the benchmark runner’s measured completed-futures rate; the export column shows whether export is included. '
    html_text += 'Continuous prefetched workloads use elapsed wall time, preserving overlap. A separate futures/request-second diagnostic is saved in report.data.json. Candidate count and batch size are separate. '
    html_text += 'p95 is withheld below 20 successful requests; headline eligibility requires at least 200 and the benchmark runner’s checks. '
    html_text += 'Compilation, cold start, and export are separate from warm request measurements. Inspect raw summaries for those costs.</p>'
    html_text += '<h2>Performance–quality observations</h2><p>Points are individual measured configurations; isolated speedups are never multiplied. '
    html_text += 'Unaccepted and out-of-tolerance results remain visible. Orange points are synthetic tests.</p>'
    html_text += _chart(points, "lpips", "Perceptual distance from reference (LPIPS; lower is closer)")
    html_text += _chart(points, "rmse", "Numerical RMSE from reference (lower is closer; not a perceptual quality score)")
    html_text += '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{x}</th>' for x in [
        "Run", "Paired futures", "RMSE", "LPIPS", "Freeze increase", "Quality-preserving", "Gate status"]) + '</tr></thead><tbody>'
    html_text += ''.join(quality_rows) + '</tbody></table></div>' + review_link
    html_text += '<h2>Stage measurements</h2><p>Means of recorded stage durations. Overlapping stage durations must not be added to infer wall time.</p>'
    html_text += '<table><thead><tr><th>Run</th><th>Stage</th><th>Mean seconds</th></tr></thead><tbody>' + ''.join(stage_rows) + '</tbody></table>'
    html_text += '<h2>Cold start and initialization</h2><p>Warmup requests include lazy compilation and initialization. '
    html_text += 'A compile-only duration is shown only when separately measured.</p><table><thead><tr>'
    html_text += '<th>Run</th><th>Model loading seconds</th><th>Warmup request seconds (in order)</th><th>Compile-only seconds</th>'
    html_text += '</tr></thead><tbody>' + ''.join(cold_rows) + '</tbody></table>'
    html_text += '<h2>Evaluation notes</h2><ul>' + ''.join('<li>' + html.escape(note) + '</li>' for note in notes) + '</ul>'
    html_text += '<p>Environment, configuration snapshots, raw measurements, failed configurations, and gate details remain in each run directory and evaluation.json.</p></body></html>'
    target.write_text(html_text)
    write_json(target.with_suffix(".data.json"), {"runs": summaries, "evaluated": bool(evaluation["runs"])})
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--evaluation")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(build_report(args.runs, evaluation_path=args.evaluation, output=args.output))


if __name__ == "__main__":
    main()
