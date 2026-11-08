#!/usr/bin/env python3
"""Bound unchanged study quality orchestration and preserve blind native frames.

No generation implementation is replaced. Hooks exist only in this process and
restore in finally. The program reads private review mappings without printing
them; reviewers must inspect only each export bundle's public/ directory.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivebench import study
from drivebench.config import digest, file_sha256
from drivebench.evaluation.metrics import as_video


TYPE = "bounded_quality_native_review_v1"
EXPORT = "blind_native_frame_export_v1"


class IntentionalPause(Exception):
    """No new unit was started; unchanged-plan resume is permitted."""


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def parse_deadline(value):
    date = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None or date.utcoffset() != dt.timedelta(0):
        raise ValueError("Deadline must specify UTC explicitly")
    return date


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def private_json(path, value, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)


def checked_file(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
        raise ValueError("Bound review artifact is missing or changed")
    return path


def _bundle_inputs(metric_path, metric_receipt, registry):
    """Read mappings privately and bind them to original immutable receipts."""
    metric_path = Path(metric_path).resolve()
    proof = json.loads(Path(metric_receipt).read_text())
    checked_file(metric_path, proof["evaluation_sha256"])
    result = json.loads(metric_path.read_text())
    if len(result["runs"]) != 1 or result["runs"][0]["complete_pairs"] is not True:
        raise ValueError("Native export requires one complete paired evaluation")
    review = result["review"]
    review_files = proof["review_artifact_sha256"]
    for path, expected in review_files.items():
        checked_file(path, expected)
    key_path = Path(review["private_key"]).resolve()
    if str(key_path) not in review_files:
        raise ValueError("Private mapping is not covered by the metric receipt")
    cases = json.loads(key_path.read_text())["cases"]
    if not cases or len(cases) != review["cases"]:
        raise ValueError("Review mapping has missing cases")
    seen = set()
    sources = {}
    for case in cases:
        case_id = case["case_id"]
        if not re.fullmatch(r"case-[0-9a-f]{24}", case_id) or case_id in seen:
            raise ValueError("Invalid or duplicate opaque review case")
        seen.add(case_id)
        if ({case["A"], case["B"]} != {case["run"], case["reference_run"]}
                or case["run"] == case["reference_run"]):
            raise ValueError("Invalid blind panel mapping")
        count = case["conditioning_frames"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Invalid conditioning frame count")
        for role in ("reference", "candidate"):
            path = str(Path(case[f"{role}_path"]).resolve())
            expected = case[f"{role}_sha256"]
            if registry.get(path) != expected:
                raise ValueError("Review array does not match committed run output hashes")
            sources[path] = expected
    binding = {"metric": str(metric_path), "metric_sha256": file_sha256(metric_path),
               "metric_receipt": str(Path(metric_receipt).resolve()),
               "metric_receipt_sha256": file_sha256(metric_receipt),
               "review_artifact_sha256": review_files, "source_arrays_sha256": sources,
               "wrapper_sha256": file_sha256(__file__)}
    return cases, binding, review


def validate_export(bundle, binding):
    bundle = Path(bundle)
    private = bundle / "private-receipt.json"
    if private.is_symlink() or not private.is_file() or private.stat().st_mode & 0o077:
        raise ValueError("Native export private receipt is missing or not private")
    receipt = json.loads(private.read_text())
    if receipt.get("receipt_type") != EXPORT or receipt.get("binding") != binding:
        raise ValueError("Native export provenance changed")
    public = bundle / "public"
    expected_files = receipt["public_artifact_sha256"]
    actual_files = {str(p.relative_to(public)) for p in public.rglob("*") if p.is_file()}
    if actual_files != set(expected_files):
        raise ValueError("Native public export file set changed")
    for name, expected in expected_files.items():
        path = public / name
        if not path.resolve().is_relative_to(public.resolve()):
            raise ValueError("Native export path escapes its public directory")
        checked_file(path, expected)
    # Missing sources are permissible only after this complete export was sealed
    # and a prior authorized retirement removed them. Existing sources rehash.
    for name, expected in binding["source_arrays_sha256"].items():
        path = Path(name)
        if path.exists() or path.is_symlink():
            checked_file(path, expected)
    return receipt


def _index(cases):
    intro = ('<!doctype html><html lang="en"><meta charset="utf-8">'
             '<title>Blind native-resolution frame review</title>'
             '<style>body{font:16px system-ui;background:#111827;color:#eee;margin:24px}'
             'a{color:#93c5fd}img{max-width:none;height:auto}section{overflow:auto;margin:24px 0}'
             '.grid{display:grid;grid-template-columns:repeat(3,max-content);gap:16px}</style>'
             '<h1>Blind native-resolution ordered-frame review</h1>'
             '<p>A/B labels match the original review. All future frames are preserved in order. '
             'Display pixels use round(value × 255), then uint8, with no resizing or palette quantization. '
             'PNG encoding is lossless for these display pixels, not for the original float arrays. '
             'Native-size panels may require horizontal scrolling. Static ordered-frame inspection is '
             'not real-time video playback or a driving-safety assessment.</p>')
    return intro + ''.join(cases) + '</html>'


def export_comparison(metric, metric_receipt, registry, destination, *, minimum_free_bytes=0, admission=None):
    cases, binding, review = _bundle_inputs(metric, metric_receipt, registry)
    token = digest({"metric": binding["metric_sha256"], "wrapper": binding["wrapper_sha256"]})[:24]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    bundle = destination / token
    if bundle.exists():
        validate_export(bundle, binding)
        return bundle
    if admission is not None:
        admission.check(admission.export_seconds, "new native review export")
    arrays = {}
    for name, expected in binding["source_arrays_sha256"].items():
        checked_file(name, expected)
        arrays[name] = np.load(name, mmap_mode="r", allow_pickle=False)
    upper_bytes = 0
    for case in cases:
        a = arrays[str(Path(case["reference_path"]).resolve())]
        b = arrays[str(Path(case["candidate_path"]).resolve())]
        if (a.shape != b.shape or a.ndim != 4 or a.shape[-1] != 3
                or case["conditioning_frames"] >= len(a)):
            raise ValueError("Native review source shapes differ or have no future frames")
        upper_bytes += (len(a) - case["conditioning_frames"]) * (a.shape[1] + 28) * a.shape[2] * 2 * 3
    # zlib framing/PNG metadata and labels are bounded conservatively; reserve
    # also protects subsequent core retirement and metadata writes.
    required = math.ceil(upper_bytes * 1.02) + 1024 * 1024
    if shutil.disk_usage(destination).free < minimum_free_bytes + required:
        raise RuntimeError("Insufficient disk reserve for native review preservation")
    staging = Path(tempfile.mkdtemp(prefix=".native-export-", dir=destination))
    os.chmod(staging, 0o700)
    public = staging / "public"
    public.mkdir()
    public_cases, links = [], []
    try:
        for case in cases:
            case_id = case["case_id"]
            reference = arrays[str(Path(case["reference_path"]).resolve())]
            candidate = arrays[str(Path(case["candidate_path"]).resolve())]
            a, b = (candidate, reference) if case["A"] == case["run"] else (reference, candidate)
            frames = []
            offset = case["conditioning_frames"]
            for i in range(offset, len(a)):
                frame_a = np.rint(as_video(a[i:i + 1])[0] * 255).astype(np.uint8)
                frame_b = np.rint(as_video(b[i:i + 1])[0] * 255).astype(np.uint8)
                h, w, _ = frame_a.shape
                image = Image.new("RGB", (2 * w, h + 28), "#111827")
                draw = ImageDraw.Draw(image)
                number = i - offset + 1
                draw.text((8, 8), f"A | Future frame {number}", fill="white")
                draw.text((w + 8, 8), f"B | Future frame {number}", fill="white")
                image.paste(Image.fromarray(frame_a), (0, 28))
                image.paste(Image.fromarray(frame_b), (w, 28))
                name = f"{case_id}-frame-{number:04d}.png"
                image.save(public / name)
                frames.append(name)
            pages = []
            for start in range(0, len(frames), 12):
                page = f"{case_id}-page-{start // 12 + 1}.html"
                panel = '<h2>' + html.escape(case_id) + '</h2><div class="grid">' + ''.join(
                    f'<figure><a href="{name}"><img loading="lazy" src="{name}" alt="A/B future frame {j + 1}"></a>'
                    f'<figcaption>Future frame {j + 1}</figcaption></figure>'
                    for j, name in enumerate(frames[start:start + 12], start)) + '</div>'
                (public / page).write_text(_index([panel]))
                pages.append(page)
            links.append('<section><h2>' + case_id + '</h2><p>' + ' | '.join(
                f'<a href="{page}">Frames {j * 12 + 1}–{min((j + 1) * 12, len(frames))}</a>'
                for j, page in enumerate(pages)) + '</p></section>')
            public_cases.append({"case_id": case_id, "future_frames": len(frames),
                                 "panel_width": w, "panel_height": h, "frames": frames, "pages": pages})
        (public / "index.html").write_text(_index(links))
        shutil.copyfile(review["csv"], public / "review.csv")
        public_manifest = {"schema_version": 1, "receipt_type": EXPORT, "cases": public_cases,
            "conversion": "np.rint(float32_in_0_1 * 255).astype(uint8); no resize or palette quantization",
            "frame_sha256": {name: file_sha256(public / name) for case in public_cases for name in case["frames"]}}
        (public / "manifest.json").write_text(json.dumps(public_manifest, indent=2) + "\n")
        hashes = {str(p.relative_to(public)): file_sha256(p) for p in sorted(public.rglob("*")) if p.is_file()}
        # Recheck immutable inputs after rendering and before committing export.
        for name, expected in binding["source_arrays_sha256"].items():
            checked_file(name, expected)
        for name, expected in binding["review_artifact_sha256"].items():
            checked_file(name, expected)
        private_json(staging / "private-receipt.json", {"schema_version": 1, "receipt_type": EXPORT,
            "created_at": utc_now().isoformat(), "binding": binding, "public_artifact_sha256": hashes,
            "public_bytes": sum((public / name).stat().st_size for name in hashes)})
        study.sync_files([p for p in staging.rglob("*") if p.is_file()])
        staging.rename(bundle)
        fd = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        validate_export(bundle, binding)
        return bundle
    except BaseException:
        # Incomplete evidence stays private and cannot authorize retirement.
        raise


class Admission:
    def __init__(self, deadline, *, run_seconds, evaluation_seconds, export_seconds,
                 cleanup_seconds=240, max_new_runs=None, now=utc_now):
        self.deadline = parse_deadline(deadline)
        self.run_seconds = positive(run_seconds, "Run unit bound")
        self.evaluation_seconds = positive(evaluation_seconds, "Evaluation bound")
        self.export_seconds = positive(export_seconds, "Export bound")
        self.cleanup_seconds = positive(cleanup_seconds, "Cleanup margin")
        if cleanup_seconds < 240:
            raise ValueError("Cleanup margin must be at least 240 seconds")
        if max_new_runs is not None and (isinstance(max_new_runs, bool) or not isinstance(max_new_runs, int) or max_new_runs < 1):
            raise ValueError("Maximum new runs must be a positive integer")
        self.max_new_runs = max_new_runs
        self.new_runs = 0
        self.now = now

    def check(self, seconds, label):
        if (self.deadline - self.now()).total_seconds() < seconds + self.cleanup_seconds:
            raise IntentionalPause(f"Insufficient declared deadline budget before {label}")

    def new_run(self):
        if self.max_new_runs is not None and self.new_runs >= self.max_new_runs:
            raise IntentionalPause("Requested maximum new variants completed")
        self.check(self.run_seconds, "next complete variant unit")


def load_ground_truth_helper():
    path = Path(__file__).with_name("export_recorded_ground_truth.py")
    spec = importlib.util.spec_from_file_location("bounded_quality_ground_truth", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def quality_hooks(destination, phase, plan, admission, *, ground_truth_helper=None, ground_truth_exports=None):
    original_run, original_retire, original_subprocess = study._execute_run, study._retire_outputs, study.subprocess
    exports = set()
    phase_dir = Path(destination) / "quality" / phase

    def execute(directory, name, cfg, records, adopted=None, adoption_source=None):
        completed = (Path(directory) / "completed" / f"{name}.json").exists()
        if not completed and not adopted:
            admission.new_run()
        result = original_run(directory, name, cfg, records, adopted, adoption_source)
        if not completed and not adopted:
            admission.new_runs += 1
        return result

    def process(command, *args, **kwargs):
        if "drivebench.evaluate" in command:
            admission.check(admission.evaluation_seconds + admission.export_seconds, "next evaluation and native export")
        elif "drivebench.benchmark" in command:
            admission.check(admission.run_seconds, "benchmark and remaining variant work")
        else:
            raise ValueError("Unexpected subprocess in bounded quality wrapper")
        return original_subprocess.run(command, *args, **kwargs)

    def retire(receipt, keep_requests, retained, *, permitted):
        for metric_receipt in sorted(phase_dir.glob("shards/*/metrics/*.receipt.json")):
            metric = metric_receipt.with_name(metric_receipt.name.replace(".receipt.json", ".json"))
            result = json.loads(metric.read_text())
            if str(Path(receipt["path"]).resolve()) not in {
                    str(Path(result["reference"]).resolve()),
                    *[str(Path(run["run_path"]).resolve()) for run in result["runs"]]}:
                continue
            shard = metric.parent.parent
            reference, candidate = metric.stem.split("--", 1)
            evidence = {name: json.loads((shard / "completed" / f"{name}.json").read_text())
                        for name in (reference, candidate)}
            study._verify_metric_receipt(metric, json.loads(metric_receipt.read_text()),
                                         evidence[reference], evidence[candidate], plan)
            registry = {str(Path(item["path"]).resolve()): item["sha256"]
                        for run in evidence.values() for item in run["outputs"]}
            bundle = export_comparison(metric, metric_receipt, registry, phase_dir / "native-review",
                                       minimum_free_bytes=plan["quality"]["minimum_free_bytes"], admission=admission)
            exports.add(str(bundle / "public" / "index.html"))
        if ground_truth_helper is not None:
            for source in sorted(phase_dir.glob(f"shards/*/completed/{plan['reference']}.json")):
                reference = json.loads(source.read_text())
                if str(Path(reference["path"]).resolve()) != str(Path(receipt["path"]).resolve()):
                    continue
                output = phase_dir / "recorded-ground-truth" / (source.parent.parent.name + ".json")
                if not output.exists() and not output.with_suffix(".receipt.json").exists():
                    admission.check(admission.export_seconds, "new recorded-ground-truth export")
                # The idempotent entry point also checks that an existing export
                # belongs to this exact run, beyond validating its own hashes.
                ground_truth_helper.export_run(receipt["path"], output)
                ground_truth_exports.add(str(output))
        original_retire(receipt, keep_requests, retained, permitted=permitted)

    study._execute_run, study._retire_outputs = execute, retire
    study.subprocess = SimpleNamespace(run=process, STDOUT=original_subprocess.STDOUT)
    try:
        yield exports
    finally:
        study._execute_run, study._retire_outputs, study.subprocess = original_run, original_retire, original_subprocess


def run(destination, phase, receipt_path, *, deadline, run_seconds, evaluation_seconds,
        export_seconds, max_new_runs=None, adopted=None, adoption_sources=None, runtime_config=None,
        max_shards=None, cleanup_seconds=240, upstream_ground_truth=False):
    receipt_path = Path(receipt_path)
    if receipt_path.exists():
        raise FileExistsError("Use a fresh wrapper execution receipt path")
    destination, plan = study._load_plan(destination)
    admission = Admission(deadline, run_seconds=run_seconds, evaluation_seconds=evaluation_seconds,
                          export_seconds=export_seconds, cleanup_seconds=cleanup_seconds, max_new_runs=max_new_runs)
    before = study.generation_sources()
    ground_truth_helper = load_ground_truth_helper() if upstream_ground_truth else None
    gt_helper_path = Path(__file__).with_name("export_recorded_ground_truth.py")
    gt_hash = file_sha256(gt_helper_path) if upstream_ground_truth else None
    initial_plan = file_sha256(destination / "plan.json")
    result = {"schema_version": 1, "receipt_type": TYPE, "started_at": utc_now().isoformat(),
              "phase": phase, "plan_sha256": initial_plan, "wrapper_sha256": file_sha256(__file__),
              "source_files": before, "absolute_deadline": admission.deadline.isoformat(),
              "bounds_seconds": {"complete_new_variant": run_seconds, "evaluation": evaluation_seconds,
                                 "native_export": export_seconds, "cleanup": cleanup_seconds},
              "maximum_new_runs": max_new_runs, "phase_complete": False,
              "ground_truth_exporter_sha256": gt_hash}
    exports = set()
    ground_truth_exports = set()
    try:
        with quality_hooks(destination, phase, plan, admission, ground_truth_helper=ground_truth_helper,
                           ground_truth_exports=ground_truth_exports) as exports:
            outcome = study.execute_quality(destination, phase, adopted=adopted,
                adoption_sources=adoption_sources, max_shards=max_shards, runtime_config=runtime_config)
        records, _ = study._manifest_info(plan["manifests"][phase]["path"])
        required = math.ceil(len(records) / plan["quality"]["shard_clips"])
        result.update(status="returned", result=outcome, phase_complete=outcome["completed_shards"] == required)
    except IntentionalPause as exc:
        result.update(status="intentionally_paused", reason=str(exc))
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        result.update(finished_at=utc_now().isoformat(), new_runs_completed=admission.new_runs,
                      native_review_indexes=sorted(exports), source_unchanged=study.generation_sources() == before,
                      plan_unchanged=file_sha256(destination / "plan.json") == initial_plan)
        result["native_export_receipt_sha256"] = {
            str(Path(index).parent.parent / "private-receipt.json"):
                file_sha256(Path(index).parent.parent / "private-receipt.json")
            for index in sorted(exports)}
        result["recorded_ground_truth_artifact_sha256"] = {
            str(path): file_sha256(path) for output in sorted(ground_truth_exports)
            for path in (Path(output), Path(output).with_suffix(".receipt.json"))}
        result["ground_truth_exporter_unchanged"] = not upstream_ground_truth or file_sha256(gt_helper_path) == gt_hash
        private_json(receipt_path, result)
    if not result["source_unchanged"] or not result["plan_unchanged"] or not result["ground_truth_exporter_unchanged"]:
        raise ValueError("Frozen source or plan changed during bounded quality execution")
    return result


def pairs(values):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError("Bindings must be unique NAME=PATH arguments")
        result[name] = path
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--phase", choices=study.PHASES, required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--run-budget-seconds", type=float, required=True,
                        help="Bound for generation, all comparisons and native export of one new variant")
    parser.add_argument("--evaluation-budget-seconds", type=float, required=True)
    parser.add_argument("--export-budget-seconds", type=float, required=True)
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--runtime-config")
    parser.add_argument("--upstream-ground-truth", action="store_true",
                        help="Preserve recorded-future proxies for each reference shard before raw retirement")
    parser.add_argument("--adopt", action="append", default=[])
    parser.add_argument("--adoption-source", action="append", default=[])
    args = parser.parse_args()
    result = run(args.study, args.phase, args.receipt, deadline=args.deadline_utc,
        run_seconds=args.run_budget_seconds, evaluation_seconds=args.evaluation_budget_seconds,
        export_seconds=args.export_budget_seconds, max_new_runs=args.max_new_runs,
        max_shards=args.max_shards, runtime_config=args.runtime_config,
        adopted=pairs(args.adopt), adoption_sources=pairs(args.adoption_source),
        upstream_ground_truth=args.upstream_ground_truth)
    # No mappings, raw source paths or variant attribution enter stdout.
    print(json.dumps({key: result[key] for key in ("status", "phase_complete", "new_runs_completed")}))


if __name__ == "__main__":
    main()
