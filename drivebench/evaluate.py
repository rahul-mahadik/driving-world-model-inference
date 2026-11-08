"""Compare saved generated futures and export reproducible blinded reviews.

    python -m drivebench.evaluate --runs outputs/finalists --reference outputs/upstream
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import io
import json
import os

import numpy as np
import yaml

from .evaluation.io import Run, artifact_sha256, candidates, experiment_identity, find_runs, get_nested, load_run, output_path, quality_rows, write_json
from .evaluation.metrics import PerceptualMetric, as_video, bootstrap_mean, compare_videos, verify_perceptual_setup
from .evaluation.review import export_review
from .config import file_sha256


METRICS = ("mae", "rmse", "max_abs_error", "lpips", "temporal_change_error",
           "freeze_fraction_increase", "temporal_derivative_mae")


def select_reference(runs: list[Run], reference: str | Path | None = None) -> Run:
    if reference is not None:
        return load_run(reference)
    for names in ({"upstream"}, {"baseline", "eager", "refactored_eager"}):
        matches = [run for run in runs if run.name in names or get_nested(run.config, "baseline_role", "name") in names]
        if len(matches) == 1:
            return matches[0]
    matches = [run for run in runs if get_nested(run.config, "model.backend", "backend") == "upstream"]
    if len(matches) == 1:
        return matches[0]
    raise ValueError("Choose an explicit --reference run; a unique upstream/baseline run was not found")


def aggregate_pairs(pairs: list[dict[str, Any]], *, seed: int = 0) -> dict[str, Any]:
    result = {}
    for metric in METRICS:
        by_scene: dict[str, list[float]] = defaultdict(list)
        values = []
        for pair in pairs:
            value = pair["metrics"].get(metric)
            if isinstance(value, (float, int)) and np.isfinite(value):
                by_scene[pair["scene_id"]].append(float(value))
                values.append(float(value))
        result[metric] = {**bootstrap_mean([float(np.mean(v)) for v in by_scene.values()], seed=seed),
                          "max": max(values), "pairs": len(values), "ci_unit": "scene"} if values else {
                              "mean": None, "max": None, "ci95": None, "units": 0, "pairs": 0, "ci_unit": "scene"}
    return result


def tolerance_gate(aggregates: dict[str, Any], tolerances: dict[str, Any], *, complete_pairs: bool,
                   synthetic: bool, evaluation_manifest_sha256: str | None = None,
                   evaluation_manifest_metadata: dict[str, Any] | None = None,
                   run_started_at: str | None = None, reviewed: bool = False,
                   evaluated_scene_ids: list[str] | None = None,
                   evaluated_request_ids: list[str] | None = None,
                   evaluated_pair_count: int | None = None) -> dict[str, Any]:
    reasons = []
    if not tolerances.get("calibrated") or not tolerances.get("frozen"):
        reasons.append("Tolerances have not been calibrated and frozen")
    dev_hash = tolerances.get("development_manifest_sha256")
    if tolerances.get("selected_on") != "development" or not isinstance(dev_hash, str) or len(dev_hash) != 64:
        reasons.append("Missing development-set tolerance provenance")
    if not evaluation_manifest_sha256 or evaluation_manifest_sha256 == dev_hash:
        reasons.append("A distinct held-out evaluation manifest is required")
    evidence = evaluation_manifest_metadata or {}
    if evidence.get("partition_role") not in {"evaluation", "evaluation_second_seed",
                                               "held_out_evaluation", "held_out_evaluation_second_seed"}:
        reasons.append("Manifest metadata must explicitly identify the held-out evaluation partition")
    if not evaluation_manifest_sha256 or evidence.get("manifest_sha256") != evaluation_manifest_sha256:
        reasons.append("Evaluation manifest metadata checksum does not match the evaluated manifest")
    development_scenes = tolerances.get("development_scene_ids")
    evaluation_scenes = evidence.get("scene_ids")
    def valid_scenes(value):
        return isinstance(value, list) and bool(value) and all(isinstance(scene, str) and scene for scene in value)
    if not valid_scenes(development_scenes) or not valid_scenes(evaluation_scenes):
        reasons.append("Development and evaluation scene IDs are required to audit scene disjointness")
    elif set(development_scenes) & set(evaluation_scenes):
        reasons.append("Development and evaluation scenes overlap")
    if not valid_scenes(evaluated_scene_ids) or not valid_scenes(evaluation_scenes) or set(evaluated_scene_ids) != set(evaluation_scenes):
        reasons.append("Saved matched outputs do not cover every scene in the evaluation manifest")
    clip_count = evidence.get("clip_count")
    if (not isinstance(clip_count, int) or isinstance(clip_count, bool) or clip_count < 1
            or not evaluated_request_ids or len(set(evaluated_request_ids)) != clip_count):
        reasons.append("Saved matched outputs do not cover every clip in the evaluation manifest")
    try:
        frozen_at = datetime.fromisoformat(str(tolerances.get("frozen_at")).replace("Z", "+00:00"))
        if not run_started_at:
            reasons.append("Run start timestamp is missing; cannot verify tolerances were frozen beforehand")
        elif frozen_at > datetime.fromisoformat(run_started_at.replace("Z", "+00:00")):
            reasons.append("Tolerances were frozen after the evaluated run started")
    except (ValueError, TypeError):
        reasons.append("Tolerance freeze or run start timestamp is invalid")
    if not complete_pairs:
        reasons.append("Reference and candidate outputs are not complete, matched pairs")
    if synthetic:
        reasons.append("Synthetic backend is a functional test only")
    checks = {}
    configured = tolerances.get("metrics", {})
    if not isinstance(configured, dict) or not configured:
        reasons.append("No metric tolerances are configured")
        configured = {}
    for metric, settings in configured.items():
        setting = settings if isinstance(settings, dict) else {}
        maximum, aggregation = setting.get("maximum"), setting.get("aggregation", "mean")
        summary = aggregates.get(metric, {})
        actual = summary.get(aggregation)
        valid = (metric in METRICS and aggregation in {"mean", "max"}
                 and isinstance(maximum, (int, float)) and not isinstance(maximum, bool)
                 and np.isfinite(maximum) and maximum >= 0)
        complete_metric = (isinstance(evaluated_pair_count, int) and evaluated_pair_count > 0
                           and summary.get("pairs") == evaluated_pair_count)
        passed = valid and complete_metric and isinstance(actual, (int, float)) and np.isfinite(actual) and actual <= maximum
        checks[metric] = {"maximum": maximum, "aggregation": aggregation, "actual": actual,
                          "passed": bool(passed)}
        if not passed:
            reasons.append(f"Metric {metric} is unavailable, invalid, or exceeds its tolerance")
    metric_pass = not reasons
    quality_reasons = list(reasons)
    if not {"rmse", "max_abs_error", "mae"}.intersection(configured):
        quality_reasons.append("No numerical error constraint")
    exact_policy = (tolerances.get("policy_type") == "exact_output_equivalence"
                    and configured.get("max_abs_error") == {"maximum": 0.0, "aggregation": "max"})
    if "lpips" not in configured and not exact_policy:
        quality_reasons.append("No perceptual metric constraint")
    if not {"freeze_fraction_increase", "temporal_derivative_mae", "temporal_change_error"}.intersection(configured):
        quality_reasons.append("No temporal diagnostic constraint")
    if not reviewed and not exact_policy:
        quality_reasons.append("Blinded review has not been completed and accepted")
    return {"within_frozen_metric_tolerances": metric_pass, "quality_preserving": not quality_reasons,
            "reasons": quality_reasons, "checks": checks}


def _ground_truth(candidate: dict, shape: tuple[int, ...]) -> np.ndarray | None:
    if candidate.get("action_kind") != "recorded":
        return None
    paths = candidate.get("ground_truth_paths")
    if not paths:
        return None
    if len(paths) != shape[0]:
        raise ValueError("Recorded ground-truth frame count must match newly generated frames; no implicit truncation")
    from .data.preprocessing import read_preprocess
    # Use exactly the same center crop, interpolation, color conversion, and
    # pixel normalization as inference, then return RGB in the metric range.
    frames = read_preprocess(paths, height=shape[1], width=shape[2])
    return np.ascontiguousarray(((frames + 1) / 2).transpose(0, 2, 3, 1))


def _expected_count(run: Run) -> int:
    return sum(int(row.get("candidate_count", len(row.get("candidates", [])))) for row in quality_rows(run))


def _review_acceptance(review_text: str | None, review: dict[str, Any]) -> set[str]:
    if review_text is None:
        return set()
    key = json.loads(Path(review["private_key"]).read_text())["cases"]
    submitted = list(csv.DictReader(io.StringIO(review_text)))
    entries = {row.get("case_id"): row for row in submitted}
    if len(entries) != len(submitted) or set(entries) != {case["case_id"] for case in key}:
        review["review_results_status"] = "rejected: case identities differ from the current export or contain duplicates"
        return set()
    review["review_results_status"] = "matched_current_cases"
    by_run: dict[str, list[bool]] = defaultdict(list)
    for case in key:
        entry = entries.get(case["case_id"], {})
        panel = "A" if case["A"] == case["run"] else "B"
        by_run[case["run"]].append(entry.get(f"acceptable_{panel}", "").strip().lower() in {"yes", "true", "1"})
    return {run for run, votes in by_run.items() if votes and all(votes)}


def action_response_pairs(reference: Run, candidate: Run) -> list[dict]:
    """Same-noise response preservation, streamed one frame from four mmap files."""
    def indexed(run):
        return {(r["request_id"], r.get("repeat", 0)): r for r in quality_rows(run)}
    left, right = indexed(reference), indexed(candidate)
    if left.keys() != right.keys():
        raise ValueError("Action-response requests differ")
    result = []
    for key in sorted(left):
        a, b = left[key], right[key]
        ca = {c["candidate_id"]: c for c in a["candidates"]}
        cb = {c["candidate_id"]: c for c in b["candidates"]}
        recorded = [c for c in ca.values() if c.get("action_kind") == "recorded"]
        if len(recorded) != 1 or ca.keys() != cb.keys() or len(ca) < 2:
            raise ValueError("Action response requires one recorded and matching alternative actions")
        base = recorded[0]
        for cid, alt in ca.items():
            if cid == base["candidate_id"]:
                continue
            records = [base, alt, cb[base["candidate_id"]], cb[cid]]
            if len({c["seed"] for c in records}) != 1:
                raise ValueError("Action response requires identical noise seeds across actions and implementations")
            hashes = [c.get("trajectory_sha256", c.get("action_sha256")) for c in records]
            if not all(hashes) or hashes[0] == hashes[1] or hashes[:2] != hashes[2:]:
                raise ValueError("Action response requires distinct actions matched across implementations")
            count = records[0].get("conditioning_frames", reference.config["generation"]["conditioning_frames"])
            paths = [output_path(run, c["frames_path"]) for run, c in zip((reference, reference, candidate, candidate), records)]
            arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths]
            if len({v.shape for v in arrays}) != 1 or any(c.get("conditioning_frames", count) != count for c in records):
                raise ValueError("Action-response output shapes or conditioning counts differ")
            totals = dict(reference_response_mae=0., candidate_response_mae=0., response_error_mae=0., response_error_squared=0.)
            maximum, elements = 0., 0
            for frame in range(count, arrays[0].shape[0]):
                values = [as_video(v[frame:frame + 1]).astype(np.float64) for v in arrays]
                original, changed = values[1] - values[0], values[3] - values[2]
                error = changed - original
                totals["reference_response_mae"] += float(np.abs(original).sum())
                totals["candidate_response_mae"] += float(np.abs(changed).sum())
                totals["response_error_mae"] += float(np.abs(error).sum())
                totals["response_error_squared"] += float(np.square(error).sum())
                maximum = max(maximum, float(np.abs(error).max()))
                elements += error.size
            if not elements:
                raise ValueError("Action response requires generated future frames")
            metrics = {k: v / elements for k, v in totals.items() if k != "response_error_squared"}
            metrics.update(response_error_rmse=float(np.sqrt(totals["response_error_squared"] / elements)), response_error_max_abs=maximum)
            result.append({"request_id": key[0], "repeat": key[1], "scene_id": a["scene_id"], "seed": base["seed"],
                           "recorded_candidate_id": base["candidate_id"], "alternative_candidate_id": cid,
                           "trajectory_sha256": hashes[:2], "output_sha256": [file_sha256(p) for p in paths],
                           "metrics": metrics, "diagnostic_only": True,
                           "note": "Same-noise image response to a changed action; neither causal trajectory adherence nor driving safety."})
    return result


def evaluate_runs(runs_path: str | Path, *, reference: str | Path | None = None,
                  tolerances_path: str | Path = "configs/tolerances.yaml", lpips: bool = False,
                  lpips_device: str = "cpu", action_response: bool = False,
                  review_limit: int = 12, review_results: str | Path | None = None,
                  output: str | Path | None = None) -> dict[str, Any]:
    runs = find_runs(runs_path)
    ref = select_reference(runs, reference)
    reference_candidates = candidates(ref)
    tolerance_path = Path(tolerances_path)
    with tolerance_path.open() as stream:
        tolerances = yaml.safe_load(stream)
    if not isinstance(tolerances, dict):
        raise ValueError("Tolerance configuration must be a mapping")
    setup_receipt = os.environ.get("DRIVEBENCH_EVALUATION_SETUP_RECEIPT")
    setup = verify_perceptual_setup(setup_receipt) if setup_receipt and lpips else None
    perceptual = PerceptualMetric(lpips, device=lpips_device)
    if setup_receipt:
        perceptual.provenance["setup_receipt_sha256"] = file_sha256(setup_receipt)
    if setup and (perceptual.provenance.get("state_dict_sha256") != setup["cpu_verification"]["model_state_dict_sha256"]
                  or not Path(perceptual.provenance.get("module_path", ".")).is_relative_to(Path(setup["package"]["target"]))):
        raise ValueError("Loaded LPIPS model or import location differs from verified isolated setup")
    review_text = Path(review_results).read_text() if review_results else None
    destination = Path(output) if output else Path(runs_path) / "evaluation.json"
    entries, review_pairs = [], []
    for run in runs:
        if run.path == ref.path:
            continue
        current = candidates(run)
        matching = sorted(reference_candidates.keys() & current.keys())
        pair_results, errors = [], []
        for key in matching:
            ref_row, ref_candidate = reference_candidates[key]
            row, candidate = current[key]
            try:
                if row["scene_id"] != ref_row["scene_id"] or candidate.get("action_kind") != ref_candidate.get("action_kind"):
                    raise ValueError("Paired candidate scene or action kind differs")
                for action_key in ("trajectory_sha256", "action_sha256"):
                    if candidate.get(action_key) != ref_candidate.get(action_key):
                        raise ValueError("Paired candidate trajectory identity differs")
                action_hash = candidate.get("trajectory_sha256", candidate.get("action_sha256"))
                if not run.synthetic or not ref.synthetic:
                    if not isinstance(action_hash, str) or len(action_hash) != 64:
                        raise ValueError("Real candidate outputs require a trajectory checksum")
                count = int(candidate.get("conditioning_frames", get_nested(run.config, "generation.conditioning_frames") or 1))
                ref_count = int(ref_candidate.get("conditioning_frames", get_nested(ref.config, "generation.conditioning_frames") or 1))
                if count < 1 or count != ref_count:
                    raise ValueError("Conditioning frame count differs or is invalid")
                ref_path, new_path = output_path(ref, ref_candidate["frames_path"]), output_path(run, candidate["frames_path"])
                videos = []
                for source, path in ((ref, ref_path), (run, new_path)):
                    video = as_video(np.load(path, allow_pickle=False))
                    declared = (get_nested(source.config, "generation.total_frames", "generation.frames", "generation.num_frames"),
                                get_nested(source.config, "generation.height"), get_nested(source.config, "generation.width"), 3)
                    if any(expected is not None and expected != actual for expected, actual in zip(declared, video.shape)):
                        raise ValueError("Saved video shape differs from the declared generation workload")
                    videos.append(video[count:])
                left, right = videos
                metrics = compare_videos(left, right)
                metrics["lpips"] = perceptual.compare(left, right)
                pair = {"request_id": key[0], "repeat": key[1], "candidate_id": key[2], "seed": key[3],
                        "scene_id": row["scene_id"], "action_kind": candidate.get("action_kind"),
                        "trajectory_sha256": action_hash, "metrics": metrics,
                        "reference_output_sha256": file_sha256(ref_path),
                        "candidate_output_sha256": file_sha256(new_path),
                        "categories": row.get("categories", ["unknown"])}
                gt = _ground_truth(candidate, right.shape)
                pair["recorded_ground_truth"] = compare_videos(gt, right) if gt is not None else None
                pair_results.append(pair)
                review_pairs.append({**{k: v for k, v in pair.items() if k not in {"metrics", "recorded_ground_truth"}},
                                     "run": run.name, "reference_run": ref.name, "reference_path": str(ref_path),
                                     "candidate_path": str(new_path), "conditioning_frames": count})
            except (ValueError, OSError, KeyError) as exc:
                errors.append({"key": list(key), "error": str(exc)})
        identity, ref_identity = experiment_identity(run), experiment_identity(ref)
        for field in ("model_commit", "checkpoint_sha256", "manifest_sha256", "generation"):
            if identity.get(field) != ref_identity.get(field):
                errors.append({"identity_field": field, "error": f"Comparison {field} differs from reference"})
            elif (not run.synthetic or not ref.synthetic) and not identity.get(field):
                errors.append({"identity_field": field, "error": f"Comparison {field} is missing"})
        for checked in (run, ref):
            if checked.summary.get("status", "ok") not in {"ok", "success"} or checked.summary.get("run_failure"):
                errors.append({"run": checked.name, "error": "Run failed before completing the planned workload"})
        complete = (len(pair_results) > 0 and len(pair_results) == len(current) == len(reference_candidates)
                    == _expected_count(run) == _expected_count(ref) and not errors
                    and not run.summary.get("failures") and not ref.summary.get("failures"))
        aggregate = aggregate_pairs(pair_results)
        entries.append({"run": run.name, "run_path": str(run.path), "reference": ref.name,
                        "artifact_sha256": artifact_sha256(run),
                        "synthetic": run.synthetic, "pairs": pair_results, "aggregates": aggregate,
                        "action_response": action_response_pairs(ref, run) if action_response else [],
                        "by_action_kind": {kind: aggregate_pairs([p for p in pair_results if p["action_kind"] == kind])
                                           for kind in sorted({p["action_kind"] or "unknown" for p in pair_results})},
                        "by_category": {category: aggregate_pairs([p for p in pair_results if category in p["categories"]])
                                        for category in sorted({c for p in pair_results for c in p["categories"]})},
                        "complete_pairs": complete, "matched_pairs": len(pair_results),
                        "missing_reference_keys": [list(k) for k in sorted(current.keys() - reference_candidates.keys())],
                        "missing_candidate_keys": [list(k) for k in sorted(reference_candidates.keys() - current.keys())],
                        "errors": errors, "identity": identity,
                        "manifest_metadata": get_nested(run.config, "data.manifest_metadata"),
                        "run_started_at": get_nested(run.config, "started_at")
                        or get_nested(run.environment, "started_at", "timestamp", "captured_at", "timestamp_utc")})
    review_directory = "review" if destination.stem == "evaluation" else f"{destination.stem}-review"
    review = export_review(review_pairs, destination.parent / review_directory, limit=review_limit)
    accepted = _review_acceptance(review_text, review)
    if (review_results and Path(review_results).resolve() == Path(review["csv"]).resolve()
            and review.get("review_results_status") == "matched_current_cases"):
        Path(review["csv"]).write_text(review_text)
    for entry in entries:
        entry["gate"] = tolerance_gate(entry["aggregates"], tolerances,
                                       complete_pairs=entry["complete_pairs"], synthetic=entry["synthetic"] or ref.synthetic,
                                       evaluation_manifest_sha256=entry["identity"]["manifest_sha256"],
                                       evaluation_manifest_metadata=entry["manifest_metadata"],
                                       run_started_at=entry["run_started_at"], reviewed=entry["run"] in accepted,
                                       evaluated_scene_ids=sorted({pair["scene_id"] for pair in entry["pairs"]}),
                                       evaluated_request_ids=sorted({pair["request_id"] for pair in entry["pairs"]}),
                                       evaluated_pair_count=entry["matched_pairs"])
    result = {"schema_version": 1, "reference": str(ref.path), "tolerances_path": str(tolerance_path.resolve()),
              "evaluator_source_sha256": {str(path.relative_to(Path(__file__).parent)): file_sha256(path)
                                           for path in [Path(__file__), *sorted((Path(__file__).parent / "evaluation").glob("*.py"))]},
              "reference_artifact_sha256": artifact_sha256(ref),
              "tolerances": tolerances, "runs": entries, "review": review,
              "perceptual": {"name": "LPIPS/alex", "status": perceptual.status, "reason": perceptual.reason,
                             "provenance": perceptual.provenance},
              "notes": ["Metrics compare newly generated frames only; conditioning frames are excluded.",
                        "All planned saved outputs must pair completely; timing-only repeats beyond benchmark.save_outputs are excluded.",
                        "Bootstrap uncertainty uses scenes as independent units, averaging repeats and candidates within scene.",
                        "Temporal metrics are image-difference diagnostics; none measures trajectory adherence.",
                        "Recorded future comparisons apply only to recorded actions and are not definitive quality scores.",
                        "PSNR is null for exact matches (mathematically infinite); exact_match records that case.",
                        "Blinded reviews include every generated future frame at 10 fps in downscaled GIF previews; contact sheets sample frames."]}
    write_json(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--tolerances", default="configs/tolerances.yaml")
    parser.add_argument("--lpips", action="store_true", help="Requires optional lpips/torch packages and model weights")
    parser.add_argument("--lpips-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--action-response", action="store_true", help="Require same-seed recorded/perturbed actions and measure response preservation")
    parser.add_argument("--review-limit", type=int, default=12)
    parser.add_argument("--review-results", help="Completed CSV with acceptable_A/acceptable_B yes/no columns")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_runs(args.runs, reference=args.reference, tolerances_path=args.tolerances,
                           lpips=args.lpips, lpips_device=args.lpips_device, action_response=args.action_response, review_limit=args.review_limit, review_results=args.review_results,
                           output=args.output)
    print(json.dumps({"compared_runs": len(result["runs"]), "perceptual": result["perceptual"],
                      "review": result["review"]["index"]}, indent=2))


if __name__ == "__main__":
    main()
