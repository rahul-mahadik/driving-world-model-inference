"""Fabricated tiny report artifacts verify presentation, never GPU outcomes."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from drivebench.config import file_sha256

SPEC = importlib.util.spec_from_file_location("full_study_report", Path(__file__).resolve().parents[1] / "scripts/report_full_study.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)
SECRET = "client_secret_SHOULD_NEVER_BE_PUBLIC"
PRIVATE = "/Users/private-account/.config/vendor/credentials.json"
UUID = "GPU-3f94db7e-0000-4444-aaaa-54c120aaa111"


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
    return path


def hashes(path):
    return {name: file_sha256(path / name) for name in report.CORE}


def fixture(tmp_path):
    study = tmp_path / "study"
    put(study / "plan.json", {"base_config": {"model": {"upstream_commit": "a" * 40, "checkpoint_sha256": "b" * 64,
        "account": SECRET, "checkpoint": PRIVATE}}, "quality": {"retain_raw_clips": 1},
        "generation_sources_at_plan": {"models/vista.py": "c" * 64}})
    folder = study / "quality/development"
    aggregate = folder / "aggregate"
    metadata = {"manifest_sha256": "d" * 64, "clip_count": 2, "scene_ids": ["scene-a", "scene-b"], "partition_role": "development"}
    variants = {"upstream": (50, 10.), "eager": (50, 9.9), "feature-2": (50, 9.45), "steps-47": (47, 9.4), "steps-40": (40, 8.)}
    for name, (steps, latency) in variants.items():
        path = aggregate / "runs" / name
        put(path / "config.json", {"model": {"backend": "synthetic", "account": SECRET}, "generation": {"steps": steps},
            "benchmark": {"candidate_count": 1}, "data": {"root": PRIVATE, "manifest_metadata": metadata}, "started_at": "2026-01-02T00:00:00+00:00"})
        put(path / "environment.json", {"gpu": {"name": "NVIDIA H100 80GB HBM3", "uuid": UUID}, "credentials": SECRET})
        put(path / "summary.json", {"backend": "synthetic", "headline_eligible": False})
        rows = [{"status": "ok", "request_id": f"request-{x}", "scene_id": f"scene-{x}", "latency_s": latency,
                 "candidates": [{"candidate_id": f"request-{x}:c00", "seed": 42, "frames_path": None}]} for x in ("a", "b")]
        (path / "measurements.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in rows))
    reviewdir = folder / "shards/000/metrics/blinded-review"
    reviewdir.mkdir(parents=True)
    (reviewdir / "index.html").write_text("<html>Fixture blinded review</html>")
    (reviewdir / "review.csv").write_text("case_id,acceptable_A,acceptable_B\ncase-a,,\n")
    put(reviewdir / "private_key.json", {"cases": [{"case_id": "case-a", "A": "upstream", "B": "eager", "secret": SECRET}]})
    review = {"index": str(reviewdir / "index.html"), "csv": str(reviewdir / "review.csv"), "private_key": str(reviewdir / "private_key.json")}
    policy = put(study / "unfrozen-tolerances.json", {"calibrated": False})
    sources = {"evaluate.py": "e" * 64}
    metric_artifacts = {}
    for reference in ("upstream", "eager"):
        entries = []
        for name in variants:
            if name == reference or (reference == "eager" and name == "upstream"):
                continue
            error = 0. if name == "eager" else .01
            pairs = [{"request_id": f"request-{x}", "scene_id": f"scene-{x}", "repeat": 0, "candidate_id": f"request-{x}:c00",
                "seed": 42, "action_kind": "recorded", "trajectory_sha256": "f" * 64,
                "reference_output_sha256": "1" * 64, "candidate_output_sha256": "2" * 64,
                "metrics": {key: error for key in report.METRICS}, "recorded_ground_truth": {key: .05 for key in report.METRICS}}
                for x in ("a", "b")]
            entry = {"run": name, "run_path": str(aggregate / "runs" / name), "artifact_sha256": hashes(aggregate / "runs" / name),
                "synthetic": True, "complete_pairs": True, "matched_pairs": 2, "pairs": pairs, "manifest_metadata": metadata,
                "gate": {"quality_preserving": True, "within_frozen_metric_tolerances": True, "checks": {}}, "errors": []}
            entries.append(entry)
            metric = put(folder / f"shards/000/metrics/{reference}--{name}.json", {"tolerances_path": str(policy), "runs": [entry],
                "reference_artifact_sha256": hashes(aggregate / "runs" / reference), "evaluator_source_sha256": sources})
            receipt = put(metric.with_suffix(".receipt.json"), {"evaluation_sha256": file_sha256(metric),
                "reference_artifact_sha256": hashes(aggregate / "runs" / reference), "candidate_artifact_sha256": entry["artifact_sha256"],
                "evaluator_source_sha256": sources, "tolerances_sha256": file_sha256(policy),
                "review_artifact_sha256": {str(p): file_sha256(p) for p in reviewdir.iterdir()}})
            metric_artifacts.update({str(metric): file_sha256(metric), str(receipt): file_sha256(receipt)})
        put(aggregate / f"evaluation-vs-{reference}.json", {"reference": str(aggregate / "runs" / reference),
            "reference_artifact_sha256": hashes(aggregate / "runs" / reference), "runs": entries, "reviews": [review],
            "perceptual": {"status": "available", "provenance": {"state_dict_sha256": "3" * 64, "module_path": PRIVATE}}})
    put(folder / "shards/000/complete.json", {"metric_artifacts": metric_artifacts,
        "runs": {name: {"path": str(aggregate / "runs" / name), "artifact_sha256": hashes(aggregate / "runs" / name)} for name in variants}})
    put(folder / "aggregation-receipt.json", {"artifact_sha256": {str(p): file_sha256(p) for p in aggregate.rglob("*") if p.is_file()}})
    coverage = put(tmp_path / "coverage.json", {"partitions": {"development": {"manifest_sha256": "d" * 64,
        "records": [{"request_id": "request-a", "labels": ["description:night-mentioned", "metadata:pedestrian-within30m"]},
                    {"request_id": "request-b", "labels": ["description:night-not-stated"]}]}}})
    return study, coverage, review


def assert_private_absent(path):
    for file in path.rglob("*"):
        if file.is_file():
            value = file.read_text()
            assert SECRET not in value and PRIVATE not in value and UUID not in value
            assert "/Users/" not in value and "/workspace/" not in value


def test_incomplete_plan_is_not_presented_as_completed(tmp_path):
    study = tmp_path / "study"
    put(study / "plan.json", {"base_config": {}, "quality": {}})
    output = tmp_path / "report"
    result = report.build(study, output, workspace_root=tmp_path, make_plots=False)
    assert result["status"].startswith("incomplete") and result["headline_claims"] == []
    assert all(p["status"] == "not_aggregated" for p in result["quality"])
    assert result["timing"]["status"] == "pending_or_ineligible"
    assert_private_absent(output / "public")


def test_report_joins_conditions_both_references_and_never_loads_arrays(tmp_path, monkeypatch):
    study, coverage, _ = fixture(tmp_path)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Report must never read generated arrays"))
    output = tmp_path / "report"
    result = report.build(study, output, workspace_root=tmp_path, coverage_path=coverage, make_plots=False)
    dev = result["quality"][0]
    assert dev["status"] == "verified"
    assert {r["reference"] for r in dev["comparisons"]} == {"upstream", "eager"}
    assert all(not r["quality_preserving"] for r in dev["comparisons"])  # Synthetic fixtures cannot pass.
    assert dev["comparisons"][0]["by_condition"]["description:night-mentioned"]["rmse"]["units"] == 1
    assert all(row["step_variant"] == "steps-47" for row in result["runtime_matches"])
    assert all(not row["equal_runtime_established"] for row in result["runtime_matches"])
    assert all(row["paired_requests"] == 2 and row["gap_resolved_at_95_percent"] for row in result["runtime_matches"])
    assert all(row["paired_scene_ci95_relative_gap"][0] == pytest.approx(9.45 / 9.4 - 1) for row in result["runtime_matches"])
    assert "Local raw evidence and blinded reviews" in (output / "index.html").read_text()
    assert "blinded-review/index.html" not in (output / "public/index.html").read_text()
    assert_private_absent(output / "public")


def test_stale_artifact_or_policy_withholds_validation(tmp_path):
    study, coverage, _ = fixture(tmp_path)
    policy = study / "unfrozen-tolerances.json"
    policy.write_text('{"calibrated":true}')
    result = report.build(study, tmp_path / "report", workspace_root=tmp_path, coverage_path=coverage, make_plots=False)
    assert result["quality"][0]["status"] == "stale_or_incomplete"
    assert all(not r["quality_preserving"] for r in result["quality"][0]["comparisons"])
    assert result["evidence_issues"]


def test_runtime_control_uncertainty_requires_matching_request_pairs(tmp_path):
    study, coverage, _ = fixture(tmp_path)
    result = report.build(study, tmp_path / "report", workspace_root=tmp_path, coverage_path=coverage, make_plots=False)
    for row in result["quality"][0]["comparisons"]:
        if row["variant"] == "feature-2":
            row["raw_request_timings"][0]["request_id"] = "unmatched"
    matches = report.runtime_matches(result["quality"])
    assert matches and all(row["paired_scene_ci95_relative_gap"] is None and row["paired_requests"] == 0 for row in matches)
    assert all(not row["equal_runtime_established"] for row in matches)


@pytest.mark.parametrize("bounds,expected", [([-.019, .019], True), ([-.019, .021], False), ([-.021, .019], False)])
def test_runtime_equivalence_uses_entire_predeclared_interval(monkeypatch, bounds, expected):
    monkeypatch.setattr(report, "_bootstrap_ratios", lambda *args, **kwargs: (0., bounds))
    timings = [{"request_id": f"r-{i}", "scene_id": f"s-{i}", "repeat": 0, "latency_s": 10.} for i in range(32)]
    def row(name, steps):
        return {"variant": name, "reference": "upstream", "status": "verified", "request_median_s": 10.,
                "steps": steps, "synthetic": False, "candidate_count": 1, "workload": "single_future", "raw_request_timings": timings}
    result = report.runtime_matches([{"phase": "development", "comparisons": [row("feature-2", 50), row("steps-47", 47)]}])[0]
    assert result["runtime_matched"] is expected
    assert result["runtime_matching_status"] == ("matched_runtime" if expected else "nearest_measured_control")
    assert result["runtime_matching_criterion_sha256"] == "6e83d60f341a36afc27355ef80f2cc4d13af46cb1e3f0c2c9cdc72859903f8fc"
    assert not result["equal_runtime_established"]


def test_agent_review_ratings_need_bound_separate_csv(tmp_path):
    study, coverage, review = fixture(tmp_path)
    votes = tmp_path / "completed.csv"
    votes.write_text("case_id,acceptable_A,acceptable_B\ncase-a,yes,no\n")
    rating = put(tmp_path / "ratings.json", {"reviewer_kind": "agent_assisted", "reviews": [{"index": review["index"],
        "index_sha256": file_sha256(review["index"]), "csv": str(votes), "csv_sha256": file_sha256(votes)}]})
    result = report.build(study, tmp_path / "report", workspace_root=tmp_path, coverage_path=coverage, review_ratings=rating, make_plots=False)
    assert result["quality"][0]["review"] == {"status": "complete", "cases": 1, "rated_cases": 1, "reviewer_kind": "agent_assisted",
        "note": "Ratings describe the predeclared blinded subset; they do not by themselves establish quality or trajectory adherence."}
    assert not result["headline_claims"]
    votes.write_text("case_id,acceptable_A,acceptable_B\ncase-other,yes,yes\n")
    result = report.build(study, tmp_path / "changed", workspace_root=tmp_path, review_ratings=rating, make_plots=False)
    assert result["quality"][0]["review"]["status"] == "pending"


def test_sidecar_checksum_mismatch_does_not_invent_condition_labels(tmp_path):
    study, coverage, _ = fixture(tmp_path)
    data = json.loads(coverage.read_text()); data["partitions"]["development"]["manifest_sha256"] = "0" * 64
    put(coverage, data)
    result = report.build(study, tmp_path / "report", workspace_root=tmp_path, coverage_path=coverage, make_plots=False)
    assert all(r["by_condition"] == {} and r["condition_status"] == "unavailable_or_manifest_mismatch" for r in result["quality"][0]["comparisons"])


def test_identifier_and_numeric_fields_reject_private_shapes():
    for value in (UUID, "3f94db7e-0000-4444-aaaa-54c120aaa111", "192.0.2.123", PRIVATE, SECRET):
        assert report.identifier(value).startswith("id-")
    assert report.interval([1, PRIVATE]) is None
    assert report.integer(SECRET) is None
    assert report.timestamp(PRIVATE) is None
    assert report.number(float("nan")) is None


def test_plot_refresh_removes_only_owned_prior_curves_and_never_inventories_unknown_files(tmp_path):
    study, coverage, _ = fixture(tmp_path)
    output = tmp_path / "report"
    result = report.build(study, output, workspace_root=tmp_path, coverage_path=coverage)
    assert result["plots"]["status"] == "rendered" and len(result["plots"]["files"]) == 4
    assert_private_absent(output / "public")
    refreshed = report.build(study, output, workspace_root=tmp_path, coverage_path=coverage, make_plots=False, replace=True)
    assert refreshed["plots"]["files"] == []
    assert not list((output / "public/plots").glob("*.svg"))
    manifest = json.loads((output / "public/projection-manifest.json").read_text())
    assert set(manifest["files"]) == {"index.html", "results.json"}
    (output / "public/secret.env").write_text(SECRET)
    with pytest.raises(ValueError, match="not owned"):
        report.build(study, output, workspace_root=tmp_path, make_plots=False, replace=True)
    assert (output / "public/secret.env").read_text() == SECRET


def timing_fixture(tmp_path):
    study = tmp_path / "study"
    put(study / "plan.json", {"base_config": {}, "quality": {}})
    environment = {"gpu": {"name": "NVIDIA H100 80GB HBM3", "uuid": UUID}, "torch": {"version": "fixture"}}
    runtime = report.reference_identity(environment)
    verification = put(tmp_path / "verification.json", {"passed": True, "environment": environment})
    numerical = put(tmp_path / "numerical.json", {"passed": True, "implementation_sha256": "9" * 64, "runtime_identity": runtime})
    for variant, factor in (("eager", 1.), ("cached", .8)):
        blocks, aggregate_rows = [], []
        for block in range(4):
            path = study / "timing/blocks" / variant / str(block) / "run"
            config = {"model": {"backend": "vista", "checkpoint_sha256": "a" * 64}, "generation": {"steps": 50},
                "optimizations": {}, "data": {"manifest_sha256": "b" * 64}, "environment_verification": str(verification),
                "benchmark": {"workload": "single_future", "candidate_count": 1, "input_condition": "uncached", "requests": 50},
                "started_at": f"2026-01-02T0{block}:00:00+00:00"}
            put(path / "config.json", config); put(path / "environment.json", environment)
            rows = [{"status": "ok", "request_id": f"request-{i}", "scene_id": f"scene-{i % 2}", "repeat": 0, "request_index": i,
                "latency_s": factor * (1 + block * .02 + i * .001), "candidate_count": 1, "batch_size": 1,
                "candidates": [{"candidate_id": f"request-{i}:c00", "seed": 42 + i, "trajectory_sha256": "c" * 64, "frames_path": None}],
                "cache_stats": {"oom_attempts": []}} for i in range(50)]
            (path / "measurements.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in rows))
            put(path / "summary.json", {"backend": "vista", "headline_eligible": False, "failures": 0,
                "throughput_measurement_wall_s": sum(r["latency_s"] for r in rows)})
            put(path / "cold_start.json", {"status": "ready", "model_load_s": 3., "warmups": [{"status": "ok", "latency_s": 1.,
                "compiler": {"stats": {"unique_graphs": 0}}} for _ in range(3)]})
            put(path / "environment_gate.json", {"passed": True, "verification_sha256": file_sha256(verification)})
            put(path / "numerical_gate.json", {"passed": True, "report_path": str(numerical), "report_sha256": file_sha256(numerical)})
            put(path.parent / "process.json", {"finished_at": f"2026-01-02T0{block}:10:00+00:00"})
            blocks.append({"block": block, "path": str(path), "artifact_sha256": hashes(path), "median_s": float(np.median([r["latency_s"] for r in rows]))})
            for r in rows:
                aggregate_rows.append({**r, "request_index": len(aggregate_rows), "repeat": block, "timing_block": block,
                    "original_request_index": r["request_index"]})
        output = study / "timing/aggregate" / variant
        config["benchmark"]["requests"] = 200
        put(output / "config.json", config); put(output / "environment.json", environment)
        put(output / "summary.json", {"backend": "vista", "headline_eligible": True, "timing_blocks": blocks,
            "latency_p50_s": float(np.median([r["latency_s"] for r in aggregate_rows])),
            "latency_p95_s": float(np.quantile([r["latency_s"] for r in aggregate_rows], .95)),
            "latency_p50_ci95_s": [.9, 1.1], "latency_p95_ci95_s": [1., 1.2]})
        (output / "measurements.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in aggregate_rows))
    return study


def test_timing_reconstructs_blocks_and_never_promotes_a_combined_claim_without_quality(tmp_path):
    study = timing_fixture(tmp_path)
    result = report.build(study, tmp_path / "report", workspace_root=tmp_path, make_plots=False)
    assert result["timing"]["status"] == "complete"
    assert all(r["headline_timing_eligible"] for r in result["timing"]["runs"])
    assert all(r["requests"] == 200 and r["unique_clips"] == 50 for r in result["timing"]["runs"])
    assert result["headline_claims"] == [] and result["status"].startswith("incomplete")
    assert_private_absent(tmp_path / "report/public")
    path = study / "timing/aggregate/eager/measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["latency_s"] *= 2
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    changed = report.build(study, tmp_path / "changed", workspace_root=tmp_path, make_plots=False)
    assert not next(r for r in changed["timing"]["runs"] if r["variant"] == "eager")["headline_timing_eligible"]


def test_timing_rechecks_warmups_and_filters_unexpected_numeric_strings(tmp_path):
    study = timing_fixture(tmp_path)
    path = study / "timing/blocks/eager/0/run/cold_start.json"
    cold = json.loads(path.read_text()); cold["warmups"] = cold["warmups"][:1]; put(path, cold)
    path = study / "timing/aggregate/eager/summary.json"
    summary = json.loads(path.read_text()); summary["latency_p50_ci95_s"] = [1, PRIVATE]; put(path, summary)
    result = report.build(study, tmp_path / "report", workspace_root=tmp_path, make_plots=False)
    eager = next(r for r in result["timing"]["runs"] if r["variant"] == "eager")
    assert not eager["headline_timing_eligible"] and eager["p50_ci95_s"] is None
    assert_private_absent(tmp_path / "report/public")


def companion_fixture(tmp_path, monkeypatch, count=1):
    """Real raw block reconstruction; authoritative helper API is a CPU-only stub.

    Production calls the helper's full _load/compare_pilot validators, whose
    manifests, calibration blocks and overlay rejection have separate tests.
    """
    from types import SimpleNamespace
    study = timing_fixture(tmp_path)
    old = study / 'timing'
    directory = tmp_path / 'companion'
    old.rename(directory)
    for path in directory.rglob('*'):
        if path.is_file():
            path.write_text(path.read_text().replace(str(old), str(directory)))
    for variant in ('eager', 'cached'):
        aggregate = directory / 'aggregate' / variant
        cfg = json.loads((aggregate / 'config.json').read_text())
        cfg['benchmark'].update(candidate_count=count, workload='single_future' if count == 1 else 'candidate_set')
        put(aggregate / 'config.json', cfg)
        summary = json.loads((aggregate / 'summary.json').read_text())
        for block in summary['timing_blocks']:
            blockpath = Path(block['path'])
            bcfg = json.loads((blockpath / 'config.json').read_text())
            bcfg['benchmark'].update(candidate_count=count, workload=cfg['benchmark']['workload'])
            put(blockpath / 'config.json', bcfg)
            rows = [json.loads(line) for line in (blockpath / 'measurements.jsonl').read_text().splitlines()]
            for row in rows:
                row['candidate_count'] = count
                base = row['candidates'][0]
                row['candidates'] = [{**base, 'candidate_id': f"{row['request_id']}:c{i:02d}", 'seed': base['seed'] + i} for i in range(count)]
            (blockpath / 'measurements.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            block['artifact_sha256'] = hashes(blockpath)
        put(aggregate / 'summary.json', summary)
        rebuilt = []
        for number, block in enumerate(summary['timing_blocks']):
            for row in map(json.loads, (Path(block['path']) / 'measurements.jsonl').read_text().splitlines()):
                rebuilt.append({**row, 'request_index': len(rebuilt), 'repeat': number, 'timing_block': number,
                                'original_request_index': row['request_index']})
        (aggregate / 'measurements.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rebuilt))
    selection = put(study / 'selection.json', {'variants': ['eager', 'cached']})
    policy = put(tmp_path / 'pilot-policy.json', {'fixture': 'fabricated CPU calibration; no GPU measurement'})
    helper_path = Path(report.__file__).with_name('run_timing_companion.py')
    plan = {'parent_study': str(study), 'parent_plan_sha256': file_sha256(study / 'plan.json'),
        'selection_sha256': file_sha256(selection), 'quality_evidence_sha256': {PRIVATE: 'e' * 64},
        'pilot_policy': str(policy), 'pilot_policy_sha256': file_sha256(policy),
        'source_identity': '9' * 64, 'helper_sha256': file_sha256(helper_path),
        'baseline': 'eager', 'candidate': 'cached', 'candidate_count': count,
        'workload': 'single_future' if count == 1 else 'candidate_set',
        'configs': {name: json.loads((directory / 'aggregate' / name / 'config.json').read_text()) for name in ('eager', 'cached')},
        'numerical_proof_sha256': {str(tmp_path / 'numerical.json'): file_sha256(tmp_path / 'numerical.json')},
        'timing_runtime_overlay': {'bound_files_sha256': {PRIVATE: 'f' * 64}, 'secret': SECRET}}
    put(directory / 'plan.json', plan)
    put(directory / 'plan-receipt.json', {'plan_sha256': file_sha256(directory / 'plan.json')})

    def authoritative_load(path):
        path = Path(path)
        value = json.loads((path / 'plan.json').read_text())
        receipt = json.loads((path / 'plan-receipt.json').read_text())
        if (receipt['plan_sha256'] != file_sha256(path / 'plan.json')
                or value['parent_plan_sha256'] != file_sha256(study / 'plan.json')
                or value['selection_sha256'] != file_sha256(selection)
                or value['pilot_policy_sha256'] != file_sha256(policy)
                or value['helper_sha256'] != file_sha256(helper_path)):
            raise ValueError('CPU authoritative fixture binding changed')
        return path, value, study

    def authoritative_compare(policy_path, baseline_path, candidate_path):
        left, right = report.Evidence(tmp_path).run(baseline_path)[2], report.Evidence(tmp_path).run(candidate_path)[2]
        metrics = {}
        for label, quantile in (('p50', .5), ('p95', .95)):
            point, bounds = report._bootstrap_ratios(np.array([r['latency_s'] for r in left]),
                np.array([r['latency_s'] for r in right]), scene_groups=[r['scene_id'] for r in left], quantile=quantile)
            metrics[label] = {'relative_latency_change': point, 'ci95': bounds, 'tolerance': .03,
                              'exceeds_tolerance_and_uncertainty': False}
        return {'status': 'pass', 'metrics': metrics, 'paired_requests': len(left), 'independent_scenes': 2,
            'pilot_policy_verified': True, 'policy_sha256': file_sha256(policy_path),
            'baseline_artifact_sha256': hashes(Path(baseline_path)), 'candidate_artifact_sha256': hashes(Path(candidate_path))}

    monkeypatch.setattr(report, 'companion_helper', lambda: SimpleNamespace(_load=authoritative_load,
        pilot=SimpleNamespace(compare_pilot=authoritative_compare),
        study=SimpleNamespace(_selection=lambda *args: json.loads(selection.read_text()))))
    aggregates = {name: str(directory / 'aggregate' / name) for name in ('eager', 'cached')}
    comparison = authoritative_compare(policy, aggregates['eager'], aggregates['cached'])
    put(directory / 'comparison.json', comparison)
    put(directory / 'completed.json', {'status': 'complete', 'finished_at': '2026-01-03T00:00:00+00:00',
        'plan_sha256': file_sha256(directory / 'plan.json'), 'aggregate_runs': aggregates,
        'artifact_sha256': {name: hashes(Path(path)) for name, path in aggregates.items()},
        'comparison_sha256': file_sha256(directory / 'comparison.json'), 'performance_status': 'pass',
        'quality_generated_here': False, 'combined_headline_claim': False})
    return study, directory


@pytest.mark.parametrize('count', [1, 4])
def test_companion_reconstructs200_and_exports_private_free_evidence_with_quality_pending(tmp_path, monkeypatch, count):
    study, directory = companion_fixture(tmp_path, monkeypatch, count)
    result = report.build(study, tmp_path / 'report', workspace_root=tmp_path, make_plots=False,
                          timing_companions={'candidate4' if count == 4 else 'single': directory})
    companion = result['timing_companions'][0]
    assert result['timing']['runs'] == []  # A companion does not require the unused parent timing path.
    assert companion['status'] == 'verified'
    assert len(companion['runs']) == 2 and all(r['requests'] == 200 and r['candidate_count'] == count for r in companion['runs'])
    assert len(companion['runs'][0]['raw_measurements']) == 200
    assert companion['provenance']['runtime_overlay_present']
    assert companion['comparisons'][0]['metrics']['p50']['ci95'][1] < 0
    assert not companion['quality']['accepted'] and not companion['headline_combined_claim']
    assert result['headline_claims'] == []
    assert_private_absent(tmp_path / 'report/public')


@pytest.mark.parametrize('fault', ['completion', 'comparison', 'aggregate', 'block', 'cold', 'gpu', 'pilot', 'selection', 'plan'])
def test_companion_tampering_fails_closed(tmp_path, monkeypatch, fault):
    study, directory = companion_fixture(tmp_path, monkeypatch)
    if fault == 'completion':
        path = directory / 'completed.json'; value = json.loads(path.read_text()); value['status'] = 'running'; put(path, value)
    elif fault == 'comparison':
        path = directory / 'comparison.json'; value = json.loads(path.read_text()); value['metrics']['p50']['relative_latency_change'] = -.99; put(path, value)
        complete = json.loads((directory / 'completed.json').read_text())
        complete['comparison_sha256'] = file_sha256(path); put(directory / 'completed.json', complete)
    elif fault in {'aggregate', 'block'}:
        path = directory / ('aggregate/eager/measurements.jsonl' if fault == 'aggregate' else 'blocks/eager/0/run/measurements.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines()]; rows[0]['latency_s'] *= 2
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    elif fault == 'cold':
        path = directory / 'blocks/eager/0/run/cold_start.json'; value = json.loads(path.read_text()); value['warmups'] = []; put(path, value)
    elif fault == 'gpu':
        path = directory / 'blocks/eager/0/run/environment.json'; value = json.loads(path.read_text()); value['gpu']['uuid'] = 'GPU-DIFFERENT'; put(path, value)
    elif fault == 'pilot':
        put(tmp_path / 'pilot-policy.json', {'changed': True})
    elif fault == 'selection':
        put(study / 'selection.json', {'changed': True})
    else:
        put(directory / 'plan.json', {'changed': True})
    result = report.build(study, tmp_path / 'report', workspace_root=tmp_path, make_plots=False,
                          timing_companions={'single': directory})
    assert result['timing_companions'][0]['status'] == 'unavailable_or_invalid'
    assert not result['timing_companions'][0]['headline_combined_claim'] and not result['headline_claims']


def test_completed_review_with_rejected_candidate_is_not_acceptance(tmp_path):
    review = tmp_path / 'review'; review.mkdir()
    index = review / 'index.html'; index.write_text('blinded fixture')
    blank = review / 'review.csv'; blank.write_text('case_id,acceptable_A,acceptable_B\ncase-1,,\n')
    private = put(review / 'private_key.json', {'cases': [{'case_id': 'case-1', 'run': 'compiled', 'reference_run': 'upstream', 'A': 'upstream', 'B': 'compiled'}]})
    metric = put(tmp_path / 'metric.json', {'review': {'index': str(index), 'csv': str(blank), 'private_key': str(private)}})
    entry = {'run': 'compiled', 'metric_artifacts': [{'path': str(metric), 'sha256': file_sha256(metric)}]}
    votes = tmp_path / 'votes.csv'; votes.write_text('case_id,acceptable_A,acceptable_B\ncase-1,yes,no\n')
    ratings = {'reviewer_kind': 'agent_assisted', 'reviews': [{'index': str(index), 'index_sha256': file_sha256(index),
                                                           'csv': str(votes), 'csv_sha256': file_sha256(votes)}]}
    assert report.accepted_variant_reviews(report.Evidence(tmp_path), entry, ratings, 'upstream')[0] is False
    votes.write_text('case_id,acceptable_A,acceptable_B\ncase-1,yes,yes\n')
    assert report.accepted_variant_reviews(report.Evidence(tmp_path), entry, ratings, 'upstream')[0] is False  # stale binding
    ratings['reviews'][0]['csv_sha256'] = file_sha256(votes)
    assert report.accepted_variant_reviews(report.Evidence(tmp_path), entry, ratings, 'upstream')[:2] == (True, 1)


@pytest.mark.parametrize('failure', [None, 'metric', 'review', 'implementation', 'source', 'runtime', 'manifest'])
def test_companion_combined_claim_requires_frozen_metric_and_accepted_review_join(tmp_path, monkeypatch, failure):
    """Exercise the real held-out gate with fabricated metadata, never GPU results."""
    from copy import deepcopy
    from drivebench.evaluate import aggregate_pairs
    study, directory = companion_fixture(tmp_path, monkeypatch)
    plan = json.loads((directory / 'plan.json').read_text())
    parent_plan = json.loads((study / 'plan.json').read_text())
    parent_plan.update(generation_sources_at_plan={'models/vista.py': '9' * 64}, manifests={})
    policy = put(study / 'reviewed-policy.json', {
        'policy_type': 'reviewed_metric_tolerances', 'calibrated': True, 'frozen': True,
        'selected_on': 'development', 'development_manifest_sha256': 'd' * 64,
        'development_scene_ids': [f'dev-{i}' for i in range(32)], 'frozen_at': '2026-01-01T00:00:00+00:00',
        'metrics': {name: {'maximum': .01, 'aggregation': 'max' if name == 'max_abs_error' else 'mean'}
                    for name in ('max_abs_error', 'lpips', 'temporal_derivative_mae')}})
    selection = put(study / 'selection.json', {'quality_policy': str(policy), 'exact_runs': [],
        'reviewed_quality_policies': {'eager': {'path': str(policy), 'calibrated_variants': ['cached']}}})
    ratings = {'reviewer_kind': 'agent_assisted', 'reviews': []}
    for phase, count in (('evaluation', 128), ('evaluation_second_seed', 16)):
        folder = study / 'quality' / phase
        aggregate = folder / 'aggregate'
        metadata = {'manifest_sha256': 'a' * 64, 'partition_role': f'held_out_{phase}',
                    'clip_count': count, 'scene_ids': [f'eval-{i}' for i in range(count)]}
        parent_plan['manifests'][phase] = {'sha256': metadata['manifest_sha256'], 'metadata': metadata}
        pairs = [{'request_id': f'request-{i}', 'scene_id': f'eval-{i}', 'candidate_id': f'request-{i}:c00',
                  'seed': i, 'repeat': 0, 'metrics': {name: .02 if failure == 'metric' else .001 for name in report.METRICS}}
                 for i in range(count)]
        for name in ('eager', 'cached'):
            run = aggregate / 'runs' / name
            cfg = deepcopy(plan['configs'][name])
            cfg['data'] = {'manifest_sha256': metadata['manifest_sha256'], 'manifest_metadata': metadata}
            cfg['study_provenance'] = {'generation_sources': parent_plan['generation_sources_at_plan']}
            cfg['benchmark']['requests'] = count
            if failure == 'implementation' and name == 'cached':
                cfg['optimizations'] = {'unexpected_option': True}
            if failure == 'manifest' and name == 'cached':
                cfg['data']['manifest_sha256'] = 'f' * 64
            environment = deepcopy(json.loads((tmp_path / 'verification.json').read_text())['environment'])
            environment['gpu']['uuid'] = 'GPU-DOCUMENTED-REPLACEMENT'
            put(run / 'config.json', cfg)
            put(run / 'environment.json', environment)
            put(run / 'summary.json', {'backend': 'vista', 'headline_eligible': False})
            rows = [{'status': 'ok', 'request_id': p['request_id'], 'scene_id': p['scene_id'], 'latency_s': 1.,
                     'candidates': [{'candidate_id': p['candidate_id'], 'seed': p['seed']}]} for p in pairs]
            (run / 'measurements.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            source = folder / 'sources' / name
            source.mkdir(parents=True)
            for filename in report.CORE:
                (source / filename).write_bytes((run / filename).read_bytes())
            if failure == 'source' and name == 'cached':
                stale = deepcopy(cfg); stale['study_provenance'] = {'generation_sources': {'models/vista.py': '0' * 64}}
                put(source / 'config.json', stale)
            if failure == 'runtime' and name == 'cached':
                environment['torch']['version'] = 'different-runtime'; put(source / 'environment.json', environment)
            cfg['study_aggregation'] = {'source_runs': [{'path': str(source), 'artifact_sha256': hashes(source), 'adopted': False}]}
            put(run / 'config.json', cfg)
        reviewdir = folder / 'shards/000/metrics/blinded-review'
        reviewdir.mkdir(parents=True)
        index = reviewdir / 'index.html'; index.write_text('Fabricated blinded metadata fixture')
        blank = reviewdir / 'review.csv'; blank.write_text('case_id,acceptable_A,acceptable_B\ncase-1,,\n')
        private = put(reviewdir / 'private_key.json', {'cases': [{'case_id': 'case-1', 'run': 'cached',
            'reference_run': 'eager', 'A': 'eager', 'B': 'cached'}]})
        review = {'index': str(index), 'csv': str(blank), 'private_key': str(private)}
        votes = folder / 'rated.csv'
        votes.write_text(f'case_id,acceptable_A,acceptable_B\ncase-1,yes,{"no" if failure == "review" else "yes"}\n')
        ratings['reviews'].append({'index': str(index), 'index_sha256': file_sha256(index),
                                  'csv': str(votes), 'csv_sha256': file_sha256(votes)})
        reference = aggregate / 'runs/eager'; candidate = aggregate / 'runs/cached'
        entry = {'run': 'cached', 'run_path': str(candidate), 'artifact_sha256': hashes(candidate),
                 'synthetic': False, 'complete_pairs': True, 'matched_pairs': count, 'pairs': pairs,
                 'aggregates': aggregate_pairs(pairs), 'manifest_metadata': metadata, 'errors': [],
                 'gate': {'quality_preserving': False, 'within_frozen_metric_tolerances': False},
                 'quality_policy': {'path': str(policy), 'sha256': file_sha256(policy), 'reference': 'eager'}}
        metric = put(folder / 'shards/000/metrics/eager--cached.json', {'review': review, 'tolerances_path': str(policy),
            'runs': [entry], 'reference_artifact_sha256': hashes(reference), 'evaluator_source_sha256': {'fixture.py': 'e' * 64}})
        receipt = put(metric.with_suffix('.receipt.json'), {'evaluation_sha256': file_sha256(metric),
            'reference_artifact_sha256': hashes(reference), 'candidate_artifact_sha256': hashes(candidate),
            'evaluator_source_sha256': {'fixture.py': 'e' * 64}, 'tolerances_sha256': file_sha256(policy),
            'review_artifact_sha256': {str(p): file_sha256(p) for p in reviewdir.iterdir()}})
        entry['metric_artifacts'] = [{'path': str(metric), 'sha256': file_sha256(metric)}]
        put(aggregate / 'evaluation-vs-eager.json', {'reference': str(reference), 'reference_artifact_sha256': hashes(reference),
            'runs': [entry], 'reviews': [review], 'perceptual': {'status': 'available'}})
        put(folder / 'shards/000/complete.json', {'metric_artifacts': {str(p): file_sha256(p) for p in (metric, receipt)},
            'runs': {name: {'path': str(aggregate / 'runs' / name), 'artifact_sha256': hashes(aggregate / 'runs' / name)}
                     for name in ('eager', 'cached')}})
        put(folder / 'aggregation-receipt.json', {'artifact_sha256': {str(p): file_sha256(p) for p in aggregate.rglob('*') if p.is_file()}})
    plan['selection_sha256'] = file_sha256(selection)
    put(study / 'plan.json', parent_plan)
    plan['parent_plan_sha256'] = file_sha256(study / 'plan.json')
    put(directory / 'plan.json', plan)
    put(directory / 'plan-receipt.json', {'plan_sha256': file_sha256(directory / 'plan.json')})
    complete = json.loads((directory / 'completed.json').read_text())
    complete['plan_sha256'] = file_sha256(directory / 'plan.json'); put(directory / 'completed.json', complete)
    result = report.build(study, tmp_path / 'report', workspace_root=tmp_path, make_plots=False,
        timing_companions={'single': directory}, review_ratings=put(tmp_path / 'ratings.json', ratings))
    companion = result['timing_companions'][0]
    assert companion['status'] == 'verified'
    assert companion['quality']['accepted'] is (failure is None)
    assert companion['headline_combined_claim'] is (failure is None)
    assert bool(result['headline_claims']) is (failure is None)
    assert_private_absent(tmp_path / 'report/public')
