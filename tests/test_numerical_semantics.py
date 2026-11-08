"""Tiny synthetic retained arrays; these are never evidence about a GPU model."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from drivebench.config import digest, file_sha256

SPEC = importlib.util.spec_from_file_location("semantics", Path(__file__).resolve().parents[1] / "scripts/verify_numerical_semantics.py")
semantics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(semantics)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def proof_fixture(tmp_path, *, cache=True):
    normal = {key: np.full((2, 2, 3, 3), value, np.float32) for key, value in [("a", .25), ("b", .5)]}
    proof = {"schema_version": 2, "passed": True, "atol": .1, "rtol": 0,
             "implementation_sha256": semantics.source_identity(),
             "runtime_identity": {"gpu": {"uuid": "GPU-CPU-FIXTURE"}},
             "input_provenance": {"candidate_ids": ["a", "b"]}, "config": {"fixture": True},
             "generation": {"steps": 50, "total_frames": 2, "height": 2, "width": 3},
             "optimizations": {"conditioning_cache_bytes": 1024 if cache else 0},
             "runs": {}, "checks": {}}
    proof["input_provenance_sha256"] = digest(proof["input_provenance"])
    values = {"upstream": {"normal": normal}, "refactored": {"normal": normal},
              "isolation": {"normal": normal, "reverse": dict(reversed(list(normal.items()))),
                            "other_candidates_after_action_change": {"b": normal["b"]}}}
    if cache:
        values.update(uncached={"normal": normal}, cached={"normal": normal})
    for run, kinds in values.items():
        details = {"outputs": {}, "outputs_sha256": {}}
        for kind, arrays in kinds.items():
            path = tmp_path / f"{run}-{kind}.npz"
            np.savez(path, **arrays)
            details["outputs"][kind], details["outputs_sha256"][kind] = str(path), file_sha256(path)
        proof["runs"][run] = details
    if cache:
        proof["runs"]["cached"].update(cache_key_checks={k: True for k in (
            "unchanged_input_hit", "unchanged_key_equal", "changed_input_missed", "changed_key_differs")},
            cache_stats={"conditioning_embedding_hits": 2, "posterior_parameter_hits": 2,
                         "conditioning_cached_embedders": ["cond_frames", "cond_frames_without_noise"]})
    path = tmp_path / "proof.json"; put(path, proof)
    return path, proof, tmp_path / "semantics.json"


def change_array(proof, run, kind, *, candidate="b", amount=.01, dtype=None):
    path = Path(proof["runs"][run]["outputs"][kind])
    with np.load(path) as source:
        values = {key: source[key] for key in source.files}
    values[candidate] = values[candidate].copy()
    values[candidate].flat[0] += amount
    if dtype: values[candidate] = values[candidate].astype(dtype)
    np.savez(path, **values)
    proof["runs"][run]["outputs_sha256"][kind] = file_sha256(path)


def test_exact_pairs_bind_all_artifacts_without_mutating_proof(tmp_path):
    path, proof, sidecar = proof_fixture(tmp_path)
    before = file_sha256(path)
    result = semantics.create(path, sidecar)
    assert result["passed"] and result["atol"] == result["rtol"] == 0
    assert len(result["outputs_sha256"]) == 7
    assert result["configuration_sha256"] == digest({k: proof[k] for k in ("config", "generation", "optimizations")})
    assert semantics.verify(path, sidecar)["sha256"] == file_sha256(sidecar)
    assert file_sha256(path) == before
    with pytest.raises(FileExistsError): semantics.create(path, sidecar)


@pytest.mark.parametrize("run,kind", [("isolation", "reverse"), ("isolation", "other_candidates_after_action_change"), ("cached", "normal")])
def test_nonzero_tolerance_cannot_hide_semantic_difference(tmp_path, run, kind):
    path, proof, sidecar = proof_fixture(tmp_path)
    change_array(proof, run, kind)
    proof["checks"] = {"candidate_order": {"passed": True, "atol": .1, "rtol": 0}}
    put(path, proof)
    result = semantics.create(path, sidecar)
    assert not result["passed"]  # Failure is preserved, never promoted.
    with pytest.raises(ValueError, match="failed"): semantics.verify(path, sidecar)
    fabricated = json.loads(sidecar.read_text()); fabricated["passed"] = True
    put(sidecar, fabricated)
    with pytest.raises(ValueError): semantics.verify(path, sidecar)


@pytest.mark.parametrize("field", ["runtime_identity", "config", "generation", "optimizations", "input_provenance"])
def test_proof_identity_mutations_invalidate_sidecar(tmp_path, field):
    path, proof, sidecar = proof_fixture(tmp_path)
    semantics.create(path, sidecar)
    proof[field]["changed"] = True
    if field == "input_provenance": proof["input_provenance_sha256"] = digest(proof[field])
    put(path, proof)
    with pytest.raises(ValueError): semantics.verify(path, sidecar)


def test_replaced_array_rejects_even_if_sidecar_pass_is_declared(tmp_path):
    path, proof, sidecar = proof_fixture(tmp_path)
    semantics.create(path, sidecar)
    change_array(proof, "isolation", "reverse")
    with pytest.raises(ValueError, match="checksum"): semantics.verify(path, sidecar)


@pytest.mark.parametrize("fault", ["missing_cache", "missing_reverse", "ids", "dtype", "nan", "cache_not_exercised", "cache_key", "source", "shape"])
def test_incomplete_or_invalid_semantic_evidence_fails_closed(tmp_path, fault):
    path, proof, sidecar = proof_fixture(tmp_path)
    if fault == "missing_cache": del proof["runs"]["uncached"]
    elif fault == "missing_reverse": del proof["runs"]["isolation"]["outputs"]["reverse"]
    elif fault == "ids": proof["input_provenance"]["candidate_ids"] = ["a", "other"]; proof["input_provenance_sha256"] = digest(proof["input_provenance"])
    elif fault == "dtype": change_array(proof, "isolation", "reverse", amount=0, dtype=np.float64)
    elif fault == "nan": change_array(proof, "isolation", "reverse", amount=np.nan)
    elif fault == "cache_not_exercised": proof["runs"]["cached"]["cache_stats"]["posterior_parameter_hits"] = 0
    elif fault == "cache_key": proof["runs"]["cached"]["cache_key_checks"]["changed_input_missed"] = False
    elif fault == "shape": proof["generation"]["height"] = 3
    else: proof["implementation_sha256"] = "0" * 64
    put(path, proof)
    if fault == "cache_not_exercised":
        assert not semantics.create(path, sidecar)["passed"]
        with pytest.raises(ValueError): semantics.verify(path, sidecar)
    else:
        with pytest.raises(ValueError): semantics.create(path, sidecar)


def test_no_cache_configuration_requires_only_isolation_arrays(tmp_path):
    path, _, sidecar = proof_fixture(tmp_path, cache=False)
    assert set(semantics.create(path, sidecar)["checks"]) == {"candidate_order", "unrelated_candidate_action_change"}
    assert semantics.verify(path, sidecar)


@pytest.mark.parametrize("value", [None, -1, True, float("nan"), float("inf")])
def test_missing_or_invalid_tolerance_never_silently_bypasses_sidecar(value):
    with pytest.raises(ValueError): semantics.requires_sidecar({"atol": value, "rtol": 0})


def test_nonzero_check_tolerance_also_requires_sidecar():
    assert semantics.requires_sidecar({"atol": 0, "rtol": 0, "checks": {"candidate_order": {"atol": .01, "rtol": 0}}})
