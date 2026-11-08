"""Offline merge tests: exact verified reports, immutable sources and root locks."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

from test_nuscenes_subset_download import archive_bytes, archive_spec, opener_for, A, B

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_nuscenes_subsets.py"
spec = importlib.util.spec_from_file_location("nuscenes_merge", SCRIPT)
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)
base = merge.base
C = "samples/CAM_FRONT/frame-c.jpg"
D = "samples/CAM_FRONT/frame-d.jpg"


@pytest.fixture
def dataset(tmp_path):
    tmp_path = tmp_path.resolve()
    required = tmp_path / "required.txt"
    required.write_text("\n".join((A, B, C)) + "\n")
    _, digest = base.required_files(required)
    roots = [tmp_path / name for name in ("original", "rescue", "extra")]
    data = [archive_bytes([(name, content, tarfile.REGTYPE)])
            for name, content in ((A, b"original"), (B, b"rescue"), (C, b"extra"))]
    catalog = {index: archive_spec(value, index) for index, value in enumerate(data, 1)}
    provenance = {"source": base.SOURCE, "objects_sha256": "1" * 64,
                  "publisher_checksums_sha256": "2" * 64}
    reports = []
    for index, root in enumerate(roots, 1):
        opener, _ = opener_for(data[index - 1], catalog[index])
        entry = base.extract_verified_archive(catalog[index], {A, B, C}, root, opener=opener)
        report = {"schema_version": 1, "required_file_list_sha256": digest,
                  "provenance": {**provenance, **({"transport": {"type": "ranges"}} if index > 1 else {})},
                  "archive_catalog": {str(key): value for key, value in catalog.items()},
                  "dataset_root": str(root), "status": "downloading", "required_count": 3,
                  "verified_archives": {str(index): entry}}
        path = tmp_path / f"report-{index}.json"
        path.write_text(json.dumps(report))
        reports.append(path)
    return {"required": required, "roots": roots, "reports": reports, "catalog": catalog,
            "provenance": provenance, "out": tmp_path / "merged.json"}


def run(data, *, apply=True, count=3):
    return merge.merge_subsets(data["required"], data["roots"][0],
                               list(zip(data["roots"], data["reports"]))[:count],
                               data["catalog"], data["provenance"], data["out"], apply=apply)


def mutate(data, index, change):
    path = data["reports"][index]
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_merges_three_sources_without_changing_source_reports_and_is_resume_compatible(dataset):
    before = [path.read_bytes() for path in dataset["reports"]]
    a_inode = (dataset["roots"][0] / A).stat().st_ino
    result = run(dataset)
    assert result["status"] == "complete" and result["verified_file_count"] == 3
    assert result["merge"]["new_file_count"] == 2
    assert len(result["merge"]["sources"]) == 3
    assert (dataset["roots"][0] / A).stat().st_ino == a_inode
    assert [path.read_bytes() for path in dataset["reports"]] == before
    assert result["provenance"] == dataset["provenance"]
    # Use a separate resume artifact: the merge receipt remains immutable.
    resume = dataset["out"].with_name("resume.json")
    resume.write_bytes(dataset["out"].read_bytes())
    def no_network(*args, **kwargs):
        pytest.fail("All archives are verified; resuming must not perform a GET")
    resumed = base.download_subset(dataset["required"], dataset["roots"][0], dataset["catalog"],
                                   dataset["provenance"], resume, archives=(1, 2, 3), opener=no_network)
    assert resumed["status"] == "complete"


def test_dry_run_does_not_publish_images_or_report(dataset):
    result = run(dataset, apply=False)
    assert result["dry_run"] and result["report_would_be"]["status"] == "complete"
    assert not dataset["out"].exists()
    assert not (dataset["roots"][0] / B).exists()


@pytest.mark.parametrize("index", [0, 1, 2])
def test_every_source_lock_blocks_merge(dataset, index):
    with (dataset["roots"][index] / ".drivebench-subset.lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(base.IntegrityError, match="busy"):
            run(dataset)
    assert not dataset["out"].exists()
    assert not (dataset["roots"][0] / B).exists()


@pytest.mark.parametrize("field,value", [("required_file_list_sha256", "0" * 64),
                                          ("dataset_root", "/wrong/root"),
                                          ("required_count", 4)])
def test_wrong_selection_or_root_is_rejected(dataset, field, value):
    mutate(dataset, 1, lambda report: report.update({field: value}))
    with pytest.raises(base.IntegrityError, match="binding/root"):
        run(dataset)
    assert not (dataset["roots"][0] / B).exists()


def test_wrong_catalog_provenance_is_rejected(dataset):
    mutate(dataset, 1, lambda report: report["provenance"].update(objects_sha256="0" * 64))
    with pytest.raises(base.IntegrityError, match="binding/root"):
        run(dataset)


def test_bad_archive_integrity_is_rejected_before_publication(dataset):
    mutate(dataset, 2, lambda report: report["verified_archives"]["3"]["integrity"].update(computed_etag="wrong"))
    with pytest.raises(base.IntegrityError, match="integrity evidence"):
        run(dataset)
    assert not (dataset["roots"][0] / B).exists()


def test_corrupted_source_is_rejected_before_any_file_is_published(dataset):
    (dataset["roots"][2] / C).write_bytes(b"corrupt")
    with pytest.raises(base.IntegrityError, match="Source frame differs"):
        run(dataset)
    assert not (dataset["roots"][0] / B).exists()


def test_symlink_source_is_rejected(dataset):
    path = dataset["roots"][2] / C
    path.unlink()
    path.symlink_to(dataset["roots"][0] / A)
    with pytest.raises(base.IntegrityError, match="symlink"):
        run(dataset)


def test_existing_conflict_rejects_complete_merge_without_overwrite(dataset):
    path = dataset["roots"][0] / C
    path.write_bytes(b"user data")
    path.chmod(0o444)
    with pytest.raises(base.IntegrityError, match="refusing overwrite"):
        run(dataset)
    assert path.read_bytes() == b"user data" and not (dataset["roots"][0] / B).exists()


def test_identical_readonly_destination_is_reused(dataset):
    path = dataset["roots"][0] / B
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((dataset["roots"][1] / B).read_bytes())
    path.chmod(0o444)
    inode = path.stat().st_ino
    result = run(dataset)
    assert path.stat().st_ino == inode and result["merge"]["new_file_count"] == 1
    assert path.stat().st_mode & 0o222 == 0


def test_partial_coverage_is_honestly_incomplete(dataset):
    result = run(dataset, count=2)
    assert result["status"] == "incomplete" and result["missing_files"] == [C]
    assert result["verified_file_count"] == 2


def test_existing_report_is_never_overwritten(dataset):
    dataset["out"].write_text("earlier evidence")
    with pytest.raises(base.IntegrityError, match="Output report exists"):
        run(dataset)
    assert dataset["out"].read_text() == "earlier evidence"


def test_duplicate_archive_with_different_bytes_is_rejected(dataset):
    first = json.loads(dataset["reports"][0].read_text())["verified_archives"]["1"]
    source = dataset["roots"][1] / A
    source.parent.mkdir(parents=True)
    source.write_bytes((dataset["roots"][0] / A).read_bytes())
    duplicate = copy.deepcopy(first)
    duplicate["integrity"]["compressed_sha256"] = "0" * 64
    mutate(dataset, 1, lambda report: report["verified_archives"].update({"1": duplicate}))
    with pytest.raises(base.IntegrityError, match="Duplicate verified archive"):
        run(dataset)


def test_agreeing_duplicate_archive_accepts_get_quote_difference(dataset):
    duplicate = json.loads(dataset["reports"][0].read_text())["verified_archives"]["1"]
    source = dataset["roots"][1] / A
    source.parent.mkdir(parents=True)
    source.write_bytes((dataset["roots"][0] / A).read_bytes())
    duplicate["integrity"]["get_etag"] = duplicate["integrity"]["get_etag"].strip('"')
    mutate(dataset, 1, lambda report: report["verified_archives"].update({"1": duplicate}))
    assert run(dataset)["status"] == "complete"
