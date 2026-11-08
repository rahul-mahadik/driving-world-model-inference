"""Subset acquisition uses tiny synthetic archives; no external requests."""
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import threading
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_nuscenes_subset.py"
spec = importlib.util.spec_from_file_location("nuscenes_subset_download", SCRIPT)
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)
A = "samples/CAM_FRONT/frame-a.jpg"
B = "sweeps/CAM_FRONT/frame-b.jpg"
OTHER = "samples/CAM_BACK/unselected.jpg"


def archive_bytes(members):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, data, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(data) if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = "../../outside.jpg"
            archive.addfile(member, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    return gzip.compress(raw.getvalue(), mtime=0)


def multipart_etag(data):
    parts = [hashlib.md5(data[i:i + dl.MULTIPART_BYTES]).digest() for i in range(0, len(data), dl.MULTIPART_BYTES)]
    return hashlib.md5(b"".join(parts)).hexdigest() + "-" + str(len(parts))


def archive_spec(data, number=1):
    key = "public/v1.0/v1.0-mini.tgz" if number == "mini" else f"public/v1.0/v1.0-trainval{number:02d}_blobs_camera.tgz"
    return {"key": key, "url": dl.BASE + key, "size_bytes": len(data), "etag": multipart_etag(data),
            "legacy_publisher_md5": hashlib.md5(data).hexdigest(), "multipart_part_size_bytes": dl.MULTIPART_BYTES}


class Response(io.BytesIO):
    status = 200

    def __init__(self, data, spec, *, fail_after=None, etag=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(spec["size_bytes"]), "ETag": '"' + (etag or spec["etag"]) + '"'}
        self.total_read, self.eof_reads, self.fail_after = 0, 0, fail_after

    def read(self, count=-1):
        if self.fail_after is not None and self.total_read >= self.fail_after:
            raise OSError("simulated network interruption")
        if self.fail_after is not None:
            count = min(count, self.fail_after - self.total_read)
        value = super().read(count)
        self.total_read += len(value)
        self.eof_reads += not value
        return value


def opener_for(data, spec, **kwargs):
    calls = []
    def open_response(*args, **kw):
        response = Response(data, spec, **kwargs)
        calls.append(response)
        return response
    return open_response, calls


def required_list(tmp_path, names=(A, B)):
    path = tmp_path / "required.txt"
    path.write_text("\n".join(names) + "\n")
    return path


def test_singapore_timezone_filenames_are_preserved_and_verified(tmp_path):
    name = "samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281441762460.jpg"
    required, _ = dl.required_files(required_list(tmp_path, (name,)))
    assert required == {name}
    data = archive_bytes([(name, b"singapore-front-camera", tarfile.REGTYPE)])
    spec = archive_spec(data)
    opener, _ = opener_for(data, spec)
    root = tmp_path / "data"
    result = dl.extract_verified_archive(spec, required, root, opener=opener)
    dl.validate_installed(root, result["files"])
    assert (root / name).read_bytes() == b"singapore-front-camera"


def test_streams_to_eof_and_publishes_only_exact_required_regular_frames(tmp_path):
    data = archive_bytes([(A, b"front", tarfile.REGTYPE), (OTHER, b"back", tarfile.REGTYPE),
                          (B, b"sweep", tarfile.REGTYPE)])
    spec = archive_spec(data)
    opener, calls = opener_for(data, spec)
    root = tmp_path / "data"
    result = dl.extract_verified_archive(spec, {A, B}, root, opener=opener)
    assert (root / A).read_bytes() == b"front"
    assert (root / B).read_bytes() == b"sweep"
    assert not (root / OTHER).exists()
    assert calls[0].total_read == len(data) and calls[0].eof_reads > 0
    assert result["integrity"]["compressed_sha256"] == hashlib.sha256(data).hexdigest()
    assert result["integrity"]["computed_etag"] == spec["etag"]
    assert result["files"][A]["sha256"] == hashlib.sha256(b"front").hexdigest()
    assert not list(root.glob(".drivebench-archive-*"))


@pytest.mark.parametrize("unsafe", ["../outside.jpg", "/tmp/outside.jpg", "samples/../../outside.jpg", "samples\\CAM_FRONT\\a.jpg"])
def test_unsafe_tar_member_rejects_entire_archive_before_publication(tmp_path, unsafe):
    data = archive_bytes([(A, b"first", tarfile.REGTYPE), (unsafe, b"bad", tarfile.REGTYPE)])
    spec = archive_spec(data)
    opener, _ = opener_for(data, spec)
    with pytest.raises(dl.IntegrityError, match="Unsafe"):
        dl.extract_verified_archive(spec, {A}, tmp_path / "data", opener=opener)
    assert not (tmp_path / "data" / A).exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_symlinks_hardlinks_and_special_members_are_never_extracted(tmp_path, kind):
    data = archive_bytes([(A, b"first", tarfile.REGTYPE), (OTHER, b"", kind)])
    spec = archive_spec(data)
    opener, _ = opener_for(data, spec)
    with pytest.raises(dl.IntegrityError, match="symlink, hardlink, or special"):
        dl.extract_verified_archive(spec, {A}, tmp_path / "data", opener=opener)
    assert not (tmp_path / "data" / A).exists()


def test_computed_etag_mismatch_keeps_all_selected_frames_unpublished(tmp_path):
    data = archive_bytes([(A, b"frame", tarfile.REGTYPE)])
    spec = archive_spec(data)
    spec["etag"] = "0" * 32 + "-1"  # GET agrees with discovery; actual bytes do not.
    opener, _ = opener_for(data, spec)
    with pytest.raises(dl.IntegrityError, match="computed multipart ETag"):
        dl.extract_verified_archive(spec, {A}, tmp_path / "data", opener=opener)
    assert not (tmp_path / "data" / A).exists()


def test_stale_legacy_md5_is_reported_when_current_multipart_checksum_passes(tmp_path):
    data = archive_bytes([(A, b"frame", tarfile.REGTYPE)])
    spec = archive_spec(data)
    spec["legacy_publisher_md5"] = "0" * 32
    opener, _ = opener_for(data, spec)
    result = dl.extract_verified_archive(spec, {A}, tmp_path / "data", opener=opener)
    assert not result["integrity"]["legacy_publisher_md5_match"]
    assert result["integrity"]["compressed_md5"] == hashlib.md5(data).hexdigest()


def test_get_etag_mismatch_is_rejected_before_reading_archive(tmp_path):
    data = archive_bytes([(A, b"frame", tarfile.REGTYPE)])
    spec = archive_spec(data)
    opener, calls = opener_for(data, spec, etag="0" * 32 + "-1")
    with pytest.raises(dl.IntegrityError, match="GET ETag"):
        dl.extract_verified_archive(spec, {A}, tmp_path / "data", opener=opener)
    assert calls[0].total_read == 0


def test_multipart_counter_handles_irregular_reads_and_exact_part_boundary():
    data = os.urandom(2 * dl.MULTIPART_BYTES + 11)
    spec = archive_spec(data)
    counted = dl.CountingReader(io.BytesIO(data), spec, threading.Event())
    while counted.read(131071):
        pass
    assert counted.computed_etag() == multipart_etag(data)
    exact = data[:2 * dl.MULTIPART_BYTES]
    counted = dl.CountingReader(io.BytesIO(exact), archive_spec(exact), threading.Event())
    while counted.read(100001):
        pass
    assert counted.computed_etag() == multipart_etag(exact)


def test_conflicting_existing_file_is_never_overwritten_or_partially_published(tmp_path):
    data = archive_bytes([(A, b"new-a", tarfile.REGTYPE), (B, b"new-b", tarfile.REGTYPE)])
    spec = archive_spec(data)
    opener, _ = opener_for(data, spec)
    root = tmp_path / "data"
    (root / B).parent.mkdir(parents=True)
    (root / B).write_bytes(b"user data")
    (root / B).chmod(0o444)
    with pytest.raises(dl.IntegrityError, match="refusing overwrite"):
        dl.extract_verified_archive(spec, {A, B}, root, opener=opener)
    assert (root / B).read_bytes() == b"user data" and not (root / A).exists()


def test_identical_readonly_existing_frame_is_reused_without_changing_inode(tmp_path):
    data = archive_bytes([(A, b"frame", tarfile.REGTYPE)])
    spec = archive_spec(data)
    opener, _ = opener_for(data, spec)
    root = tmp_path / "data"
    (root / A).parent.mkdir(parents=True)
    (root / A).write_bytes(b"frame")
    (root / A).chmod(0o444)
    inode = (root / A).stat().st_ino
    dl.extract_verified_archive(spec, {A}, root, opener=opener)
    assert (root / A).stat().st_ino == inode
    assert (root / A).stat().st_mode & 0o222 == 0


def test_existing_destination_symlink_is_refused(tmp_path):
    data = archive_bytes([(A, b"frame", tarfile.REGTYPE)])
    spec = archive_spec(data)
    opener, _ = opener_for(data, spec)
    root = tmp_path / "data"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    (root / "samples").symlink_to(outside, target_is_directory=True)
    with pytest.raises(dl.IntegrityError, match="symlink"):
        dl.extract_verified_archive(spec, {A}, root, opener=opener)
    assert not list(outside.iterdir())


def test_partial_network_failure_removes_staging_and_retry_is_bounded(tmp_path):
    data = archive_bytes([(A, b"frame", tarfile.REGTYPE)])
    spec = archive_spec(data)
    opener, calls = opener_for(data, spec, fail_after=20)
    root = tmp_path / "data"
    with pytest.raises(OSError, match="interruption"):
        dl.archive_with_retries(spec, {A}, root, retries=1, opener=opener)
    assert len(calls) == 2
    assert not (root / A).exists() and not list(root.glob(".drivebench-archive-*"))


def test_truncated_http_and_content_length_mismatch_do_not_publish(tmp_path):
    data = archive_bytes([(A, b"frame", tarfile.REGTYPE)])
    spec = archive_spec(data)
    opener, _ = opener_for(data[:-8], spec)
    with pytest.raises((EOFError, dl.IntegrityError)):
        dl.extract_verified_archive(spec, {A}, tmp_path / "data", opener=opener)
    assert not (tmp_path / "data" / A).exists()
    def wrong_length(*args, **kwargs):
        result = Response(data, spec)
        result.headers["Content-Length"] = str(len(data) + 1)
        return result
    with pytest.raises(dl.IntegrityError, match="Content-Length"):
        dl.extract_verified_archive(spec, {A}, tmp_path / "data", opener=wrong_length)


def test_missing_required_frames_leave_verified_resume_progress_but_fail_completion(tmp_path):
    data = archive_bytes([(A, b"frame", tarfile.REGTYPE)])
    spec = archive_spec(data)
    opener, _ = opener_for(data, spec)
    report = tmp_path / "report.json"
    with pytest.raises(dl.IntegrityError, match="1 required frames"):
        dl.download_subset(required_list(tmp_path), tmp_path / "data", {1: spec}, {}, report,
                           archives=(1,), opener=opener)
    proof = json.loads(report.read_text())
    assert proof["status"] == "incomplete" and proof["missing_files"] == [B]
    assert proof["verified_archives"]["1"]["status"] == "verified"


def test_verified_archive_resumes_only_with_same_list_and_existing_frame_hashes(tmp_path):
    data = archive_bytes([(A, b"frame", tarfile.REGTYPE)])
    spec = archive_spec(data)
    opener, calls = opener_for(data, spec)
    required, root, report = required_list(tmp_path, (A,)), tmp_path / "data", tmp_path / "report.json"
    args = (required, root, {1: spec}, {"source": "test"}, report)
    dl.download_subset(*args, archives=(1,), opener=opener)
    dl.download_subset(*args, archives=(1,), opener=opener)
    assert len(calls) == 1
    (root / A).write_bytes(b"tampered")
    with pytest.raises(dl.IntegrityError, match="verified SHA256"):
        dl.download_subset(*args, archives=(1,), opener=opener)
    assert len(calls) == 1
    (root / A).write_bytes(b"frame")
    required.write_text(A + "\n" + B + "\n")
    with pytest.raises(dl.IntegrityError, match="required list or publisher provenance"):
        dl.download_subset(*args, archives=(1,), opener=opener)
    assert len(calls) == 1


def test_parallel_workers_are_bounded_and_all_frames_reported(tmp_path):
    specs, payloads = {}, {}
    for number, name in enumerate((A, B, "samples/CAM_FRONT/frame-c.jpg"), 1):
        data = archive_bytes([(name, bytes([number]), tarfile.REGTYPE)])
        specs[number] = archive_spec(data, number)
        payloads[specs[number]["url"]] = (data, specs[number])
    active = maximum = 0
    lock = threading.Lock()
    class SlowResponse(Response):
        def __enter__(self):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(active, maximum)
            time.sleep(0.02)
            return self
        def __exit__(self, *args):
            nonlocal active
            with lock:
                active -= 1
            super().__exit__(*args)
    def opener(request, **kwargs):
        return SlowResponse(*payloads[request.full_url])
    result = dl.download_subset(required_list(tmp_path, (A, B, "samples/CAM_FRONT/frame-c.jpg")),
                                tmp_path / "data", specs, {}, tmp_path / "report.json",
                                archives=(1, 2, 3), workers=2, opener=opener)
    assert result["status"] == "complete" and result["verified_file_count"] == 3
    assert maximum == 2


def test_catalog_uses_current_size_and_etag_and_retains_legacy_md5(tmp_path):
    objects, lines = [], []
    for number in [*range(1, 11), "mini"]:
        data = archive_bytes([(A, b"frame", tarfile.REGTYPE)])
        spec = archive_spec(data, number)
        objects.append({"key": spec["key"], "size": spec["size_bytes"], "etag": '"' + spec["etag"] + '"'})
        lines.append(spec["legacy_publisher_md5"] + "  " + Path(spec["key"]).name)
    path = tmp_path / "objects.json"
    path.write_text(json.dumps({"bucket": "motional-nuscenes", "region": "ap-northeast-1", "objects": objects}))
    catalog, provenance = dl.load_catalog(path, "\n".join(lines).encode())
    assert len(catalog) == 11
    assert catalog["mini"]["key"] == "public/v1.0/v1.0-mini.tgz"
    assert catalog[1]["etag"] == multipart_etag(data)
    assert provenance["objects_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    objects[0]["etag"] = '"' + "0" * 32 + '-2"'
    path.write_text(json.dumps({"bucket": "motional-nuscenes", "region": "ap-northeast-1", "objects": objects}))
    with pytest.raises(dl.IntegrityError, match="8 MiB multipart layout"):
        dl.load_catalog(path, "\n".join(lines).encode())


def test_named_mini_archive_extracts_only_allowlisted_camera_frames(tmp_path):
    data = archive_bytes([(A, b"front", tarfile.REGTYPE), (OTHER, b"back", tarfile.REGTYPE),
                          ("v1.0-mini/sample.json", b"[]", tarfile.REGTYPE),
                          ("samples/LIDAR_TOP/points.pcd.bin", b"lidar", tarfile.REGTYPE)])
    spec = archive_spec(data, "mini")
    opener, calls = opener_for(data, spec)
    root = tmp_path / "data"
    result = dl.download_subset(required_list(tmp_path, (A,)), root, {"mini": spec}, {}, tmp_path / "report.json",
                                archives=("mini",), workers=10, opener=opener)
    assert result["status"] == "complete"
    assert set(result["verified_archives"]) == {"mini"}
    assert (root / A).read_bytes() == b"front"
    assert not (root / OTHER).exists()
    assert not (root / "v1.0-mini").exists()
    assert not (root / "samples/LIDAR_TOP").exists()
    assert len(calls) == 1


def test_archive_cli_allowlist_and_worker_bound(tmp_path):
    assert dl.archive_argument("mini") == "mini"
    assert dl.archive_argument("1") == 1 and dl.archive_argument("10") == 10
    for value in ("../mini", "test", "0", "11"):
        with pytest.raises(dl.argparse.ArgumentTypeError):
            dl.archive_argument(value)
    for workers in (0, 11):
        with pytest.raises(dl.IntegrityError, match="1..10"):
            dl.download_subset(tmp_path / "unused", tmp_path / "unused", {}, {}, tmp_path / "report", workers=workers)
