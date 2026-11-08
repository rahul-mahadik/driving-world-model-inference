"""Transport fixtures only; no public data or network requests."""
import hashlib
import importlib.util
import io
from pathlib import Path
import re
import threading
import urllib.error

import pytest

from test_nuscenes_subset_download import archive_bytes, archive_spec, A, OTHER
import tarfile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_nuscenes_ranges.py"
spec = importlib.util.spec_from_file_location("nuscenes_ranges", SCRIPT)
ranges = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ranges)


class Response(io.BytesIO):
    def __init__(self, payload, archive, start, end, url, mutate=None):
        super().__init__(payload)
        self.status = 206
        self.url = url
        self.headers = {"Content-Range": f"bytes {start}-{end}/{archive['size_bytes']}",
                        "Content-Length": str(end - start + 1), "ETag": '"' + archive["etag"] + '"'}
        if mutate:
            mutate(self)

    def geturl(self):
        return self.url


def range_fixture(data, archive, *, mutate=None, failure=None):
    calls = []
    lock = threading.Lock()
    def opener(request, timeout):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        assert match
        start, end = map(int, match.groups())
        assert request.get_header("If-match") == '"' + archive["etag"] + '"'
        assert request.get_header("Accept-encoding") == "identity"
        assert timeout <= 30
        assert request.full_url == ranges.CDN + archive["key"]
        with lock:
            calls.append((start, end))
            attempt = len(calls)
        if failure:
            failure(attempt)
        return Response(data[start:end + 1], archive, start, end, request.full_url, mutate)
    return opener, calls


def test_ordered_read_matches_complete_archive_and_original_integrity(tmp_path):
    data = archive_bytes([(A, b"selected" * 100, tarfile.REGTYPE), (OTHER, b"unselected", tarfile.REGTYPE)])
    archive = archive_spec(data)
    opener, calls = range_fixture(data, archive)
    with ranges.RangeResponse(archive, opener=opener, chunk_bytes=17) as stream:
        parts = []
        while block := stream.read(11):
            parts.append(block)
        assert b"".join(parts) == data
    assert sorted(calls) == [(s, min(s + 17, len(data)) - 1) for s in range(0, len(data), 17)]
    transport = ranges.range_opener({1: archive}, opener=opener, chunk_bytes=31)
    result = ranges.base.extract_verified_archive(archive, {A}, tmp_path, opener=transport)
    assert (tmp_path / A).read_bytes() == b"selected" * 100
    assert not (tmp_path / OTHER).exists()
    assert result["integrity"]["compressed_sha256"] == hashlib.sha256(data).hexdigest()
    assert result["integrity"]["computed_etag"] == archive["etag"]


@pytest.mark.parametrize("mutate", [
    lambda r: setattr(r, "status", 200),
    lambda r: setattr(r, "url", "https://other.example/changed"),
    lambda r: r.headers.update({"Content-Range": "bytes 1-16/99"}),
    lambda r: r.headers.update({"Content-Length": "999"}),
    lambda r: r.headers.update({"ETag": '"changed-1"'}),
    lambda r: r.headers.update({"Content-Encoding": "gzip"}),
])
def test_range_identity_failures_fail_closed_without_retry(mutate):
    data = b"archive bytes"
    archive = archive_spec(data)
    opener, calls = range_fixture(data, archive, mutate=mutate)
    with ranges.RangeResponse(archive, workers=1, opener=opener) as stream:
        with pytest.raises(ranges.base.IntegrityError, match="Range status"):
            stream.read()
    assert len(calls) == 1


def test_partial_network_body_retries_only_same_range_then_succeeds():
    data = b"a" * 55
    archive = archive_spec(data)
    good, calls = range_fixture(data, archive)
    def opener(*args, **kwargs):
        response = good(*args, **kwargs)
        if len(calls) == 1:
            response.truncate(8)
        return response
    with ranges.RangeResponse(archive, workers=1, opener=opener, retries=1) as stream:
        assert stream.read() == data
    assert calls == [(0, 54), (0, 54)]


def test_retry_bound_and_object_change_http412_not_retried():
    data = b"a" * 32
    archive = archive_spec(data)
    def unavailable(_):
        raise urllib.error.URLError("temporary network failure")
    opener, calls = range_fixture(data, archive, failure=unavailable)
    with ranges.RangeResponse(archive, workers=1, opener=opener, retries=1) as stream:
        with pytest.raises(urllib.error.URLError):
            stream.read()
    assert len(calls) == 2
    def changed(_):
        raise urllib.error.HTTPError(ranges.CDN + archive["key"], 412, "changed", {}, None)
    opener, calls = range_fixture(data, archive, failure=changed)
    with ranges.RangeResponse(archive, workers=1, opener=opener) as stream:
        with pytest.raises(ranges.base.IntegrityError, match="HTTP412"):
            stream.read()
    assert len(calls) == 1


def test_only_four_chunks_can_be_requested_before_consumption():
    data = bytes(range(128))
    archive = archive_spec(data)
    opener, calls = range_fixture(data, archive)
    four_started = threading.Event()
    def signal(*args, **kwargs):
        response = opener(*args, **kwargs)
        if len(calls) == 4:
            four_started.set()
        return response
    with ranges.RangeResponse(archive, opener=signal, chunk_bytes=16) as stream:
        assert four_started.wait(2)
        assert len(calls) == 4 and stream.next_start == 64
        assert stream.read(1) == data[:1]
        assert len(calls) == 4 and stream.next_start == 64
        assert stream.read(15) == data[1:16]
        assert stream.next_start == 80


def test_valid_tar_with_wrong_compressed_integrity_is_never_published(tmp_path):
    data = archive_bytes([(A, b"selected", tarfile.REGTYPE)])
    archive = archive_spec(data)
    changed = bytearray(data)
    changed[4] = 1  # Valid gzip header mtime; same decompressed tar, wrong whole-object hash.
    opener, _ = range_fixture(bytes(changed), archive)
    transport = ranges.range_opener({1: archive}, opener=opener, chunk_bytes=32)
    with pytest.raises(ranges.base.IntegrityError, match="computed multipart ETag"):
        ranges.base.extract_verified_archive(archive, {A}, tmp_path, opener=transport)
    assert not (tmp_path / A).exists()
    assert not list(tmp_path.glob(".drivebench-archive-*"))


def test_unknown_catalog_object_and_limits_rejected():
    archive = archive_spec(b"x")
    with pytest.raises(ranges.base.IntegrityError, match="exact catalog"):
        ranges.range_opener({1: archive})("https://unlisted.example/object")
    for kwargs in ({"workers": 5}, {"workers": 0}, {"retries": 3}, {"chunk_bytes": ranges.RANGE_BYTES + 1}):
        with pytest.raises(ValueError):
            ranges.RangeResponse(archive, **kwargs)
