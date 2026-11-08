#!/usr/bin/env python3
"""Stream official nuScenes camera archives; publish only checksum-verified frames.

Run this CLI on the Linux GPU host. No complete archive is stored locally.
Default: all ten trainval camera archives, two streams, one network retry.
Use --archives mini for the official mini archive; at most ten workers.
An incomplete archive selection exits nonzero but retains verified progress.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request

SOURCE = "https://registry.opendata.aws/motional-nuscenes/"
BASE = "https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/"
CHECKSUM_KEY = "public/v1.0/md5.checksum"
CHUNK = 1024 * 1024
MAX_FRAME_BYTES = 20 * 1024 * 1024
MULTIPART_BYTES = 8 * 1024 * 1024
PRINT_LOCK = threading.Lock()


class IntegrityError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


def emit(event, **values):
    with PRINT_LOCK:
        print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_path(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise IntegrityError("Unsafe archive path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise IntegrityError("Unsafe archive path")
    return path.as_posix()


def camera_path(name):
    normalized = canonical_path(name)
    # Singapore recordings include a literal +0800 timezone in their filenames.
    if not re.fullmatch(r"(?:samples|sweeps)/CAM_FRONT/[A-Za-z0-9_.+-]+\.jpg", normalized):
        raise IntegrityError("Required paths must be exact samples/sweeps CAM_FRONT JPEG paths")
    return normalized


def required_files(path):
    raw = Path(path).read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise IntegrityError("Required-file list exceeds 16 MiB")
    names = [camera_path(line) for line in raw.decode("utf-8").splitlines() if line]
    if not names or len(names) != len(set(names)):
        raise IntegrityError("Required-file list is empty or contains duplicate paths")
    return frozenset(names), sha256_bytes(raw)


def directory_without_symlinks(path):
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        if not stat.S_ISDIR(current.lstat().st_mode):
            raise IntegrityError("Destination directory contains a symlink or non-directory")
    return path


def file_identity(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise IntegrityError("Existing frame is not a regular file")
            digest, size = hashlib.sha256(), 0
            for block in iter(lambda: stream.read(CHUNK), b""):
                digest.update(block)
                size += len(block)
        return {"sha256": digest.hexdigest(), "size_bytes": size}
    except OSError as exc:
        raise IntegrityError("Existing frame is missing, unreadable, or a symlink") from exc


def load_catalog(objects_path, checksum_bytes):
    raw = Path(objects_path).read_bytes()
    objects = json.loads(raw)
    if objects.get("bucket") != "motional-nuscenes" or objects.get("region") != "ap-northeast-1":
        raise IntegrityError("Object discovery must identify the official nuScenes bucket and region")
    sizes, etags = {}, {}
    for item in objects["objects"]:
        if item["key"] in sizes:
            raise IntegrityError("Duplicate object discovery key")
        sizes[item["key"]] = item["size"]
        etags[item["key"]] = item.get("etag", "").strip('"')
    hashes = {}
    for line in checksum_bytes.decode("ascii").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{32})\s+\*?([^/\s]+)", line.strip())
        if not match or match[2] in hashes:
            raise IntegrityError("Invalid publisher checksum list")
        hashes[match[2]] = match[1].lower()
    catalog = {}
    for number in [*range(1, 11), "mini"]:
        name = "v1.0-mini.tgz" if number == "mini" else f"v1.0-trainval{number:02d}_blobs_camera.tgz"
        key = "public/v1.0/" + name
        size = sizes.get(key)
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or name not in hashes:
            raise IntegrityError("Missing archive size or publisher MD5")
        etag = etags.get(key, "")
        match = re.fullmatch(r"[0-9a-f]{32}-([1-9][0-9]*)", etag)
        if not match or int(match[1]) != (size + MULTIPART_BYTES - 1) // MULTIPART_BYTES:
            raise IntegrityError("Object ETag does not match the expected 8 MiB multipart layout")
        catalog[number] = {"key": key, "url": BASE + key, "size_bytes": size,
                           "legacy_publisher_md5": hashes[name], "etag": etag,
                           "multipart_part_size_bytes": MULTIPART_BYTES}
    provenance = {"source": SOURCE, "bucket": objects["bucket"], "region": objects["region"],
                  "objects_sha256": sha256_bytes(raw), "publisher_checksums_url": BASE + CHECKSUM_KEY,
                  "publisher_checksums_sha256": sha256_bytes(checksum_bytes)}
    return catalog, provenance


def fetch_checksums(opener=urllib.request.urlopen):
    with opener(BASE + CHECKSUM_KEY, timeout=30) as response:
        if response.status != 200:
            raise IntegrityError("Publisher checksum request did not return HTTP 200")
        value = response.read(5001)
    if len(value) > 5000:
        raise IntegrityError("Publisher checksum response exceeds 5 KB")
    return value


class CountingReader:
    def __init__(self, response, archive, cancel):
        self.response, self.archive, self.cancel = response, archive, cancel
        self.digest, self.size = hashlib.md5(usedforsecurity=False), 0
        self.sha256 = hashlib.sha256()
        self.part_digest = hashlib.md5(usedforsecurity=False)
        self.part_size, self.parts = 0, []
        self.last_progress = time.monotonic()

    def read(self, size=-1):
        if self.cancel.is_set():
            raise Cancelled("Another archive failed; transfer cancelled")
        data = self.response.read(CHUNK if size < 0 else min(size, CHUNK))
        self.size += len(data)
        if self.size > self.archive["size_bytes"]:
            raise IntegrityError("Compressed response exceeds publisher object size")
        self.digest.update(data)
        self.sha256.update(data)
        remaining = memoryview(data)
        while remaining:
            count = min(len(remaining), MULTIPART_BYTES - self.part_size)
            self.part_digest.update(remaining[:count])
            self.part_size += count
            remaining = remaining[count:]
            if self.part_size == MULTIPART_BYTES:
                self.parts.append(self.part_digest.digest())
                self.part_digest = hashlib.md5(usedforsecurity=False)
                self.part_size = 0
        if time.monotonic() - self.last_progress >= 30:
            emit("download_progress", key=self.archive["key"], received_bytes=self.size,
                 expected_bytes=self.archive["size_bytes"])
            self.last_progress = time.monotonic()
        return data

    def computed_etag(self):
        parts = self.parts + ([self.part_digest.digest()] if self.part_size else [])
        return hashlib.md5(b"".join(parts), usedforsecurity=False).hexdigest() + "-" + str(len(parts))


def validate_installed(root, files):
    for name, identity in files.items():
        if camera_path(name) != name:
            raise IntegrityError("Noncanonical path in verification report")
        directory_without_symlinks(root / PurePosixPath(name).parent)
        if file_identity(root / name) != identity:
            raise IntegrityError("Installed frame differs from its verified SHA256/length")


def publish(root, staging, files, lock):
    # Validate the complete selected set before publishing any member of this archive.
    with lock:
        for name, identity in files.items():
            destination = root / name
            directory_without_symlinks(destination.parent)
            if os.path.lexists(destination) and file_identity(destination) != identity:
                raise IntegrityError("Existing frame conflicts with verified archive; refusing overwrite")
        for name, identity in files.items():
            destination = root / name
            try:
                os.link(staging / name, destination, follow_symlinks=False)
            except FileExistsError:
                if file_identity(destination) != identity:
                    raise IntegrityError("Concurrent frame differs; refusing overwrite")


def extract_verified_archive(archive, required, root, *, opener=urllib.request.urlopen,
                             cancel=None, publication_lock=None):
    cancel = cancel or threading.Event()
    publication_lock = publication_lock or threading.Lock()
    root = directory_without_symlinks(root)
    with tempfile.TemporaryDirectory(prefix=".drivebench-archive-", dir=root) as temp:
        staging, files = Path(temp), {}
        request = urllib.request.Request(archive["url"], headers={"User-Agent": "drivebench-nuscenes/1", "Accept-Encoding": "identity"})
        with opener(request, timeout=60) as response:
            if response.status != 200:
                raise IntegrityError("Archive request did not return HTTP 200")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != archive["size_bytes"]:
                raise IntegrityError("Archive Content-Length differs from discovered object size")
            etag = response.headers.get("ETag", "")
            if etag not in {archive["etag"], '"' + archive["etag"] + '"'}:
                raise IntegrityError("GET ETag differs from current object discovery")
            counted = CountingReader(response, archive, cancel)
            with gzip.GzipFile(fileobj=counted, mode="rb") as decompressed:
                with tarfile.open(fileobj=decompressed, mode="r|") as bundle:
                    for member in bundle:
                        name = canonical_path(member.name)
                        if not (member.isdir() or member.isreg()):
                            raise IntegrityError("Archive contains a symlink, hardlink, or special member")
                        if name not in required:
                            continue
                        if not member.isreg() or not 0 < member.size <= MAX_FRAME_BYTES or name in files:
                            raise IntegrityError("Required frame is duplicate, nonregular, empty, or oversized")
                        destination = staging / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        digest, size = hashlib.sha256(), 0
                        source = bundle.extractfile(member)
                        if source is None:
                            raise IntegrityError("Required regular member has no data")
                        with source, destination.open("xb") as target:
                            for block in iter(lambda: source.read(CHUNK), b""):
                                digest.update(block)
                                size += len(block)
                                target.write(block)
                        if size != member.size:
                            raise IntegrityError("Truncated selected frame")
                        files[name] = {"sha256": digest.hexdigest(), "size_bytes": size}
                # tar iteration can stop before gzip trailer/HTTP EOF. Verify both.
                while decompressed.read(CHUNK):
                    pass
            while counted.read(CHUNK):
                pass
            if counted.size != archive["size_bytes"] or counted.computed_etag() != archive["etag"]:
                raise IntegrityError("Compressed archive byte count or computed multipart ETag mismatch")
            integrity = {"compressed_size_bytes": counted.size, "compressed_sha256": counted.sha256.hexdigest(),
                         "compressed_md5": counted.digest.hexdigest(), "computed_etag": counted.computed_etag(),
                         "get_etag": etag, "multipart_part_size_bytes": MULTIPART_BYTES,
                         "legacy_publisher_md5_match": counted.digest.hexdigest() == archive["legacy_publisher_md5"]}
            if not integrity["legacy_publisher_md5_match"]:
                emit("legacy_md5_disagrees", key=archive["key"], current_etag_verified=True)
        if cancel.is_set():
            raise Cancelled("Transfer cancelled before publication")
        publish(root, staging, files, publication_lock)
    return {"archive": archive, "status": "verified", "files": files, "integrity": integrity,
            "verified_at": datetime.now(timezone.utc).isoformat()}


def archive_with_retries(archive, required, root, *, retries=1, opener=urllib.request.urlopen,
                         cancel=None, publication_lock=None):
    for attempt in range(retries + 1):
        try:
            return extract_verified_archive(archive, required, root, opener=opener,
                                            cancel=cancel, publication_lock=publication_lock)
        except (OSError, urllib.error.URLError, http.client.IncompleteRead, EOFError, tarfile.ReadError) as exc:
            if attempt == retries or (cancel is not None and cancel.is_set()):
                raise
            emit("archive_retry", key=archive["key"], next_attempt=attempt + 2, error=type(exc).__name__)
    raise AssertionError("Unreachable")


def atomic_report(path, value):
    path = Path(path)
    directory_without_symlinks(path.parent)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".subset-report-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def download_subset(required_path, root, catalog, provenance, report_path, *, archives=tuple(range(1, 11)),
                    workers=2, retries=1, opener=urllib.request.urlopen):
    if workers not in range(1, 11) or retries not in {0, 1, 2}:
        raise IntegrityError("Workers must be 1..10 and retries 0..2")
    if not archives or len(set(archives)) != len(archives) or any(n not in catalog for n in archives):
        raise IntegrityError("Invalid or duplicate archive selection")
    required, list_hash = required_files(required_path)
    root = directory_without_symlinks(root)
    report_path = Path(os.path.abspath(report_path))
    directory_without_symlinks(report_path.parent)
    lock_path = root / ".drivebench-subset.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise IntegrityError("Another subset downloader holds the dataset lock") from None
        return _download_locked(required, list_hash, root, catalog, provenance, report_path,
                                archives, workers, retries, opener)
    finally:
        os.close(lock_fd)


def _download_locked(required, list_hash, root, catalog, provenance, report_path, archives, workers, retries, opener):
    binding = {"required_file_list_sha256": list_hash, "provenance": provenance,
               "archive_catalog": {str(n): spec for n, spec in catalog.items()}}
    report = {"schema_version": 1, **binding, "verified_archives": {}, "required_count": len(required)}
    if os.path.lexists(report_path):
        if report_path.is_symlink() or not report_path.is_file():
            raise IntegrityError("Verification report must be a regular file")
        old = json.loads(report_path.read_text())
        if old.get("schema_version") != 1 or any(old.get(key) != value for key, value in binding.items()):
            raise IntegrityError("Resume report does not match required list or publisher provenance")
        if not isinstance(old.get("verified_archives"), dict):
            raise IntegrityError("Invalid verified archive report")
        for key, entry in old["verified_archives"].items():
            if key not in binding["archive_catalog"] or entry.get("archive") != binding["archive_catalog"][key] or entry.get("status") != "verified":
                raise IntegrityError("Invalid verified archive identity")
            integrity = entry.get("integrity", {})
            if (integrity.get("computed_etag") != entry["archive"]["etag"]
                    or integrity.get("compressed_size_bytes") != entry["archive"]["size_bytes"]
                    or integrity.get("multipart_part_size_bytes") != MULTIPART_BYTES
                    or not re.fullmatch(r"[0-9a-f]{64}", integrity.get("compressed_sha256", ""))):
                raise IntegrityError("Verified archive lacks current multipart integrity evidence")
            if not isinstance(entry.get("files"), dict) or not set(entry["files"]).issubset(required):
                raise IntegrityError("Verified archive contains unexpected selected paths")
            validate_installed(root, entry["files"])
        report["verified_archives"] = old["verified_archives"]
    cancel, publication_lock = threading.Event(), threading.Lock()
    pending = [number for number in archives if str(number) not in report["verified_archives"]]
    report.update(status="downloading", requested_archives=list(archives), dataset_root=str(root))
    atomic_report(report_path, report)
    emit("subset_start", archives_pending=pending, archives_verified=len(report["verified_archives"]),
         required_count=len(required), workers=workers)
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    errors = []
    try:
        for number in pending:
            futures[executor.submit(archive_with_retries, catalog[number], required, root, retries=retries,
                                    opener=opener, cancel=cancel, publication_lock=publication_lock)] = number
        for future in as_completed(futures):
            number = futures[future]
            try:
                entry = future.result()
            except Exception as exc:
                errors.append({"archive": number, "error": type(exc).__name__, "message": str(exc)})
                cancel.set()
                for queued in futures:
                    queued.cancel()
                emit("archive_failed", archive=number, error=type(exc).__name__)
            else:
                report["verified_archives"][str(number)] = entry
                atomic_report(report_path, report)
                emit("archive_verified", archive=number, selected_files=len(entry["files"]))
    except BaseException:
        cancel.set()
        for queued in futures:
            queued.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    installed = set()
    for entry in report["verified_archives"].values():
        validate_installed(root, entry["files"])
        installed.update(entry["files"])
    missing = sorted(required - installed)
    report.update(status="complete" if not errors and not missing else "incomplete", missing_files=missing,
                  verified_file_count=len(installed), errors=errors, updated_at=datetime.now(timezone.utc).isoformat())
    atomic_report(report_path, report)
    if errors:
        raise IntegrityError(f"Archive transfer failed; verified progress retained in {report_path}")
    if missing:
        raise IntegrityError(f"{len(missing)} required frames are absent from verified archives; see {report_path}")
    emit("subset_complete", verified_files=len(installed), report=str(report_path))
    return report


def archive_argument(value):
    if value == "mini":
        return value
    if value.isdigit() and 1 <= int(value) <= 10:
        return int(value)
    raise argparse.ArgumentTypeError("Archive must be mini or a number from 1 to 10")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-file-list", type=Path, default=Path("data/selection-plan/required-camera-files.txt"))
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--objects", type=Path, default=Path("outputs/verda-access/nuscenes-public-objects.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/nuscenes-subset.json"))
    parser.add_argument("--archives", type=archive_argument, nargs="+", default=list(range(1, 11)),
                        help="Archive numbers 1..10 or mini (default: all 10 camera archives)")
    parser.add_argument("--workers", type=int, choices=range(1, 11), default=2,
                        help="Parallel archive transfers, 1..10 (default: 2)")
    parser.add_argument("--retries", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args(argv)
    if platform.system() != "Linux":
        parser.error("Run archive transfers on the Linux GPU host; local large downloads are disabled")
    try:
        catalog, provenance = load_catalog(args.objects, fetch_checksums())
        download_subset(args.required_file_list, args.root, catalog, provenance, args.report,
                        archives=args.archives, workers=args.workers, retries=args.retries)
    except (IntegrityError, OSError, ValueError, KeyError) as exc:
        emit("subset_failed", error=type(exc).__name__, message=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
