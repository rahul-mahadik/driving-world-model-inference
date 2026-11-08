#!/usr/bin/env python3
"""Merge verified nuScenes subset reports without altering the source reports.

All source roots and the destination must be idle: their existing downloader
locks are taken exclusively and nonblockingly. Only files from verified archive
entries are eligible; temporary archive staging files are never considered.
The default is a dry run. --apply copies missing verified files and creates a
new, non-overwriting report compatible with the original downloader's resume
binding. Partial coverage is explicitly reported as incomplete.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_nuscenes_subset as base


def checked_directory(path, *, missing_ok=False):
    """Check every existing ancestor without creating directories in dry runs."""
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if missing_ok:
                continue
            raise base.IntegrityError("Dataset/source directory is missing") from None
        if not stat.S_ISDIR(mode):
            raise base.IntegrityError("Directory contains a symlink or non-directory")
    return path


def read_report(path):
    path = Path(os.path.abspath(path))
    checked_directory(path.parent)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise base.IntegrityError("Source report must be a regular file")
        raw = stream.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise base.IntegrityError("Source report exceeds 32 MiB")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def identity_valid(identity):
    return (isinstance(identity, dict) and set(identity) == {"sha256", "size_bytes"}
            and isinstance(identity.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", identity["sha256"])
            and type(identity.get("size_bytes")) is int
            and 0 < identity["size_bytes"] <= base.MAX_FRAME_BYTES)


def checked_entry(key, entry, catalog, required, root):
    if (key not in catalog or not isinstance(entry, dict)
            or entry.get("archive") != catalog[key] or entry.get("status") != "verified"):
        raise base.IntegrityError("Invalid verified archive identity")
    spec, integrity = catalog[key], entry.get("integrity", {})
    if (not isinstance(integrity, dict)
            or integrity.get("computed_etag") != spec["etag"]
            or integrity.get("get_etag") not in {spec["etag"], '"' + spec["etag"] + '"'}
            or integrity.get("compressed_size_bytes") != spec["size_bytes"]
            or integrity.get("multipart_part_size_bytes") != base.MULTIPART_BYTES
            or not isinstance(integrity.get("compressed_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", integrity["compressed_sha256"])
            or not isinstance(integrity.get("compressed_md5"), str)
            or not re.fullmatch(r"[0-9a-f]{32}", integrity["compressed_md5"])
            or integrity.get("legacy_publisher_md5_match") is not
                (integrity.get("compressed_md5") == spec["legacy_publisher_md5"])):
        raise base.IntegrityError("Verified archive lacks current integrity evidence")
    files = entry.get("files")
    if not isinstance(files, dict) or not set(files).issubset(required):
        raise base.IntegrityError("Verified archive contains unexpected paths")
    for name, identity in files.items():
        if base.camera_path(name) != name or not identity_valid(identity):
            raise base.IntegrityError("Invalid selected frame identity")
        checked_directory((root / name).parent)
        if base.file_identity(root / name) != identity:
            raise base.IntegrityError("Source frame differs from verified SHA256/length")


def comparable_entry(entry):
    # GET ETag quoting and verification time may differ between valid transports.
    integrity = dict(entry["integrity"])
    integrity["get_etag"] = integrity["get_etag"].strip('"')
    return {"archive": entry["archive"], "files": entry["files"], "integrity": integrity}


def exclusive_roots(stack, roots):
    for root in sorted(set(roots), key=str):
        lock_path = root / ".drivebench-subset.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        stack.callback(os.close, fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise base.IntegrityError("Dataset lock must be a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise base.IntegrityError(f"Dataset is busy; refusing merge: {root}") from None


def create_report(path, report):
    """Create the report atomically, never replacing an earlier artifact."""
    base.directory_without_symlinks(path.parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".merge-report-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def merge_subsets(required_path, root, sources, catalog, provenance, report_path, *, apply=False):
    required, list_hash = base.required_files(required_path)
    root = checked_directory(root)
    sources = [(checked_directory(source_root), Path(os.path.abspath(path))) for source_root, path in sources]
    if not sources or len({path for _, path in sources}) != len(sources):
        raise base.IntegrityError("Provide distinct source reports")
    report_path = Path(os.path.abspath(report_path))
    checked_directory(report_path.parent, missing_ok=True)
    if os.path.lexists(report_path):
        raise base.IntegrityError("Output report exists; choose a new artifact path")
    catalog = {str(key): value for key, value in catalog.items()}
    binding = {"required_file_list_sha256": list_hash, "archive_catalog": catalog,
               "provenance": provenance}
    with ExitStack() as stack:
        exclusive_roots(stack, [root, *(source_root for source_root, _ in sources)])
        archives, frame_sources, frame_identities, source_records, archive_sources = {}, {}, {}, [], {}
        for source_root, path in sources:
            old, report_hash = read_report(path)
            old_provenance = old.get("provenance", {})
            if not isinstance(old_provenance, dict):
                raise base.IntegrityError("Invalid source provenance")
            canonical_provenance = {key: value for key, value in old_provenance.items() if key != "transport"}
            if (old.get("schema_version") != 1 or old.get("required_file_list_sha256") != list_hash
                    or old.get("archive_catalog") != catalog or canonical_provenance != provenance
                    or old.get("required_count") != len(required)
                    or old.get("dataset_root") != str(source_root)
                    or old.get("status") not in {"downloading", "complete", "incomplete"}
                    or not isinstance(old.get("verified_archives"), dict)):
                raise base.IntegrityError("Source report binding/root does not match this selection/catalog")
            source_records.append({"report_path": str(path), "report_sha256": report_hash,
                                   "dataset_root": str(source_root), "status_at_merge": old["status"],
                                   "provenance": old_provenance,
                                   "verified_archives": sorted(old["verified_archives"])})
            for key, entry in old["verified_archives"].items():
                checked_entry(key, entry, catalog, required, source_root)
                if key in archives and comparable_entry(archives[key]) != comparable_entry(entry):
                    raise base.IntegrityError("Duplicate verified archive identities disagree")
                if key not in archives:
                    archives[key] = entry
                    archive_sources[key] = {"report_path": str(path), "dataset_root": str(source_root)}
                for name, identity in entry["files"].items():
                    if name in frame_identities and frame_identities[name] != identity:
                        raise base.IntegrityError("Verified sources disagree about a selected frame")
                    frame_identities[name] = identity
                    frame_sources.setdefault(name, source_root / name)
        missing = sorted(required - frame_identities.keys())
        new_files = {}
        # Check every conflict before any images or directories are published.
        for name, identity in frame_identities.items():
            destination = root / name
            checked_directory(destination.parent, missing_ok=True)
            if os.path.lexists(destination):
                if base.file_identity(destination) != identity:
                    raise base.IntegrityError("Existing destination frame conflicts; refusing overwrite")
            else:
                new_files[name] = identity
        report = {"schema_version": 1, **binding, "dataset_root": str(root),
                  "required_count": len(required), "verified_archives": archives,
                  "verified_file_count": len(frame_identities), "missing_files": missing,
                  "status": "incomplete" if missing else "complete", "errors": [],
                  "updated_at": datetime.now(timezone.utc).isoformat(),
                  "merge": {"schema_version": 1, "sources": source_records,
                            "selected_archive_sources": archive_sources, "new_file_count": len(new_files),
                            "scope": "Only complete verified archives; source reports preserved verbatim"}}
        if not apply:
            return {"dry_run": True, "report_would_be": report, "output_report": str(report_path)}
        with tempfile.TemporaryDirectory(prefix=".drivebench-merge-", dir=root) as directory:
            staging = Path(directory)
            for name, identity in new_files.items():
                destination = staging / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(frame_sources[name], os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as source, destination.open("xb") as target:
                    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                        raise base.IntegrityError("Source became a nonregular file")
                    shutil.copyfileobj(source, target, base.CHUNK)
                    target.flush()
                    os.fsync(target.fileno())
                if base.file_identity(destination) != identity:
                    raise base.IntegrityError("Copied frame differs from verified SHA256/length")
            base.publish(root, staging, new_files, threading.Lock())
        base.validate_installed(root, frame_identities)
        create_report(report_path, report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-file-list", type=Path, default=Path("data/selection-plan/required-camera-files.txt"))
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--source", nargs=2, action="append", required=True, metavar=("ROOT", "REPORT"))
    parser.add_argument("--objects", type=Path, default=Path("outputs/verda-access/nuscenes-public-objects.json"))
    parser.add_argument("--publisher-checksums", type=Path, default=Path("data/nuscenes/publisher-md5.checksum"))
    parser.add_argument("--report", type=Path, default=Path("outputs/nuscenes-merged.json"))
    parser.add_argument("--apply", action="store_true", help="Publish missing verified files and create a new report (default: dry run)")
    args = parser.parse_args(argv)
    try:
        catalog, provenance = base.load_catalog(args.objects, args.publisher_checksums.read_bytes())
        result = merge_subsets(args.required_file_list, args.root, args.source, catalog, provenance,
                               args.report, apply=args.apply)
        report = result.get("report_would_be", result)
        base.emit("merge_applied" if args.apply else "merge_dry_run", status=report["status"],
                  verified_file_count=report["verified_file_count"], missing_file_count=len(report["missing_files"]),
                  new_file_count=report["merge"]["new_file_count"], verified_archives=sorted(report["verified_archives"]),
                  report=str(args.report))
    except (base.IntegrityError, OSError, ValueError, KeyError, TypeError) as exc:
        base.emit("merge_failed", error=type(exc).__name__, message=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
