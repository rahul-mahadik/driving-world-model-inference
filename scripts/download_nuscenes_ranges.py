#!/usr/bin/env python3
"""Stream ordered, bounded parallel ranges from the official nuScenes CDN.

Uses a separate dataset root/report/lock from the original downloader. At most
four 16 MiB ranges per archive and four archives run concurrently. Only selected
JPEGs are stored; a complete compressed archive is never retained on disk.
Completed verified archives resume; in-flight gzip streams do not. All selected
images remain unpublished until complete gzip, size, SHA256 and multipart ETag
verification by download_nuscenes_subset. A partial archive set exits nonzero
with verified per-archive results retained, because the full 4358-image selection
is still the requirement. This script never merges the two dataset roots.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import http.client
import io
from pathlib import Path
import platform
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_nuscenes_subset as base

CDN = "https://d36yt3mvayqw5m.cloudfront.net/"
RANGE_BYTES = 16 * 1024 * 1024
SOCKET_TIMEOUT = 30
RANGE_DEADLINE = 120


class RangeResponse:
    """HTTP-like ordered view; each real request must return verified HTTP206.

The current chunk counts toward the four-chunk window. A replacement request
is submitted only after that chunk is consumed, never when merely dequeued.
"""

    status = 200  # Logical complete-object view expected by the existing reader.

    def __init__(self, archive, *, workers=4, retries=2, chunk_bytes=RANGE_BYTES,
                 opener=urllib.request.urlopen, timeout=SOCKET_TIMEOUT):
        if workers not in range(1, 5) or retries not in range(3) or not 0 < chunk_bytes <= RANGE_BYTES:
            raise ValueError("Ranges require 1..4 workers, 0..2 retries and chunks <=16 MiB")
        self.archive, self.workers, self.retries = archive, workers, retries
        self.chunk_bytes, self.opener, self.timeout = chunk_bytes, opener, min(timeout, SOCKET_TIMEOUT)
        self.url = CDN + archive["key"]
        self.headers = {"Content-Length": str(archive["size_bytes"]), "ETag": '"' + archive["etag"] + '"'}
        self.pending = deque()
        self.current, self.offset, self.next_start = b"", 0, 0
        self.cancel = threading.Event()
        self.executor = None
        self.closed = False

    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=self.workers)
        for _ in range(self.workers):
            self._submit()
        base.emit("range_stream_start", key=self.archive["key"], endpoint=CDN,
                  chunk_bytes=self.chunk_bytes, range_workers=self.workers,
                  timestamp_utc=datetime.now(timezone.utc).isoformat())
        return self

    def _submit(self):
        if self.next_start < self.archive["size_bytes"] and not self.cancel.is_set():
            start = self.next_start
            end = min(start + self.chunk_bytes, self.archive["size_bytes"]) - 1
            self.next_start = end + 1
            self.pending.append(self.executor.submit(self._fetch, start, end))

    def _fetch(self, start, end):
        expected = end - start + 1
        for attempt in range(self.retries + 1):
            if self.cancel.is_set():
                raise base.Cancelled("Range stream closed")
            request = urllib.request.Request(self.url, headers={
                "Range": f"bytes={start}-{end}", "If-Match": '"' + self.archive["etag"] + '"',
                "Accept-Encoding": "identity", "User-Agent": "drivebench-nuscenes-ranges/1"})
            began = time.monotonic()
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    headers = response.headers
                    if (response.status != 206 or response.geturl() != self.url
                            or headers.get("Content-Range") != f"bytes {start}-{end}/{self.archive['size_bytes']}"
                            or headers.get("ETag", "").strip('"') != self.archive["etag"]
                            or headers.get("Content-Length") != str(expected)
                            or headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}):
                        raise base.IntegrityError("Range status, URL, identity, encoding, or bounds differ from catalog")
                    data = io.BytesIO()
                    while data.tell() < expected:
                        if self.cancel.is_set():
                            raise base.Cancelled("Range stream closed")
                        if time.monotonic() - began > RANGE_DEADLINE:
                            raise TimeoutError("Range total deadline exceeded")
                        block = response.read(min(base.CHUNK, expected - data.tell()))
                        if not block:
                            raise http.client.IncompleteRead(b"", expected - data.tell())
                        data.write(block)
                    if response.read(1):
                        raise base.IntegrityError("Range body exceeds exact Content-Length")
                    return data.getvalue()
            except urllib.error.HTTPError as exc:
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise base.IntegrityError(f"Range HTTP{exc.code}; object identity/access not retried") from exc
                error = exc
            except (OSError, urllib.error.URLError, http.client.IncompleteRead) as exc:
                error = exc
            if attempt == self.retries or self.cancel.is_set():
                raise error
            base.emit("range_retry", key=self.archive["key"], start=start, end=end,
                      next_attempt=attempt + 2, error=type(error).__name__)
            self.cancel.wait(0.25 * (attempt + 1))
        raise AssertionError("Unreachable")

    def read(self, size=-1):
        if self.closed or self.executor is None:
            raise ValueError("Range response is not open")
        if self.cancel.is_set():
            raise base.Cancelled("Range response cancelled")
        remaining = base.CHUNK if size < 0 else min(size, base.CHUNK)
        result = io.BytesIO()
        while remaining:
            if not self.current:
                if not self.pending:
                    break
                self.current = self.pending.popleft().result()
                self.offset = 0
            take = min(remaining, len(self.current) - self.offset)
            result.write(self.current[self.offset:self.offset + take])
            self.offset += take
            remaining -= take
            if self.offset == len(self.current):
                self.current = b""
                self.offset = 0
                self._submit()
        return result.getvalue()

    def close(self):
        self.cancel.set()
        for future in self.pending:
            future.cancel()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
        self.current = b""
        self.closed = True

    def __exit__(self, *args):
        self.close()


def range_opener(catalog, *, workers=4, retries=2, opener=urllib.request.urlopen, chunk_bytes=RANGE_BYTES):
    identities = {archive["url"]: archive for archive in catalog.values()}
    def open_archive(request, timeout=SOCKET_TIMEOUT):
        address = request.full_url if isinstance(request, urllib.request.Request) else request
        if address not in identities:
            raise base.IntegrityError("Range transport only serves exact catalog objects")
        return RangeResponse(identities[address], workers=workers, retries=retries,
                             chunk_bytes=chunk_bytes, opener=opener, timeout=timeout)
    return open_archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-file-list", type=Path, default=Path("data/selection-plan/required-camera-files.txt"))
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes-ranges"))
    parser.add_argument("--objects", type=Path, default=Path("outputs/host/nuscenes-public-objects.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/nuscenes-ranges.json"))
    parser.add_argument("--archives", type=base.archive_argument, nargs="+", default=[10, 1])
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4, help="Concurrent archives; default4")
    parser.add_argument("--range-workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--retries", type=int, choices=range(3), default=2, help="Retries per range; no whole-archive retry")
    args = parser.parse_args(argv)
    if platform.system() != "Linux":
        parser.error("Run large transfers on the Linux host only")
    if args.root.resolve() == Path("data/nuscenes").resolve():
        parser.error("Use a separate staging root; the original dataset is still owned by its downloader")
    try:
        catalog, provenance = base.load_catalog(args.objects, base.fetch_checksums())
        provenance["transport"] = {"type": "bounded_ordered_ranges_v1", "endpoint": CDN,
            "chunk_bytes": RANGE_BYTES, "range_workers": args.range_workers,
            "max_buffered_chunks_per_archive": args.range_workers,
            "range_retries": args.retries, "socket_timeout_seconds": SOCKET_TIMEOUT,
            "range_deadline_seconds": RANGE_DEADLINE,
            "checks": "HTTP206, exact Content-Range/Length, If-Match, ETag, identity encoding; full streamed multipart verification"}
        opener = range_opener(catalog, workers=args.range_workers, retries=args.retries)
        base.download_subset(args.required_file_list, args.root, catalog, provenance, args.report,
                             archives=args.archives, workers=args.workers, retries=0, opener=opener)
    except (base.IntegrityError, OSError, ValueError, KeyError, http.client.IncompleteRead) as exc:
        base.emit("range_subset_failed", error=type(exc).__name__, message=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
