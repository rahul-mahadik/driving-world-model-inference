#!/usr/bin/env python3
"""Prepare Vista's exact public OpenCLIP init asset without constructing a model.

Default prints a plan. --download explicitly populates the normal HF cache and
verifies published bytes. Run in the CUDA11.8 container without --gpus.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from datetime import datetime, timezone

MODEL = "ViT-H-14"
TAG = "laion2b_s32b_b79k"
REPO = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
FILENAME = "open_clip_pytorch_model.bin"
REVISION = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"
SHA256 = "9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4"
SIZE = 3944692325
METADATA_SOURCE = f"https://huggingface.co/api/models/{REPO}?blobs=true"


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--hf-home", type=Path, default=Path("/workspace/outputs/model-cache/huggingface"))
    parser.add_argument("--report", type=Path, default=Path("/workspace/outputs/model-cache/openclip-pretrained.json"))
    args = parser.parse_args(argv)
    plan = {"model": MODEL, "tag": TAG, "repository": REPO, "filename": FILENAME,
            "revision": REVISION, "sha256": SHA256, "size_bytes": SIZE,
            "metadata_source": METADATA_SOURCE,
            "url": f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}",
            "hf_home": str(args.hf_home), "cache_dir": str(args.hf_home / "hub"),
            "scope": "public pretrained cache only; no model construction or GPU work"}
    if not args.download:
        print(json.dumps({"status": "plan_only", **plan}, indent=2))
        return 0
    require(platform.system() == "Linux", "Download on the Linux host, not the local workstation")
    require(args.hf_home.is_absolute(), "HF_HOME must be absolute")
    require(not args.report.exists(), "Choose a fresh report path; existing evidence is preserved")
    require(os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") == "https://huggingface.co",
            "Expected the official Hugging Face endpoint")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    from open_clip.pretrained import download_pretrained, get_pretrained_cfg
    from huggingface_hub import constants, get_hf_file_metadata, hf_hub_url, try_to_load_from_cache

    require(importlib.metadata.version("open-clip-torch") == "2.20.0", "Expected pinned OpenCLIP2.20.0")
    require(Path(constants.HF_HUB_CACHE) == args.hf_home / "hub", "HF cache override differs from planned default")
    cfg = get_pretrained_cfg(MODEL, TAG)
    require(not cfg.get("url") and cfg.get("hf_hub") == REPO + "/", "Installed OpenCLIP asset config changed")
    metadata = get_hf_file_metadata(hf_hub_url(REPO, FILENAME), token=False)
    require(metadata.commit_hash == REVISION, "Upstream main revision changed; review instead of silently replacing pin")
    require(metadata.etag.strip('"') == SHA256 and metadata.size == SIZE, "Published asset hash or size changed")
    # Exactly the API and cache defaults used by untouched Vista initialization.
    cached = Path(download_pretrained(cfg))
    require(cached.name == FILENAME and cached.parent.name == REVISION, "Returned snapshot differs from pin")
    require(cached.resolve().is_relative_to((args.hf_home / "hub").resolve()), "Asset escaped planned cache")
    require(cached.stat().st_size == SIZE and sha256_file(cached) == SHA256, "Downloaded bytes failed integrity verification")
    default_cached = try_to_load_from_cache(REPO, FILENAME, revision="main")
    require(isinstance(default_cached, str) and Path(default_cached).resolve() == cached.resolve(),
            "Default main cache lookup does not resolve to verified asset")
    report = {"schema_version": 1, "status": "verified", **plan,
              "created_at": datetime.now(timezone.utc).isoformat(), "cached_path": str(cached),
              "open_clip_version": importlib.metadata.version("open-clip-torch"),
              "huggingface_hub_version": importlib.metadata.version("huggingface-hub"),
              "download_api": "open_clip.pretrained.download_pretrained(cfg), cache_dir=None",
              "default_main_cache_verified": True, "helper_sha256": sha256_file(Path(__file__))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
