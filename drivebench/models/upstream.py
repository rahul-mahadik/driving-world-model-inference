"""Integrity checks and lazy imports for the untouched Vista reference checkout."""

from __future__ import annotations

import hashlib
import importlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


VISTA_COMMIT = "cc9821b4253ca7987c32757613d2fc2448fa9f5d"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkout(path: str | Path, expected_commit: str = VISTA_COMMIT) -> dict[str, str]:
    """Reject a different revision or modified source, including untracked files."""
    root = Path(path).expanduser().resolve()
    if expected_commit != VISTA_COMMIT:
        raise ValueError(f"This adapter was audited only for Vista {VISTA_COMMIT}")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
        changed = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True, stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot verify Vista checkout at {root}: {exc}") from exc
    if revision != expected_commit:
        raise ValueError(f"Vista commit mismatch: expected {expected_commit}, found {revision}")
    if changed:
        raise ValueError(f"Vista checkout must be pristine; changes found:\n{changed}")
    return {"upstream_path": str(root), "upstream_commit": revision}


def verify_checkpoint(path: str | Path, expected_sha256: str) -> dict[str, str]:
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("model.checkpoint_sha256 must be the independently recorded 64-digit SHA-256")
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Vista checkpoint is missing: {checkpoint}")
    if checkpoint.suffix != ".safetensors":
        raise ValueError("Use the published .safetensors checkpoint (pickle checkpoint loading is disabled)")
    observed = sha256_file(checkpoint)
    if observed != expected_sha256.lower():
        raise ValueError(f"Checkpoint SHA-256 mismatch: expected {expected_sha256.lower()}, found {observed}")
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": observed}


def import_reference(root: str | Path) -> Any:
    """Import original sample_utils only after checking module provenance."""
    root = Path(root).resolve()
    for name in ("vwm", "sample_utils", "train", "init_proj_path"):
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, "__file__", None):
            if not Path(existing.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f"Module {name!r} is already imported from outside pinned Vista; use a fresh process")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("sample_utils")
    if Path(module.__file__).resolve() != root / "sample_utils.py":
        raise RuntimeError("Imported sample_utils is not the pinned Vista reference")
    return module


def observed_only_padding(frames: Any, num_frames: int) -> Any:
    """Make a fresh T-frame array using observed inputs exclusively.

    Padding is necessary because original do_sample encodes all T inputs even
    though its conditioning mask uses only the observed prefix.
    """
    import numpy as np

    observed = np.asarray(frames)
    if observed.ndim != 4 or observed.shape[1] != 3:
        raise ValueError("Observed frames must have shape [observed_T, 3, H, W]")
    if observed.dtype != np.float32 or not np.isfinite(observed).all():
        raise ValueError("Observed frames must be finite float32 values")
    if not 1 <= len(observed) < num_frames:
        raise ValueError("Provide a nonempty observed prefix shorter than the generated clip")
    if observed.min() < -1.0 or observed.max() > 1.0:
        raise ValueError("Vista inputs must be normalized to [-1, 1]")
    padded = np.repeat(observed[-1:], num_frames, axis=0)
    padded[:len(observed)] = observed
    return np.ascontiguousarray(padded)


class UpstreamBackend:
    """Construct the adapter in untouched original-do_sample reference mode."""

    def __new__(cls, model_config, generation, optimizations):
        from drivebench.models.vista import VistaBackend

        return VistaBackend(model_config, generation, optimizations, _upstream=True)
