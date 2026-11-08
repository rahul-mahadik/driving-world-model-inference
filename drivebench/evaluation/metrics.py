"""Metrics on RGB video in [0, 1]. No metric here measures driving safety.

Temporal differences are diagnostic proxies, not optical flow or trajectory
adherence. LPIPS is optional, imports lazily, and never falls back to a made-up
perceptual score.
"""
from __future__ import annotations

from typing import Any
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def verify_perceptual_setup(path):
    """Verify isolated evaluator assets before importing checkpoint-loading code."""
    from ..config import file_sha256
    receipt = json.loads(Path(path).read_text())
    if (receipt.get("passed") is not True or receipt.get("status") != "cpu_verified"
            or receipt.get("cpu_verification", {}).get("passed") is not True):
        raise ValueError("Perceptual evaluator must have a passed CPU setup receipt")
    target = Path(receipt["package"]["target"]).resolve()
    for name, record in receipt["package"]["files"].items():
        source = (target / name).resolve()
        if not source.is_relative_to(target) or source.stat().st_size != record["bytes"] or file_sha256(source) != record["sha256"]:
            raise ValueError("Isolated LPIPS package changed after setup")
    for name in ("wheel", "alexnet_cached_weight", "lpips_learned_weight"):
        record = receipt[name]
        source = Path(record["path"])
        if source.stat().st_size != record["bytes"] or file_sha256(source) != record["sha256"]:
            raise ValueError("Perceptual evaluator weight or wheel changed after setup")
    return receipt


def as_video(value: np.ndarray) -> np.ndarray:
    """Validate a THWC RGB array, accepting uint8 or floating [0, 1]."""
    array = np.asarray(value)
    if array.ndim != 4 or array.shape[-1] != 3 or min(array.shape) == 0:
        raise ValueError(f"Expected nonempty THWC RGB video, got {array.shape}")
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    elif np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32, copy=False)
    else:
        raise ValueError("Video must be uint8 or floating-point RGB in [0, 1]")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("Video contains nonfinite values or values outside [0, 1]")
    return array


def temporal_diagnostics(video: np.ndarray, freeze_threshold: float = 1 / 255) -> dict[str, Any]:
    video = as_video(video)
    if video.shape[0] < 2:
        return {"temporal_change": None, "frozen_transition_fraction": None,
                "temporal_acceleration": None, "transitions": 0}
    delta = np.diff(video, axis=0)
    change = np.mean(np.abs(delta), axis=(1, 2, 3), dtype=np.float64)
    return {
        "temporal_change": float(change.mean()),
        "frozen_transition_fraction": float(np.mean(change <= freeze_threshold)),
        "temporal_acceleration": float(np.mean(np.abs(np.diff(delta, axis=0)), dtype=np.float64))
        if len(delta) > 1 else None,
        "transitions": len(change),
    }


def compare_videos(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference, candidate = as_video(reference), as_video(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"Video shape mismatch: {reference.shape} versus {candidate.shape}")
    error = candidate - reference
    mse = float(np.mean(np.square(error, dtype=np.float64)))
    ref_temporal = temporal_diagnostics(reference)
    new_temporal = temporal_diagnostics(candidate)
    result: dict[str, Any] = {
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "rmse": float(np.sqrt(mse)),
        "max_abs_error": float(np.max(np.abs(error))),
        "psnr_db": float(-10 * np.log10(mse)) if mse > 0 else None,
        "exact_match": mse == 0,
        "reference_temporal": ref_temporal,
        "candidate_temporal": new_temporal,
        "temporal_change_error": None,
        "freeze_fraction_increase": None,
        "temporal_derivative_mae": None,
    }
    if reference.shape[0] > 1:
        result.update(
            temporal_change_error=abs(new_temporal["temporal_change"] - ref_temporal["temporal_change"]),
            freeze_fraction_increase=max(0.0, new_temporal["frozen_transition_fraction"]
                                         - ref_temporal["frozen_transition_fraction"]),
            temporal_derivative_mae=float(np.mean(np.abs(np.diff(candidate, axis=0)
                                                        - np.diff(reference, axis=0)), dtype=np.float64)),
        )
    return result


class PerceptualMetric:
    """Explicitly opt in to LPIPS; weights may need a separate download."""

    def __init__(self, enabled: bool = False, device: str = "cpu"):
        if device not in {"cpu", "cuda"}:
            raise ValueError("LPIPS device must be cpu or cuda")
        self.enabled = enabled
        self.device = device
        self.model = None
        self.status = "not_requested"
        self.reason: str | None = None
        self.provenance: dict[str, Any] = {"device": device}
        if enabled:
            try:
                import lpips
                model = lpips.LPIPS(net="alex", verbose=False).eval()
                weights = hashlib.sha256()
                for name, value in sorted(model.state_dict().items()):
                    array = value.detach().cpu().contiguous().numpy()
                    weights.update(f"{name}:{array.dtype}:{array.shape}".encode())
                    weights.update(array.tobytes())
                self.provenance.update(package_version=importlib.metadata.version("lpips"),
                                       state_dict_sha256=weights.hexdigest(), network="alex", module_path=str(Path(lpips.__file__).resolve()))
                self.model = model.to(device)
                self.status = "available"
            except Exception as exc:
                self.status = "unavailable"
                self.reason = f"{type(exc).__name__}: {exc}"

    def compare(self, reference: np.ndarray, candidate: np.ndarray) -> float | None:
        if self.model is None:
            return None
        import torch
        reference, candidate = as_video(reference), as_video(candidate)
        if reference.shape != candidate.shape:
            raise ValueError("LPIPS requires matching video shapes")
        values = []
        with torch.inference_mode():
            for left, right in zip(reference, candidate):
                a = torch.from_numpy(np.ascontiguousarray(left.transpose(2, 0, 1))).unsqueeze(0).to(self.device) * 2 - 1
                b = torch.from_numpy(np.ascontiguousarray(right.transpose(2, 0, 1))).unsqueeze(0).to(self.device) * 2 - 1
                values.append(float(self.model(a, b).item()))
        return float(np.mean(values))


def bootstrap_mean(values: list[float], *, seed: int = 0, resamples: int = 2000) -> dict[str, Any]:
    """Percentile CI; callers pass one value per independent scene."""
    data = np.asarray(values, dtype=np.float64)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("Bootstrap needs finite, nonempty observations")
    mean = float(data.mean())
    if len(data) < 2:
        return {"mean": mean, "ci95": None, "units": len(data)}
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    for start in range(0, resamples, 128):
        count = min(128, resamples - start)
        estimates[start:start + count] = data[rng.integers(0, len(data), (count, len(data)))].mean(axis=1)
    return {"mean": mean, "ci95": [float(x) for x in np.quantile(estimates, [0.025, 0.975])],
            "units": len(data)}
