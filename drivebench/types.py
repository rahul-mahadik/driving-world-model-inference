"""Small, explicit contracts shared by model adapters and the executor."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GenerationConfig:
    total_frames: int = 25
    conditioning_frames: int = 1
    height: int = 576
    width: int = 1024
    steps: int = 50
    guidance_scale: float = 2.5
    cond_aug: float = 0.0
    dtype: str = "fp16"
    decode_chunk_size: int = 14

    def __post_init__(self):
        for name in ("total_frames", "conditioning_frames", "height", "width", "steps", "decode_chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.conditioning_frames >= self.total_frames:
            raise ValueError("conditioning_frames must be smaller than total_frames")
        if self.height % 8 or self.width % 8:
            raise ValueError("height and width must be divisible by 8")
        if self.dtype not in {"fp16", "bf16", "fp32"}:
            raise ValueError("dtype must be fp16, bf16, or fp32")
        if any(type(value) not in (int, float) or not np.isfinite(value) for value in (self.guidance_scale, self.cond_aug)) or self.cond_aug < 0:
            raise ValueError("guidance_scale and cond_aug must be finite; cond_aug must be nonnegative")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class OptimizationConfig:
    compile_denoiser: bool = False
    compile_decoder: bool = False
    compile_mode: str = "default"
    dynamic_shapes: bool = False
    fullgraph: bool = False
    conditioning_cache_bytes: int = 0
    max_batch_size: int = 1
    feature_cache_interval: int = 1
    feature_cache_module: str | None = None
    keep_model_resident: bool = True
    pinned_memory: bool = False
    async_transfer: bool = False
    prefetch_depth: int = 0
    attention_backend: str = "upstream"

    def __post_init__(self):
        for name in ("compile_denoiser", "compile_decoder", "dynamic_shapes", "fullgraph", "keep_model_resident", "pinned_memory", "async_transfer"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean, not a quoted string or number")
        for name in ("conditioning_cache_bytes", "max_batch_size", "feature_cache_interval", "prefetch_depth"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.max_batch_size < 1 or self.feature_cache_interval < 1:
            raise ValueError("batch size and cache refresh interval must be positive")
        if self.conditioning_cache_bytes < 0 or self.prefetch_depth < 0:
            raise ValueError("cache and prefetch limits must be nonnegative")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}:
            raise ValueError("unsupported torch.compile mode")
        if self.feature_cache_interval > 1 and not self.feature_cache_module:
            raise ValueError("approximate reuse requires an explicitly selected feature_cache_module")
        if self.async_transfer and not self.pinned_memory:
            raise ValueError("asynchronous transfer requires pinned_memory")
        if self.feature_cache_module is not None and (not isinstance(self.feature_cache_module, str) or not self.feature_cache_module.strip()):
            raise ValueError("feature_cache_module must be a nonempty string or null")
        if not isinstance(self.attention_backend, str) or not self.attention_backend.strip():
            raise ValueError("attention_backend must be a nonempty string")

    def to_dict(self):
        return asdict(self)


@dataclass
class BackendBatch:
    frames: np.ndarray  # [candidate, frame, height, width, RGB], float32 [0, 1], HOST
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    denoiser_evaluations: int = 0  # actual network invocations, CFG counted as one batched call
    cache_stats: dict[str, Any] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    gpu_inference_s: float | None = None


@dataclass
class PreparedContext:
    payload: Any
    cache_key: str
    preparation_s: float
    cache_hit: bool
    config_identity: str


@dataclass
class GenerationResult:
    frames: dict[str, np.ndarray]
    candidate_ids: list[str]
    seeds: list[int]
    effective_config: dict[str, Any]
    stage_timings_s: dict[str, float]
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    cache_stats: dict[str, Any]
    denoiser_evaluations: int
    fallbacks: list[str]
    status: str
    failures: list[dict[str, Any]] = field(default_factory=list)
    gpu_inference_s: float | None = None
