"""Portable, explicit boundaries between workloads, measurements and RL."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Protocol


@dataclass(frozen=True)
class KernelConfig:
    block_size: int
    num_warps: int

    def __post_init__(self):
        if (type(self.block_size) is not int or not 64 <= self.block_size <= 4096
                or self.block_size & (self.block_size - 1)):
            raise ValueError("block_size must be a power of two between 64 and 4096")
        if type(self.num_warps) is not int or self.num_warps not in (1, 2, 4, 8):
            raise ValueError("num_warps must be one of 1, 2, 4, 8")


@dataclass(frozen=True)
class Workload:
    operator: str
    shape: tuple[int, ...]
    dtype: str = "float16"
    num_groups: int = 32
    eps: float = 1e-5
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.operator not in ("geglu", "groupnorm_silu"):
            raise ValueError("Unknown operator")
        if not self.shape or any(type(x) is not int or x <= 0 for x in self.shape):
            raise ValueError("shape must contain positive integer dimensions")
        if math.prod(self.shape) >= 2**31:
            raise ValueError("Workload exceeds the supported 32-bit element-index limit")
        if len(self.shape) > 4:
            raise ValueError("At most four dimensions are supported")
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("Unsupported dtype")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32")
        if (type(self.eps) not in (int, float) or not math.isfinite(self.eps)
                or self.eps <= 0):
            raise ValueError("eps must be finite and positive")
        if type(self.num_groups) is not int or self.num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if self.operator == "geglu" and self.shape[-1] % 2:
            raise ValueError("GEGLU needs an even final dimension")
        if self.operator == "groupnorm_silu":
            if len(self.shape) != 4 or self.shape[1] % self.num_groups:
                raise ValueError("GroupNorm needs NCHW with channels divisible by groups")
            if math.prod(self.shape[1:]) // self.num_groups <= 1:
                raise ValueError("GroupNorm needs at least two values per group")

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Measurement:
    workload_key: str
    config: KernelConfig | None
    status: str
    latency_ms: float | None
    peak_allocated_bytes: int | None
    correct: bool
    backend: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Evaluator(Protocol):
    def baseline(self, workload: Workload) -> Measurement: ...
    def evaluate(self, workload: Workload, config: KernelConfig) -> Measurement: ...


DEFAULT_CONFIGS = tuple(KernelConfig(b, w) for b in (128, 256, 512, 1024) for w in (4, 8))
