"""Inference-only fused operators for the separate kernel-tuning study.

The eager references reproduce the relevant Vista precision boundaries.  The
Triton implementations require NVIDIA CUDA and Triton 3.x; importing this module
does not import Torch or Triton.  Neither operator silently falls back to eager.

These are standalone operator candidates, not replacements installed into the
frozen Vista model.  Numerical and performance qualification is workload-specific.
"""

from __future__ import annotations

import importlib
import math
from numbers import Integral, Real

SUPPORTED_BLOCK_SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
SUPPORTED_NUM_WARPS = (1, 2, 4, 8)
MAX_GROUP_CHUNKS = 4096


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("The kernel study requires the optional PyTorch dependency") from exc


def _check_tensor(torch, x, name):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.layout != torch.strided or not x.is_contiguous():
        raise ValueError(f"{name} must be a contiguous, strided tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{name} must have float16, bfloat16, or float32 dtype")
    if x.numel() == 0 or any(size <= 0 for size in x.shape):
        raise ValueError(f"{name} must be nonempty")
    if x.numel() >= 2**31:
        raise ValueError(f"{name} exceeds the supported 32-bit indexing range")


def _check_inference(torch, *tensors):
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise RuntimeError("These inference-only operators have no custom backward; use no_grad or inference_mode")


def _check_launch(block_size, num_warps):
    if isinstance(block_size, bool) or not isinstance(block_size, Integral) or block_size not in SUPPORTED_BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {SUPPORTED_BLOCK_SIZES}")
    if isinstance(num_warps, bool) or not isinstance(num_warps, Integral) or num_warps not in SUPPORTED_NUM_WARPS:
        raise ValueError(f"num_warps must be one of {SUPPORTED_NUM_WARPS}")


def _cuda_kernels(torch, x):
    if x.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("Custom kernels require NVIDIA CUDA tensors; CPU and HIP fallback is disabled")
    if x.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 requires a CUDA device with bfloat16 support")
    try:
        triton = importlib.import_module("triton")
    except ImportError as exc:
        raise RuntimeError("Custom kernels require Triton 3.x in the CUDA runtime") from exc
    if str(triton.__version__).split(".", 1)[0] != "3":
        raise RuntimeError("This separate kernel study targets Triton 3.x; historical Triton 2.x is unsupported")
    return importlib.import_module("drivebench_tuning._triton_kernels")


def _geglu_args(x):
    torch = _torch()
    _check_tensor(torch, x, "x")
    if x.ndim < 1 or x.shape[-1] % 2:
        raise ValueError("GEGLU input must have shape [..., 2D] with an even last dimension")
    _check_inference(torch, x)
    return torch


def geglu_reference(x):
    """Vista's exact-GELU activation and gate, excluding its preceding linear map.

    GELU produces an intermediate in the input dtype before multiplication.  This
    distinction matters for float16 and bfloat16 and is retained by the kernel.
    CPU tensors are accepted here for reference/contract tests only.
    """
    torch = _geglu_args(x)
    value, gate = x.chunk(2, dim=-1)
    return value * torch.nn.functional.gelu(gate, approximate="none")


def fused_geglu(x, *, block_size=256, num_warps=4):
    """Fuse exact erf-GELU and gating without materializing the GELU tensor.

    Supports nonempty contiguous [..., 2D] CUDA tensors.  Compilation, launches,
    and allocation errors propagate; there is no correctness or eager fallback.
    """
    torch = _geglu_args(x)
    _check_launch(block_size, num_warps)
    with torch.cuda.device(x.device) if x.device.type == "cuda" else _nullcontext():
        kernels = _cuda_kernels(torch, x)
        out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device)
        kernels.geglu_forward[((out.numel() + block_size - 1) // block_size,)](
            x, out, out.numel(), x.shape[-1] // 2,
            BLOCK_SIZE=int(block_size), num_warps=int(num_warps), enable_fp_fusion=False,
        )
    return out


def _groupnorm_args(x, weight, bias, num_groups, eps):
    torch = _torch()
    _check_tensor(torch, x, "x")
    _check_tensor(torch, weight, "weight")
    _check_tensor(torch, bias, "bias")
    if x.ndim != 4:
        raise ValueError("GroupNorm+SiLU supports contiguous NCHW input only")
    if isinstance(num_groups, bool) or not isinstance(num_groups, Integral) or num_groups <= 0:
        raise ValueError("num_groups must be a positive integer")
    if x.shape[1] % num_groups:
        raise ValueError("channels must be divisible by num_groups")
    if weight.shape != (x.shape[1],) or bias.shape != (x.shape[1],):
        raise ValueError("weight and bias must each have shape [channels]")
    if weight.device != x.device or bias.device != x.device:
        raise ValueError("input and affine tensors must be on the same device")
    if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    _check_inference(torch, x, weight, bias)
    group_size = x.shape[1] // num_groups * x.shape[2] * x.shape[3]
    # F.group_norm rejects one value per channel/group for a singleton batch.
    if x.shape[0] * group_size <= 1:
        raise ValueError("GroupNorm needs more than one value across batch and group")
    return torch, group_size


def groupnorm_silu_reference(x, weight, bias, *, num_groups=32, eps=1e-5):
    """GroupNorm32 followed by SiLU, including Vista's intermediate dtype cast.

    Accumulation and affine parameters use float32; the normalized output is
    converted back to input dtype *before* SiLU.  This models GroupNorm32's
    explicit float input and type(x.dtype) output rather than a low-precision
    GroupNorm variant.  Affine tensors are promoted explicitly when necessary.
    """
    torch, _ = _groupnorm_args(x, weight, bias, num_groups, eps)
    normalized = torch.nn.functional.group_norm(
        x.float(), int(num_groups), weight.float(), bias.float(), float(eps)
    ).to(dtype=x.dtype)
    return torch.nn.functional.silu(normalized)


def fused_groupnorm_silu(x, weight, bias, *, num_groups=32, eps=1e-5, block_size=256, num_warps=4):
    """Three-stage GroupNorm statistics and fused normalization+SiLU output.

    Each group uses bounded chunks and stable mean/M2 merging.  Only compact
    float32 partial statistics are materialized; there is no full float32 input
    copy or normalized intermediate.  The precision cast before SiLU is kept.
    At most MAX_GROUP_CHUNKS chunks per group are supported.  Reduction order
    differs from PyTorch, so bitwise equality is not promised.
    """
    torch, group_size = _groupnorm_args(x, weight, bias, num_groups, eps)
    _check_launch(block_size, num_warps)
    chunks = (group_size + block_size - 1) // block_size
    if chunks > MAX_GROUP_CHUNKS:
        raise ValueError(f"Group has {chunks} chunks; maximum supported is {MAX_GROUP_CHUNKS}")
    with torch.cuda.device(x.device) if x.device.type == "cuda" else _nullcontext():
        kernels = _cuda_kernels(torch, x)
        groups = x.shape[0] * int(num_groups)
        partials = torch.empty((3, groups, chunks), device=x.device, dtype=torch.float32)
        moments = torch.empty((2, groups), device=x.device, dtype=torch.float32)
        out = torch.empty_like(x)
        kernels.groupnorm_partial[(groups * chunks,)](
            x, partials, group_size, groups, CHUNKS=chunks,
            BLOCK_SIZE=int(block_size), num_warps=int(num_warps), enable_fp_fusion=False,
        )
        kernels.groupnorm_merge[(groups,)](
            partials, moments, groups, float(eps), CHUNKS=chunks,
            REDUCE_SIZE=1 << (chunks - 1).bit_length(),
            num_warps=int(num_warps), enable_fp_fusion=False,
        )
        kernels.groupnorm_silu_forward[(groups * chunks,)](
            x, weight, bias, moments, out, group_size, groups,
            SPATIAL=x.shape[2] * x.shape[3], GROUPS=int(num_groups),
            CHANNELS_PER_GROUP=x.shape[1] // int(num_groups), CHUNKS=chunks,
            BLOCK_SIZE=int(block_size), num_warps=int(num_warps), enable_fp_fusion=False,
        )
    return out


def _nullcontext():
    # Keep CPU reference/import paths free of optional runtime imports.
    from contextlib import nullcontext
    return nullcontext()
