"""Executable schedules for the separate, live transformation track.

The published tabular kernels remain unchanged.  Fusion here joins GEGLU's
GELU/multiply or GroupNorm's affine output/SiLU, not the preceding linear layer
or all GroupNorm statistics.  Each active launch field controls a real launch.
No CUDA failure silently falls back to a reference implementation.
"""

from __future__ import annotations

import importlib
import math

from . import kernels

MAX_SINGLE_GROUP_SIZE = 65536


def validate_schedule(workload, schedule) -> None:
    """Check structural validity without importing Torch or Triton.

    Inactive parameters have one canonical value.  This prevents duplicate
    effective schedules from inflating the advertised search space.
    """
    kernels._check_launch(schedule.block_size, schedule.num_warps)
    kernels._check_launch(schedule.activation_block_size, schedule.activation_num_warps)
    kernels._check_launch(schedule.reduction_block_size, schedule.reduction_num_warps)
    if type(schedule.fused) is not bool:
        raise ValueError("fused must be a boolean")
    if schedule.reduction not in ("chunked", "single"):
        raise ValueError("reduction must be chunked or single")
    if schedule.fused and (schedule.activation_block_size, schedule.activation_num_warps) != (256, 4):
        raise ValueError("fused schedules require canonical inactive activation settings")
    if workload.operator == "geglu":
        if (schedule.reduction, schedule.reduction_block_size, schedule.reduction_num_warps) != ("chunked", 256, 4):
            raise ValueError("GEGLU requires canonical inactive reduction settings")
    elif workload.operator == "groupnorm_silu":
        size = math.prod(workload.shape[1:]) // workload.num_groups
        if schedule.reduction == "single":
            if schedule.reduction_block_size != 256:
                raise ValueError("single reduction requires canonical inactive reduction block size")
            if size > MAX_SINGLE_GROUP_SIZE:
                raise ValueError(f"single reduction supports at most {MAX_SINGLE_GROUP_SIZE} values per group")
        elif (size + schedule.reduction_block_size - 1) // schedule.reduction_block_size > kernels.MAX_GROUP_CHUNKS:
            raise ValueError("chunked reduction exceeds the maximum supported group chunks")
    else:
        raise ValueError("unsupported schedule operator")


def execute_schedule(workload, schedule, inputs: tuple):
    """Run the requested schedule, preserving the operator precision boundaries."""
    validate_schedule(workload, schedule)
    if not isinstance(inputs, tuple) or len(inputs) != (1 if workload.operator == "geglu" else 3):
        raise ValueError("inputs must be a tuple matching the operator arguments")
    x = inputs[0]
    if workload.operator == "geglu":
        torch = kernels._geglu_args(x)
        group_size = None
    else:
        torch, group_size = kernels._groupnorm_args(
            *inputs, workload.num_groups, workload.eps,
        )
    if tuple(x.shape) != workload.shape or str(x.dtype).removeprefix("torch.") != workload.dtype:
        raise ValueError("input shape and dtype must match the bound workload")
    # Validate support on the input device before allocating candidate storage.
    with torch.cuda.device(x.device) if x.device.type == "cuda" else kernels._nullcontext():
        existing = kernels._cuda_kernels(torch, x)
        impl = importlib.import_module("drivebench_tuning._schedule_triton")
        if workload.operator == "geglu":
            out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
            count, width = out.numel(), x.shape[-1] // 2
            grid = ((count + schedule.block_size - 1) // schedule.block_size,)
            if schedule.fused:
                existing.geglu_forward[grid](
                    x, out, count, width, BLOCK_SIZE=schedule.block_size,
                    num_warps=schedule.num_warps, enable_fp_fusion=False,
                )
            else:
                activated = torch.empty_like(out)
                impl.geglu_activation[((count + schedule.activation_block_size - 1) // schedule.activation_block_size,)](
                    x, activated, count, width, BLOCK_SIZE=schedule.activation_block_size,
                    num_warps=schedule.activation_num_warps, enable_fp_fusion=False,
                )
                impl.geglu_multiply[grid](
                    x, activated, out, count, width, BLOCK_SIZE=schedule.block_size,
                    num_warps=schedule.num_warps, enable_fp_fusion=False,
                )
            return out

        weight, bias = inputs[1:]
        groups = x.shape[0] * workload.num_groups
        moments = torch.empty((2, groups), device=x.device, dtype=torch.float32)
        if schedule.reduction == "chunked":
            chunks = (group_size + schedule.reduction_block_size - 1) // schedule.reduction_block_size
            partials = torch.empty((3, groups, chunks), device=x.device, dtype=torch.float32)
            existing.groupnorm_partial[(groups * chunks,)](
                x, partials, group_size, groups, CHUNKS=chunks,
                BLOCK_SIZE=schedule.reduction_block_size, num_warps=schedule.reduction_num_warps,
                enable_fp_fusion=False,
            )
            existing.groupnorm_merge[(groups,)](
                partials, moments, groups, float(workload.eps), CHUNKS=chunks,
                REDUCE_SIZE=1 << (chunks - 1).bit_length(),
                num_warps=schedule.reduction_num_warps, enable_fp_fusion=False,
            )
        else:
            impl.groupnorm_single[(groups,)](
                x, moments, group_size, groups, float(workload.eps),
                REDUCE_SIZE=1 << (group_size - 1).bit_length(),
                num_warps=schedule.reduction_num_warps, enable_fp_fusion=False,
            )
        out = torch.empty_like(x)
        normalized = out if schedule.fused else torch.empty_like(x)
        output_chunks = (group_size + schedule.block_size - 1) // schedule.block_size
        impl.groupnorm_output[(groups * output_chunks,)](
            x, weight, bias, moments, normalized, group_size, groups,
            SPATIAL=x.shape[2] * x.shape[3], GROUPS=workload.num_groups,
            CHANNELS_PER_GROUP=x.shape[1] // workload.num_groups, CHUNKS=output_chunks,
            BLOCK_SIZE=schedule.block_size, FUSED=schedule.fused,
            num_warps=schedule.num_warps, enable_fp_fusion=False,
        )
        if not schedule.fused:
            impl.silu_forward[((x.numel() + schedule.activation_block_size - 1) // schedule.activation_block_size,)](
                normalized, out, x.numel(), BLOCK_SIZE=schedule.activation_block_size,
                num_warps=schedule.activation_num_warps, enable_fp_fusion=False,
            )
        return out
