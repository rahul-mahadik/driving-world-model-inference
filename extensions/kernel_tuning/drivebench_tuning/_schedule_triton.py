"""Additional lowering choices for live schedule transformations only."""

import triton
import triton.language as tl
from triton.language.extra import libdevice

from ._triton_kernels import _combine_moments


@triton.jit
def geglu_activation(X, Activated, N, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row, column = index // D, index % D
    gate = tl.load(X + row * (2 * D) + D + column, index < N, other=0).to(tl.float32)
    value = (0.5 * gate) * (1.0 + libdevice.erf(gate * 0.7071067811865476))
    # Storage in Activated performs the required input-dtype round before mul.
    tl.store(Activated + index, value, index < N)


@triton.jit
def geglu_multiply(X, Activated, Y, N, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row, column = index // D, index % D
    value = tl.load(X + row * (2 * D) + column, index < N, other=0).to(tl.float32)
    gate = tl.load(Activated + index, index < N, other=0).to(tl.float32)
    tl.store(Y + index, value * gate, index < N)


@triton.jit
def groupnorm_single(X, Moments, GROUP_SIZE, TOTAL_GROUPS, EPS, REDUCE_SIZE: tl.constexpr):
    group = tl.program_id(0)
    offset = tl.arange(0, REDUCE_SIZE)
    valid = offset < GROUP_SIZE
    value = tl.load(X + group * GROUP_SIZE + offset, valid, other=0).to(tl.float32)
    zero = tl.full((REDUCE_SIZE,), 0.0, tl.float32)
    mean, m2, count = tl.reduce((value, zero, valid.to(tl.float32)), 0, _combine_moments)
    inverse_std = tl.rsqrt(tl.maximum(m2 / count, 0.0) + EPS)
    tl.store(Moments + group, mean)
    tl.store(Moments + TOTAL_GROUPS + group, inverse_std)


@triton.jit
def groupnorm_output(
    X, Weight, Bias, Moments, Y, GROUP_SIZE, TOTAL_GROUPS,
    SPATIAL: tl.constexpr, GROUPS: tl.constexpr, CHANNELS_PER_GROUP: tl.constexpr,
    CHUNKS: tl.constexpr, BLOCK_SIZE: tl.constexpr, FUSED: tl.constexpr,
):
    program = tl.program_id(0)
    group, chunk = program // CHUNKS, program % CHUNKS
    offset = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offset < GROUP_SIZE
    index = group * GROUP_SIZE + offset
    value = tl.load(X + index, valid, other=0)
    channel = (group % GROUPS) * CHANNELS_PER_GROUP + offset // SPATIAL
    weight = tl.load(Weight + channel, valid, other=0).to(tl.float32)
    bias = tl.load(Bias + channel, valid, other=0).to(tl.float32)
    mean = tl.load(Moments + group)
    inverse_std = tl.load(Moments + TOTAL_GROUPS + group)
    normalized = ((value.to(tl.float32) - mean) * inverse_std) * weight + bias
    rounded = normalized.to(value.dtype).to(tl.float32)
    if FUSED:
        rounded = rounded / (1.0 + libdevice.exp(-rounded))
    tl.store(Y + index, rounded, valid)


@triton.jit
def silu_forward(X, Y, N, BLOCK_SIZE: tl.constexpr):
    index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value = tl.load(X + index, index < N, other=0).to(tl.float32)
    tl.store(Y + index, value / (1.0 + libdevice.exp(-value)), index < N)
