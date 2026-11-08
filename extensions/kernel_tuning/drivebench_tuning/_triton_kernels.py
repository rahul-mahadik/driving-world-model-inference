"""CUDA implementations, imported lazily by kernels.py.

The mean/M2 merge uses the parallel variance identity (Chan/Welford), avoiding
the cancellation of E[x*x] - E[x]**2 for large-offset groups.  Tail lanes have
zero population and cannot contribute to the statistics.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def geglu_forward(X, Y, N, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = index // D
    column = index % D
    value = tl.load(X + row * (2 * D) + column, index < N, other=0)
    gate = tl.load(X + row * (2 * D) + D + column, index < N, other=0)
    gate_f32 = gate.to(tl.float32)
    gelu = (0.5 * gate_f32) * (1.0 + libdevice.erf(gate_f32 * 0.7071067811865476))
    # Eager GELU writes the input dtype before its separate multiply operation.
    rounded_gelu = gelu.to(gate.dtype).to(tl.float32)
    result = value.to(tl.float32) * rounded_gelu
    tl.store(Y + index, result, index < N)


@triton.jit
def _combine_moments(mean_a, m2_a, count_a, mean_b, m2_b, count_b):
    count = count_a + count_b
    denominator = tl.maximum(count, 1.0)
    delta = mean_b - mean_a
    mean = mean_a + delta * (count_b / denominator)
    m2 = m2_a + m2_b + delta * delta * (count_a * (count_b / denominator))
    return mean, m2, count


@triton.jit
def groupnorm_partial(X, Partial, GROUP_SIZE, TOTAL_GROUPS, CHUNKS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program = tl.program_id(0)
    group = program // CHUNKS
    chunk = program % CHUNKS
    offset = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offset < GROUP_SIZE
    value = tl.load(X + group * GROUP_SIZE + offset, valid, other=0).to(tl.float32)
    count = valid.to(tl.float32)
    m2 = tl.full((BLOCK_SIZE,), 0.0, tl.float32)
    mean, sum_squared_deviation, population = tl.reduce((value, m2, count), 0, _combine_moments)
    plane = TOTAL_GROUPS * CHUNKS
    tl.store(Partial + program, mean)
    tl.store(Partial + plane + program, sum_squared_deviation)
    tl.store(Partial + 2 * plane + program, population)


@triton.jit
def groupnorm_merge(Partial, Moments, TOTAL_GROUPS, EPS, CHUNKS: tl.constexpr, REDUCE_SIZE: tl.constexpr):
    group = tl.program_id(0)
    chunk = tl.arange(0, REDUCE_SIZE)
    valid = chunk < CHUNKS
    plane = TOTAL_GROUPS * CHUNKS
    partial_mean = tl.load(Partial + group * CHUNKS + chunk, valid, other=0)
    partial_m2 = tl.load(Partial + plane + group * CHUNKS + chunk, valid, other=0)
    partial_count = tl.load(Partial + 2 * plane + group * CHUNKS + chunk, valid, other=0)
    mean, m2, count = tl.reduce((partial_mean, partial_m2, partial_count), 0, _combine_moments)
    inverse_std = tl.rsqrt(tl.maximum(m2 / count, 0.0) + EPS)
    tl.store(Moments + group, mean)
    tl.store(Moments + TOTAL_GROUPS + group, inverse_std)


@triton.jit
def groupnorm_silu_forward(
    X, Weight, Bias, Moments, Y, GROUP_SIZE, TOTAL_GROUPS,
    SPATIAL: tl.constexpr, GROUPS: tl.constexpr, CHANNELS_PER_GROUP: tl.constexpr,
    CHUNKS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    program = tl.program_id(0)
    group = program // CHUNKS
    chunk = program % CHUNKS
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
    # Vista GroupNorm32 converts its output before the subsequent SiLU.
    rounded = normalized.to(value.dtype).to(tl.float32)
    result = rounded / (1.0 + libdevice.exp(-rounded))
    tl.store(Y + index, result, valid)
