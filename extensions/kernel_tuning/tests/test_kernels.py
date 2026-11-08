"""Reference contracts run on CPU; CUDA correctness cases never use a fallback."""

import importlib.util
import subprocess
import sys

import pytest

from drivebench_tuning import kernels
from drivebench_tuning.contract import DEFAULT_CONFIGS

torch = pytest.importorskip("torch", reason="PyTorch is optional for the RL contract tests")


def test_import_does_not_load_gpu_runtime():
    code = "import sys; import drivebench_tuning.kernels; assert 'triton' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("operator", ["geglu", "groupnorm_silu"])
def test_references_support_fullgraph_capture_without_gpu_claim(operator):
    # The eager Dynamo backend checks full-graph capture only.  This deliberately
    # does not count as CUDA execution or Inductor numerical validation.
    if operator == "geglu":
        reference = kernels.geglu_reference
        args = (torch.randn(2, 66),)
    else:
        def reference(x, w, b):
            return kernels.groupnorm_silu_reference(x, w, b, num_groups=2)
        args = (torch.randn(2, 4, 3, 5), torch.ones(4), torch.zeros(4))
    captured = torch.compile(reference, backend="eager", fullgraph=True)
    torch.testing.assert_close(captured(*args), reference(*args), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_geglu_reference_keeps_eager_intermediate_dtype(dtype):
    generator = torch.Generator().manual_seed(17)
    x = torch.randn((2, 7, 66), generator=generator).to(dtype)
    value, gate = x.chunk(2, dim=-1)
    expected = value * torch.nn.functional.gelu(gate, approximate="none")
    result = kernels.geglu_reference(x)
    assert result.shape == (2, 7, 33)
    assert result.dtype == dtype
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_groupnorm_reference_preserves_cast_before_silu(dtype):
    generator = torch.Generator().manual_seed(31)
    x = torch.randn((2, 6, 3, 5), generator=generator).to(dtype)
    weight = torch.randn((6,), generator=generator)
    bias = torch.randn((6,), generator=generator)
    normalized = torch.nn.functional.group_norm(x.float(), 3, weight, bias, 1e-5)
    expected = torch.nn.functional.silu(normalized.to(dtype))
    result = kernels.groupnorm_silu_reference(x, weight, bias, num_groups=3)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    if dtype != torch.float32:
        # The forbidden simplification would move this cast after SiLU.
        wrong_boundary = torch.nn.functional.silu(normalized).to(dtype)
        assert not torch.equal(expected, wrong_boundary)


def test_cpu_custom_kernels_raise_instead_of_falling_back():
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        kernels.fused_geglu(torch.ones(2, 8))
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        kernels.fused_groupnorm_silu(torch.ones(2, 4, 3, 5), torch.ones(4), torch.zeros(4), num_groups=2)


@pytest.mark.parametrize("shape", [(), (3,), (2, 7), (0, 8)])
def test_geglu_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        kernels.geglu_reference(torch.empty(shape))


def test_layout_dtype_and_backward_are_explicit():
    with pytest.raises(ValueError, match="contiguous"):
        kernels.geglu_reference(torch.randn(8, 4).transpose(0, 1))
    with pytest.raises(TypeError, match="dtype"):
        kernels.geglu_reference(torch.ones(2, 8, dtype=torch.float64))
    requires_grad = torch.randn(2, 8, requires_grad=True)
    with pytest.raises(RuntimeError, match="no custom backward"):
        kernels.geglu_reference(requires_grad)
    with torch.no_grad():
        assert not kernels.geglu_reference(requires_grad).requires_grad


@pytest.mark.parametrize("block_size,num_warps", [(0, 4), (192, 4), (8192, 4), (256, 3), (256, 16), (True, 4), (256, True)])
def test_launch_config_rejected_before_device_execution(block_size, num_warps):
    with pytest.raises(ValueError):
        kernels.fused_geglu(torch.ones(2, 8), block_size=block_size, num_warps=num_warps)


@pytest.mark.parametrize("groups,eps", [(0, 1e-5), (3, 1e-5), (True, 1e-5), (2, 0), (2, -1), (2, float("nan")), (2, float("inf"))])
def test_groupnorm_rejects_invalid_statistics_configuration(groups, eps):
    with pytest.raises(ValueError):
        kernels.groupnorm_silu_reference(torch.ones(2, 4, 3, 5), torch.ones(4), torch.zeros(4), num_groups=groups, eps=eps)


def test_groupnorm_rejects_invalid_layout_affine_and_gradients():
    x = torch.ones(2, 4, 3, 5)
    with pytest.raises(ValueError, match="NCHW"):
        kernels.groupnorm_silu_reference(x.reshape(2, 4, 15), torch.ones(4), torch.zeros(4), num_groups=2)
    with pytest.raises(ValueError, match="shape"):
        kernels.groupnorm_silu_reference(x, torch.ones(3), torch.zeros(4), num_groups=2)
    with pytest.raises(RuntimeError, match="no custom backward"):
        kernels.groupnorm_silu_reference(x, torch.ones(4, requires_grad=True), torch.zeros(4), num_groups=2)
    with pytest.raises(ValueError, match="more than one"):
        kernels.groupnorm_silu_reference(torch.ones(1, 4, 1, 1), torch.ones(4), torch.zeros(4), num_groups=4)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32, torch.bool])
def test_groupnorm_rejects_unsupported_input_and_affine_dtypes(dtype):
    x = torch.ones(2, 4, 3, 5)
    with pytest.raises(TypeError, match="dtype"):
        kernels.groupnorm_silu_reference(x.to(dtype), torch.ones(4), torch.zeros(4), num_groups=2)
    with pytest.raises(TypeError, match="dtype"):
        kernels.groupnorm_silu_reference(x, torch.ones(4, dtype=dtype), torch.zeros(4), num_groups=2)
    with pytest.raises(TypeError, match="dtype"):
        kernels.groupnorm_silu_reference(x, torch.ones(4), torch.zeros(4, dtype=dtype), num_groups=2)


def test_bfloat16_affine_reference_uses_explicit_float32_conversion():
    x = torch.linspace(-4, 4, 120, dtype=torch.bfloat16).reshape(2, 4, 3, 5)
    weight = torch.tensor([0.3, 0.7, 1.2, 1.6], dtype=torch.bfloat16)
    bias = torch.tensor([-0.2, -0.1, 0.3, 0.4], dtype=torch.float16)
    expected = torch.nn.functional.silu(
        torch.nn.functional.group_norm(x.float(), 2, weight.float(), bias.float()).to(torch.bfloat16)
    )
    torch.testing.assert_close(kernels.groupnorm_silu_reference(x, weight, bias, num_groups=2), expected, rtol=0, atol=0)


def test_indexing_range_rejected_without_allocation():
    with pytest.raises(ValueError, match="32-bit indexing"):
        kernels.geglu_reference(torch.empty((2**30, 2), device="meta"))


def test_groupnorm_chunk_limit_is_explicit():
    x = torch.empty(1, 1, 1, kernels.MAX_GROUP_CHUNKS * 64 + 1)
    with pytest.raises(ValueError, match="maximum supported"):
        kernels.fused_groupnorm_silu(x, torch.ones(1), torch.zeros(1), num_groups=1, block_size=64)


CUDA_READY = torch.cuda.is_available() and torch.version.hip is None and importlib.util.find_spec("triton") is not None
cuda_only = pytest.mark.skipif(not CUDA_READY, reason="Requires NVIDIA CUDA and Triton; no GPU correctness evidence on CPU")


def _cuda_dtype(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    return dtype


def _tolerance(dtype):
    return {torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2), torch.float32: (3e-5, 3e-5)}[dtype]


@cuda_only
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape,block_size,num_warps", [((66,), 64, 1), ((3, 7, 254), 128, 8), ((2, 513, 640), 512, 4)])
def test_cuda_geglu_random_and_tail(dtype, shape, block_size, num_warps):
    torch.manual_seed(7)
    x = torch.randn(shape, device="cuda", dtype=_cuda_dtype(dtype))
    expected = kernels.geglu_reference(x)
    result = kernels.fused_geglu(x, block_size=block_size, num_warps=num_warps)
    torch.cuda.synchronize()
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


@cuda_only
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_geglu_extreme_gate_values(dtype):
    gate = torch.tensor([-100, -10, -6, -3, -1, 0, 1e-4, 1, 3, 6, 10, 100], device="cuda", dtype=_cuda_dtype(dtype))
    values = torch.linspace(-4, 4, gate.numel(), device="cuda", dtype=dtype)
    x = torch.cat((values, gate))
    result = kernels.fused_geglu(x)
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(result, kernels.geglu_reference(x), rtol=rtol, atol=atol)


@cuda_only
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape,groups,block_size,num_warps", [((2, 6, 3, 5), 3, 64, 1), ((3, 32, 7, 11), 32, 128, 8), ((2, 320, 72, 128), 32, 1024, 4)])
def test_cuda_groupnorm_silu_random_multiple_chunks_and_tail(dtype, shape, groups, block_size, num_warps):
    torch.manual_seed(37)
    x = torch.randn(shape, device="cuda", dtype=_cuda_dtype(dtype))
    weight = torch.randn(shape[1], device="cuda")
    bias = torch.randn(shape[1], device="cuda")
    expected = kernels.groupnorm_silu_reference(x, weight, bias, num_groups=groups)
    result = kernels.fused_groupnorm_silu(x, weight, bias, num_groups=groups, block_size=block_size, num_warps=num_warps)
    torch.cuda.synchronize()
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


@cuda_only
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["constant", "large_offset"])
def test_cuda_groupnorm_stable_statistics(mode, record_property):
    # Exactly representable increments make the large-offset test exercise
    # variance stability without confusing it with low-precision input loss.
    values = torch.arange(2 * 6 * 7 * 11, device="cuda", dtype=torch.float32)
    x = (10000 + (values % 17) * 0.125).reshape(2, 6, 7, 11)
    if mode == "constant":
        x.fill_(3.0)
    weight = torch.linspace(0.5, 1.5, 6, device="cuda")
    bias = torch.linspace(-0.5, 0.5, 6, device="cuda")
    result = kernels.fused_groupnorm_silu(x, weight, bias, num_groups=3, block_size=128)
    reference = kernels.groupnorm_silu_reference(x, weight, bias, num_groups=3)
    grouped = x.double().reshape(2, 3, -1)
    mean = grouped.mean(-1, keepdim=True)
    variance = grouped.var(-1, unbiased=False, keepdim=True)
    normalized = ((grouped - mean) * torch.rsqrt(variance + 1e-5)).reshape_as(x)
    affine = normalized * weight.double()[None, :, None, None] + bias.double()[None, :, None, None]
    oracle = torch.nn.functional.silu(affine.float())
    assert torch.isfinite(result).all()
    record_property("custom_eager_max_abs_error", (result - reference).abs().max().item())
    record_property("custom_double_statistics_oracle_max_abs_error", (result - oracle).abs().max().item())
    record_property("eager_double_statistics_oracle_max_abs_error", (reference - oracle).abs().max().item())
    if mode == "constant":
        # For a constant group, the centered input and variance are exactly zero,
        # so the mathematical output is SiLU(bias), independent of the weight.
        # The original eager-equivalence gate failed on A100. The inspected
        # CUDA scale/shift form is consistent with affine cancellation in this
        # diagnostic. That failure remains historical evidence; this gate checks the
        # mathematical identity, not equivalence to that eager rounding path.
        analytic = (bias.double() / (1 + torch.exp(-bias.double()))).float()
        analytic = analytic[None, :, None, None].expand_as(x)
        # Outputs have magnitude below one; one float32 epsilon is a strict
        # absolute rounding budget, smaller than the original 3e-5 threshold.
        atol = torch.finfo(torch.float32).eps
        record_property("correctness_reference", "constant_group_analytic_and_double_statistics")
        record_property("custom_analytic_max_abs_error", (result - analytic).abs().max().item())
        record_property("absolute_tolerance", atol)
        torch.testing.assert_close(oracle, analytic, rtol=0, atol=atol)
        torch.testing.assert_close(result, analytic, rtol=0, atol=atol)
        torch.testing.assert_close(result, oracle, rtol=0, atol=atol)
        return
    # Prospective absolute threshold for the intentionally ill-conditioned
    # large-offset case; the ordinary random case has much tighter thresholds.
    atol = 3e-3
    torch.testing.assert_close(result, reference, rtol=atol, atol=atol)
    torch.testing.assert_close(result, oracle, rtol=atol, atol=atol)


@cuda_only
@pytest.mark.gpu
def test_cuda_uses_input_device_and_current_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(19, 510, device="cuda")
        result = kernels.fused_geglu(x)
        expected = kernels.geglu_reference(x)
    stream.synchronize()
    torch.testing.assert_close(result, expected, rtol=3e-5, atol=3e-5)


@cuda_only
@pytest.mark.gpu
@pytest.mark.parametrize("config", DEFAULT_CONFIGS, ids=lambda c: f"block{c.block_size}-warps{c.num_warps}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("operator", ["geglu", "groupnorm_silu"])
def test_cuda_entire_tuning_action_space_on_tail_shapes(operator, dtype, config):
    torch.manual_seed(109)
    dtype = _cuda_dtype(dtype)
    options = {"block_size": config.block_size, "num_warps": config.num_warps}
    if operator == "geglu":
        args = (torch.randn(3, 7, 254, device="cuda", dtype=dtype),)
        result = kernels.fused_geglu(*args, **options)
        expected = kernels.geglu_reference(*args)
    else:
        args = (torch.randn(2, 6, 7, 11, device="cuda", dtype=dtype),
                torch.randn(6, device="cuda"), torch.randn(6, device="cuda"))
        result = kernels.fused_groupnorm_silu(*args, num_groups=3, **options)
        expected = kernels.groupnorm_silu_reference(*args, num_groups=3)
    torch.cuda.synchronize()
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


@cuda_only
@pytest.mark.gpu
def test_cuda_bfloat16_with_mixed_precision_affine():
    _cuda_dtype(torch.bfloat16)
    x = torch.linspace(-4, 4, 120, device="cuda", dtype=torch.bfloat16).reshape(2, 4, 3, 5)
    weight = torch.tensor([0.3, 0.7, 1.2, 1.6], device="cuda", dtype=torch.bfloat16)
    bias = torch.tensor([-0.2, -0.1, 0.3, 0.4], device="cuda", dtype=torch.float16)
    result = kernels.fused_groupnorm_silu(x, weight, bias, num_groups=2)
    reference = kernels.groupnorm_silu_reference(x, weight, bias, num_groups=2)
    torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-2)
