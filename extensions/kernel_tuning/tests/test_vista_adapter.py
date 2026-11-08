"""Adapter contracts and optional isolated CUDA integration checks.

Most tests use CPU test doubles. GPU-marked tests exercise real custom kernels
through representative modules with mocked source verification. They do not
load a Vista checkpoint, measure performance, or validate full-model quality.
"""

from collections import OrderedDict

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from drivebench_tuning.contract import KernelConfig
from drivebench_tuning.vista_adapter import capture_vista_workloads, vista_kernel_overrides


class GEGLU(nn.Module):
    def __init__(self, dim_in=4, dim_out=3):
        super().__init__()
        self.proj = nn.Linear(dim_in, 2 * dim_out)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


# Explicitly representative test doubles. Their source is NOT the pinned
# checkout; only the private test seam permits these names to be recognized.
GEGLU.__module__ = "vwm.modules.attention"
GroupNorm32.__module__ = "vwm.modules.diffusionmodules.util"


CONFIG = KernelConfig(256, 4)


def fake_kernels(calls):
    def geglu(x, *, block_size, num_warps):
        calls.append(("geglu", tuple(x.shape), block_size, num_warps))
        value, gate = x.chunk(2, dim=-1)
        return value * F.gelu(gate)

    def groupnorm(x, weight, bias, *, num_groups, eps, block_size, num_warps):
        calls.append(("groupnorm_silu", tuple(x.shape), block_size, num_warps, eps))
        normalized = F.group_norm(x.float(), num_groups, weight, bias, eps).to(x.dtype)
        return F.silu(normalized)

    return {"geglu": geglu, "groupnorm_silu": groupnorm}


def make_model():
    model = nn.ModuleDict({
        "feed_forward": GEGLU(),
        "convolution": nn.Sequential(OrderedDict([
            ("norm", GroupNorm32(2, 4, eps=3e-5)),
            ("activation", nn.SiLU()),
            ("convolution", nn.Conv2d(4, 4, 1)),
        ])),
        "ordinary_norm": nn.Sequential(nn.GroupNorm(2, 4), nn.SiLU()),
    })
    return model.eval()


def test_operator_semantics_and_exact_restoration():
    torch.manual_seed(11)
    model = make_model()
    calls = []
    modules_before = {name: id(module) for name, module in model.named_modules()}
    parameters_before = {name: id(parameter) for name, parameter in model.named_parameters()}
    keys_before = list(model.state_dict())
    state_before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    x, image = torch.randn(2, 5, 4), torch.randn(2, 4, 3, 3)
    with torch.no_grad():
        expected_geglu = model["feed_forward"](x)
        expected_norm = model["convolution"](image)
        expected_ordinary = model["ordinary_norm"](image)
        with vista_kernel_overrides(model, geglu_config=CONFIG, groupnorm_config=CONFIG,
                                    _test_kernels=fake_kernels(calls)) as selection:
            assert selection["execution"] == "cpu_contract_test_only"
            assert selection["gpu_validation_performed"] is False
            assert selection["geglu_paths"] == ["feed_forward"]
            assert selection["groupnorm_silu_paths"] == ["convolution.norm"]
            assert list(model.state_dict()) == keys_before
            torch.testing.assert_close(model["feed_forward"](x), expected_geglu, rtol=0, atol=0)
            torch.testing.assert_close(model["convolution"](image), expected_norm, rtol=0, atol=0)
            torch.testing.assert_close(model["ordinary_norm"](image), expected_ordinary, rtol=0, atol=0)
        assert calls == [("geglu", (2, 5, 6), 256, 4), ("groupnorm_silu", (2, 4, 3, 3), 256, 4, 3e-5)]
        torch.testing.assert_close(model["feed_forward"](x), expected_geglu, rtol=0, atol=0)
        torch.testing.assert_close(model["convolution"](image), expected_norm, rtol=0, atol=0)
    assert {name: id(module) for name, module in model.named_modules()} == modules_before
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameters_before
    assert all("forward" not in module.__dict__ for module in model.modules())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, state_before[name], rtol=0, atol=0)


def test_groupnorm_preserves_cast_before_silu():
    model = nn.Sequential(GroupNorm32(2, 4), nn.SiLU()).eval()
    x = torch.linspace(-3, 3, 72, dtype=torch.float16).reshape(2, 4, 3, 3)
    calls = []
    with torch.no_grad():
        expected = model(x)
        with vista_kernel_overrides(model, groupnorm_config=CONFIG, _test_kernels=fake_kernels(calls)):
            actual = model(x)
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_exception_restores_forwards_and_does_not_fallback():
    model = make_model()
    calls = []

    def fail(*args, **kwargs):
        calls.append("failed")
        raise RuntimeError("unsupported test workload")

    with torch.no_grad():
        with pytest.raises(RuntimeError, match="unsupported test workload"):
            with vista_kernel_overrides(model, geglu_config=CONFIG, _test_kernels={"geglu": fail}):
                model["feed_forward"](torch.ones(1, 4))
        assert "forward" not in model["feed_forward"].__dict__
        # A failed scope releases its reservation and can be entered again.
        with vista_kernel_overrides(model, geglu_config=CONFIG, _test_kernels=fake_kernels([])):
            assert model["feed_forward"](torch.ones(1, 4)).shape == (1, 3)
    assert calls == ["failed"]


def test_entry_requires_eval_and_no_grad():
    model = make_model()
    with pytest.raises(ValueError, match="no_grad"):
        with vista_kernel_overrides(model, geglu_config=CONFIG, _test_kernels=fake_kernels([])):
            pass
    model["feed_forward"].train()
    with torch.no_grad(), pytest.raises(ValueError, match="eval mode"):
        with vista_kernel_overrides(model, geglu_config=CONFIG, _test_kernels=fake_kernels([])):
            pass


def test_enabling_gradients_inside_scope_fails():
    model = make_model()
    with torch.no_grad():
        with vista_kernel_overrides(model, geglu_config=CONFIG, _test_kernels=fake_kernels([])):
            with torch.enable_grad(), pytest.raises(ValueError, match="without gradients"):
                model["feed_forward"](torch.ones(1, 4))


def test_unrecognized_and_nonadjacent_modules_are_not_patched():
    class DifferentGEGLU(GEGLU):
        pass

    unrecognized = nn.Sequential(DifferentGEGLU()).eval()
    nonadjacent = nn.Sequential(GroupNorm32(2, 4), nn.Identity(), nn.SiLU()).eval()
    with torch.no_grad():
        for model, kwargs in ((unrecognized, {"geglu_config": CONFIG}),
                              (nonadjacent, {"groupnorm_config": CONFIG})):
            with pytest.raises(ValueError, match="No supported"):
                with vista_kernel_overrides(model, _test_kernels=fake_kernels([]), **kwargs):
                    pass
            assert all("forward" not in module.__dict__ for module in model.modules())


def test_sequential_subclasses_and_child_hooks_are_not_bypassed():
    class CustomSequential(nn.Sequential):
        def forward(self, x):
            return super().forward(x) + 1

    model = CustomSequential(GroupNorm32(2, 4), nn.SiLU()).eval()
    with torch.no_grad(), pytest.raises(ValueError, match="No supported"):
        with vista_kernel_overrides(model, groupnorm_config=CONFIG, _test_kernels=fake_kernels([])):
            pass
    model = make_model()
    hook = model["convolution"].norm.register_forward_hook(lambda m, args, output: output + 1)
    with torch.no_grad(), pytest.raises(ValueError, match="Remove hooks"):
        with vista_kernel_overrides(model, groupnorm_config=CONFIG, _test_kernels=fake_kernels([])):
            pass
    hook.remove()


def test_preserves_shared_norm_calls_outside_the_fused_sequence():
    norm = GroupNorm32(2, 4).eval()
    model = nn.ModuleDict({"fused": nn.Sequential(norm, nn.SiLU()), "unfused": norm}).eval()
    x = torch.randn(2, 4, 3, 3)
    with torch.no_grad():
        expected_norm = model["unfused"](x)
        with vista_kernel_overrides(model, groupnorm_config=CONFIG, _test_kernels=fake_kernels([])):
            torch.testing.assert_close(model["unfused"](x), expected_norm, rtol=0, atol=0)
            torch.testing.assert_close(model["fused"](x), F.silu(expected_norm), rtol=0, atol=0)


def test_production_rejects_test_double_source():
    with torch.no_grad(), pytest.raises(ValueError, match="(Cannot verify|does not match)"):
        with vista_kernel_overrides(make_model(), geglu_config=CONFIG):
            pass


def test_plan_rejects_unsupported_second_family_without_partial_patch():
    model = nn.ModuleDict({"geglu": GEGLU(), "norm": nn.Sequential(GroupNorm32(2, 4), nn.SiLU(inplace=True))}).eval()
    with torch.no_grad(), pytest.raises(ValueError, match="Unsupported GroupNorm32"):
        with vista_kernel_overrides(model, geglu_config=CONFIG, groupnorm_config=CONFIG,
                                    _test_kernels=fake_kernels([])):
            pass
    assert all("forward" not in module.__dict__ for module in model.modules())


def test_capture_actual_projection_and_norm_workloads_and_remove_hooks():
    model = make_model()
    with torch.no_grad():
        with capture_vista_workloads(model, _allow_test_modules=True) as records:
            model["feed_forward"](torch.randn(2, 5, 4))
            model["feed_forward"](torch.randn(2, 5, 4))
            model["convolution"](torch.randn(2, 4, 3, 3))
        assert len(records) == 2
        assert records[0]["operator"] == "geglu"
        assert records[0]["shape"] == [2, 5, 6]
        assert records[0]["calls"] == 2
        assert records[0]["observation"] == "cpu_contract_test_only"
        assert records[1]["shape"] == [2, 4, 3, 3]
        assert records[1]["num_groups"] == 2
        assert records[1]["eps"] == 3e-5
        assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_capture_removes_hooks_on_exception():
    model = make_model()
    with torch.no_grad(), pytest.raises(RuntimeError, match="stop"):
        with capture_vista_workloads(model, _allow_test_modules=True):
            raise RuntimeError("stop")
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_nested_overrides_do_not_damage_the_outer_scope():
    model = make_model()
    calls = []
    with torch.no_grad():
        with vista_kernel_overrides(model, geglu_config=CONFIG, _test_kernels=fake_kernels(calls)):
            outer_forward = model["feed_forward"].forward
            with pytest.raises(ValueError, match="already has an instance forward"):
                with vista_kernel_overrides(model, geglu_config=CONFIG, _test_kernels=fake_kernels([])):
                    pass
            assert model["feed_forward"].forward is outer_forward
            model["feed_forward"](torch.ones(1, 4))
    assert len(calls) == 1
    assert "forward" not in model["feed_forward"].__dict__


def test_capture_rejects_ambiguous_shared_module_paths():
    norm = GroupNorm32(2, 4)
    model = nn.ModuleDict({"fused": nn.Sequential(norm, nn.SiLU()), "elsewhere": norm}).eval()
    with torch.no_grad(), pytest.raises(ValueError, match="unshared subtree"):
        with capture_vista_workloads(model, _allow_test_modules=True):
            pass
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_rejects_module_compile_call_override():
    model = make_model()
    model["feed_forward"]._compiled_call_impl = lambda x: x
    with torch.no_grad(), pytest.raises(ValueError, match="eager model"):
        with vista_kernel_overrides(model, geglu_config=CONFIG, _test_kernels=fake_kernels([])):
            pass


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                    reason="isolated adapter integration requires NVIDIA CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_isolated_cuda_adapter_uses_real_kernels_and_restores(monkeypatch, dtype):
    """CUDA operator integration only; representative classes, no checkpoint."""
    triton = pytest.importorskip("triton")
    if triton.__version__.split(".")[0] != "3":
        pytest.skip("custom kernels require Triton 3.x")
    import drivebench_tuning.kernels as kernels
    import drivebench_tuning.vista_adapter as adapter

    # Only source identity is mocked: all production dispatch and Triton calls
    # are real. These test classes reproduce the pinned operations but are not
    # presented as an actual Vista model/checkpoint execution.
    monkeypatch.setattr(adapter, "_verify_source", lambda *args, **kwargs: None)
    model = make_model().to(device="cuda")
    model["feed_forward"].to(dtype=dtype)
    model["convolution"].convolution.to(dtype=dtype)
    module_ids = {name: id(module) for name, module in model.named_modules()}
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    state_before = {name: value.clone() for name, value in model.state_dict().items()}
    calls = []
    real_geglu, real_groupnorm = kernels.fused_geglu, kernels.fused_groupnorm_silu

    def tracked_geglu(x, **kwargs):
        calls.append(("geglu", tuple(x.shape)))
        return real_geglu(x, **kwargs)

    def tracked_groupnorm(x, weight, bias, **kwargs):
        calls.append(("groupnorm_silu", tuple(x.shape)))
        return real_groupnorm(x, weight, bias, **kwargs)

    monkeypatch.setattr(kernels, "fused_geglu", tracked_geglu)
    monkeypatch.setattr(kernels, "fused_groupnorm_silu", tracked_groupnorm)
    inputs = [(torch.randn(2, 5, 4, device="cuda", dtype=dtype),
               torch.randn(2, 4, 3, 3, device="cuda", dtype=dtype)),
              (torch.randn(3, 7, 4, device="cuda", dtype=dtype),
               torch.randn(3, 4, 2, 5, device="cuda", dtype=dtype))]
    originals = [(x.clone(), image.clone()) for x, image in inputs]
    tolerance = {torch.float16: dict(rtol=3e-3, atol=3e-3),
                 torch.float32: dict(rtol=5e-5, atol=1e-5)}[dtype]
    with torch.no_grad():
        expected = [(model["feed_forward"](x), model["convolution"](image)) for x, image in inputs]
        with vista_kernel_overrides(model, geglu_config=KernelConfig(128, 4),
                                    groupnorm_config=KernelConfig(128, 4)) as selection:
            assert selection["execution"] == "custom_triton"
            for (x, image), (reference_geglu, reference_norm) in zip(inputs, expected):
                torch.testing.assert_close(model["feed_forward"](x), reference_geglu, **tolerance)
                torch.testing.assert_close(model["convolution"](image), reference_norm, **tolerance)
            torch.cuda.synchronize()
        for (x, image), (reference_geglu, reference_norm) in zip(inputs, expected):
            torch.testing.assert_close(model["feed_forward"](x), reference_geglu, rtol=0, atol=0)
            torch.testing.assert_close(model["convolution"](image), reference_norm, rtol=0, atol=0)
    assert calls == [("geglu", (2, 5, 6)), ("groupnorm_silu", (2, 4, 3, 3)),
                     ("geglu", (3, 7, 6)), ("groupnorm_silu", (3, 4, 2, 5))]
    assert {name: id(module) for name, module in model.named_modules()} == module_ids
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids
    assert all("forward" not in module.__dict__ for module in model.modules())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name], rtol=0, atol=0)
    for (x, image), (original_x, original_image) in zip(inputs, originals):
        torch.testing.assert_close(x, original_x, rtol=0, atol=0)
        torch.testing.assert_close(image, original_image, rtol=0, atol=0)
