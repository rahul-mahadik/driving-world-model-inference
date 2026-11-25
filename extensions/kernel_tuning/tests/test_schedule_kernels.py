"""CPU structural checks and opt-in CUDA numerical checks for new schedules."""

from contextlib import nullcontext
from dataclasses import replace
import importlib.util
import math
import subprocess
import sys
from types import SimpleNamespace

import pytest

from drivebench_tuning.contract import DTYPE_TOLERANCES, Workload
from drivebench_tuning import schedule_kernels as sk


def schedule(**changes):
    fields = dict(fused=True, block_size=256, num_warps=4,
                  activation_block_size=256, activation_num_warps=4,
                  reduction="chunked", reduction_block_size=256, reduction_num_warps=4)
    fields.update(changes)
    return SimpleNamespace(**fields)


GEGLU = Workload("geglu", (2, 66))
NORM = Workload("groupnorm_silu", (2, 6, 3, 5), num_groups=3)


def test_import_has_no_torch_or_triton_dependency():
    subprocess.run([sys.executable, "-c", "import sys; import drivebench_tuning.schedule_kernels; "
                    "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)


@pytest.mark.parametrize("workload,changes", [
    (GEGLU, {"activation_block_size": 128}),
    (GEGLU, {"activation_num_warps": 8}),
    (GEGLU, {"reduction": "single"}),
    (GEGLU, {"reduction_block_size": 128}),
    (GEGLU, {"reduction_num_warps": 8}),
    (NORM, {"reduction": "single", "reduction_block_size": 128}),
    (NORM, {"reduction": "atomic"}),
    (GEGLU, {"fused": 1}),
    (GEGLU, {"block_size": True}),
    (GEGLU, {"num_warps": 3}),
])
def test_inactive_or_invalid_fields_rejected(workload, changes):
    with pytest.raises(ValueError):
        sk.validate_schedule(workload, schedule(**changes))


def test_reduction_limits_depend_on_workload_and_strategy():
    size = sk.MAX_SINGLE_GROUP_SIZE + 1
    workload = Workload("groupnorm_silu", (1, 1, 1, size), num_groups=1)
    with pytest.raises(ValueError, match="single reduction"):
        sk.validate_schedule(workload, schedule(reduction="single"))
    sk.validate_schedule(workload, schedule(reduction="chunked"))
    huge = replace(workload, shape=(1, 1, 1, 4096 * 64 + 1))
    with pytest.raises(ValueError, match="group chunks"):
        sk.validate_schedule(huge, schedule(reduction_block_size=64))
    sk.validate_schedule(huge, schedule(reduction_block_size=128))


def test_distinct_structural_space_counts_do_not_include_inactive_knobs():
    # These are executable schedule counts for small groups, not measured cells.
    launches = [(b, w) for b in sk.kernels.SUPPORTED_BLOCK_SIZES for w in sk.kernels.SUPPORTED_NUM_WARPS]
    activation = [(True, 256, 4)] + [(False, b, w) for b, w in launches]
    stats = [("chunked", b, w) for b, w in launches] + [("single", 256, w) for w in sk.kernels.SUPPORTED_NUM_WARPS]
    count = 0
    for b, w in launches:
        for fused, a, aw in activation:
            s = schedule(fused=fused, block_size=b, num_warps=w,
                         activation_block_size=a, activation_num_warps=aw)
            sk.validate_schedule(GEGLU, s)
            count += 1
            for strategy, rb, rw in stats:
                fields = dict(vars(s), reduction=strategy, reduction_block_size=rb,
                              reduction_num_warps=rw)
                sk.validate_schedule(NORM, schedule(**fields))
    assert count == 812
    assert count * len(stats) == 25984


class LaunchRecorder:
    """Record lowering without executing, so CPU tests make no CUDA claim."""
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((self.name, grid, args, kwargs))
        return launch


def fake_launches(monkeypatch, torch):
    calls = []
    names = ("geglu_forward", "geglu_activation", "geglu_multiply", "groupnorm_partial",
             "groupnorm_merge", "groupnorm_single", "groupnorm_output", "silu_forward")
    runtime = SimpleNamespace(**{name: LaunchRecorder(name, calls) for name in names})
    monkeypatch.setattr(sk.kernels, "_cuda_kernels", lambda *_: runtime)
    monkeypatch.setattr(sk.importlib, "import_module", lambda _: runtime)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    return calls


def test_unfusing_geglu_creates_two_independently_scheduled_stages(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = fake_launches(monkeypatch, torch)
    x = torch.ones(GEGLU.shape, dtype=torch.float16)
    sk.execute_schedule(GEGLU, schedule(fused=False, block_size=64, num_warps=1,
                                      activation_block_size=128, activation_num_warps=8), (x,))
    assert [c[0] for c in calls] == ["geglu_activation", "geglu_multiply"]
    assert [(c[3]["BLOCK_SIZE"], c[3]["num_warps"]) for c in calls] == [(128, 8), (64, 1)]
    assert calls[0][2][1] is calls[1][2][1]
    assert calls[0][2][1].dtype == x.dtype
    calls.clear()
    sk.execute_schedule(GEGLU, schedule(), (x,))
    assert [c[0] for c in calls] == ["geglu_forward"]


@pytest.mark.parametrize("strategy", ["chunked", "single"])
@pytest.mark.parametrize("fused", [True, False])
def test_groupnorm_lowering_tracks_fusion_and_reduction(monkeypatch, strategy, fused):
    torch = pytest.importorskip("torch")
    calls = fake_launches(monkeypatch, torch)
    args = (torch.ones(NORM.shape, dtype=torch.float16), torch.ones(6), torch.zeros(6))
    s = schedule(fused=fused, block_size=128, num_warps=1, reduction=strategy,
                 reduction_block_size=64 if strategy == "chunked" else 256,
                 reduction_num_warps=2,
                 activation_block_size=256 if fused else 512,
                 activation_num_warps=4 if fused else 8)
    sk.execute_schedule(NORM, s, args)
    expected = ["groupnorm_partial", "groupnorm_merge"] if strategy == "chunked" else ["groupnorm_single"]
    expected += ["groupnorm_output"] + ([] if fused else ["silu_forward"])
    assert [c[0] for c in calls] == expected
    assert calls[0][3]["num_warps"] == 2
    output = next(c for c in calls if c[0] == "groupnorm_output")
    assert output[3]["BLOCK_SIZE"] == 128 and output[3]["num_warps"] == 1
    assert output[3]["FUSED"] is fused
    if not fused:
        assert calls[-1][3]["BLOCK_SIZE"] == 512 and calls[-1][3]["num_warps"] == 8
        assert output[2][4] is calls[-1][2][0]
        assert output[2][4].dtype == args[0].dtype


def test_cpu_inputs_cannot_be_presented_as_cuda_execution():
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        sk.execute_schedule(GEGLU, schedule(), (torch.ones(GEGLU.shape, dtype=torch.float16),))
    with pytest.raises(ValueError, match="shape and dtype"):
        sk.execute_schedule(GEGLU, schedule(), (torch.ones(GEGLU.shape),))
    with pytest.raises(ValueError, match="tuple"):
        sk.execute_schedule(GEGLU, schedule(), [torch.ones(GEGLU.shape)])


def test_cpu_reference_contract_rounds_gelu_before_multiplication():
    torch = pytest.importorskip("torch")
    # Scalar double-precision erf is independent of Torch's GELU implementation.
    gates = [-3.0, -1.2, -0.1, 0.4, 1.1, 2.2]
    x = torch.tensor([1.7] * len(gates) + gates, dtype=torch.float16)
    rounded = torch.tensor([0.5 * float(g) * (1 + math.erf(float(g) / math.sqrt(2)))
                            for g in x[len(gates):]], dtype=torch.float16)
    expected = x[:len(gates)] * rounded
    actual = sk.kernels.geglu_reference(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cpu_reference_contract_casts_groupnorm_before_silu():
    torch = pytest.importorskip("torch")
    x = torch.linspace(-4, 4, 180).reshape(2, 6, 3, 5).to(torch.bfloat16)
    weight = torch.tensor([0.3, 0.7, 1.2, 1.6, 0.8, 0.2])
    bias = torch.tensor([-0.2, -0.1, 0.3, 0.4, 0.2, -0.5])
    groups = x.double().reshape(2, 3, -1)
    norm = ((groups - groups.mean(-1, keepdim=True)) * torch.rsqrt(groups.var(-1, unbiased=False, keepdim=True) + 1e-5)).reshape_as(x)
    affine = (norm * weight.double()[None, :, None, None] + bias.double()[None, :, None, None]).to(x.dtype)
    expected = torch.nn.functional.silu(affine)
    actual = sk.kernels.groupnorm_silu_reference(x, weight, bias, num_groups=3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("operator,reduction", [("geglu", "chunked"),
                                              ("groupnorm_silu", "chunked"),
                                              ("groupnorm_silu", "single")])
def test_cuda_schedule_numerical_contract(dtype, fused, operator, reduction):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.version.hip is not None or importlib.util.find_spec("triton") is None:
        pytest.skip("Requires NVIDIA CUDA and Triton; CPU checks are not GPU evidence")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported by this device")
    torch.manual_seed(51)
    workload = (replace(GEGLU, dtype=dtype) if operator == "geglu" else
                replace(NORM, dtype=dtype, shape=(2, 6, 7, 19)))
    args = (torch.randn(workload.shape, device="cuda", dtype=getattr(torch, dtype)),)
    if operator == "groupnorm_silu":
        args += (torch.randn(6, device="cuda"), torch.randn(6, device="cuda"))
        expected = sk.kernels.groupnorm_silu_reference(*args, num_groups=3)
    else:
        expected = sk.kernels.geglu_reference(*args)
    s = schedule(fused=fused, block_size=64, num_warps=1, reduction=reduction,
                 activation_block_size=256 if fused else 128,
                 activation_num_warps=4 if fused else 2)
    actual = sk.execute_schedule(workload, s, args)
    torch.cuda.synchronize()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, **DTYPE_TOLERANCES[dtype])
