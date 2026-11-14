"""CPU contracts, not GPU performance or perceptual-quality validation."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from drivebench.types import GenerationConfig, OptimizationConfig
from drivebench_tuning.modern_attention import NativeSDPACrossAttention
from drivebench_tuning import modern_vista as modern


def reference_attention(q, k, v, *, causal=False):
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if causal:
        allowed = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))
    return scores.softmax(-1) @ v


def reference_forward(module, x, context=None, additional_tokens=None, crossframe=0):
    if additional_tokens is not None:
        x = torch.cat((additional_tokens, x), dim=1)
    context = x if context is None else context
    action = None
    if module.action_control:
        context, action = context[..., :module.context_dim], context[..., module.context_dim:]
    q, k, v = module.to_q(x), module.to_k(context), module.to_v(context)
    if module.add_lora:
        q = q + module.q_adapter_up(module.q_adapter_down(x)) * module.lora_scale
        k = k + module.k_adapter_up(module.k_adapter_down(context)) * module.lora_scale
        v = v + module.v_adapter_up(module.v_adapter_down(context)) * module.lora_scale
    if action is not None:
        k = k + module.k_adapter_action_control(action)
        v = v + module.v_adapter_action_control(action)
    if crossframe:
        k = torch.stack([k[(i // crossframe) * crossframe] for i in range(len(k))])
        v = torch.stack([v[(i // crossframe) * crossframe] for i in range(len(v))])
    # Separate head loops are intentionally independent from the port's reshape.
    outputs = []
    for head in range(module.heads):
        part = slice(head * module.dim_head, (head + 1) * module.dim_head)
        outputs.append(reference_attention(q[..., part], k[..., part], v[..., part], causal=module.causal))
    out = torch.cat(outputs, dim=-1)
    if additional_tokens is not None:
        out = out[:, additional_tokens.shape[1]:]
    result = module.to_out(out)
    if module.add_lora:
        result = result + module.out_adapter_up(module.out_adapter_down(out)) * module.lora_scale
    return result


@pytest.mark.parametrize("action,lora,additional,crossframe,causal", [
    (False, False, False, 0, False), (True, False, False, 0, False),
    (True, True, True, 2, False), (False, True, True, 3, False),
    (False, False, False, 0, True),
])
def test_native_attention_math(action, lora, additional, crossframe, causal):
    torch.manual_seed(17)
    module = NativeSDPACrossAttention(8, context_dim=8, heads=2, dim_head=4,
                                     action_control=action, add_lora=lora, lora_rank=3,
                                     causal=causal).double().eval()
    # Ensure trained action/LoRA projections participate instead of remaining zero.
    for name, parameter in module.named_parameters():
        if "adapter" in name:
            torch.nn.init.normal_(parameter, std=0.05)
    x = torch.randn(6, 4, 8, dtype=torch.float64)
    context = torch.randn(6, 5, 8 + (2432 if action else 0), dtype=torch.float64) if action else None
    tokens = torch.randn(6, 2, 8, dtype=torch.float64) if additional else None
    expected = reference_forward(module, x, context, tokens, crossframe)
    actual = module(x, context, additional_tokens=tokens,
                    n_times_crossframe_attn_in_self=crossframe, batchify_xformers=True)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_large_temporal_batch_chunk_preserves_order(monkeypatch):
    module = NativeSDPACrossAttention(4, heads=2, dim_head=2).eval()
    x = torch.randn(16385, 1, 4)
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention
    def observe(q, k, v, **kwargs):
        calls.append(len(q))
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", observe)
    actual = module(x, batchify_xformers=True)
    assert calls == [16384, 1]
    torch.testing.assert_close(actual, reference_forward(module, x))


def test_action_features_change_output_and_shape_is_enforced():
    module = NativeSDPACrossAttention(4, context_dim=3, heads=1, dim_head=4, action_control=True)
    torch.nn.init.normal_(module.v_adapter_action_control.weight, std=0.1)
    x, context = torch.randn(1, 2, 4), torch.randn(1, 3, 2435)
    a = module(x, context)
    context[..., 3:] += 0.5
    assert not torch.allclose(a, module(x, context))
    with pytest.raises(ValueError, match="19 x 128"):
        module(x, context[..., :-1])


def test_unsupported_attention_requests_fail():
    module = NativeSDPACrossAttention(4, heads=1, dim_head=4)
    with pytest.raises(NotImplementedError):
        module(torch.randn(2, 3, 4), mask=torch.ones(3, 3))
    with pytest.raises(ValueError, match="divide"):
        module(torch.randn(3, 3, 4), n_times_crossframe_attn_in_self=2)
    with pytest.raises(ValueError, match="constructor"):
        NativeSDPACrossAttention(4, imaginary_backend=True)


class GroupNorm32(torch.nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class SmallUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.norm = GroupNorm32(2, 4)
        self.temporal = torch.nn.Conv3d(4, 4, (3, 1, 1), padding=(1, 0, 0))
    def forward(self, x):
        self.input_stride = x.stride()
        return self.norm(self.conv(x))


def test_half_layout_policy_preserves_norm_and_state_keys():
    network = SmallUNet().eval()
    keys = set(network.state_dict())
    original_values = {name: value.clone() for name, value in network.state_dict().items()}
    temporal_stride = network.temporal.weight.stride()
    record = modern.apply_parameter_policy(network, torch)
    assert set(network.state_dict()) == keys
    assert network.norm.weight.dtype == torch.float32
    assert network.conv.weight.dtype == torch.float16
    assert network.conv.weight.is_contiguous(memory_format=torch.channels_last)
    assert network.temporal.weight.stride() == temporal_stride
    assert not any(p.requires_grad for p in network.parameters())
    for name, value in network.state_dict().items():
        # Casting is intentionally lossy; the exact FP16 roundtrip is specified.
        expected = original_values[name].half()
        if "norm" in name:
            expected = original_values[name]
        assert torch.equal(value, expected)
    assert torch.isfinite(network(torch.randn(2, 4, 5, 6).half())).all()
    assert network.input_stride[1] == 1
    assert record["conv3d_layout"] == "unchanged"
    with pytest.raises(RuntimeError, match="override"):
        modern.apply_parameter_policy(network, torch)


def test_factory_scope_restores_even_after_failure(monkeypatch):
    owners = [type("Basic", (), {"ATTENTION_MODES": {"old": object()}}),
              type("Video", (), {"ATTENTION_MODES": {"old": object()}})]
    old_maps = [x.ATTENTION_MODES for x in owners]
    vae = SimpleNamespace(make_attn=lambda c, t, k: (c, t, k))
    reference = SimpleNamespace(initial_model_load=lambda model: model)
    old_attn, old_load = vae.make_attn, reference.initial_model_load
    modules = {"vwm.modules.attention": SimpleNamespace(BasicTransformerBlock=owners[0]),
               "vwm.modules.video_attention": SimpleNamespace(VideoTransformerBlock=owners[1]),
               "vwm.modules.diffusionmodules.model": vae}
    monkeypatch.setattr(modern.importlib, "import_module", modules.__getitem__)
    with pytest.raises(RuntimeError, match="intentional"):
        with modern.native_construction(reference, torch):
            assert owners[0].ATTENTION_MODES["softmax"] is NativeSDPACrossAttention
            assert owners[1].ATTENTION_MODES["softmax-xformers"] is NativeSDPACrossAttention
            assert vae.make_attn(32, "vanilla-xformers") == (32, "vanilla", None)
            raise RuntimeError("intentional")
    assert all(x.ATTENTION_MODES is y for x, y in zip(owners, old_maps))
    assert vae.make_attn is old_attn and reference.initial_model_load is old_load


@pytest.mark.parametrize("changes", [
    {"feature_cache_interval": 2, "feature_cache_module": "output_blocks.9"},
    {"compile_decoder": True}, {"fullgraph": True},
    {"max_batch_size": 2}, {"conditioning_cache_bytes": 1}, {"pinned_memory": True},
])
def test_baseline_has_one_compile_toggle(changes):
    model = {"backend": "modern_vista", "modern_runtime": dict(modern.RUNTIME)}
    modern.validate_options(model, GenerationConfig(), OptimizationConfig(compile_denoiser=True))
    with pytest.raises(ValueError, match="paired denoiser"):
        modern.validate_options(model, GenerationConfig(), OptimizationConfig(**changes))


def test_modern_runtime_requires_actual_gpu_stack():
    with pytest.raises(RuntimeError, match="exactly Torch|requires CUDA"):
        modern.runtime_diagnostics(torch)


def test_numerical_runtime_identity_records_actual_dispatch_flags():
    before = modern.numerical_runtime_settings(torch)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        inside = modern.numerical_runtime_settings(torch)
        assert inside["sdpa_enabled"] == {"flash": False, "mem_efficient": False,
                                           "math": True, "cudnn": False}
        assert inside["float32_matmul_precision"] == torch.get_float32_matmul_precision()
        assert inside["deterministic_warn_only"] == torch.is_deterministic_algorithms_warn_only_enabled()
    assert modern.numerical_runtime_settings(torch) == before


def test_checkpoint_layout_matches_actual_upstream_class_when_checkout_supplied():
    root = os.environ.get("VISTA_SOURCE")
    if not root:
        pytest.skip("Set VISTA_SOURCE to the pristine pinned checkout for original-class parity")
    path = Path(root) / "vwm/modules/attention.py"
    spec = importlib.util.spec_from_file_location("vista_original_attention_oracle", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    # Execute the original forward's layout/projection logic; only its final
    # CUDA-only xFormers operation is replaced by explicit CPU attention math.
    legacy.xformers = SimpleNamespace(ops=SimpleNamespace(memory_efficient_attention=
        lambda q, k, v, **kwargs: reference_attention(q, k, v)))
    for action, lora in ((False, False), (True, False), (True, True)):
        kwargs = dict(query_dim=8, context_dim=8, heads=2, dim_head=4,
                      action_control=action, add_lora=lora, lora_rank=3)
        original = legacy.MemoryEfficientCrossAttention(**kwargs).double().eval()
        port = NativeSDPACrossAttention(**kwargs).double().eval()
        assert set(original.state_dict()) == set(port.state_dict())
        for name, parameter in original.named_parameters():
            if "adapter" in name:
                torch.nn.init.normal_(parameter, std=0.02)
        port.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(6, 3, 8, dtype=torch.float64)
        context = torch.randn(6, 4, 8 + (2432 if action else 0), dtype=torch.float64)
        tokens = torch.randn(6, 1, 8, dtype=torch.float64)
        args = dict(context=context, additional_tokens=tokens,
                    n_times_crossframe_attn_in_self=2, batchify_xformers=True)
        torch.testing.assert_close(port(x, **args), original(x, **args), atol=1e-10, rtol=1e-10)


def test_native_attention_can_be_dynamo_captured_on_cpu():
    torch.manual_seed(9)
    module = NativeSDPACrossAttention(4, heads=1, dim_head=4).eval()
    x = torch.randn(2, 3, 4)
    # This checks capture semantics only, not GPU Inductor code or its speed.
    captured = torch.compile(module, backend="eager", fullgraph=True)
    torch.testing.assert_close(captured(x), module(x))


def test_actual_reduced_vista_unet_when_full_dependencies_supplied():
    root = os.environ.get("VISTA_SOURCE")
    if not root:
        pytest.skip("Set VISTA_SOURCE for original reduced-network CPU integration")
    for name in ("open_clip", "pytorch_lightning", "transformers", "kornia", "omegaconf"):
        pytest.importorskip(name)
    from drivebench.models.upstream import import_reference
    reference = import_reference(root)
    from vwm.modules.diffusionmodules.video_model import VideoUNet
    with modern.native_construction(reference, torch):
        network = VideoUNet(
            in_channels=8, model_channels=32, out_channels=4, num_res_blocks=1,
            attention_resolutions=[1], channel_mult=[1], num_head_channels=32,
            use_linear_in_transformer=True, context_dim=16, use_spatial_context=True,
            spatial_transformer_attn_type="softmax-xformers", num_classes="sequential",
            adm_in_channels=8, extra_ff_mix_layer=True, merge_strategy="learned_with_images",
            video_kernel_size=[3, 1, 1], action_control=True,
        ).eval()
        assert sum(isinstance(x, NativeSDPACrossAttention) for x in network.modules()) == 16
        policy = modern.apply_parameter_policy(network, torch)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
            output = network(torch.randn(2, 8, 8, 8), timesteps=torch.ones(2),
                             context=torch.randn(2, 1, 2448), y=torch.randn(2, 8),
                             cond_mask=torch.tensor([1, 0]), num_frames=2)
        assert tuple(output.shape) == (2, 4, 8, 8)
        assert output.dtype == torch.float16 and torch.isfinite(output).all()
        assert len(policy["fp32_groupnorm_modules"]) == 25
