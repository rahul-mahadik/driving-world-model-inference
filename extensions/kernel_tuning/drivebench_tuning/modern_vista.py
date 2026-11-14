"""Explicit modern Vista candidate; the historical backend remains unchanged.

Fresh-process use only. CPU unit tests establish local tensor/layout contracts;
this module does not assert checkpoint quality or CUDA performance validation.
"""
from __future__ import annotations

import contextlib
import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from drivebench.models.upstream import import_reference, verify_checkout, VISTA_COMMIT
from drivebench.models.vista import VistaBackend, _EXECUTION_LOCK


RUNTIME = {
    "schema_version": 1,
    "torch_version": "2.14.1+cu130",
    "cuda_version": "13.0",
    "attention": "native_sdpa",
    "parameter_policy": "unet_fp16_norm_fp32_vae_fp32",
    "memory_format": "channels_last_2d",
}
PORT_DEPENDENCIES = {"torchvision": "0.29.1", "einops": "0.8.2",
                     "open-clip-torch": "2.20.0", "timm": "0.9.2",
                     "transformers": "4.57.6", "pytorch-lightning": "2.6.6",
                     "kornia": "0.8.3", "omegaconf": "2.3.0"}


def numerical_runtime_settings(torch):
    """Record dispatch permissions, not an assertion about the chosen kernel."""
    return {
        "sdpa_enabled": {name: bool(getattr(torch.backends.cuda, name + "_sdp_enabled")())
                         for name in ("flash", "mem_efficient", "math", "cudnn")},
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def validate_options(model, generation, optimizations):
    if model.get("backend") != "modern_vista" or model.get("modern_runtime") != RUNTIME:
        raise ValueError("Explicit modern_vista backend and exact modern_runtime contract required")
    if any(model.get(k) is not None for k in ("compiler_runtime", "compiler_feature_boundary")):
        raise ValueError("Historical compiler adapters cannot be mixed with the modern runtime")
    if (generation.dtype != "fp16" or generation.decode_chunk_size != 14
            or generation.total_frames != 25 or generation.conditioning_frames != 1):
        raise ValueError("Modern baseline requires FP16 autocast, 25 frames, one observed frame, decoder chunk 14")
    if (optimizations.compile_decoder or optimizations.compile_mode != "default"
            or optimizations.fullgraph or optimizations.dynamic_shapes
            or optimizations.conditioning_cache_bytes or optimizations.feature_cache_interval != 1
            or optimizations.feature_cache_module is not None or optimizations.max_batch_size != 1
            or not optimizations.keep_model_resident or optimizations.pinned_memory
            or optimizations.async_transfer or optimizations.prefetch_depth
            or optimizations.attention_backend != "upstream"):
        raise ValueError("Modern baseline permits only the paired denoiser compile toggle; caching/batching/extra variants are disabled")


def runtime_diagnostics(torch):
    """Fail closed on unsupported actual runtime; never label CPU as CUDA proof."""
    if torch.__version__ != RUNTIME["torch_version"] or torch.version.cuda != RUNTIME["cuda_version"]:
        raise RuntimeError("Modern runtime requires exactly Torch 2.14.1+cu130 / CUDA 13.0")
    if not torch.cuda.is_available():
        raise RuntimeError("Modern Vista inference requires CUDA; CPU attention tests do not admit inference")
    versions = {name: importlib.metadata.version(name) for name in PORT_DEPENDENCIES}
    if any(versions[name].split("+")[0] != expected for name, expected in PORT_DEPENDENCIES.items()):
        raise RuntimeError("Installed modern inference dependencies differ from the reviewed recipe")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before starting the worker; pre-acquire CLIP assets")
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None and not constants.HF_HUB_OFFLINE:
        raise RuntimeError("Hugging Face was imported before offline mode; use a fresh process")
    try:
        importlib.metadata.version("xformers")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("Use an isolated modern environment without xformers installed")
    if any(x == "xformers" or x.startswith("xformers.") for x in sys.modules):
        raise RuntimeError("xformers is already imported; use a fresh process")
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        raise RuntimeError("Native SDPA is unavailable")
    if (os.environ.get("TORCH_COMPILE_DISABLE") not in (None, "", "0")
            or torch._dynamo.config.suppress_errors or torch._dynamo.config.disable):
        raise RuntimeError("Compiler disable/error suppression would invalidate paired runtime evidence")
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        raise RuntimeError("This modern baseline currently requires an Ampere-or-newer CUDA device")
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(capability), "sdpa_dispatch": "native_auto_actual_kernel_requires_profiler",
            "numerical_settings": numerical_runtime_settings(torch),
            "dependency_versions": versions, "asset_network_access": "offline_cache_only",
            "xformers_installed": False, "gpu_validation_completed": False}


def apply_parameter_policy(network, torch):
    """Convert only the UNet, preserving FP32 GroupNorm32 accumulation weights."""
    if "forward" in network.__dict__:
        raise RuntimeError("UNet already has an instance forward override")
    original_keys = set(network.state_dict())
    norm_states = {name: {key: value.detach().clone() for key, value in module.state_dict().items()}
                   for name, module in network.named_modules() if isinstance(module, torch.nn.GroupNorm)}
    network.half()
    normalizations = []
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.GroupNorm):
            module.float()
            module.load_state_dict(norm_states[name], strict=True)
            normalizations.append(name)
    torch.nn.utils.convert_conv2d_weight_memory_format(network, torch.channels_last)
    network.requires_grad_(False)
    original_forward = network.forward

    def forward(x, *args, **kwargs):
        if x.ndim != 4:
            raise ValueError("Vista UNet expects flattened-frame NCHW input")
        return original_forward(x.to(memory_format=torch.channels_last), *args, **kwargs)

    network.forward = forward
    if set(network.state_dict()) != original_keys:
        raise RuntimeError("Parameter conversion changed checkpoint state keys")
    convs = [name for name, module in network.named_modules()
             if isinstance(module, torch.nn.Conv2d)]
    return {"converted_unet_conv2d_modules": convs, "fp32_groupnorm_modules": normalizations,
            "unet_input": "NCHW logical shape, channels_last physical strides",
            "conv3d_layout": "unchanged", "vae_and_conditioner_parameters": "unchanged_FP32",
            "channels_last_propagation": "not_assumed_profile_actual_operators"}


@contextlib.contextmanager
def native_construction(reference, torch):
    """Scoped factories only; restored on success/failure under execution lock."""
    from .modern_attention import NativeSDPACrossAttention
    attention = importlib.import_module("vwm.modules.attention")
    video = importlib.import_module("vwm.modules.video_attention")
    vae = importlib.import_module("vwm.modules.diffusionmodules.model")
    owners = (attention.BasicTransformerBlock, video.VideoTransformerBlock)
    old_maps = [owner.ATTENTION_MODES for owner in owners]
    old_make_attn, old_load = vae.make_attn, reference.initial_model_load
    record = {}

    def make_attn(channels, attn_type="vanilla", attn_kwargs=None):
        if attn_type not in ("vanilla", "vanilla-xformers", "none"):
            raise ValueError("Unaudited VAE attention variant")
        return old_make_attn(channels, "vanilla" if attn_type == "vanilla-xformers" else attn_type, attn_kwargs)

    def initial_model_load(model):
        record.update(apply_parameter_policy(model.model.diffusion_model, torch))
        # Upstream deliberately disables VAE autocast. Global .half() would
        # break that policy and can overflow the video decoder.
        if not model.disable_first_stage_autocast:
            raise ValueError("Pinned Vista must retain FP32 first-stage execution")
        if any(p.is_floating_point() and p.dtype != torch.float32
               for part in (model.first_stage_model, model.conditioner) for p in part.parameters()):
            raise ValueError("The modern baseline retains FP32 VAE and conditioner parameters")
        count = action_count = 0
        for module in model.modules():
            if isinstance(module, NativeSDPACrossAttention):
                count += 1
                action_count += int(module.action_control)
            elif type(module).__module__.startswith("vwm.") and type(module).__name__ in (
                    "MemoryEfficientCrossAttention", "MemoryEfficientAttnBlock", "CrossAttention"):
                raise RuntimeError("An unported attention module remains in the model")
        if not count or not action_count:
            raise RuntimeError("Native attention/action projection inventory is empty")
        record.update(native_attention_modules=count, action_attention_modules=action_count)
        return old_load(model)

    try:
        for owner in owners:
            owner.ATTENTION_MODES = {"softmax": NativeSDPACrossAttention,
                                     "softmax-xformers": NativeSDPACrossAttention}
        vae.make_attn = make_attn
        reference.initial_model_load = initial_model_load
        yield record
    finally:
        for owner, original in zip(owners, old_maps):
            owner.ATTENTION_MODES = original
        vae.make_attn, reference.initial_model_load = old_make_attn, old_load


class ModernVistaBackend(VistaBackend):
    """Modern eager/compiled pair sharing input, sampler and output semantics."""
    max_batch_size = 1

    def __init__(self, model_config, generation, optimizations):
        validate_options(model_config, generation, optimizations)
        import torch
        observed_runtime = runtime_diagnostics(torch)
        root = model_config.get("upstream_dir", "vendor/Vista")
        verify_checkout(root, model_config.get("upstream_commit", VISTA_COMMIT))
        with _EXECUTION_LOCK:
            reference = import_reference(root)
            with native_construction(reference, torch) as port_record:
                # Parent loader and sampler remain unchanged; compilation is
                # selected explicitly after dtype/layout conversion below.
                super().__init__(model_config, generation,
                                 dataclasses.replace(optimizations, compile_denoiser=False))
            self.optimizations = optimizations
            if optimizations.compile_denoiser:
                self._denoiser = torch.compile(self._denoiser, backend="inductor", mode="default",
                                               dynamic=False, fullgraph=False)
        implementation = {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        attention_path = Path(__file__).with_name("modern_attention.py")
        implementation[attention_path.name] = hashlib.sha256(attention_path.read_bytes()).hexdigest()
        inventory = {}
        for parameter in self.model.parameters():
            item = inventory.setdefault(str(parameter.dtype), {"tensors": 0, "elements": 0, "bytes": 0})
            item["tensors"] += 1
            item["elements"] += parameter.numel()
            item["bytes"] += parameter.numel() * parameter.element_size()
        self.identity.update(
            backend="modern_vista", modern_runtime=dict(RUNTIME), actual_modern_runtime=observed_runtime,
            modern_port=port_record, implementation_sha256=implementation,
            parameter_dtypes=inventory, parameter_grad_policy="UNet_frozen_others_upstream_flags",
            compilation={"enabled": optimizations.compile_denoiser, "target": "denoiser",
                         "backend": "inductor", "mode": "default", "fullgraph": False, "dynamic": False,
                         "compile_cost": "lazy_first_use_must_be_measured", "fallback_free_proven": False},
            upstream_agreement="unmeasured_modern_port_requires_output_quality_validation",
            quality_preserving=False, modern_gpu_validated=False,
        )


def main():
    """Print the required contract without loading model weights or using CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-runtime", action="store_true",
                        help="Check actual CUDA/runtime compatibility; does not load Vista or validate model output")
    args = parser.parse_args()
    result = {"modern_runtime": RUNTIME, "gpu_validated": False,
              "entrypoint": "drivebench_tuning.modern_vista.ModernVistaBackend"}
    if args.check_runtime:
        import torch
        result["actual_runtime"] = runtime_diagnostics(torch)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
