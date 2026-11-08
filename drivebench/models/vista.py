"""Candidate-isolated Vista inference with an unchanged reference sampler.

The module is CPU-importable. CUDA, torch, and Vista dependencies are loaded
only when a real backend is constructed. No model weights are bundled.
"""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from drivebench.models.upstream import (
    VISTA_COMMIT, import_reference, observed_only_padding, sha256_file,
    verify_checkpoint, verify_checkout,
)
from drivebench.runtime.feature_cache import FixedIntervalFeatureCache
from drivebench.runtime.compiler import create_compiler_runtime, validate_compiler_runtime
from drivebench.types import BackendBatch, GenerationConfig, OptimizationConfig


UC_ZERO_KEYS = ["cond_frames", "cond_frames_without_noise", "command", "trajectory", "speed", "angle", "goal"]
# Only full blocks whose leading dimension retains the ordered CFG frame batch.
# output_blocks.9 was selected from a separate measured development block profile.
FEATURE_CACHE_MODULES = {
    "middle_block": "middle_block",
    "model.diffusion_model.middle_block": "middle_block",
    "output_blocks.9": "output_blocks.9",
    "model.diffusion_model.output_blocks.9": "output_blocks.9",
}
# Global RNG and upstream's residency flag are process global. All instances
# participate in the same lock; use separate worker processes for concurrency.
_EXECUTION_LOCK = threading.RLock()


class VistaBackend:
    # Experimental implementation bound, not a measured hardware capacity.
    # The default requested batch remains one. Explicit larger requests retain
    # serial conditioning/decoding; the engine reports and splits an OOM batch.
    max_batch_size = 16

    def __init__(
        self, model_config: dict, generation: GenerationConfig,
        optimizations: OptimizationConfig, *, _upstream: bool = False,
    ):
        self.generation = generation
        self.optimizations = optimizations
        self.is_upstream = _upstream
        self.max_batch_size = 1 if _upstream else type(self).max_batch_size
        self._compiler_feature_boundary = model_config.get("compiler_feature_boundary")
        validate_compiler_runtime(model_config, optimizations, upstream=_upstream)
        self._validate_options()
        root = model_config.get("upstream_dir", "vendor/Vista")
        self.identity = verify_checkout(root, model_config.get("upstream_commit", VISTA_COMMIT))
        self.identity.update(verify_checkpoint(model_config.get("checkpoint", ""), model_config.get("checkpoint_sha256", "")))
        config_path = Path(model_config.get("inference_config", Path(root) / "configs/inference/vista.yaml")).resolve()
        pinned_config = Path(root).resolve() / "configs/inference/vista.yaml"
        if config_path != pinned_config:
            raise ValueError("Use the untouched pinned configs/inference/vista.yaml; unsupported model configs require a new audit")
        self.identity.update({
            "inference_config": str(config_path), "inference_config_sha256": sha256_file(config_path),
            "backend": "upstream" if _upstream else "vista", "dtype": generation.dtype,
            "candidate_execution": "candidate_major_joint_denoising" if not _upstream and optimizations.max_batch_size > 1 else "serial",
            "candidate_batch_limit": self.max_batch_size,
            "candidate_batch_limit_basis": "original_serial_reference" if _upstream else "experimental_implementation_bound_not_hardware_capacity",
            "candidate_batch_agreement": "requires_gpu_validation" if not _upstream and optimizations.max_batch_size > 1 else "not_applicable",
            "upstream_agreement": "requires_gpu_validation",
            "checkpoint_loading": "original Vista loader followed by strict state-key verification",
            "random_stream": "per-candidate CPU and CUDA fork; original draw order",
            "decoder_chunk_quality": "requires_gpu_validation" if generation.decode_chunk_size != 14 else "upstream_default",
        })
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Vista requires the pinned CUDA GPU environment; see docs/vista-integration.md") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Vista inference requires a CUDA GPU; use the synthetic backend only for CPU contract tests")
        self.torch = torch
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[generation.dtype]
        if generation.dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but this CUDA device does not support BF16")
        self.reference = import_reference(root)
        with _EXECUTION_LOCK:
            self._compiler_runtime = create_compiler_runtime(model_config, optimizations, torch, upstream=_upstream)
            if self._compiler_runtime is not None:
                self.identity["compiler_runtime"] = self._compiler_runtime.identity
            self.reference.set_lowvram_mode(False)
            self.model = self.reference.init_model({"config": str(config_path), "ckpt": self.identity["checkpoint"]})
        from safetensors import safe_open
        with safe_open(self.identity["checkpoint"], framework="pt", device="cpu") as checkpoint:
            checkpoint_keys = set(checkpoint.keys())
        model_keys = set(self.model.state_dict())
        if checkpoint_keys != model_keys:
            raise RuntimeError(
                "Checkpoint state keys do not match pinned Vista; "
                f"missing={sorted(model_keys - checkpoint_keys)}, unexpected={sorted(checkpoint_keys - model_keys)}"
            )
        self.model.eval()
        # Preserve the upstream loader's selective parameter flags. Torch 2.0.1
        # autocast caches FP32 weight casts only for requires_grad leaf tensors;
        # globally freezing the UNet changes its inference timing behavior.
        inventory: dict[str, dict[str, int]] = {}
        for parameter in self.model.parameters():
            entry = inventory.setdefault(str(parameter.dtype), {
                "tensors": 0, "elements": 0, "bytes": 0,
                "requires_grad_tensors": 0, "requires_grad_elements": 0,
            })
            entry["tensors"] += 1
            entry["elements"] += parameter.numel()
            entry["bytes"] += parameter.numel() * parameter.element_size()
            if parameter.requires_grad:
                entry["requires_grad_tensors"] += 1
                entry["requires_grad_elements"] += parameter.numel()
        self.identity["parameter_dtypes"] = inventory
        self.identity["autocast_dtype"] = generation.dtype
        self.identity["parameter_grad_policy"] = "preserve_upstream_loader_flags"
        self.identity["execution_grad_mode"] = "no_grad"
        self.identity["autocast_cache_enabled"] = torch.is_autocast_cache_enabled()
        self._profile_enabled = os.environ.get("DRIVEBENCH_PROFILE") == "1"
        self._denoiser = lambda x, sigma, cond, mask: self.model.denoiser(self.model.model, x, sigma, cond, mask)
        self.feature_cache = FixedIntervalFeatureCache(optimizations.feature_cache_interval)
        self._feature_cache_module = None
        self._feature_step = 0
        self._active_feature_scope = False
        self._compiled_feature_forward = None
        if optimizations.feature_cache_interval > 1 or self._compiler_feature_boundary:
            self._install_feature_cache()
        with _EXECUTION_LOCK, self._compiler_scope(require_graphs=False), self._feature_forward_scope():
            if optimizations.compile_denoiser:
                self._denoiser = torch.compile(
                    self._denoiser, mode=optimizations.compile_mode,
                    dynamic=optimizations.dynamic_shapes, fullgraph=optimizations.fullgraph,
                )
            if optimizations.compile_decoder:
                # Keep VideoDecoder's type intact: decode_first_stage checks isinstance.
                self.model.first_stage_model.decoder.forward = torch.compile(
                    self.model.first_stage_model.decoder.forward,
                    mode=optimizations.compile_mode, dynamic=optimizations.dynamic_shapes,
                    fullgraph=optimizations.fullgraph,
                )
        self._samplers: dict[tuple[int, float], Any] = {}
        self._transfer_buffers = None

    def _validate_options(self):
        g, o = self.generation, self.optimizations
        if g.total_frames != 25:
            raise ValueError("The pinned Vista adapter currently validates only a 25-frame, single-round clip")
        if g.height % 64 or g.width % 64:
            raise ValueError("Vista spatial dimensions must be divisible by 64 for the VAE and UNet downsampling path")
        if g.dtype not in {"fp16", "bf16"}:
            raise ValueError("Vista supports fp16 or bf16 autocast; FP32 requires auditing upstream autocast decorators")
        if g.decode_chunk_size in {4, 5}:
            raise ValueError("Upstream overlap=3 decoder requires chunk size 1-3 or >=6; chunks 4/5 lose overlap context")
        if o.attention_backend != "upstream":
            raise ValueError("Only upstream xFormers attention is supported; alternative masks/scaling/layouts are unvalidated")
        if not o.keep_model_resident:
            raise ValueError("This adapter keeps model components resident; offload is not implemented")
        if o.feature_cache_interval > 1 and o.compile_denoiser and not getattr(self, "_compiler_feature_boundary", None):
            raise ValueError("Feature reuse with compiled denoiser needs a graph-break audit; run these experiments separately")
        if o.feature_cache_interval > 1 and o.max_batch_size > 1:
            raise ValueError("Feature reuse is scoped to one candidate; run candidate batching and feature reuse separately")
        if o.feature_cache_interval > 1 and o.feature_cache_module not in FEATURE_CACHE_MODULES:
            raise ValueError("Only middle_block and output_blocks.9 are audited for fixed-interval reuse")
        if self.is_upstream:
            if g.dtype != "fp16" or g.decode_chunk_size != 14:
                raise ValueError("The untouched upstream baseline requires its FP16 autocast and decoder chunk size 14")
            if o.compile_denoiser or o.compile_decoder or o.feature_cache_interval > 1 or o.conditioning_cache_bytes:
                raise ValueError("Upstream baseline cannot enable compiler or conditioning/feature cache changes")

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def reset_peak_memory(self):
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory(self):
        return (self.torch.cuda.max_memory_allocated(self.device),
                self.torch.cuda.max_memory_reserved(self.device))

    def release_after_oom(self):
        # Candidate scopes and temporary forward replacements are restored by
        # finally blocks before the executor invokes this method.
        self.torch.cuda.empty_cache()

    def _autocast(self):
        return self.torch.autocast("cuda", dtype=self.dtype)

    def prepare_context(self, frames: np.ndarray) -> dict[str, Any]:
        g, o, torch = self.generation, self.optimizations, self.torch
        padded = observed_only_padding(frames, g.total_frames)
        if frames.shape != (g.conditioning_frames, 3, g.height, g.width):
            raise ValueError("Observed input shape does not match generation conditioning_frames/height/width")
        with _EXECUTION_LOCK, torch.cuda.device(self.device), torch.no_grad(), self._autocast(), self.model.ema_scope("Context"):
            if o.async_transfer:
                from drivebench.runtime.buffers import PinnedDoubleBuffer
                if self._transfer_buffers is None:
                    self._transfer_buffers = PinnedDoubleBuffer(padded.shape, torch.float32, self.device)
                with self._transfer_buffers.transfer(padded) as staged:
                    # A retained context must own its storage after a buffer
                    # slot is released. Stream waits are managed by the pool.
                    images = staged.clone()
            else:
                host = torch.from_numpy(padded)
                if o.pinned_memory:
                    host = host.pin_memory()
                images = host.to(self.device)
            payload: dict[str, Any] = {"images": images, "observed_count": len(frames), "image_embeddings": {}, "posterior_parameters": []}
            # Hold pinned memory until same-stream transfer completes. Every
            # consumer runs on that stream; there is no cross-stream dependency.
            self.synchronize()
            remaining = o.conditioning_cache_bytes
            if remaining and not self.is_upstream:
                value = self.reference.init_embedder_options([x.input_key for x in self.model.conditioner.embedders])
                value.update(cond_frames_without_noise=images[:1], cond_frames=images[:1], cond_aug=g.cond_aug)
                batch, _ = self.reference.get_batch(list(value), value, [g.total_frames])
                for index, embedder in enumerate(self.model.conditioner.embedders):
                    key = embedder.input_key
                    if key == "cond_frames_without_noise":
                        deterministic = embedder.open_clip.ucg_rate == 0 and embedder.open_clip.max_crops == 0
                    elif key == "cond_frames" and g.cond_aug == 0:
                        deterministic = embedder.sigma_sampler is None and not embedder.encoder.regularization.sample
                    else:
                        continue
                    if not deterministic:
                        continue
                    feature = embedder(batch[key]).detach()
                    size = feature.numel() * feature.element_size()
                    if size <= remaining:
                        payload["image_embeddings"][index] = feature
                        remaining -= size
                # Cache deterministic posterior parameters, not sampled latents.
                # Sampling remains in each candidate and consumes the same CPU
                # random draws with the same upstream chunk shapes/order.
                encoder_chunk = self.model.en_and_decode_n_samples_a_time
                params = []
                with torch.autocast("cuda", enabled=not self.model.disable_first_stage_autocast):
                    for chunk in images.split(encoder_chunk):
                        parameters, _ = self.model.first_stage_model.encode(chunk, unregularized=True)
                        params.append(parameters.detach())
                size = sum(x.numel() * x.element_size() for x in params)
                if size <= remaining:
                    payload["posterior_parameters"] = params
                    remaining -= size
                payload["conditioning_cache_bytes"] = o.conditioning_cache_bytes - remaining
            return payload

    @contextlib.contextmanager
    def _cached_embeddings(self, payload, stats):
        originals = []
        try:
            for index, feature in payload["image_embeddings"].items():
                embedder = self.model.conditioner.embedders[index]
                originals.append((embedder, embedder.forward))
                def cached_forward(_input, *, _feature=feature):
                    stats["conditioning_embedding_hits"] += 1
                    return _feature.clone()
                embedder.forward = cached_forward
            yield
        finally:
            for embedder, original in originals:
                embedder.forward = original

    def _encode(self, payload, stats):
        torch = self.torch
        if not payload["posterior_parameters"]:
            return self.model.encode_first_stage(payload["images"])
        stats["posterior_parameter_hits"] += 1
        with torch.autocast("cuda", enabled=not self.model.disable_first_stage_autocast):
            latents = [self.model.first_stage_model.regularization(parameters)[0] for parameters in payload["posterior_parameters"]]
            return torch.cat(latents, dim=0) * self.model.scale_factor

    def _install_feature_cache(self):
        module_path = FEATURE_CACHE_MODULES.get(self.optimizations.feature_cache_module)
        if module_path is None:
            raise ValueError("Only middle_block and output_blocks.9 are audited for fixed-interval reuse")
        network = self.model.model.diffusion_model
        if module_path == "middle_block":
            module = network.middle_block
        else:
            if len(network.output_blocks) != 12:
                raise ValueError("output_blocks.9 reuse requires the pinned twelve-output-block topology")
            module = network.output_blocks[9]
        self._feature_cache_module = module_path
        original = module.forward
        def forward(*args, **kwargs):
            # Interval one is a boundary-only control: no cache allocation or
            # clone overhead is introduced into the compiled reference.
            if not self._active_feature_scope or self.optimizations.feature_cache_interval == 1:
                return original(*args, **kwargs)
            x = args[0]
            if len(x.shape) != 4 or x.shape[0] != 2 * self.generation.total_frames:
                raise RuntimeError("Feature reuse expected exactly [unconditional_T, conditional_T] for one candidate")
            signature = (tuple(x.shape), tuple(x.stride()), str(x.dtype), str(x.device))
            return self.feature_cache.evaluate(
                block=module_path, step=self._feature_step, cfg_scope="unconditional_then_conditional",
                signature=signature, compute=lambda: original(*args, **kwargs),
            )
        if getattr(self, "_compiler_feature_boundary", None):
            # Create exactly once. Dynamo sees a stable graph boundary, never
            # the changing Python step index or the cached feature tensor.
            disabled = self.torch._dynamo.disable(forward)
            if getattr(disabled, "_torchdynamo_orig_callable", None) is not forward:
                raise RuntimeError("Compiled feature boundary did not preserve its original callable")
            self._compiled_feature_forward = (module, original, disabled)
            self.identity["compiler_feature_boundary"] = {
                "module": module_path, "kind": "explicit_full_block_dynamo_disable",
                "stable_wrapper_per_backend": True, "installed_only_during_execution": True,
                "original_forward_restored_after_scope": True,
                "cache_state_hidden_from_dynamo": True,
                "interval_one_policy": "original_block_without_cache_or_clone",
                "validation": "requires_independent_gpu_compile_stability_and_quality_evidence",
            }
        else:
            module.forward = forward

    def _compiler_scope(self, *, require_graphs):
        runtime = getattr(self, "_compiler_runtime", None)
        return runtime.activate(require_graphs=require_graphs) if runtime is not None else contextlib.nullcontext()

    @contextlib.contextmanager
    def _feature_forward_scope(self):
        replacement = getattr(self, "_compiled_feature_forward", None)
        if replacement is None:
            yield
            return
        module, original, wrapper = replacement
        current = module.forward
        # A class-defined forward produces a fresh bound-method object on each
        # read. Compare its owner/function, not that transient object's identity.
        same_bound_method = (getattr(current, "__self__", None) is module
                             and getattr(current, "__func__", None) is not None
                             and getattr(current, "__func__", None) is getattr(original, "__func__", None))
        if current is not original and not same_bound_method:
            raise RuntimeError("Selected feature block forward changed outside its execution scope")
        module.forward = wrapper
        try:
            yield
        finally:
            module.forward = original

    def _sampler(self, config):
        key = (config.steps, config.guidance_scale)
        if key not in self._samplers:
            self._samplers[key] = self.reference.init_sampling(
                guider="VanillaCFG", steps=config.steps,
                cfg_scale=config.guidance_scale, num_frames=config.total_frames,
            )
        return self._samplers[key]

    def _join_candidate_conditions(self, conditions, num_frames):
        """Keep candidate-major frame blocks intact before VanillaCFG doubles them.

        The pinned conditioner emits these three frame-indexed tensors. Reject
        other layouts instead of silently assigning another clip's action to a
        temporal block. All vendor temporal reshapes still use num_frames=25.
        """
        keys = {"vector", "crossattn", "concat"}
        if any(set(condition) != keys for condition in conditions):
            raise ValueError("Candidate batching requires the pinned vector/crossattn/concat conditioning layout")
        for key in keys:
            expected = tuple(conditions[0][key].shape)
            if not expected or expected[0] != num_frames:
                raise ValueError(f"Candidate {key} conditioning must contain exactly {num_frames} frames")
            if any(tuple(condition[key].shape) != expected for condition in conditions):
                raise ValueError(f"Candidate {key} conditioning shapes differ")
        return {key: self.torch.cat([condition[key] for condition in conditions], dim=0)
                for key in sorted(keys)}

    def _generate_joint_batch(self, payload, trajectories, seeds, config, stats, timed, outputs):
        """Prepare random inputs separately, denoise together, decode separately.

        This is a real larger UNet batch, not concurrent serial calls. CUDA
        kernels may choose different floating-point algorithms at this shape;
        strict upstream agreement is an external prerequisite for accepting it.
        """
        torch = self.torch
        sampler = self._sampler(config)
        if getattr(sampler, "s_churn", None) != 0:
            raise ValueError("Candidate batching requires the pinned deterministic sampler with s_churn=0")
        prepared = []
        evaluations = 0
        with torch.random.fork_rng(devices=[self.device.index]), torch.no_grad(), self._autocast(), self.model.ema_scope("Sampling"):
            with timed("gpu_inference"):
                for trajectory, seed in zip(trajectories, seeds):
                    with torch.random.fork_rng(devices=[self.device.index]):
                        torch.random.default_generator.manual_seed(seed)
                        torch.cuda.manual_seed(seed)
                        value = self.reference.init_embedder_options([x.input_key for x in self.model.conditioner.embedders])
                        cond_img = payload["images"][:1]
                        value.update(
                            cond_frames_without_noise=cond_img, cond_aug=config.cond_aug,
                            # The zero-augmentation draw is part of the seed contract.
                            cond_frames=cond_img + config.cond_aug * torch.randn_like(cond_img),
                            trajectory=torch.as_tensor(np.asarray(trajectory, dtype=np.float32).reshape(8), device=self.device),
                        )
                        with timed("conditioning_gpu"), self._cached_embeddings(payload, stats):
                            c, uc = self.reference.get_condition(self.model, value, config.total_frames, UC_ZERO_KEYS, "cuda")
                        with timed("encode_gpu"):
                            z = self._encode(payload, stats)
                        # Original VAE posterior CPU draws (14 then 11 frames)
                        # precede this candidate's CUDA latent draw.
                        noise = torch.randn_like(z)
                        prepared.append((c, uc, z, noise))
                with timed("denoising_gpu"):
                    c = self._join_candidate_conditions([item[0] for item in prepared], config.total_frames)
                    uc = self._join_candidate_conditions([item[1] for item in prepared], config.total_frames)
                    z = torch.cat([item[2] for item in prepared], dim=0)
                    noise = torch.cat([item[3] for item in prepared], dim=0)
                    del prepared  # Do not retain duplicate candidate tensors during UNet execution.
                    mask = torch.zeros(len(seeds) * config.total_frames, device=self.device)
                    for index in range(len(seeds)):
                        start = index * config.total_frames
                        mask[start:start + config.conditioning_frames] = 1

                    def denoiser(x, sigma, cond, cond_mask):
                        nonlocal evaluations
                        evaluations += 1
                        return self._denoiser(x, sigma, cond, cond_mask)

                    latent = sampler(denoiser, noise, cond=c, uc=uc, cond_frame=z, cond_mask=mask)
                    del c, uc, noise, mask
            # VideoDecoder uses overlapping windows. Feeding B*T frames to
            # it would leak a previous candidate into the next candidate.
            for index in range(len(seeds)):
                with timed("gpu_inference"):
                    start = index * config.total_frames
                    clip = latent[start:start + config.total_frames]
                    clip[0] = z[start]
                    samples_z = torch.zeros_like(clip, dtype=torch.float32)
                    samples_z[:] = clip
                    with timed("decode_gpu"):
                        original_chunk = self.model.en_and_decode_n_samples_a_time
                        self.model.en_and_decode_n_samples_a_time = config.decode_chunk_size
                        try:
                            decoded = self.model.decode_first_stage(samples_z)
                        finally:
                            self.model.en_and_decode_n_samples_a_time = original_chunk
                        samples = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
                with timed("device_to_host_gpu"):
                    outputs.append(samples.float().permute(0, 2, 3, 1).contiguous().cpu().numpy())
                del clip, samples_z, decoded, samples
        return evaluations

    def generate_batch(self, payload, trajectories, seeds, config: GenerationConfig) -> BackendBatch:
        for name in ("total_frames", "conditioning_frames", "height", "width", "dtype", "cond_aug"):
            if getattr(config, name) != getattr(self.generation, name):
                raise ValueError("Context-dependent configuration changed; create a new backend/context")
        if self.is_upstream and config.decode_chunk_size != 14:
            raise ValueError("Upstream baseline requires its original decoder chunk size 14")
        if config.decode_chunk_size in {4, 5}:
            raise ValueError("Upstream overlap=3 decoder requires chunk size 1-3 or >=6")
        if not trajectories or len(trajectories) != len(seeds):
            raise ValueError("Provide equal, nonempty trajectory and seed lists")
        for trajectory in trajectories:
            if np.asarray(trajectory).shape not in {(4, 2), (8,)} or not np.isfinite(trajectory).all():
                raise ValueError("Vista trajectory must be four finite coordinate pairs [4, 2] or their flat [8] representation")
        for seed in seeds:
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError("Seeds must be integers in [0, 2**63)")
        if len(seeds) > min(self.max_batch_size, self.optimizations.max_batch_size):
            raise ValueError("Candidate batch exceeds the configured backend limit; use InferenceEngine for bounded splitting")
        torch = self.torch
        outputs, events = [], []
        stats = {"conditioning_embedding_hits": 0, "posterior_parameter_hits": 0,
                 "conditioning_cache_bytes": payload.get("conditioning_cache_bytes", 0),
                 "conditioning_cached_embedders": sorted(self.model.conditioner.embedders[index].input_key
                                                        for index in payload["image_embeddings"])}
        measured_evaluations = 0
        fallbacks = []
        feature_hits_before, feature_misses_before = self.feature_cache.hits, self.feature_cache.misses
        @contextlib.contextmanager
        def timed(name):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                if self._profile_enabled:
                    with torch.profiler.record_function(f"drivebench::{name}"), torch.cuda.nvtx.range(f"drivebench::{name}"):
                        yield
                else:
                    yield
            finally:
                end.record()
                events.append((name, start, end))
        with _EXECUTION_LOCK, torch.cuda.device(self.device), self._compiler_scope(require_graphs=True), self._feature_forward_scope():
            self.reference.set_lowvram_mode(False)
            if len(seeds) > 1:
                measured_evaluations = self._generate_joint_batch(payload, trajectories, seeds, config, stats, timed, outputs)
            for candidate_index, (trajectory, seed) in enumerate(zip(trajectories, seeds) if len(seeds) == 1 else ()):
                with torch.random.fork_rng(devices=[self.device.index]), torch.no_grad(), self._autocast(), self.model.ema_scope("Sampling"):
                    torch.random.default_generator.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    # Seed CPU and the current device only. fork_rng restores
                    # both after each candidate without touching other GPUs.
                    self._feature_step = 0
                    with timed("gpu_inference"), self.feature_cache.candidate((candidate_index, seed)):
                        value = self.reference.init_embedder_options([x.input_key for x in self.model.conditioner.embedders])
                        cond_img = payload["images"][:1]
                        value.update(
                            cond_frames_without_noise=cond_img, cond_aug=config.cond_aug,
                            # Do not remove this draw when cond_aug==0: upstream consumes it.
                            cond_frames=cond_img + config.cond_aug * torch.randn_like(cond_img),
                            trajectory=torch.as_tensor(np.asarray(trajectory, dtype=np.float32).reshape(8), device=self.device),
                        )
                        sampler = self._sampler(config)
                        if self.is_upstream:
                            def count_call(_module, _inputs):
                                nonlocal measured_evaluations
                                measured_evaluations += 1
                            handle = self.model.denoiser.register_forward_pre_hook(count_call)
                            try:
                                with timed("upstream_sample_gpu"):
                                    samples, _, _ = self.reference.do_sample(
                                        payload["images"], self.model, sampler, value,
                                        num_rounds=1, num_frames=config.total_frames,
                                        force_uc_zero_embeddings=UC_ZERO_KEYS,
                                        initial_cond_indices=list(range(config.conditioning_frames)),
                                        device="cuda",
                                    )
                            finally:
                                handle.remove()
                        else:
                            with timed("conditioning_gpu"), self._cached_embeddings(payload, stats):
                                c, uc = self.reference.get_condition(
                                    self.model, value, config.total_frames, UC_ZERO_KEYS, "cuda",
                                )
                            with timed("encode_gpu"):
                                z = self._encode(payload, stats)
                            def denoiser(x, sigma, cond, mask):
                                nonlocal measured_evaluations
                                self._feature_step = measured_evaluations_candidate[0]
                                measured_evaluations += 1
                                measured_evaluations_candidate[0] += 1
                                self._active_feature_scope = True
                                try:
                                    return self._denoiser(x, sigma, cond, mask)
                                finally:
                                    self._active_feature_scope = False
                            measured_evaluations_candidate = [0]
                            with timed("denoising_gpu"):
                                mask = torch.zeros(config.total_frames, device=self.device)
                                mask[:config.conditioning_frames] = 1
                                noise = torch.randn_like(z)
                                latent = sampler(denoiser, noise, cond=c, uc=uc, cond_frame=z, cond_mask=mask)
                                latent[0] = z[0]
                                # Original do_sample copies into float32 samples_z.
                                samples_z = torch.zeros_like(z, dtype=torch.float32)
                                samples_z[:] = latent
                            with timed("decode_gpu"):
                                original_chunk = self.model.en_and_decode_n_samples_a_time
                                self.model.en_and_decode_n_samples_a_time = config.decode_chunk_size
                                try:
                                    decoded = self.model.decode_first_stage(samples_z)
                                finally:
                                    self.model.en_and_decode_n_samples_a_time = original_chunk
                                samples = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
                    with timed("device_to_host_gpu"):
                        outputs.append(samples.float().permute(0, 2, 3, 1).contiguous().cpu().numpy())
            self.synchronize()
        runtime = getattr(self, "_compiler_runtime", None)
        if runtime is not None:
            stats["compiler_runtime"] = runtime.last_execution
        if getattr(self, "_compiled_feature_forward", None) is not None:
            stats["compiler_feature_boundary"] = {
                "module": self._feature_cache_module, "restoration_complete": True,
                "interval": self.optimizations.feature_cache_interval,
                "boundary_only_control": self.optimizations.feature_cache_interval == 1,
            }
        durations: dict[str, float] = {}
        for name, start, end in events:
            durations[name] = durations.get(name, 0.0) + start.elapsed_time(end) / 1000
        gpu_inference_s = durations.pop("gpu_inference", 0.0)
        stats.update(
            feature_hits=self.feature_cache.hits - feature_hits_before,
            feature_misses=self.feature_cache.misses - feature_misses_before,
            feature_policy="approximate_fixed_interval_full_block_output" if self.optimizations.feature_cache_interval > 1 else "disabled",
            feature_cache_module=getattr(self, "_feature_cache_module", None),
            # Metadata stays nested so the engine does not add the interval
            # across separately executed candidates as it does hit counters.
            feature_cache_configuration={
                "module": getattr(self, "_feature_cache_module", None),
                "interval": self.optimizations.feature_cache_interval,
                "candidate_scope": "single_candidate",
                "cfg_layout": "unconditional_T_then_conditional_T",
            } if self.optimizations.feature_cache_interval > 1 else None,
            joint_denoising_batches=int(len(seeds) > 1),
            serial_denoising_batches=int(len(seeds) == 1),
        )
        return BackendBatch(
            frames=np.stack(outputs), stage_timings_s=durations,
            denoiser_evaluations=measured_evaluations, cache_stats=stats, fallbacks=fallbacks,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device), gpu_inference_s=gpu_inference_s,
        )
