from __future__ import annotations

import hashlib
import time
from collections import deque

import numpy as np

from .config import digest
from .models import create_backend
from .runtime.cache import ContextCache
from .types import GenerationConfig, GenerationResult, OptimizationConfig, PreparedContext


class InferenceEngine:
    """One engine per worker; prepared contexts are action-independent.

    This object is deliberately not thread-safe. Background workers may decode
    images; all model calls run on the owning inference thread.
    """
    def __init__(self, model=None, generation=None, optimizations=None, backend=None):
        self.generation = generation or GenerationConfig()
        self.optimizations = optimizations or OptimizationConfig()
        self.model_config = model or {"backend": "vista"}
        self.backend = backend or create_backend(self.model_config, self.generation, self.optimizations)
        self.cache = ContextCache(self.optimizations.conditioning_cache_bytes)
        self.identity = digest({"model": self.model_config, "generation": self._context_config(self.generation),
                                "preprocess": "vista-center-crop-lanczos-minus1-plus1-v1"})

    @staticmethod
    def _context_config(config):
        return {key: getattr(config, key) for key in ("height", "width", "conditioning_frames", "dtype", "cond_aug", "total_frames")}

    def synchronize(self):
        fn = getattr(self.backend, "synchronize", None)
        if fn:
            fn()

    def reset_peak_memory(self):
        fn = getattr(self.backend, "reset_peak_memory", None)
        if fn:
            fn()

    def prepare_context(self, observed_frames) -> PreparedContext:
        self.synchronize()
        started = time.perf_counter()
        if isinstance(observed_frames, np.ndarray):
            frames = np.asarray(observed_frames, dtype=np.float32)
        else:
            from .data.preprocess import read_preprocess
            frames = read_preprocess(observed_frames, self.generation.height, self.generation.width)
        expected = (self.generation.conditioning_frames, 3, self.generation.height, self.generation.width)
        if frames.shape != expected:
            raise ValueError(f"expected observed frames {expected}, got {frames.shape}")
        if not np.isfinite(frames).all() or frames.min() < -1.0001 or frames.max() > 1.0001:
            raise ValueError("observed frames must be finite and normalized to [-1,1]")
        key = hashlib.sha256(self.identity.encode() + np.ascontiguousarray(frames).tobytes()).hexdigest()
        # Nonzero augmentation is candidate stochastic computation. Conservatively
        # disable cross-request reuse instead of omitting a random seed from a key.
        cacheable = self.generation.cond_aug == 0 and self.cache.max_bytes > 0
        payload = self.cache.get(key) if cacheable else None
        hit = payload is not None
        if not hit:
            payload = self.backend.prepare_context(frames)
            if cacheable:
                self.cache.put(key, payload)
        self.synchronize()
        return PreparedContext(payload, key, time.perf_counter() - started, hit, self.identity)

    def generate(self, context, trajectories, seeds, config=None, candidate_ids=None):
        config = config or self.generation
        if context.config_identity != self.identity or self._context_config(config) != self._context_config(self.generation):
            raise ValueError("context belongs to a different model or preprocessing configuration")
        n = len(trajectories)
        if n == 0 or n != len(seeds):
            raise ValueError("supply one seed per trajectory and at least one trajectory")
        ids = list(candidate_ids) if candidate_ids is not None else [f"candidate-{i:02d}" for i in range(n)]
        if len(ids) != n or len(set(ids)) != n or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("candidate IDs must be nonempty, unique, and match trajectory count")
        actions = [np.asarray(x, dtype=np.float32) for x in trajectories]
        if any(x.shape not in {(8,), (4, 2)} or not np.isfinite(x).all() for x in actions):
            raise ValueError("Vista trajectories require eight finite numeric features (four XY pairs)")
        if any(type(s) not in (int, np.int64, np.int32) or not 0 <= int(s) < 2**63 for s in seeds):
            raise ValueError("seeds must be integers in [0, 2**63)")
        requested_batch = self.optimizations.max_batch_size
        maximum = min(requested_batch, getattr(self.backend, "max_batch_size", requested_batch))
        fallbacks = []
        if maximum < requested_batch:
            fallbacks.append(f"backend candidate-batch limit is {maximum}; requested {requested_batch}")
        batches = deque([list(range(i, min(n, i + maximum))) for i in range(0, n, maximum)])
        outputs, failures, timings, cache_stats = {}, [], {}, {}
        evaluations = peak_allocated = peak_reserved = 0
        oom_attempts = []
        gpu_time = None
        self.synchronize()
        started = time.perf_counter()
        actual_batches = []
        while batches:
            indices = batches.popleft()
            recover_oom = False
            try:
                batch = self.backend.generate_batch(context.payload, [actions[i] for i in indices],
                                                    [int(seeds[i]) for i in indices], config)
                expected = (len(indices), config.total_frames, config.height, config.width, 3)
                if (batch.frames.shape != expected or not np.isfinite(batch.frames).all()
                        or batch.frames.min() < 0 or batch.frames.max() > 1):
                    raise ValueError(f"backend returned invalid output; expected {expected}")
                actual_batches.append(len(indices))
                for local, index in enumerate(indices):
                    outputs[ids[index]] = batch.frames[local]
                evaluations += batch.denoiser_evaluations
                for k, v in batch.stage_timings_s.items():
                    timings[k] = timings.get(k, 0) + v
                for k, v in batch.cache_stats.items():
                    if isinstance(v, (int, float)):
                        cache_stats[k] = cache_stats.get(k, 0) + v
                    else:
                        cache_stats[k] = v
                fallbacks.extend(batch.fallbacks)
                peak_allocated = max(peak_allocated, batch.peak_allocated_bytes)
                peak_reserved = max(peak_reserved, batch.peak_reserved_bytes)
                if batch.gpu_inference_s is not None:
                    gpu_time = (gpu_time or 0) + batch.gpu_inference_s
            except Exception as error:
                oom = isinstance(error, MemoryError) or type(error).__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"}
                if not oom:
                    raise
                oom_attempts.append({"batch_size": len(indices), "candidate_ids": [ids[i] for i in indices],
                                     "error": str(error), "retried": len(indices) > 1})
                if len(indices) > 1:
                    midpoint = len(indices) // 2
                    batches.appendleft(indices[midpoint:])
                    batches.appendleft(indices[:midpoint])
                    fallbacks.append(f"out of memory at batch size {len(indices)}; retried smaller batches")
                else:
                    failures.append({"candidate_id": ids[indices[0]], "status": "out_of_memory", "error": str(error)})
                recover_oom = True
            # The exception traceback retains the failing backend's tensors.
            # Release the allocator only after leaving except, when Python has
            # dropped that traceback, and before retrying a smaller batch.
            if recover_oom:
                release = getattr(self.backend, "release_after_oom", None)
                if release:
                    release()
        self.synchronize()
        timings["generate_wall"] = time.perf_counter() - started
        timings["prepare_context"] = context.preparation_s
        cache_stats["context"] = self.cache.stats()
        cache_stats["context_hit"] = context.cache_hit
        cache_stats["oom_attempts"] = oom_attempts
        cache_stats["denoiser_evaluations_complete"] = not oom_attempts
        memory = getattr(self.backend, "peak_memory", None)
        if memory:
            allocated, reserved = memory()
            peak_allocated, peak_reserved = max(peak_allocated, allocated), max(peak_reserved, reserved)
        # Batch sizes and input conditions are recorded independently from count.
        effective = {**config.to_dict(), "candidate_count": n, "requested_batch_size": requested_batch,
                     "actual_batch_sizes": actual_batches, "optimizations": self.optimizations.to_dict()}
        status = "ok" if not failures else "partial" if outputs else "out_of_memory"
        return GenerationResult(outputs, ids, list(map(int, seeds)), effective, timings,
                                peak_allocated, peak_reserved, cache_stats, evaluations,
                                list(dict.fromkeys(fallbacks)), status, failures, gpu_time)
