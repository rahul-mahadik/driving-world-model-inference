"""Deterministic CPU test double. It is NOT a driving world model."""
import time

import numpy as np

from drivebench.types import BackendBatch


class SyntheticBackend:
    max_batch_size = 16
    identity = {"backend": "synthetic", "checkpoint_sha256": "synthetic-v1"}

    def __init__(self, model_config, generation, optimizations):
        self.optimizations = optimizations
        self.generation = generation

    def prepare_context(self, frames):
        return np.array(frames, copy=True)

    def synchronize(self):
        pass

    def generate_batch(self, payload, trajectories, seeds, config):
        started = time.perf_counter()
        outputs = []
        evaluations = 0
        reused = 0
        for trajectory, seed in zip(trajectories, seeds):
            rng = np.random.default_rng(seed)
            initial = np.transpose(payload[0], (1, 2, 0)) * 0.5 + 0.5
            frames = np.repeat(initial[None], config.total_frames, axis=0)
            noise = rng.normal(0, .025, frames.shape).astype(np.float32)
            action = np.asarray(trajectory).reshape(-1, 2).mean(axis=0)
            feature = None
            for step in range(config.steps):
                if step % self.optimizations.feature_cache_interval == 0:
                    feature = noise * (1 - step / config.steps)
                    evaluations += 1
                else:
                    reused += 1
            for t in range(config.conditioning_frames, config.total_frames):
                shift = int(round(float(action[0]) * t / config.total_frames))
                frames[t] = np.roll(initial, shift, axis=1) + feature[t]
            frames[:config.conditioning_frames] = np.transpose(payload, (0, 2, 3, 1)) * .5 + .5
            outputs.append(np.clip(frames, 0, 1).astype(np.float32))
        return BackendBatch(frames=np.stack(outputs),
                            stage_timings_s={"synthetic_generation": time.perf_counter() - started},
                            denoiser_evaluations=evaluations,
                            cache_stats={"feature_reuses": reused},
                            fallbacks=["SYNTHETIC TEST DOUBLE: no Vista inference or GPU measurement"])
