#!/usr/bin/env python3
"""Bounded BF16 denoiser probe with FP16 conditioning and temporal decoding.

This separate launcher preserves the failed global-BF16 experiment. It changes
autocast only around the existing denoiser callable and never casts weights.
Two dtype-observation hooks make all timings diagnostic, not headline results.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivebench.config import digest, file_sha256
from scripts import probe_vista_optimization as base

KIND = "precision_denoiser_bf16"
DECLARATION = {"denoiser_autocast": "bf16", "conditioning_autocast": "fp16",
               "decoder_autocast": "fp16", "weights": "unchanged_fp32",
               "dtype_hooks": ["input_blocks.0.0", "out.2"]}
HELPERS = {
    "scripts/probe_vista_optimization.py": "7ba82bf7efe5ca86ad498867a25ba4fef92d4610bdb85fc2df37ac9156d168dd",
    "scripts/validate_compiled_attention_boundary.py": "f266438c63ca8fa5b7a954906fe4c1ecdf88a7d969cf4d8e2693aa8f2c553854",
    "scripts/validate_compiled_preflight.py": "6c1aa61b6a8c656a4c541d6bf11a2dc62a9c6350769d52af36c1ebabc96a3334",
}


def bound_files(config):
    import yaml
    root = Path(__file__).resolve().parents[1]
    files = {str(Path(__file__).resolve()): file_sha256(__file__)}
    for name, expected in HELPERS.items():
        path = root / name
        if file_sha256(path) != expected:
            raise ValueError("Historical helper changed: " + name)
        files[str(path)] = expected
    path, seen = Path(config).resolve(), set()
    while True:
        if path in seen:
            raise ValueError("Config inheritance cycle")
        seen.add(path)
        files[str(path)] = file_sha256(path)
        value = yaml.safe_load(path.read_text())
        if not value.get("extends"):
            break
        path = (path.parent / value["extends"]).resolve()
    return files


@contextlib.contextmanager
def precision_backend(state):
    from drivebench.models.vista import VistaBackend
    original = VistaBackend.__init__
    handles = []

    def initialize(self, model, generation, optimizations, **kwargs):
        if kwargs.get("_upstream") or model.get("compiler_runtime") or model.get("compiler_feature_boundary"):
            raise ValueError("Precision probe must use an independent refactored eager backend")
        original(self, model, generation, optimizations, **kwargs)
        if not self.torch.cuda.is_bf16_supported():
            raise RuntimeError("Device does not support BF16")
        original_denoiser = self._denoiser
        observations = {name: {} for name in DECLARATION["dtype_hooks"]}
        network = self.model.model.diffusion_model
        modules = {"input_blocks.0.0": network.input_blocks[0][0], "out.2": network.out[2]}
        for name, module in modules.items():
            def record(_module, _args, output, name=name):
                dtype = str(output.dtype)
                observations[name][dtype] = observations[name].get(dtype, 0) + 1
            handles.append(module.register_forward_hook(record))

        def denoiser(*args, **kw):
            with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                return original_denoiser(*args, **kw)

        self._denoiser = denoiser
        self.identity["precision_probe"] = {"declaration": deepcopy(DECLARATION),
                                             "dtype_observations": observations}
        state["dtype_observations"] = observations
        state["backend_initialized"] = True

    VistaBackend.__init__ = initialize
    try:
        yield
    finally:
        VistaBackend.__init__ = original
        for handle in handles:
            handle.remove()
        state["restoration_complete"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    files = bound_files(args.config)
    state = {"declaration": deepcopy(DECLARATION), "wrapper_sha256": file_sha256(__file__),
             "bound_files": files, "headline_eligible": False, "quality_validated": False,
             "timing_note": "Two module dtype/call-count hooks; diagnostic timing only"}
    output = args.output.resolve()
    report_path = output / "probe.json"
    original_save, original_boundary = base.save, base.attention_boundary
    original_kinds = set(base.KINDS)
    base.KINDS.add(KIND)

    def save(path, value):
        if Path(path) == report_path:
            value = {**value, "precision_wrapper": deepcopy(state)}
            for execution in value.get("executions", []):
                execution["instrumented"] = True
        original_save(path, value)

    @contextlib.contextmanager
    def boundary(enabled):
        if enabled:
            raise ValueError("Precision probe cannot enable compilation")
        yield {"kind": KIND, "declaration": DECLARATION,
               "wrapper_sha256": state["wrapper_sha256"], "bound_files_sha256": digest(files)}

    code = 1
    try:
        config, _, _ = base.inspect_config(args.config)
        if config["probe"]["kind"] != KIND or config.get("precision_runtime") != DECLARATION:
            raise ValueError("Select the exact stage-specific precision declaration")
        base.save, base.attention_boundary = save, boundary
        with precision_backend(state):
            code = base.main(["--config", str(args.config), "--output", str(output)])
    finally:
        base.save, base.attention_boundary = original_save, original_boundary
        base.KINDS.clear()
        base.KINDS.update(original_kinds)
        if report_path.exists():
            report = json.loads(report_path.read_text())
            state["bound_files_unchanged"] = all(file_sha256(p) == h for p, h in files.items())
            observed = state.get("dtype_observations", {})
            state["bf16_observed_at_both_boundaries"] = bool(observed) and all(
                item.get("torch.bfloat16", 0) > 0 and set(item) == {"torch.bfloat16"}
                for item in observed.values())
            if not state["bound_files_unchanged"] or (code in (0, 2) and not state["bf16_observed_at_both_boundaries"]):
                code = 1
                report["status"] = "failed_precision_evidence"
            report.update(precision_wrapper=state, exit_code=code)
            original_save(report_path, report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
