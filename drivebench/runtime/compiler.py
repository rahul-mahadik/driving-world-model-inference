"""Scoped, explicitly selected compiler runtime for the pinned Vista stack.

The caller must hold Vista's process execution lock for construction and every
activation. No registration is installed at import time. This module does not
change kernels, widen numerical tolerances, or enable compiler-error fallback.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


RUNTIME_NAME = "registered_einops_attention_v1"
EINOPS_SHA256 = {
    "_torch_specific.py": "3b2e546ea034e446afbe60cb79450d271f6202b2db212c7a40f0753aa2278530",
    "einops.py": "6d5af8549c684fb0a4b2261e8207a18ffcb638102f7e2f52a4d1e101f0f722a6",
    "packing.py": "cfd55f3fc49680d4ef8bbe9ab6662c970eb4f004e47fcbca64749351af6caeb2",
}
NATIVE_SOURCE_SHA256 = "6ee2ed49dfd70c3ccce4e046c3d15c2da35e5963f77eaa591627883a11f08fb9"
FUNCTIONS = ("rearrange", "reduce", "repeat", "einsum", "pack", "unpack")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_compiler_runtime(model_config, optimizations, *, upstream=False):
    """Validate the opt-in without importing Torch or changing historic options."""
    name = model_config.get("compiler_runtime")
    boundary = model_config.get("compiler_feature_boundary")
    if boundary is not None and (boundary != "output_blocks.9" or name != RUNTIME_NAME
            or optimizations.max_batch_size != 1
            or optimizations.feature_cache_module not in {"output_blocks.9", "model.diffusion_model.output_blocks.9"}):
        raise ValueError("Compiler feature boundary requires registered runtime, output_blocks.9 and a single-candidate batch")
    if name is None:
        return None
    if name != RUNTIME_NAME:
        raise ValueError("Unknown model.compiler_runtime; select registered_einops_attention_v1 explicitly")
    if upstream or model_config.get("backend", "vista") != "vista":
        raise ValueError("Compiler runtime is available only for the refactored Vista backend")
    if (not optimizations.compile_denoiser or optimizations.compile_decoder
            or optimizations.compile_mode != "default"
            or optimizations.fullgraph or (optimizations.feature_cache_interval > 1 and boundary is None)):
        raise ValueError("Registered compiler runtime requires default/partial denoiser compilation; feature reuse needs the explicit output9 boundary")
    return name


def _load_runtime(torch):
    import torch._dynamo as dynamo
    import torch._dynamo.utils as utils
    import torch._inductor.metrics as metrics
    import einops
    import einops.einops
    import einops.packing
    import einops._torch_specific
    import xformers
    import xformers.ops as ops
    if (torch.__version__ != "2.0.1+cu118" or einops.__version__ != "0.6.1"
            or xformers.__version__ != "0.0.20"):
        raise ValueError("Registered compiler runtime requires Torch2.0.1+cu118/einops0.6.1/xFormers0.0.20")
    registration = einops._torch_specific.allow_ops_in_compiled_graph
    if (registration.__module__ != "einops._torch_specific"
            or registration.__name__ != "allow_ops_in_compiled_graph"):
        raise ValueError("Official einops registration API was replaced")
    root = Path(inspect.getsourcefile(registration) or "").parent
    sources = {name: _sha(root / name) for name in EINOPS_SHA256}
    if sources != EINOPS_SHA256:
        raise ValueError("Installed einops source differs from the pinned official wheel")
    functions = {}
    for name in FUNCTIONS:
        owner = einops.packing if name in {"pack", "unpack"} else einops.einops
        fn = getattr(einops, name)
        if fn is not getattr(owner, name) or fn.__module__ != owner.__name__:
            raise ValueError("Registered einops function was replaced: " + name)
        functions[name] = fn
    native = ops.memory_efficient_attention
    if (native.__module__ != "xformers.ops.fmha" or native.__name__ != "memory_efficient_attention"
            or hasattr(native, "_torchdynamo_orig_callable")
            or _sha(inspect.getsourcefile(native) or "") != NATIVE_SOURCE_SHA256):
        raise ValueError("Native xFormers attention differs from the pinned implementation")
    source_hashes = {"einops/" + k: v for k, v in sources.items()}
    disable_source = Path(inspect.getsourcefile(dynamo.disable))
    source_hashes.update({"xformers/ops/fmha/__init__.py": NATIVE_SOURCE_SHA256,
                         "torch/_dynamo/allowed_functions.py": _sha(dynamo.allowed_functions.__file__),
                         "torch/_dynamo/" + disable_source.name: _sha(disable_source),
                         "drivebench/runtime/compiler.py": _sha(__file__)})
    return SimpleNamespace(dynamo=dynamo, registration=registration, functions=functions,
                           ops=ops, native=native, counters=utils.counters, metrics=metrics,
                           source_hashes=source_hashes)


class RegisteredCompilerRuntime:
    """One stable native-attention wrapper, activated only while its owner runs."""

    def __init__(self, dependencies, *, feature_boundary=None, dynamic_shapes=False):
        self._d = dependencies
        self._active = False
        self._graphs_created = 0
        self._kernels_created = 0
        self.last_execution = None
        self._wrapper = self._d.dynamo.disable(self._d.native)
        if getattr(self._wrapper, "_torchdynamo_orig_callable", None) is not self._d.native:
            raise RuntimeError("Dynamo attention boundary did not retain the native callable")
        self._identity = {
            "schema_version": 1, "name": RUNTIME_NAME,
            "registration_api": "einops._torch_specific.allow_ops_in_compiled_graph",
            "registered_functions": list(FUNCTIONS), "cache_size_limit": 512,
            "suppress_errors": False, "native_attention_boundary": True,
            "native_operator_unchanged": True, "stable_wrapper_per_backend": True,
            "scope": "Owner compile construction and generation under Vista execution lock; restored between calls",
            "torch_version": "2.0.1+cu118", "einops_version": "0.6.1", "xformers_version": "0.0.20",
            "compilation": {"mode": "default", "dynamic_shapes": dynamic_shapes, "fullgraph": False, "target": "denoiser"},
            "source_sha256": dict(self._d.source_hashes),
        }
        if feature_boundary is not None:
            self._identity["feature_boundary"] = feature_boundary
        self._identity["implementation_sha256"] = hashlib.sha256(json.dumps(
            self._identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def _state(self):
        return {
            "unique_graphs": int(self._d.counters.get("stats", {}).get("unique_graphs", 0)),
            "generated_kernels": int(self._d.metrics.generated_kernel_count),
            "unimplemented": {str(k): int(v) for k, v in self._d.counters.get("unimplemented", {}).items() if v},
        }

    @contextlib.contextmanager
    def activate(self, *, require_graphs=False):
        """Fail closed on silent eager frames; always restore process globals.

        `require_graphs` is false only for constructing torch.compile's lazy
        callable. Successful generation must have actual captured graphs and
        generated Inductor kernels. These counters do not prove CUDA graph replay.
        """
        if self._active:
            raise RuntimeError("Compiler runtime scopes must not overlap or nest")
        d = self._d
        cfg = d.dynamo.config
        allowed = d.dynamo.allowed_functions._allowed_function_ids
        disallowed = d.dynamo.allowed_functions._disallowed_function_ids
        before_registry = {name: (id(fn) in allowed, id(fn) in disallowed) for name, fn in d.functions.items()}
        if (cfg.disable or cfg.suppress_errors is not False or cfg.cache_size_limit != 64
                or any(a or b for a, b in before_registry.values()) or d.ops.memory_efficient_attention is not d.native):
            raise RuntimeError("Compiler runtime requires clean pinned defaults and no leaked registration/attention wrapper")
        before = self._state()
        if before["unimplemented"]:
            raise RuntimeError("Prior unimplemented compiler frames make fallback-free execution unverifiable")
        previous = (cfg.cache_size_limit, cfg.suppress_errors, cfg.disable)
        self._active = True
        record = {"require_graphs": require_graphs, "before": before, "restoration_complete": False,
                  "runtime_implementation_sha256": self._identity["implementation_sha256"]}
        self.last_execution = record
        try:
            d.registration()
            if not all(id(fn) in allowed and id(fn) not in disallowed for fn in d.functions.values()):
                raise RuntimeError("Official einops registration did not take effect")
            cfg.cache_size_limit = 512
            d.ops.memory_efficient_attention = self._wrapper
            yield
            after = self._state()
            record["after"] = after
            self._graphs_created += max(0, after["unique_graphs"] - before["unique_graphs"])
            self._kernels_created += max(0, after["generated_kernels"] - before["generated_kernels"])
            record.update(graphs_created_by_runtime=self._graphs_created, kernels_created_by_runtime=self._kernels_created)
            if cfg.suppress_errors is not False or cfg.disable or cfg.cache_size_limit != 512:
                raise RuntimeError("Compiler settings changed while the runtime was active")
            if d.ops.memory_efficient_attention is not self._wrapper:
                raise RuntimeError("Attention wrapper changed while the runtime was active")
            if after["unimplemented"]:
                raise RuntimeError("Unimplemented compiler frames caused an undeclared eager fallback")
            if require_graphs and (self._graphs_created < 1 or self._kernels_created < 1):
                raise RuntimeError("No positive compiled graph and Inductor kernel evidence")
            record["passed"] = True
        except BaseException as error:
            record.update(passed=False, error_type=type(error).__name__)
            raise
        finally:
            counter_error = None
            try:
                if "after" not in record:
                    after = self._state()
                    record["after"] = after
                    self._graphs_created += max(0, after["unique_graphs"] - before["unique_graphs"])
                    self._kernels_created += max(0, after["generated_kernels"] - before["generated_kernels"])
                    record.update(graphs_created_by_runtime=self._graphs_created, kernels_created_by_runtime=self._kernels_created)
            except BaseException as error:
                counter_error = error
                record.update(passed=False, counter_evidence_error=type(error).__name__)
            d.ops.memory_efficient_attention = d.native
            cfg.cache_size_limit, cfg.suppress_errors, cfg.disable = previous
            for name, fn in d.functions.items():
                for registry, present in ((allowed, before_registry[name][0]), (disallowed, before_registry[name][1])):
                    if present:
                        registry.add(id(fn))
                    elif id(fn) in registry:
                        registry.remove(id(fn))
            record["restoration_complete"] = (
                d.ops.memory_efficient_attention is d.native and (cfg.cache_size_limit, cfg.suppress_errors, cfg.disable) == previous
                and all((id(fn) in allowed, id(fn) in disallowed) == before_registry[name] for name, fn in d.functions.items()))
            self._active = False
            if not record["restoration_complete"]:
                raise RuntimeError("Compiler runtime failed to restore process globals")
            if counter_error is not None:
                raise RuntimeError("Compiler counter evidence became unavailable") from counter_error


def create_compiler_runtime(model_config, optimizations, torch, *, upstream=False):
    name = validate_compiler_runtime(model_config, optimizations, upstream=upstream)
    return RegisteredCompilerRuntime(_load_runtime(torch), feature_boundary=model_config.get("compiler_feature_boundary"),
                                     dynamic_shapes=optimizations.dynamic_shapes) if name else None
