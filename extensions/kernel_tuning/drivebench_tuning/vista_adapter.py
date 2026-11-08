"""Scoped, opt-in kernel overrides for the pinned Vista eager model.

The source contract is Vista cc9821b4253ca7987c32757613d2fc2448fa9f5d:
``vwm/modules/attention.py::GEGLU`` and
``vwm/modules/diffusionmodules/util.py::GroupNorm32``. GEGLU retains its
projection; GroupNorm32 computes in FP32, casts to the input dtype, and then
applies SiLU. This adapter does not modify the frozen DriveBench backend.

Use on an exclusively owned, eager model, inside ``torch.no_grad()`` or
``torch.inference_mode()``. Do not compile, train, or concurrently use the model
while overrides are active. Successful installation is not GPU validation or
evidence of a full-model speedup. Production kernels reject unsupported inputs;
there is no implicit eager fallback.
"""

from contextlib import contextmanager
import hashlib
import inspect
import math
from pathlib import Path
import threading
from types import MethodType
from weakref import WeakSet

from .contract import KernelConfig


VISTA_COMMIT = "cc9821b4253ca7987c32757613d2fc2448fa9f5d"
_GEGLU_CLASS = ("vwm.modules.attention", "GEGLU")
_GROUPNORM_CLASS = ("vwm.modules.diffusionmodules.util", "GroupNorm32")
_SOURCE_HASHES = {
    _GEGLU_CLASS: "f03333445a6b8471ad8a693c4a7a98a9fe44c998bc3268e325fcc7f6bbf28b6b",
    _GROUPNORM_CLASS: "3aaf638d3c03fb1cd95545be6eb11b9a16ba7f3cb43952eb2b3345826fd7ed6f",
}
_ACTIVE_MODULES = WeakSet()
_ACTIVE_LOCK = threading.RLock()
_MISSING = object()


def _class_key(module):
    cls = type(module)
    return cls.__module__, cls.__qualname__


def _assert_inference(model, torch):
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if model.training or any(module.training for module in model.modules()):
        raise ValueError("Vista kernel overrides require every module in eval mode")
    if torch.is_grad_enabled():
        raise ValueError("Vista kernel overrides require no_grad or inference_mode")
    if any(type(module).__module__.startswith("torch._dynamo")
           or getattr(module, "_compiled_call_impl", None) is not None
           for module in model.modules()):
        raise ValueError("Use an eager model before compiling or installing kernel overrides")


def _verify_source(module, verified, *, test_only=False):
    key = _class_key(module)
    if type(module) in verified or test_only:
        return
    try:
        path = inspect.getsourcefile(type(module))
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    except (TypeError, OSError) as exc:
        raise ValueError("Cannot verify the pinned Vista module source") from exc
    if actual != _SOURCE_HASHES[key]:
        raise ValueError(f"{key[0]}.{key[1]} does not match pinned Vista {VISTA_COMMIT}")
    verified.add(type(module))


def _discover(model, torch, *, geglu, groupnorm, test_only=False):
    """Resolve the complete patch plan before mutating any module."""
    geglus, sequences, verified = [], [], set()
    for path, module in model.named_modules():
        if geglu and _class_key(module) == _GEGLU_CLASS:
            _verify_source(module, verified, test_only=test_only)
            projection = getattr(module, "proj", None)
            if (type(projection) is not torch.nn.Linear
                    or projection.out_features % 2 != 0):
                raise ValueError(f"Unsupported Vista GEGLU projection at {path!r}")
            if "forward" in module.__dict__:
                raise ValueError(f"GEGLU already has an instance forward override at {path!r}")
            geglus.append((path, module))
        # TimestepEmbedSequential and other subclasses can have additional
        # forward arguments/semantics. Only plain Sequential is supported.
        if not groupnorm or type(module) is not torch.nn.Sequential:
            continue
        children = list(module._modules.items())
        pairs = {}
        for index in range(len(children) - 1):
            norm_name, norm = children[index]
            _, activation = children[index + 1]
            if (_class_key(norm) != _GROUPNORM_CLASS
                    or type(activation) is not torch.nn.SiLU):
                continue
            _verify_source(norm, verified, test_only=test_only)
            if (not isinstance(norm, torch.nn.GroupNorm) or not norm.affine
                    or norm.weight is None or norm.bias is None
                    or not math.isfinite(norm.eps) or norm.eps <= 0
                    or activation.inplace):
                raise ValueError(f"Unsupported GroupNorm32/SiLU pair at {path!r}")
            if "forward" in norm.__dict__ or "forward" in activation.__dict__:
                raise ValueError(f"Normalization pair already has a forward override at {path!r}")
            # Fusing bypasses the two child calls. Hooks on these children
            # could change values or collect misleading observations.
            if any(getattr(child, name) for child in (norm, activation)
                   for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks")):
                raise ValueError(f"Remove hooks from the normalization pair at {path!r} before fusion")
            pairs[index] = (f"{path}.{norm_name}".lstrip("."), norm, activation)
        if pairs:
            if "forward" in module.__dict__:
                raise ValueError(f"Sequential already has an instance forward override at {path!r}")
            sequences.append((path, module, children, pairs))
    if geglu and not geglus:
        raise ValueError("No supported pinned Vista GEGLU modules were found")
    if groupnorm and not sequences:
        raise ValueError("No supported adjacent Vista GroupNorm32/SiLU pairs were found")
    return geglus, sequences


@contextmanager
def vista_kernel_overrides(model, *, geglu_config=None, groupnorm_config=None,
                           _test_kernels=None):
    """Temporarily run recognized Vista operators with the custom kernels.

    ``geglu_config`` / ``groupnorm_config`` are ``KernelConfig`` objects; None
    leaves that operator family untouched. The yielded dictionary describes
    selected module paths and source binding, not measured performance.

    All original module/parameter identities and state-dict keys are retained.
    Instance forwards are restored on normal exit and exceptions. Unsupported
    input shapes/dtypes fail at execution rather than silently falling back.
    ``_test_kernels`` is an explicit CPU contract-test seam that bypasses source
    hashing; its metadata is always labeled ``cpu_contract_test_only``.
    """
    import torch

    _assert_inference(model, torch)
    for config in (geglu_config, groupnorm_config):
        if config is not None and not isinstance(config, KernelConfig):
            raise TypeError("kernel configurations must be KernelConfig objects")
    if geglu_config is None and groupnorm_config is None:
        raise ValueError("Select at least one kernel configuration")
    geglus, sequences = _discover(
        model, torch, geglu=geglu_config is not None,
        groupnorm=groupnorm_config is not None, test_only=_test_kernels is not None,
    )
    if _test_kernels is None:
        from .kernels import fused_geglu, fused_groupnorm_silu
        implementations = {"geglu": fused_geglu, "groupnorm_silu": fused_groupnorm_silu}
    else:
        implementations = _test_kernels
    selected = [module for _, module in geglus] + [module for _, module, _, _ in sequences]
    for operator, config in (("geglu", geglu_config), ("groupnorm_silu", groupnorm_config)):
        if config is not None and not callable(implementations.get(operator)):
            raise TypeError(f"Missing callable for {operator}")
    with _ACTIVE_LOCK:
        if any(module in _ACTIVE_MODULES for module in selected):
            raise RuntimeError("Kernel overrides are already active on a selected module")
        _ACTIVE_MODULES.update(selected)
    originals = []
    try:
        def install(module, forward):
            originals.append((module, module.__dict__.get("forward", _MISSING)))
            module.forward = MethodType(forward, module)

        def geglu_forward(module, x):
            if torch.is_grad_enabled() or model.training or module.training:
                raise ValueError("Kernel overrides only support eager evaluation without gradients")
            projected = module.proj(x)
            return implementations["geglu"](
                projected, block_size=geglu_config.block_size, num_warps=geglu_config.num_warps)

        for _, module in geglus:
            install(module, geglu_forward)
        for _, sequence, children, pairs in sequences:
            # Bind each sequence's original children, including any shared
            # children, without changing their own forwards or registrations.
            def sequence_forward(module, x, *, _children=children, _pairs=pairs):
                if torch.is_grad_enabled() or model.training or module.training:
                    raise ValueError("Kernel overrides only support eager evaluation without gradients")
                index = 0
                while index < len(_children):
                    if index in _pairs:
                        _, norm, activation = _pairs[index]
                        if norm.training or activation.training:
                            raise ValueError("Normalization pair must remain in eval mode")
                        x = implementations["groupnorm_silu"](
                            x, norm.weight, norm.bias, num_groups=norm.num_groups, eps=norm.eps,
                            block_size=groupnorm_config.block_size, num_warps=groupnorm_config.num_warps)
                        index += 2
                    else:
                        x = _children[index][1](x)
                        index += 1
                return x
            install(sequence, sequence_forward)
        yield {
            "source_commit": VISTA_COMMIT,
            "execution": "cpu_contract_test_only" if _test_kernels is not None else "custom_triton",
            "gpu_validation_performed": False,
            "geglu_paths": [path for path, _ in geglus],
            "groupnorm_silu_paths": [pair[0] for _, _, _, pairs in sequences for pair in pairs.values()],
        }
    finally:
        for module, original in reversed(originals):
            if original is _MISSING:
                del module.forward
            else:
                module.forward = original
        with _ACTIVE_LOCK:
            _ACTIVE_MODULES.difference_update(selected)


@contextmanager
def capture_vista_workloads(model, *, _allow_test_modules=False):
    """Collect actual eager operator input shapes, without retaining tensors.

    The caller executes its own representative model forward inside the scope.
    Records are deduplicated by module path, shape, dtype, and device, and count
    observed calls. This helper does not generate a clip or load a checkpoint.
    Run it separately from kernel overrides; existing hooks are not modified.
    """
    import torch

    _assert_inference(model, torch)
    # Discover each family only if its recognized class is present. A model
    # containing neither is an error rather than an empty successful capture.
    geglu = any(_class_key(m) == _GEGLU_CLASS for m in model.modules())
    groupnorm = any(_class_key(m) == _GROUPNORM_CLASS for m in model.modules())
    if not geglu and not groupnorm:
        raise ValueError("No recognized Vista operators were found")
    geglus, sequences = _discover(model, torch, geglu=geglu, groupnorm=groupnorm,
                                  test_only=_allow_test_modules)
    # A module hook cannot distinguish which parent called a shared child.
    # Reject ambiguous capture paths instead of attributing calls incorrectly.
    all_paths = {}
    for path, module in model.named_modules(remove_duplicate=False):
        all_paths.setdefault(id(module), []).append(path)
    observed = [module.proj for _, module in geglus]
    observed += [pair[1] for _, _, _, pairs in sequences for pair in pairs.values()]
    if any(len(all_paths[id(module)]) != 1 for module in observed):
        raise ValueError("Capture an unshared subtree to give each observed operator one module path")
    records, index, handles = [], {}, []

    def record(path, operator, tensor, norm=None):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected a tensor at the Vista operator boundary")
        key = (path, operator, tuple(tensor.shape), tuple(tensor.stride()),
               str(tensor.dtype), str(tensor.device))
        if key not in index:
            item = {"module_path": path, "operator": operator, "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."), "device": str(tensor.device),
                    "contiguous": tensor.is_contiguous(), "strides": list(tensor.stride()),
                    "calls": 0, "source_commit": VISTA_COMMIT,
                    "observation": "cpu_contract_test_only" if _allow_test_modules else "eager_forward_hook"}
            if norm is not None:
                item.update(num_groups=norm.num_groups, eps=norm.eps)
            records.append(item)
            index[key] = item
        index[key]["calls"] += 1

    try:
        for path, module in geglus:
            def geglu_hook(_module, _args, output, *, _path=path):
                record(_path, "geglu", output)
            handles.append(module.proj.register_forward_hook(geglu_hook))
        for _, _, _, pairs in sequences:
            for path, norm, _ in pairs.values():
                def norm_hook(module, args, *, _path=path):
                    record(_path, "groupnorm_silu", args[0], module)
                handles.append(norm.register_forward_pre_hook(norm_hook))
        yield records
    finally:
        for handle in handles:
            handle.remove()
