"""Real CUDA operator measurements, separate from the sealed Vista timing cohort.

No synthetic/CPU fallback. Timing is one synchronized invocation (CUDA events and
host wall time); allocation deltas include the returned output and temporary
buffers, exclude already resident inputs, and are not total model GPU memory.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import time
import uuid

from .contract import KernelConfig, Measurement, Workload


class DeadlineExceeded(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    root = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}


def percentile(values, fraction):
    if not values or not 0 <= fraction <= 1:
        raise ValueError('Nonempty samples and a fraction in [0,1] required')
    xs = sorted(values)
    index = (len(xs) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return xs[lo] + (xs[hi] - xs[lo]) * (index - lo)


class EvidenceJournal:
    """Exclusive new run directory; append every started, failed and finished trial."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / 'trials.jsonl'
        self.path.touch(exist_ok=False)
        self.sequence = 0

    def write(self, name, value):
        if Path(name).name != name:
            raise ValueError('Journal artifact must be a basename')
        with (self.directory / name).open('x') as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n')

    def append(self, value):
        import os
        self.sequence += 1
        row = {'sequence': self.sequence, 'at': utc_now(), **value}
        with self.path.open('a') as f:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        return self.sequence


class CudaEvaluator:
    """Serial, single-device evaluator. Caller must provide an outer job timeout.

    No warmup, compile, correctness, export, or tensor creation time is charged
    to latency. Those costs are recorded separately and consume wall-time budget.
    Every evaluate() performs a fresh sample set; only inputs and compilation
    are reused. Does not allocate cloud resources, add funding, or launch guards.
    """
    def __init__(self, journal: EvidenceJournal, *, repetitions=30, warmups=5,
                 max_seconds=900, device=0):
        if type(repetitions) is not int or not 3 <= repetitions <= 1000:
            raise ValueError('repetitions must be 3..1000')
        if type(warmups) is not int or not 1 <= warmups <= 100:
            raise ValueError('warmups must be 1..100')
        if not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('max_seconds must be finite and positive')
        import torch
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError('An NVIDIA CUDA GPU is required; CPU/synthetic fallback is disabled')
        import triton
        from triton.runtime.errors import OutOfResources
        self._out_of_resources = OutOfResources
        if torch.__version__.split('+')[0] != '2.10.0' or triton.__version__ != '3.6.0':
            raise RuntimeError('This prospective cohort requires torch 2.10.0 / triton 3.6.0')
        if type(device) is not int or not 0 <= device < torch.cuda.device_count():
            raise ValueError('Invalid CUDA device')
        self.torch, self.device, self.journal = torch, torch.device(f'cuda:{device}'), journal
        self.repetitions, self.warmups = repetitions, warmups
        self.deadline = time.monotonic() + max_seconds
        self._current = None
        self._compiled = None
        self._inputs = None
        self._expected = None
        props = torch.cuda.get_device_properties(self.device)
        gpu_uuid = getattr(props, 'uuid', None)
        if not gpu_uuid:
            raise RuntimeError('Physical GPU UUID is required for evidence binding')
        import subprocess
        driver = subprocess.run(['nvidia-smi', '--query-gpu=uuid,driver_version',
                                 '--format=csv,noheader'], capture_output=True, text=True,
                                check=True, timeout=10).stdout.strip().splitlines()
        self.runtime = {
            'gpu_uuid': str(gpu_uuid), 'gpu_name': props.name,
            'compute_capability': [props.major, props.minor], 'total_memory': props.total_memory,
            'driver_inventory': driver, 'torch': torch.__version__, 'triton': triton.__version__,
            'cuda': torch.version.cuda, 'python': platform.python_version(),
            'packages': {p: importlib.metadata.version(p) for p in ('torch', 'triton', 'numpy', 'gymnasium')},
            'source_hashes': source_hashes(), 'device_index': device,
        }
        self.cohort_id = hashlib.sha256(json.dumps(self.runtime, sort_keys=True).encode()).hexdigest()
        journal.write('runtime.json', {**self.runtime, 'cohort_id': self.cohort_id})
        journal.append({'event': 'evaluator_ready', 'cohort_id': self.cohort_id,
                        'repetitions': repetitions, 'warmups': warmups, 'max_seconds': max_seconds})

    def _check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise DeadlineExceeded('Operator study wall-time budget exhausted')

    def _prepare(self, workload):
        self._check_deadline()
        if self._current == workload:
            return
        torch = self.torch
        self._inputs, self._expected, self._compiled, self._current = None, None, None, None
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        generator = torch.Generator(device='cpu').manual_seed(workload.seed)
        cpu_x = torch.randn(workload.shape, generator=generator, dtype=torch.float32).to(getattr(torch, workload.dtype))
        cpu_inputs = [cpu_x]
        if workload.operator == 'groupnorm_silu':
            cpu_inputs += [torch.randn(workload.shape[1], generator=generator),
                           torch.randn(workload.shape[1], generator=generator)]
        self._input_hashes = [hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                              for x in cpu_inputs]
        self._inputs = [x.to(self.device) for x in cpu_inputs]
        with torch.no_grad():
            self._expected = self._reference(workload)(*self._inputs).cpu()
        if not torch.isfinite(self._expected).all().item():
            raise ValueError('Eager reference produced nonfinite values')
        self._current = workload
        self.journal.append({'event': 'workload_prepared', 'workload': asdict(workload),
                             'workload_key': workload.key(), 'input_sha256': self._input_hashes})

    def _reference(self, workload=None):
        workload = workload or self._current
        from .kernels import geglu_reference, groupnorm_silu_reference
        if workload.operator == 'geglu':
            return geglu_reference
        def reference(x, weight, bias):
            return groupnorm_silu_reference(x, weight, bias, num_groups=workload.num_groups, eps=workload.eps)
        return reference

    def _verify_inputs(self):
        torch = self.torch
        actual = [hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                  for x in self._inputs]
        if actual != self._input_hashes:
            raise RuntimeError('Operator mutated its inputs; cohort aborted')

    def baseline(self, workload):
        return self.measure(workload, None, implementation='eager')

    def evaluate(self, workload, config):
        return self.measure(workload, config, implementation='triton')

    def compiled_baseline(self, workload):
        return self.measure(workload, None, implementation='compiled')

    def measure(self, workload: Workload, config: KernelConfig | None, *, implementation):
        if implementation not in ('eager', 'compiled', 'triton'):
            raise ValueError('Unknown implementation')
        if (implementation == 'triton') != isinstance(config, KernelConfig):
            raise ValueError('Only a Triton measurement takes a KernelConfig')
        torch = self.torch
        trial_id = str(uuid.uuid4())
        evidence = {'trial_id': trial_id, 'cohort_id': self.cohort_id,
                    'implementation': implementation, 'boundary': 'isolated_operator',
                    'latency_boundary': 'CUDA events around one invocation; includes stream gaps',
                    'memory_boundary': 'peak live allocator increase above resident inputs; includes output',
                    'warmups': self.warmups, 'repetitions': self.repetitions}
        self.journal.append({'event': 'started', 'trial_id': trial_id, 'workload': asdict(workload),
                             'config': asdict(config) if config else None, 'implementation': implementation})
        try:
            with torch.cuda.device(self.device), torch.no_grad():
                preparation_start = time.perf_counter()
                self._prepare(workload)
                evidence['input_sha256'] = self._input_hashes
                if implementation == 'eager':
                    fn = self._reference()
                elif implementation == 'compiled':
                    if self._compiled is None:
                        self._compiled = torch.compile(self._reference(), backend='inductor', fullgraph=True,
                                                       dynamic=False, mode='default')
                    fn = self._compiled
                else:
                    from .kernels import fused_geglu, fused_groupnorm_silu
                    if workload.operator == 'geglu':
                        fn = lambda x: fused_geglu(x, block_size=config.block_size, num_warps=config.num_warps)
                    else:
                        fn = lambda x, w, b: fused_groupnorm_silu(x, w, b, num_groups=workload.num_groups,
                                                                eps=workload.eps, block_size=config.block_size,
                                                                num_warps=config.num_warps)
                actual = fn(*self._inputs).cpu()
                self._verify_inputs()
                reference = self._expected
                rtol, atol = {'float16': (2e-3, 2e-3), 'bfloat16': (2e-2, 2e-2),
                              'float32': (2e-5, 3e-6)}[workload.dtype]
                finite = bool(torch.isfinite(actual).all().item())
                shape_ok = actual.shape == reference.shape and actual.dtype == reference.dtype
                correct = finite and shape_ok and bool(torch.allclose(actual, reference, rtol=rtol, atol=atol))
                error = (actual.float() - reference.float()).abs() if finite and shape_ok else None
                evidence['correctness'] = {'passed': correct, 'rtol': rtol, 'atol': atol,
                    'max_abs_error': float(error.max()) if error is not None else None,
                    'rmse': float((error.square().mean()).sqrt()) if error is not None else None,
                    'shape_dtype_match': shape_ok, 'finite': finite}
                del actual, error
                if not correct:
                    result = Measurement(workload.key(), config, 'incorrect', None, None, False, 'cuda', evidence)
                    self.journal.append({'event': 'finished', 'measurement': result.to_dict()})
                    return result
                for _ in range(self.warmups):
                    self._check_deadline()
                    output = fn(*self._inputs)
                    del output
                torch.cuda.synchronize(self.device)
                evidence['preparation_seconds'] = time.perf_counter() - preparation_start
                samples, host_samples, memory, reserved = [], [], [], []
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                # Prime lazily created events outside measurement.
                start.record(); end.record(); end.synchronize()
                for _ in range(self.repetitions):
                    self._check_deadline()
                    torch.cuda.synchronize(self.device)
                    before = torch.cuda.memory_allocated(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)
                    host_start = time.perf_counter()
                    start.record()
                    output = fn(*self._inputs)
                    end.record()
                    end.synchronize()
                    host_samples.append((time.perf_counter() - host_start) * 1000)
                    samples.append(float(start.elapsed_time(end)))
                    memory.append(int(torch.cuda.max_memory_allocated(self.device) - before))
                    reserved.append(int(torch.cuda.max_memory_reserved(self.device)))
                    output_bytes = output.numel() * output.element_size()
                    del output
                    self.journal.append({'event': 'sample', 'trial_id': trial_id,
                        'sample_index': len(samples)-1, 'cuda_event_ms': samples[-1],
                        'host_wall_ms': host_samples[-1], 'peak_allocated_delta_bytes': memory[-1],
                        'resident_allocated_bytes': before, 'peak_reserved_total_bytes': reserved[-1]})
                self._verify_inputs()
                evidence["input_integrity_verified"] = True
                if any(not math.isfinite(v) or v <= 0 for v in samples):
                    raise RuntimeError('Invalid CUDA event timing')
                median = statistics.median(samples)
                evidence.update({'cuda_event_ms': samples, 'host_wall_ms': host_samples,
                    'p95_latency_ms': percentile(samples, .95), 'invocations_per_second': 1000 / median,
                    'peak_allocated_delta_bytes': memory, 'peak_reserved_total_bytes': reserved,
                    'output_bytes': output_bytes, 'temporary_overhead_estimate_bytes': max(0, max(memory)-output_bytes),
                    'note': 'Allocator deltas and operator timings do not establish whole-model speed or memory reduction.'})
                result = Measurement(workload.key(), config, 'ok', median, max(memory), True, 'cuda', evidence)
                self.journal.append({'event': 'finished', 'measurement': result.to_dict()})
                return result
        except Exception as exc:
            evidence['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            result = Measurement(workload.key(), config, 'error', None, None, False, 'cuda', evidence)
            self.journal.append({'event': 'failed', 'measurement': result.to_dict()})
            # A bad launch may poison a CUDA context. Abort rather than collecting
            # measurements after an unknown execution error. OOM/unsupported configs
            # can be treated as failed actions after explicitly releasing cached memory.
            if isinstance(exc, (ValueError, torch.cuda.OutOfMemoryError, self._out_of_resources)):
                torch.cuda.empty_cache()
                return result
            raise

    def close(self):
        self._inputs, self._expected, self._compiled, self._current = None, None, None, None
        self.torch.cuda.synchronize(self.device)
        self.torch.cuda.empty_cache()
