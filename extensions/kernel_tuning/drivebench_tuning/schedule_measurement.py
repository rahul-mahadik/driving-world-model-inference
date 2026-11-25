"""Fresh CUDA measurements for the schedule-transformation track.

A persistent, owned subprocess keeps compilation state for one episode. Every
query executes new CUDA-event samples. All setup, first-call compilation,
correctness checks, warmups, errors and process cleanup consume caller wall time.
The parent kills and reaps the process group on timeout. This is an execution
boundary for trusted built-in schedules, not a security sandbox for submitted code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import selectors
import signal
import statistics
import subprocess
import sys
import time
import uuid

from .contract import Workload


@dataclass(frozen=True)
class ScheduleMeasurement:
    workload_key: str
    schedule: dict | None
    status: str
    correct: bool
    latency_ms: float | None
    raw_samples_ms: list[float]
    peak_allocated_bytes: int | None
    elapsed_seconds: float
    evidence: dict = field(default_factory=dict)
    backend: str = 'cuda'

    def to_dict(self):
        return asdict(self)


class CudaScheduleEvaluator:
    """One process/device per episode, with an enforced timeout per query.

    ``evidence_directory`` must be a new path. No CUDA imports, cloud allocation,
    credential handling or external process management happen in the constructor.
    The directory retains requests, results, worker samples and stderr, including
    partial attempts. ``seed`` controls the draw from fresh timing samples only.
    The worker uses the existing pinned Torch 2.10 / Triton 3.6 evaluator runtime.
    """
    def __init__(self, evidence_directory, *, repetitions=7, warmups=2, device=0, seed=0):
        if type(repetitions) is not int or not 3 <= repetitions <= 1000:
            raise ValueError('repetitions must be 3..1000')
        if type(warmups) is not int or not 1 <= warmups <= 100:
            raise ValueError('warmups must be 1..100')
        if type(device) is not int or device < 0:
            raise ValueError('device must be a nonnegative integer')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('seed must be a uint32')
        if os.name != 'posix':
            raise RuntimeError('Bounded CUDA subprocess execution requires POSIX process groups')
        self.directory = Path(evidence_directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.repetitions, self.warmups, self.device = repetitions, warmups, device
        self._random = random.Random(seed)
        self._process = None
        self._selector = None
        self._stderr = None
        self._buffer = b''
        self._closed = False
        self._runtime = None
        self._cohort_id = None
        self._workload_key = None
        self._input_hashes = None
        self._baseline_done = False
        self._record({'event': 'created', 'sampling_seed': seed,
                      'repetitions': repetitions, 'warmups': warmups, 'device': device})

    def _record(self, record):
        with (self.directory / 'queries.jsonl').open('a') as f:
            f.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def _start(self):
        env = dict(os.environ)
        env['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
        self._stderr = (self.directory / 'worker-stderr.log').open('ab', buffering=0)
        self._process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'drivebench_tuning.schedule_measurement', '--worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            env=env, start_new_session=True, bufsize=0,
        )
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._process.stdout, selectors.EVENT_READ)

    def _read_response(self, deadline):
        while b'\n' not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._selector.select(remaining):
                raise TimeoutError('Schedule measurement exceeded its remaining wall-clock budget')
            chunk = os.read(self._process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError('CUDA measurement worker exited without a complete response')
            self._buffer += chunk
            if len(self._buffer) > 2_000_000:
                raise ValueError('CUDA measurement response exceeds the protocol size limit')
        line, self._buffer = self._buffer.split(b'\n', 1)
        response = json.loads(line)
        if not isinstance(response, dict):
            raise ValueError('CUDA measurement worker returned a non-object response')
        return response

    def _stop_worker(self):
        process = self._process
        if process is not None:
            # The group may outlive its leader (e.g. compiler subprocesses).
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        if self._selector is not None:
            self._selector.close()
        if self._stderr is not None:
            self._stderr.close()
        self._process = self._selector = self._stderr = None
        self._buffer = b''

    def baseline(self, workload: Workload, *, timeout_seconds: float) -> ScheduleMeasurement:
        if self._baseline_done:
            raise RuntimeError('Create a new evaluator for each episode; baseline already measured')
        return self._query(workload, None, timeout_seconds=timeout_seconds)

    def evaluate(self, workload: Workload, schedule, *, timeout_seconds: float) -> ScheduleMeasurement:
        if not self._baseline_done:
            raise RuntimeError('A successful baseline is required before candidate measurements')
        from .schedule_kernels import validate_schedule
        validate_schedule(workload, schedule)
        return self._query(workload, asdict(schedule), timeout_seconds=timeout_seconds)

    def _query(self, workload, schedule, *, timeout_seconds):
        if self._closed:
            raise RuntimeError('Evaluator is closed')
        if not isinstance(workload, Workload):
            raise TypeError('workload must be a Workload')
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError('timeout_seconds must be finite and positive')
        if self._workload_key is not None and workload.key() != self._workload_key:
            raise ValueError('An episode may not change workload or inputs')
        started = time.monotonic()
        deadline = started + timeout_seconds
        trial_id = str(uuid.uuid4())
        request = {'trial_id': trial_id, 'workload': asdict(workload), 'schedule': schedule,
                   'timeout_seconds': timeout_seconds, 'repetitions': self.repetitions,
                   'warmups': self.warmups, 'device': self.device,
                   'worker_directory': str(self.directory / 'worker')}
        evidence = {'trial_id': trial_id, 'evidence_directory': str(self.directory),
                    'latency_boundary': 'one CUDA-event invocation drawn from a fresh query sample set',
                    'cost_boundary': 'parent wall time including setup, compilation, checks, warmups and cleanup',
                    'timeout_seconds': timeout_seconds}
        self._record({'event': 'query_started', **request})
        try:
            if self._process is None:
                self._start()
            payload = (json.dumps(request, allow_nan=False) + '\n').encode()
            if len(payload) > 8192:
                raise ValueError('Measurement request exceeds protocol size limit')
            self._process.stdin.write(payload)
            self._process.stdin.flush()
            response = self._read_response(deadline)
            if response.get('trial_id') != trial_id:
                raise ValueError('Worker response belongs to a different query')
            evidence.update(response.get('evidence', {}))
            evidence['trial_id'] = trial_id
            status, correct = response['status'], response['correct']
            if status not in ('ok', 'incorrect', 'error') or type(correct) is not bool:
                raise ValueError('Invalid measurement status')
            if correct != (status == 'ok'):
                raise ValueError('Worker status and correctness disagree')
            samples = response.get('raw_samples_ms', [])
            peak = response.get('peak_allocated_bytes')
            if not isinstance(samples, list) or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError('Invalid fresh timing samples')
            runtime = evidence.get('runtime')
            hashes = evidence.get('input_sha256')
            if status == 'ok':
                if len(samples) != self.repetitions or type(peak) is not int or peak <= 0:
                    raise ValueError('Successful trial needs complete timings and positive allocation')
                if not isinstance(runtime, dict) or not runtime.get('gpu_uuid') or not evidence.get('cohort_id'):
                    raise ValueError('Missing physical GPU/runtime binding')
                if (not isinstance(hashes, list) or not hashes or any(
                        not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h)
                        for h in hashes)):
                    raise ValueError('Missing input hash binding')
                if self._runtime is not None and (runtime != self._runtime or hashes != self._input_hashes
                                                 or evidence['cohort_id'] != self._cohort_id):
                    raise ValueError('Hardware/runtime/inputs changed within the episode')
                self._runtime, self._input_hashes = runtime, hashes
                self._cohort_id = evidence['cohort_id']
                self._workload_key = workload.key()
                index = self._random.randrange(len(samples))
                latency = float(samples[index])
                evidence.update(sampled_index=index, median_ms=statistics.median(samples))
                if schedule is None:
                    self._baseline_done = True
            else:
                latency = None
            if response.get('fatal'):
                self._stop_worker()
                self._closed = True
                evidence['worker_reaped'] = True
            elapsed = time.monotonic() - started
            if elapsed >= timeout_seconds:
                # A response arriving after the deadline cannot earn reward.
                raise TimeoutError('Measurement finished after the wall-clock deadline')
            result = ScheduleMeasurement(workload.key(), schedule, status, correct, latency,
                                         samples, peak, elapsed, evidence)
        except Exception as exc:
            self._stop_worker()
            self._closed = True
            evidence['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            evidence['worker_reaped'] = True
            result = ScheduleMeasurement(workload.key(), schedule,
                'timeout' if isinstance(exc, TimeoutError) else 'error', False, None, [], None,
                time.monotonic() - started, evidence)
        self._record({'event': 'query_finished', 'measurement': result.to_dict()})
        return result

    def close(self):
        if not self._closed or self._process is not None:
            started = time.monotonic()
            self._stop_worker()
            self._closed = True
            self._record({'event': 'closed', 'cleanup_seconds': time.monotonic() - started})


def _measure_candidate(runner, workload, schedule, trial_id):
    """Measure a built-in schedule using the existing reference and input contract."""
    from .contract import DTYPE_TOLERANCES
    from .schedule_kernels import execute_schedule, validate_schedule
    validate_schedule(workload, schedule)
    torch = runner.torch
    evidence = {'trial_id': trial_id, 'runtime': runner.runtime, 'cohort_id': runner.cohort_id,
                'implementation': 'transformed-triton', 'warmups': runner.warmups,
                'repetitions': runner.repetitions}
    started = time.perf_counter()
    runner.journal.append({'event': 'schedule_started', 'trial_id': trial_id,
                           'workload': asdict(workload), 'schedule': asdict(schedule)})
    try:
        with torch.cuda.device(runner.device), torch.no_grad():
            runner._prepare(workload)
            evidence['input_sha256'] = runner._input_hashes
            fn = lambda: execute_schedule(workload, schedule, tuple(runner._inputs))
            first = time.perf_counter()
            actual = fn()
            torch.cuda.synchronize(runner.device)
            evidence['compile_and_first_call_seconds'] = time.perf_counter() - first
            actual = actual.cpu()
            reference = runner._expected
            tolerance = DTYPE_TOLERANCES[workload.dtype]
            finite = bool(torch.isfinite(actual).all().item())
            shape_ok = actual.shape == reference.shape and actual.dtype == reference.dtype
            correct = shape_ok and finite and bool(torch.allclose(actual, reference, **tolerance))
            delta = (actual.float() - reference.float()).abs() if finite and shape_ok else None
            evidence['correctness'] = {'passed': correct, 'finite': finite,
                'shape_dtype_match': shape_ok, **tolerance,
                'max_abs_error': float(delta.max()) if delta is not None else None}
            runner._verify_inputs()
            del actual, delta
            if not correct:
                return {'status': 'incorrect', 'correct': False, 'raw_samples_ms': [],
                        'peak_allocated_bytes': None, 'evidence': evidence}
            for _ in range(runner.warmups):
                runner._check_deadline()
                output = fn()
                del output
            torch.cuda.synchronize(runner.device)
            evidence['preparation_seconds'] = time.perf_counter() - started
            samples, memories = [], []
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); end.record(); end.synchronize()
            for index in range(runner.repetitions):
                runner._check_deadline()
                torch.cuda.synchronize(runner.device)
                resident = torch.cuda.memory_allocated(runner.device)
                torch.cuda.reset_peak_memory_stats(runner.device)
                begin.record()
                output = fn()
                end.record(); end.synchronize()
                sample = float(begin.elapsed_time(end))
                peak = int(torch.cuda.max_memory_allocated(runner.device) - resident)
                del output
                samples.append(sample); memories.append(peak)
                runner.journal.append({'event': 'schedule_sample', 'trial_id': trial_id,
                    'sample_index': index, 'cuda_event_ms': sample,
                    'peak_allocated_delta_bytes': peak})
            runner._verify_inputs()
            evidence.update(input_integrity_verified=True, peak_allocated_delta_bytes=memories)
            return {'status': 'ok', 'correct': True, 'raw_samples_ms': samples,
                    'peak_allocated_bytes': max(memories), 'evidence': evidence}
    except Exception as exc:
        recoverable = isinstance(exc, (ValueError, torch.cuda.OutOfMemoryError, runner._out_of_resources))
        evidence['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        if recoverable:
            torch.cuda.empty_cache()
        return {'status': 'error', 'correct': False, 'raw_samples_ms': [],
                'peak_allocated_bytes': None, 'evidence': evidence, 'fatal': not recoverable}


def _bind_schedule_runtime(runner):
    # Keep the original evaluator/runtime.json untouched as provenance.
    # This track adds measured hardware context and gives the augmented
    # document its own cohort identity, used by every subsequent trial.
    properties = runner.torch.cuda.get_device_properties(runner.device)
    count = properties.multi_processor_count
    if type(count) is not int or count <= 0:
        raise RuntimeError('CUDA runtime must expose a positive multiprocessor count')
    runtime = {**runner.runtime, 'schedule_runtime_schema': 1,
               'parent_runtime_cohort_id': runner.cohort_id, 'multiprocessor_count': count}
    runner.runtime = runtime
    runner.cohort_id = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    runner.journal.write('schedule-runtime.json', {**runtime, 'cohort_id': runner.cohort_id})


def _worker_request(request, runner):
    from .benchmark import CudaEvaluator, EvidenceJournal
    from .schedule import KernelSchedule
    workload = Workload(**request['workload'])
    if runner is None:
        runner = CudaEvaluator(EvidenceJournal(request['worker_directory']),
            repetitions=request['repetitions'], warmups=request['warmups'],
            max_seconds=request['timeout_seconds'], device=request['device'])
        _bind_schedule_runtime(runner)
    # Parent deadline is authoritative and includes worker setup and IPC.
    runner.deadline = time.monotonic() + request['timeout_seconds']
    if request['schedule'] is None:
        eager = runner.baseline(workload)
        if eager.status != 'ok':
            raise RuntimeError('Correct eager baseline is required')
        compiled = runner.compiled_baseline(workload)
        eligible = [row for row in (eager, compiled) if row.status == 'ok' and row.correct]
        selected = min(eligible, key=lambda row: row.latency_ms)
        evidence = {**selected.evidence, 'runtime': runner.runtime,
                    'baseline_policy': 'lower fresh median of correct eager and compiled references',
                    'selected_reference': selected.evidence['implementation'],
                    'baseline_roles': {'eager': eager.to_dict(), 'compiled': compiled.to_dict()},
                    'compiled_fallback': selected is eager}
        result = {'status': 'ok', 'correct': True, 'raw_samples_ms': selected.evidence['cuda_event_ms'],
                  'peak_allocated_bytes': selected.peak_allocated_bytes, 'evidence': evidence}
    else:
        schedule = KernelSchedule(**request['schedule'])
        result = _measure_candidate(runner, workload, schedule, request['trial_id'])
    result['trial_id'] = request['trial_id']
    runner.journal.append({'event': 'schedule_query_finished', 'result': result})
    return result, runner


def _worker_main():
    import contextlib
    runner = None
    protocol_stdout = sys.stdout
    for line in sys.stdin:
        request = json.loads(line)
        try:
            # Compiler diagnostics belong in preserved stderr, not the protocol.
            with contextlib.redirect_stdout(sys.stderr):
                result, runner = _worker_request(request, runner)
        except Exception as exc:
            result = {'trial_id': request['trial_id'], 'status': 'error', 'correct': False,
                      'raw_samples_ms': [], 'peak_allocated_bytes': None, 'fatal': True,
                      'evidence': {'error': {'type': type(exc).__name__, 'message': str(exc)}}}
        protocol_stdout.write(json.dumps(result, allow_nan=False) + '\n')
        protocol_stdout.flush()
        if result.get('fatal'):
            break


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('This module is an evaluator worker; use the registered environment.')
    _worker_main()
