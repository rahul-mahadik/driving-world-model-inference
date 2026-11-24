"""CPU compiler regressions, never GPU performance evidence or a fallback."""
from contextlib import nullcontext
from dataclasses import replace
import json
import time

import pytest
import torch

from drivebench_tuning.benchmark import CudaEvaluator, EvidenceJournal
from drivebench_tuning.contract import Workload, DTYPE_TOLERANCES


@pytest.fixture
def cpu_preparation(tmp_path, monkeypatch):
    # Exercise production preparation/cleanup using CPU tensors only. The public
    # constructor still refuses CPU execution; no synthetic measurements escape.
    torch.compiler.reset()
    evaluator = object.__new__(CudaEvaluator)
    evaluator.torch = torch
    evaluator.device = torch.device('cpu')
    evaluator.journal = EvidenceJournal(tmp_path/'cpu-regression')
    evaluator.deadline = time.monotonic() + 120
    evaluator._inputs = evaluator._expected = evaluator._compiled = evaluator._current = None
    calls = []
    real_reset = torch.compiler.reset
    def reset():
        calls.append('reset')
        real_reset()
    monkeypatch.setattr(torch.compiler, 'reset', reset)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append('synchronize'))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append('empty_cache'))
    try:
        yield evaluator, calls
    finally:
        real_reset()


def workloads(operator):
    # Twelve distinct shape/dtype pairs share the same Python reference code
    # object, exceeding the default eight-specialization limit without isolation.
    return [Workload(operator, (n, 32) if operator == 'geglu' else (n, 8, 3, 3),
                     dtype=dtype, num_groups=4)
            for n in (2, 3, 4, 5) for dtype in ('float16', 'bfloat16', 'float32')]


@pytest.mark.parametrize('operator', ['geglu', 'groupnorm_silu'])
def test_more_than_eight_workloads_compile_with_real_dynamo(cpu_preparation, operator):
    evaluator, calls = cpu_preparation
    graphs = []
    def cpu_backend(graph, examples):
        graphs.append(graph)
        return graph.forward
    from torch._dynamo import config
    original_limit = config.recompile_limit
    assert original_limit == 8  # Pinned torch 2.10 default; never raise the limit.
    for index, workload in enumerate(workloads(operator)):
        evaluator._prepare(workload)
        assert calls[-3:] == ['synchronize', 'reset', 'empty_cache']
        compiled = torch.compile(evaluator._reference(), backend=cpu_backend,
                                 fullgraph=True, dynamic=False)
        evaluator._compiled = compiled
        with torch.no_grad():
            actual = compiled(*evaluator._inputs)
        tolerance = DTYPE_TOLERANCES[workload.dtype]
        assert torch.allclose(actual, evaluator._expected, **tolerance)
        reset_count = calls.count('reset')
        previous_inputs = evaluator._inputs
        evaluator._prepare(workload)
        assert evaluator._compiled is compiled
        assert evaluator._inputs is previous_inputs
        assert calls.count('reset') == reset_count == index + 1
        with torch.no_grad():
            assert torch.equal(compiled(*evaluator._inputs), actual)
        assert len(graphs) == index + 1
    evaluator.close()
    assert calls[-3:] == ['synchronize', 'reset', 'empty_cache']
    assert calls.count('reset') == 13
    assert evaluator._inputs is evaluator._expected is evaluator._compiled is evaluator._current is None
    assert config.recompile_limit == original_limit
    assert not (evaluator.journal.directory/'completed.json').exists()


def test_control_without_reset_reproduces_the_ninth_specialization_failure(cpu_preparation, monkeypatch):
    evaluator, calls = cpu_preparation
    # This fault injection would fail if the regression ceased to exercise the
    # shared code-object cache that ended the archived GPU attempt.
    monkeypatch.setattr(torch.compiler, 'reset', lambda: None)
    from torch._dynamo.exc import FailOnRecompileLimitHit
    for workload in workloads('geglu')[:8]:
        evaluator._prepare(workload)
        evaluator._compiled = torch.compile(evaluator._reference(), backend='eager',
                                            fullgraph=True, dynamic=False)
        with torch.no_grad(): evaluator._compiled(*evaluator._inputs)
    evaluator._prepare(workloads('geglu')[8])
    evaluator._compiled = torch.compile(evaluator._reference(), backend='eager',
                                        fullgraph=True, dynamic=False)
    with pytest.raises(FailOnRecompileLimitHit):
        with torch.no_grad(): evaluator._compiled(*evaluator._inputs)


def test_seed_transition_also_resets_but_identical_workload_does_not(cpu_preparation):
    evaluator, calls = cpu_preparation
    workload = workloads('geglu')[0]
    evaluator._prepare(workload)
    first = evaluator._input_hashes[:]
    evaluator._prepare(workload)
    assert calls.count('reset') == 1
    evaluator._prepare(replace(workload, seed=1))
    assert calls.count('reset') == 2
    assert evaluator._input_hashes != first


def test_reset_remains_outside_samples_and_same_workload_reuses_compile(cpu_preparation, monkeypatch):
    evaluator, calls = cpu_preparation
    evaluator.cohort_id = 'cpu-unit-test-not-a-gpu-cohort'
    evaluator.repetitions = 3
    evaluator.warmups = 1
    evaluator._out_of_resources = type('UnitTestOutOfResources', (Exception,), {})
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 1024)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: 1024)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda device: calls.append('sample_start'))
    class CpuTestEvent:
        def __init__(self, **kwargs): pass
        def record(self): calls.append('event_record')
        def synchronize(self): pass
        def elapsed_time(self, end): return 1.0
    monkeypatch.setattr(torch.cuda, 'Event', CpuTestEvent)
    compile_calls = []
    def compile_stub(function, **kwargs):
        compile_calls.append(kwargs)
        calls.append('compile_stub')
        return function
    monkeypatch.setattr(torch, 'compile', compile_stub)
    workload = workloads('geglu')[0]
    result = evaluator.compiled_baseline(workload)
    assert result.status == 'ok'
    assert calls.index('reset') < calls.index('compile_stub') < calls.index('event_record') < calls.index('sample_start')
    assert 'reset' not in calls[calls.index('event_record'):]
    assert result.evidence['preparation_seconds'] >= 0
    evaluator.compiled_baseline(workload)
    assert len(compile_calls) == 1
    assert calls.count('reset') == 1
    events = [json.loads(line) for line in evaluator.journal.path.read_text().splitlines()]
    assert len([x for x in events if x['event'] == 'sample']) == 6
    assert len([x for x in events if x['event'] == 'compiler_state_reset']) == 1


def test_compile_threads_uses_effective_config_not_an_environment_claim(monkeypatch):
    from torch._inductor import config
    from drivebench_tuning.benchmark import checked_compiler_thread_count
    monkeypatch.setenv('TORCHINDUCTOR_COMPILE_THREADS', '1')
    with config.patch(compile_threads=4):
        with pytest.raises(RuntimeError, match='requires Inductor compile_threads=1'):
            checked_compiler_thread_count()
    monkeypatch.delenv('TORCHINDUCTOR_COMPILE_THREADS')
    with config.patch(compile_threads=1):
        assert checked_compiler_thread_count() == 1
    with config.patch(compile_threads=True):
        with pytest.raises(RuntimeError, match='requires Inductor compile_threads=1'):
            checked_compiler_thread_count()
