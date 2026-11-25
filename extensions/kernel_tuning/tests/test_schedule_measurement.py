"""CPU protocol/timeout tests; these fixtures are not CUDA performance evidence."""
from dataclasses import asdict, replace
import json
import hashlib
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from drivebench_tuning.contract import Workload
from drivebench_tuning.schedule import KernelSchedule
from drivebench_tuning.schedule_measurement import CudaScheduleEvaluator, ScheduleMeasurement


WORKLOAD = Workload('geglu', (2, 256))


def fake_start(monkeypatch, code):
    def start(self):
        self._stderr = (self.directory / 'worker-stderr.log').open('ab', buffering=0)
        self._process = subprocess.Popen([sys.executable, '-u', '-c', code],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            start_new_session=True, bufsize=0)
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
    monkeypatch.setattr(CudaScheduleEvaluator, '_start', start)


# Full fixture response deliberately contains no real hardware claims in a file
# outside pytest's temporary directory. Tests only verify protocol bookkeeping.
WORKER = r'''
import json,sys,time
count=0
for line in sys.stdin:
    req=json.loads(line); count+=1
    time.sleep(.02)
    result={'trial_id':req['trial_id'],'status':'ok','correct':True,
        'raw_samples_ms':[float(count+i) for i in range(req['repetitions'])],
        'peak_allocated_bytes':256,
        'evidence':{'runtime':{'gpu_uuid':'test-only-fixture','gpu_name':'no GPU measured'},
                    'cohort_id':'test-only-fixture','input_sha256':['0'*64],
                    'fixture_not_hardware':True}}
    print(json.dumps(result),flush=True)
'''


def test_each_query_draws_from_new_samples_and_charges_wall_time(tmp_path, monkeypatch):
    fake_start(monkeypatch, WORKER)
    evaluator = CudaScheduleEvaluator(tmp_path / 'evidence', repetitions=3, seed=9)
    before = time.monotonic()
    baseline = evaluator.baseline(WORKLOAD, timeout_seconds=5)
    first = evaluator.evaluate(WORKLOAD, KernelSchedule(), timeout_seconds=5)
    second = evaluator.evaluate(WORKLOAD, KernelSchedule(), timeout_seconds=5)
    evaluator.close()
    assert all(x.status == 'ok' for x in (baseline, first, second))
    assert first.raw_samples_ms == [2., 3., 4.]
    assert second.raw_samples_ms == [3., 4., 5.]
    for result in (baseline, first, second):
        assert result.latency_ms == result.raw_samples_ms[result.evidence['sampled_index']]
        assert result.elapsed_seconds >= .02
        assert result.evidence['fixture_not_hardware']
    assert sum(x.elapsed_seconds for x in (baseline, first, second)) <= time.monotonic() - before
    assert len({x.evidence['trial_id'] for x in (baseline, first, second)}) == 3
    rows = [json.loads(x) for x in (tmp_path / 'evidence/queries.jsonl').read_text().splitlines()]
    assert sum(x['event'] == 'query_started' for x in rows) == 3
    assert sum(x['event'] == 'query_finished' for x in rows) == 3
    assert rows[-1]['event'] == 'closed'


def test_sample_draw_is_seeded_without_reusing_measurements(tmp_path, monkeypatch):
    fake_start(monkeypatch, WORKER)
    draws = []
    for i in range(2):
        evaluator = CudaScheduleEvaluator(tmp_path / str(i), repetitions=3, seed=87)
        result = evaluator.baseline(WORKLOAD, timeout_seconds=5)
        draws.append(result.evidence['sampled_index'])
        evaluator.close()
    assert draws[0] == draws[1]


def test_timeout_reaps_process_and_refuses_reuse(tmp_path, monkeypatch):
    fake_start(monkeypatch, 'import sys,time\nsys.stdin.readline()\ntime.sleep(60)')
    evaluator = CudaScheduleEvaluator(tmp_path / 'run')
    before = time.monotonic()
    result = evaluator.baseline(WORKLOAD, timeout_seconds=.15)
    assert result.status == 'timeout' and not result.correct
    assert result.latency_ms is None and result.raw_samples_ms == []
    assert .15 <= result.elapsed_seconds < 3
    assert time.monotonic() - before < 3
    assert result.evidence['worker_reaped']
    assert evaluator._process is None
    with pytest.raises(RuntimeError, match='closed'):
        evaluator.baseline(WORKLOAD, timeout_seconds=1)


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='POSIX descendant cleanup')
def test_timeout_kills_compiler_descendants(tmp_path, monkeypatch):
    pid_file = tmp_path / 'descendant.pid'
    code = '\n'.join([
        'import os,sys,time', 'sys.stdin.readline()', 'pid=os.fork()',
        'if pid==0:', '    time.sleep(60)',
        'else:', f'    open({str(pid_file)!r},"w").write(str(pid))',
        '    time.sleep(60)',
    ])
    fake_start(monkeypatch, code)
    evaluator = CudaScheduleEvaluator(tmp_path / 'run')
    result = evaluator.baseline(WORKLOAD, timeout_seconds=.5)
    assert result.status == 'timeout'
    assert pid_file.exists()
    child = int(pid_file.read_text())
    # Orphan reaping is owned by the OS; a zombie cannot run or retain CUDA work.
    state = subprocess.run(['ps', '-o', 'stat=', '-p', str(child)], capture_output=True, text=True).stdout.strip()
    assert not state or state.startswith('Z')


def test_partial_json_line_cannot_evade_timeout(tmp_path, monkeypatch):
    code = 'import sys,time\nsys.stdin.readline()\nsys.stdout.write("{\\\"status\\\":")\nsys.stdout.flush()\ntime.sleep(60)'
    fake_start(monkeypatch, code)
    evaluator = CudaScheduleEvaluator(tmp_path / 'run')
    result = evaluator.baseline(WORKLOAD, timeout_seconds=.15)
    assert result.status == 'timeout'
    assert evaluator._process is None


@pytest.mark.parametrize('mutation', [
    "result['evidence']['runtime']['gpu_uuid'] = str(count)",
    "result['evidence']['cohort_id'] = str(count)",
    "result['evidence']['input_sha256'] = [str(count)*64]",
    "result['trial_id'] = 'wrong'",
    "result['raw_samples_ms'] = [float('nan')]",
    "result['raw_samples_ms'] = [1.]",
    "result['correct'] = False",
])
def test_rejects_identity_or_measurement_corruption(tmp_path, monkeypatch, mutation):
    code = WORKER.replace('    print(json.dumps(result),flush=True)',
        '    if count > 1:\n        ' + mutation + '\n    print(json.dumps(result),flush=True)')
    fake_start(monkeypatch, code)
    evaluator = CudaScheduleEvaluator(tmp_path / 'run', repetitions=3)
    assert evaluator.baseline(WORKLOAD, timeout_seconds=5).correct
    result = evaluator.evaluate(WORKLOAD, KernelSchedule(), timeout_seconds=5)
    assert result.status == 'error' and not result.correct
    assert result.evidence['worker_reaped']
    assert evaluator._process is None


def test_failed_compile_is_charged_and_preserved(tmp_path, monkeypatch):
    code = WORKER.replace('    print(json.dumps(result),flush=True)', '''
    if count>1:
        result.update(status='error',correct=False,raw_samples_ms=[],peak_allocated_bytes=None)
        result['evidence']['error']={'type':'FixtureCompileError','message':'test fixture'}
    print(json.dumps(result),flush=True)''')
    fake_start(monkeypatch, code)
    evaluator = CudaScheduleEvaluator(tmp_path / 'run', repetitions=3)
    assert evaluator.baseline(WORKLOAD, timeout_seconds=5).correct
    result = evaluator.evaluate(WORKLOAD, KernelSchedule(), timeout_seconds=5)
    assert result.status == 'error' and not result.correct
    assert result.elapsed_seconds >= .02
    assert result.evidence['error']['type'] == 'FixtureCompileError'
    assert 'FixtureCompileError' in (evaluator.directory / 'queries.jsonl').read_text()
    evaluator.close()


def test_constructor_and_episode_contract(tmp_path, monkeypatch):
    fake_start(monkeypatch, WORKER)
    evaluator = CudaScheduleEvaluator(tmp_path / 'run')
    with pytest.raises(FileExistsError):
        CudaScheduleEvaluator(tmp_path / 'run')
    with pytest.raises(RuntimeError, match='baseline'):
        evaluator.evaluate(WORKLOAD, KernelSchedule(), timeout_seconds=5)
    for timeout in (0, -1, True, float('inf')):
        with pytest.raises(ValueError):
            evaluator.baseline(WORKLOAD, timeout_seconds=timeout)
    assert evaluator.baseline(WORKLOAD, timeout_seconds=5).correct
    with pytest.raises(RuntimeError, match='already measured'):
        evaluator.baseline(WORKLOAD, timeout_seconds=5)
    with pytest.raises(ValueError, match='change workload'):
        evaluator.evaluate(replace(WORKLOAD, seed=1), KernelSchedule(), timeout_seconds=5)
    evaluator.close()


def test_import_does_not_import_optional_gpu_libraries():
    code = ('import sys; import drivebench_tuning.schedule_measurement; '
            'assert "torch" not in sys.modules; assert "triton" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_real_worker_rejects_missing_cuda_without_fallback(tmp_path, monkeypatch):
    # Exercise the actual subprocess entry point while explicitly hiding GPUs.
    # This tests failure handling only; it does not execute a GPU workload.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    evaluator = CudaScheduleEvaluator(tmp_path / 'run')
    result = evaluator.baseline(WORKLOAD, timeout_seconds=15)
    assert result.status == 'error' and not result.correct
    assert result.raw_samples_ms == [] and result.latency_ms is None
    assert result.evidence['worker_reaped']
    assert 'CUDA GPU' in result.evidence['error']['message']
    assert evaluator._process is None


def test_augmented_runtime_has_a_new_cohort_and_preserves_parent():
    from drivebench_tuning.schedule_measurement import _bind_schedule_runtime
    original = {'gpu_uuid': 'test-only', 'compute_capability': [8, 6]}
    writes = []
    runner = SimpleNamespace(runtime=original, cohort_id='original-cohort', device=0,
        torch=SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(multi_processor_count=82))),
        journal=SimpleNamespace(write=lambda name, value: writes.append((name, value))))
    _bind_schedule_runtime(runner)
    assert 'multiprocessor_count' not in original
    assert runner.runtime['multiprocessor_count'] == 82
    assert runner.runtime['parent_runtime_cohort_id'] == 'original-cohort'
    assert runner.cohort_id == hashlib.sha256(json.dumps(runner.runtime, sort_keys=True).encode()).hexdigest()
    assert writes == [('schedule-runtime.json', {**runner.runtime, 'cohort_id': runner.cohort_id})]


@pytest.mark.gpu
@pytest.mark.parametrize('workload', [
    Workload('geglu', (4, 256), dtype='float32'),
    Workload('groupnorm_silu', (2, 8, 7, 11), dtype='float16', num_groups=4),
])
def test_cuda_environment_baseline_repeated_measurements_and_submission(tmp_path, workload):
    """Executable live check, skipped on CPU; no speedup expectation."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip('Requires NVIDIA CUDA; CPU tests are not GPU validation')
    triton = pytest.importorskip('triton')
    if torch.__version__.split('+')[0] != '2.10.0' or triton.__version__ != '3.6.0':
        pytest.skip('Requires the declared Torch 2.10 / Triton 3.6 live runtime')
    from drivebench_tuning.schedule import ACTIONS
    from drivebench_tuning.schedule_env import ScheduleTuningEnv
    env = ScheduleTuningEnv(workloads=[workload], budget_seconds=600,
                           trial_timeout_seconds=300, evidence_dir=tmp_path / 'episode')
    try:
        observation, initial = env.reset(seed=47)
        assert env.observation_space.contains(observation)
        baseline = initial['baseline']
        assert baseline['correct'] and baseline['backend'] == 'cuda'
        assert baseline['evidence']['selected_reference'] in ('eager', 'compiled')
        assert set(baseline['evidence']['baseline_roles']) == {'eager', 'compiled'}
        runtime = baseline['evidence']['runtime']
        assert runtime['multiprocessor_count'] > 0
        assert runtime['parent_runtime_cohort_id'] != baseline['evidence']['cohort_id']
        records = [baseline]
        # Make a real edit, then issue repeated measurements of the resulting
        # schedule. The final submit is another fresh invocation sample set.
        _, _, terminated, truncated, _ = env.step(ACTIONS.index('unfuse'))
        assert not terminated and not truncated
        for name in ('measure', 'measure', 'submit'):
            observation, reward, terminated, truncated, info = env.step(ACTIONS.index(name))
            assert not truncated
            assert terminated == (name == 'submit')
            assert env.observation_space.contains(observation)
            result = info['measurement']
            assert result['correct'] and result['status'] == 'ok'
            assert result['latency_ms'] in result['raw_samples_ms']
            assert result['elapsed_seconds'] > 0
            assert result['evidence']['compile_and_first_call_seconds'] > 0
            for key in ('runtime', 'cohort_id', 'input_sha256'):
                assert result['evidence'][key] == baseline['evidence'][key]
            records.append(result)
        assert len({row['evidence']['trial_id'] for row in records}) == 4
        assert env.evaluator._process is None
        worker = Path(info['evidence_directory']) / 'evaluator/worker'
        original = json.loads((worker / 'runtime.json').read_text())
        augmented = json.loads((worker / 'schedule-runtime.json').read_text())
        assert original['cohort_id'] == augmented['parent_runtime_cohort_id']
        assert augmented['cohort_id'] == baseline['evidence']['cohort_id']
    finally:
        env.close()
