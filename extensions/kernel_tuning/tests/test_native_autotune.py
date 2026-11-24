"""CPU contracts for the native-engine adapter; these invent no GPU evidence."""
from types import SimpleNamespace
import uuid

import pytest

from drivebench_tuning.native_autotune import (
    WholeOperatorLaunch, confirm_selections, native_candidates, ppo_select,
    reference_fallback, validate_input_bindings,
)
from drivebench_tuning.search import BudgetedTuningEnv, CandidateSpec, TrialResult
from drivebench_tuning.suite import load_suite


class Journal:
    def __init__(self):
        self.rows = []

    def append(self, row):
        self.rows.append(row)


class ControlledEvaluator:
    """Identity-check fixture only; never used for performance or policy results."""
    def __init__(self, latencies=None):
        self.runner = SimpleNamespace(cohort_id='fixture-cohort')
        self.latencies = latencies or {}
        self.calls = []

    def baseline(self, task):
        return self.evaluate(task, None)

    def evaluate(self, task, candidate):
        key = candidate.candidate_id if candidate else None
        self.calls.append(key)
        return TrialResult(str(uuid.uuid4()), task.task_id, key, 'cuda',
                           'fixture-cohort', 'a' * 64, 'ok', True,
                           self.latencies.get(key, 1.0), 10,
                           {'test_fixture': True, 'not_gpu_evidence': True})


def test_grid_is_complete_and_no_outcome_based_pruning():
    suite = load_suite('operator-generalization-v2')
    assert len(native_candidates(suite)) == 24
    broken = SimpleNamespace(candidates=suite.candidates[:-1])
    with pytest.raises(ValueError, match='exact 24'):
        native_candidates(broken)


def test_input_binding_rejects_wrong_suite_and_missing_test_task():
    suite = load_suite('operator-generalization-v2')
    doc = {'suite_sha256': suite.sha256,
           'expected_input_digests': {suite.tasks[i].task_id: 'a' * 64 for i in suite.splits['test']}}
    assert len(validate_input_bindings(doc, suite)) == 18
    with pytest.raises(ValueError, match='suite'):
        validate_input_bindings({**doc, 'suite_sha256': 'b' * 64}, suite)
    doc['expected_input_digests'].pop(suite.tasks[suite.splits['test'][0]].task_id)
    with pytest.raises(ValueError, match='Every test'):
        validate_input_bindings(doc, suite)


def test_whole_operator_adapter_delegates_exact_shared_configuration(monkeypatch):
    from drivebench_tuning.contract import Workload
    from drivebench_tuning import kernels
    config = CandidateSpec('candidate', {'implementation': 'triton', 'block_size': 128, 'num_warps': 2})
    calls = []
    monkeypatch.setattr(kernels, 'fused_groupnorm_silu', lambda *a, **kw: calls.append((a, kw)) or 'output')
    w = Workload(operator='groupnorm_silu', shape=(1, 32, 8, 8), dtype='float16')
    launch = WholeOperatorLaunch(w, {(128, 2): config})
    assert launch.run('x', 'w', 'b', 'task', BLOCK_SIZE=128, num_warps=2) == 'output'
    assert calls == [(('x', 'w', 'b'), {'num_groups': w.num_groups, 'eps': w.eps,
                                      'block_size': 128, 'num_warps': 2})]
    assert launch.current == config
    with pytest.raises(ValueError, match='outside'):
        launch.run('x', 'w', 'b', 'task', BLOCK_SIZE=64, num_warps=2)
    with pytest.raises(ValueError, match='compiler metadata'):
        launch.run('x', 'w', 'b', 'task', BLOCK_SIZE=128, num_warps=2, num_stages=4)


def test_fallback_is_selected_before_independent_confirmation():
    suite = load_suite('operator-generalization-v2')
    task = suite.tasks[suite.splits['test'][0]]
    compiled = suite.candidates[0]
    native = native_candidates(suite)[0]
    ev = ControlledEvaluator({None: 1.0, compiled.candidate_id: .7, native.candidate_id: .8})
    journal = Journal()
    fallback = reference_fallback(ev, task, native, compiled, expected_digest='a' * 64,
                                  offset=1, journal=journal)
    assert fallback['selected_role'] == 'compiled'
    # Reverse confirmation outcomes. The previously chosen fallback must remain
    # compiled, never switch to the reference that got lucky during confirmation.
    ev.latencies = {None: .3, compiled.candidate_id: .7, native.candidate_id: .2}
    confirmed = confirm_selections(ev, task, {'eager': None, 'compiled': compiled,
        'native': native, 'ppo': native}, expected_digest='a' * 64, offset=2, journal=journal)
    assert fallback['selected_role'] == 'compiled'
    assert len(ev.calls) == 7
    assert len({r['trial_id'] for r in confirmed['measurements'].values()}) == 4
    assert {r['trial_id'] for r in fallback['probes'].values()}.isdisjoint(
        r['trial_id'] for r in confirmed['measurements'].values())


def test_confirmation_aborts_mismatched_inputs():
    suite = load_suite('operator-generalization-v2')
    task = suite.tasks[suite.splits['test'][0]]
    ev = ControlledEvaluator()
    with pytest.raises(ValueError, match='binding failed'):
        confirm_selections(ev, task, {r: None for r in ('eager', 'compiled', 'native', 'ppo')},
                           expected_digest='b' * 64, offset=0, journal=Journal())
    assert len(ev.calls) == 1


def test_live_ppo_spends_eight_turns_including_repeat_queries():
    suite = load_suite('operator-generalization-v2')
    ev = ControlledEvaluator()
    env = BudgetedTuningEnv(suite.tasks, suite.candidates, ev, trial_budget=8, memory_weight=0)
    journal = Journal()
    result = ppo_select(env, lambda obs, info: 0, suite.splits['test'][0], 'a' * 64, 0, journal)
    assert result['candidate_turns'] == 8
    assert len(ev.calls) == 9  # fresh eager reset, eight actual repeated queries
    assert result['final_info']['trials_used'] == 8
    assert all(row['action'] == 0 for row in result['turns'])
    assert result['selected_candidate_id'] is None  # ties do not replace eager
