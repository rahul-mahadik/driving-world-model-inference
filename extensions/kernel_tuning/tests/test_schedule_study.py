"""Prospective study bookkeeping tests. All evaluators here are CPU fixtures."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import random

import pytest

from drivebench_tuning import schedule_study as study
from drivebench_tuning.contract import Workload
from drivebench_tuning.schedule import ACTIONS, KernelSchedule, transform, initial_schedule
from drivebench_tuning.schedule_env import ScheduleTuningEnv
from drivebench_tuning.schedule_measurement import ScheduleMeasurement


def test_design_has_balanced_six_family_three_seed_three_policy_pairing():
    doc = study.protocol()
    assert len(doc['tasks']) == 6 and len(doc['episodes']) == 54
    assert all(t['metadata']['split'] == 'test' for t in doc['tasks'])
    assert all(t['metadata']['workload']['dtype'] == 'float16' for t in doc['tasks'])
    for policy in study.POLICIES:
        rows = [e for e in doc['episodes'] if e['policy'] == policy]
        assert len(rows) == 18
        assert all(sum(r['order_position'] == position for r in rows) == 6 for position in range(3))
        assert len({(r['task_id'], r['seed']) for r in rows}) == 18
    assert doc['episode_seconds'] == 90
    assert doc['setup_timeout_seconds'] == 45
    assert doc['final_reserve_seconds'] == 20
    assert doc['max_exploration_queries'] == 128
    assert study.validate_protocol(doc) == study.digest(doc)
    doc['episode_seconds'] = 91
    with pytest.raises(ValueError, match='frozen'):
        study.validate_protocol(doc)


@pytest.mark.parametrize('workload', [Workload('geglu', (2, 256)),
    Workload('groupnorm_silu', (2, 32, 17, 19), num_groups=4),
    Workload('groupnorm_silu', (2, 32, 512, 512), num_groups=4)])
def test_restoration_uses_only_legal_edits_for_reachable_states(workload):
    rng = random.Random(42)
    state = initial_schedule(workload)
    states = [state]
    for _ in range(200):
        action = rng.choice(study.legal_edits(workload, state))
        state = transform(workload, state, action)
        states.append(state)
    for current, target in zip(states[::7], reversed(states[::7])):
        restored = current
        for action in study.navigation_actions(workload, current, target):
            assert ACTIONS[action] not in ('measure', 'submit')
            restored = transform(workload, restored, action)
        assert restored == target


class FixtureEvaluator:
    def __init__(self, evidence_directory, **kwargs):
        self.sequence = 0
        self.closed = False
        self.kwargs = kwargs
    def _row(self, workload, schedule, latency, **extra):
        self.sequence += 1
        return ScheduleMeasurement(workload.key(), asdict(schedule) if schedule else None,
            'ok', True, latency, [latency, latency, latency], 256, .01,
            {'trial_id': str(self.sequence), 'selected_reference': 'compiled',
             'cohort_id': 'CPU-fixture', 'runtime': {'gpu_uuid': 'CPU-fixture'},
             'input_sha256': ['0'*64], **extra}, backend='test')
    def baseline(self, workload, **kwargs):
        return self._row(workload, None, 2.)
    def evaluate(self, workload, schedule, **kwargs):
        return self._row(workload, schedule, 1. if schedule.block_size >= 256 else 3.)
    def measure_reference(self, workload, implementation, **kwargs):
        assert implementation == 'compiled'
        return self._row(workload, None, 2., fixed_reference_confirmation=implementation)
    def close(self):
        self.closed = True


def fixture_env(**kwargs):
    return ScheduleTuningEnv(**kwargs, allow_test_evaluator=True)


@pytest.mark.parametrize('policy', study.POLICIES)
def test_policy_preserves_fresh_confirmation_and_explicit_baseline_semantics(tmp_path, monkeypatch, policy):
    monkeypatch.setattr(study, 'MAX_QUERIES', 4)
    doc = study.protocol()
    spec = next(e for e in doc['episodes'] if e['policy'] == policy)
    task = next(t for t in doc['tasks'] if t['task_id'] == spec['task_id'])
    row = study._run_episode(spec, task, tmp_path / 'evidence', tmp_path / 'cache',
                            env_factory=fixture_env, evaluator_factory=FixtureEvaluator)
    assert row['successful']
    assert row['baseline']['evidence']['trial_id'] != row['confirmation']['evidence']['trial_id']
    assert len({row[name]['evidence']['trial_id'] for name in ('baseline','paired_final_reference','confirmation')}) == 3
    assert row['confirmed_latency_ratio'] == row['confirmed_median_ms']/row['paired_final_reference_median_ms']
    assert row['confirmation']['backend'] == 'test'
    assert (tmp_path / 'evidence/result.json').is_file()
    if policy == 'compiled_eager_no_edit':
        assert row['environment_return'] is None
        assert row['exploration_queries'] == 0
        assert row['confirmation']['evidence']['fixed_reference_confirmation'] == 'compiled'
    else:
        assert row['exploration_queries'] == 4
        assert row['stop_reason'] == 'query_safety_cap'
        assert row['confirmation']['schedule'] == row['selected_schedule']
        assert row['environment_completed']
        paths = list((tmp_path / 'evidence/environment').glob('episode-*/environment.jsonl'))
        steps = [json.loads(line) for line in paths[0].read_text().splitlines()]
        assert sum(r['event'] == 'trial_returned' for r in steps) == 5
        assert any(r.get('action') == 'submit' for r in steps)


def test_reset_failure_and_interruption_are_preserved_and_counted(tmp_path):
    doc = study.protocol(); spec = doc['episodes'][0]; task = doc['tasks'][0]
    for exception, label in ((RuntimeError('setup failure'), 'failed'), (KeyboardInterrupt('stop'), 'interrupted')):
        class Broken(FixtureEvaluator):
            def baseline(self, workload, **kwargs):
                raise exception
        row = study._run_episode(spec, task, tmp_path / label, tmp_path / ('cache-'+label),
                                env_factory=fixture_env, evaluator_factory=Broken)
        assert row['status'] == label and not row['successful']
        assert row['failure_adjusted_speed_score'] == -1.
        assert row['confirmation'] is None
        result = json.loads((tmp_path / label / 'result.json').read_text())
        assert result['error']['message'] == str(exception)
        summary = study.summarize([row], doc)
        assert summary['attempted_episodes'] == 1
        assert summary['policies'][spec['policy']]['failed_or_unconfirmed'] == 1


def fixture_result(spec, cohort='fixture'):
    return {**spec, 'successful': False, 'stop_reason': 'fixture_failure', 'failure_adjusted_speed_score': -1.,
        'elapsed_seconds': 1., 'exploration_queries': 0, 'failed_queries': 0, 'invalid_actions': 0,
        'baseline': {'evidence': {'cohort_id': cohort}}}


def test_summary_rejects_duplicate_and_misbound_episodes():
    doc = study.protocol(); row = fixture_result(doc['episodes'][0])
    for key, value in [('task_id', 'wrong'), ('seed', 0), ('policy', 'random_walk'), ('order_position', 7)]:
        with pytest.raises(ValueError):
            study.summarize([{**row, key: value}], doc)
    with pytest.raises(ValueError, match='Duplicate'):
        study.summarize([row, row], doc)


def test_hardware_change_preserves_rows_but_suppresses_pooled_aggregates(tmp_path, monkeypatch):
    count = 0
    def execute(spec, task, directory, cache_directory):
        nonlocal count
        count += 1
        return fixture_result(spec, str(count))
    monkeypatch.setattr(study, '_run_episode', execute)
    summary = study.run_study(tmp_path / 'out', tmp_path / 'cache')
    assert summary['attempted_episodes'] == 2
    assert summary['policies'] == {} and not summary['comparison_valid']
    assert len((tmp_path / 'out/episodes.jsonl').read_text().splitlines()) == 2
    assert (tmp_path / 'out/partial.json').is_file()
    assert not (tmp_path / 'out/completed.json').exists()


def test_full_campaign_includes_all_failed_attempts_without_success_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(study, '_run_episode', lambda spec, *_: fixture_result(spec))
    summary = study.run_study(tmp_path / 'out', tmp_path / 'cache')
    assert summary['attempted_episodes'] == 54
    assert all(p['confirmed'] == 0 and p['failed_or_unconfirmed'] == 18 for p in summary['policies'].values())
    completed = json.loads((tmp_path / 'out/completed.json').read_text())
    assert completed['status'] == 'complete'
    assert not completed['all_episodes_successful'] and not completed['gpu_validation_claim']
    assert completed['summary_sha256'] == study.digest(summary)


def test_nonfitting_episode_is_not_started(tmp_path, monkeypatch):
    monkeypatch.setattr(study, '_run_episode', lambda *_: pytest.fail('must not admit episode'))
    summary = study.run_study(tmp_path / 'out', tmp_path / 'cache', max_seconds=1)
    assert summary['attempted_episodes'] == 0
    assert len(summary['missing_episode_ids']) == 54


def test_cleanup_failure_stops_campaign_and_preserves_attempt(tmp_path, monkeypatch):
    def execute(spec, *_):
        return {**fixture_result(spec), 'cleanup_failed': True}
    monkeypatch.setattr(study, '_run_episode', execute)
    summary = study.run_study(tmp_path / 'out', tmp_path / 'cache')
    assert summary['attempted_episodes'] == 1
    assert 'cleanup failed' in summary['failure']


def test_cache_must_be_outside_scientific_evidence(tmp_path):
    with pytest.raises(ValueError, match='separate'):
        study.run_study(tmp_path / 'out', tmp_path / 'out/cache')


def test_protocol_can_be_written_without_gpu_execution(tmp_path):
    path = tmp_path / 'prospective.json'
    assert study.main(['--write-protocol', str(path)]) == 0
    assert json.loads(path.read_text()) == study.protocol()


class FixtureClock:
    """Deterministic elapsed time; never sleeps or measures GPU performance."""
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def timed_fixture_evaluator(clock, *, setup_seconds, exploration_seconds):
    class Timed(FixtureEvaluator):
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []
            self.confirming = False
            self.instances.append(self)

        def _timed(self, kind, workload, schedule, duration, timeout_seconds):
            assert not self.closed
            self.calls.append((kind, timeout_seconds))
            if duration >= timeout_seconds:
                clock.advance(timeout_seconds)
                self.sequence += 1
                self.closed = True
                return ScheduleMeasurement(workload.key(), asdict(schedule) if schedule else None,
                    'timeout', False, None, [], None, timeout_seconds,
                    {'trial_id': str(self.sequence), 'worker_reaped': True}, backend='test')
            clock.advance(duration)
            row = self._row(workload, schedule, 1. if schedule else 2.,
                            fixed_reference_confirmation='compiled' if kind == 'reference' else None)
            return replace(row, elapsed_seconds=duration)

        def baseline(self, workload, *, timeout_seconds):
            return self._timed('setup', workload, None, setup_seconds, timeout_seconds)

        def evaluate(self, workload, schedule, *, timeout_seconds):
            return self._timed('submit' if self.confirming else 'explore', workload, schedule,
                               1. if self.confirming else exploration_seconds, timeout_seconds)

        def measure_reference(self, workload, implementation, *, timeout_seconds):
            assert implementation == 'compiled'
            self.confirming = True
            return self._timed('reference', workload, None, 1., timeout_seconds)

    return Timed


@pytest.mark.parametrize('policy', study.POLICIES[1:])
def test_exploration_stops_before_a_short_timeout_can_destroy_confirmation(tmp_path, monkeypatch, policy):
    clock = FixtureClock()
    monkeypatch.setattr(study.time, 'monotonic', clock.monotonic)
    evaluator = timed_fixture_evaluator(clock, setup_seconds=39.99, exploration_seconds=14.99)
    doc = study.protocol()
    spec = next(e for e in doc['episodes'] if e['policy'] == policy)
    task = next(t for t in doc['tasks'] if t['task_id'] == spec['task_id'])
    row = study._run_episode(spec, task, tmp_path/'evidence', tmp_path/'cache',
                            env_factory=fixture_env, evaluator_factory=evaluator)
    assert row['successful']
    assert row['exploration_queries'] == 2
    assert row['failed_queries'] == 0
    assert row['stop_reason'] == 'confirmation_reserve'
    calls = evaluator.instances[0].calls
    assert [timeout for kind, timeout in calls if kind == 'explore'] == [15., 15.]
    assert [kind for kind, _ in calls[-2:]] == ['reference', 'submit']
    assert row['elapsed_seconds'] < study.EPISODE_SECONDS


@pytest.mark.parametrize('policy', study.POLICIES[1:])
def test_exploration_window_is_rechecked_after_navigation(tmp_path, monkeypatch, policy):
    clock = FixtureClock()
    monkeypatch.setattr(study.time, 'monotonic', clock.monotonic)
    evaluator = timed_fixture_evaluator(clock, setup_seconds=30., exploration_seconds=14.)

    class SlowFirstEdit(ScheduleTuningEnv):
        delayed = False

        def step(self, action):
            if ACTIONS[action] not in ('measure', 'submit') and not self.delayed:
                self.delayed = True
                clock.advance(12.)
            return super().step(action)

    doc = study.protocol()
    spec = next(e for e in doc['episodes'] if e['policy'] == policy)
    task = next(t for t in doc['tasks'] if t['task_id'] == spec['task_id'])
    row = study._run_episode(spec, task, tmp_path/'evidence', tmp_path/'cache',
        env_factory=lambda **kwargs: SlowFirstEdit(**kwargs, allow_test_evaluator=True),
        evaluator_factory=evaluator)
    assert row['successful']
    assert row['exploration_queries'] == 1
    assert row['confirmation']['schedule'] == row['selected_schedule']
    assert row['restore_actions'] > 0
    assert [timeout for kind, timeout in evaluator.instances[0].calls if kind == 'explore'] == [15.]


@pytest.mark.parametrize('policy', study.POLICIES[1:])
def test_full_window_timeout_remains_a_failed_episode(tmp_path, monkeypatch, policy):
    clock = FixtureClock()
    monkeypatch.setattr(study.time, 'monotonic', clock.monotonic)
    evaluator = timed_fixture_evaluator(clock, setup_seconds=30., exploration_seconds=16.)
    doc = study.protocol()
    spec = next(e for e in doc['episodes'] if e['policy'] == policy)
    task = next(t for t in doc['tasks'] if t['task_id'] == spec['task_id'])
    row = study._run_episode(spec, task, tmp_path/'evidence', tmp_path/'cache',
                            env_factory=fixture_env, evaluator_factory=evaluator)
    assert not row['successful']
    assert row['failure_adjusted_speed_score'] == -1.
    assert row['exploration_queries'] == row['failed_queries'] == 1
    assert row['stop_reason'] == 'worker_failure_or_budget'
    assert row['confirmation'] is None and row['paired_final_reference'] is None
    assert evaluator.instances[0].closed
