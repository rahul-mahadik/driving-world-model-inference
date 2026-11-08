"""Actual core CPU fixture wiring; none of these are Vista/GPU measurements."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from drivebench.engine import InferenceEngine
from drivebench.types import GenerationConfig, OptimizationConfig
from drivebench_tuning.inference import InferenceEvaluator, InferenceRequest, QualityDecision
from drivebench_tuning.evaluation import EvaluationError, evaluate_policy, grid_policy
from drivebench_tuning.search import BudgetedTuningEnv, CandidateSpec, TaskSpec


TASK = TaskSpec('cpu-fixture', (8., 8., 2.), {'fixture': True})
REFERENCE = {'generation': {'total_frames': 3, 'height': 8, 'width': 8, 'steps': 2},
             'optimizations': {'max_batch_size': 1}}
CANDIDATE = CandidateSpec('batch-2', {'optimizations': {'max_batch_size': 2}})


def request(task):
    return InferenceRequest(np.zeros((1, 3, 8, 8), np.float32),
                            (np.zeros(8, np.float32), np.ones(8, np.float32)), (11, 12))


def factory(task, config):
    return InferenceEngine(model={'backend': 'synthetic'},
                           generation=GenerationConfig(**config['generation']),
                           optimizations=OptimizationConfig(**config['optimizations']))


def gate(task, candidate, result, reference):
    same = reference is None or all(np.array_equal(value, reference.frames[key]) for key, value in result.frames.items())
    return QualityDecision(same, {'policy': 'CPU fixture exact equality', 'visual_quality_established': False})


def evaluator(**kwargs):
    return InferenceEvaluator(kwargs.pop('request_loader', request), kwargs.pop('engine_factory', factory),
                              kwargs.pop('quality_gate', gate), REFERENCE,
                              backend=kwargs.pop('backend', 'synthetic'), cohort_id='cpu-fixture-v1', **kwargs)


def mutate_result(mutator):
    def wrapped_factory(task, config):
        engine = factory(task, config)
        generate = engine.generate

        def wrapped_generate(*args, **kwargs):
            result = generate(*args, **kwargs)
            if config['optimizations']['max_batch_size'] == 2:
                mutator(result)
            return result

        engine.generate = wrapped_generate
        return engine
    return wrapped_factory


def test_actual_engine_maps_config_and_gates_both_roles_with_fresh_trials():
    calls, cleanups = [], []

    def recording_gate(task, candidate, result, reference):
        calls.append((candidate, reference is None))
        return gate(task, candidate, result, reference)

    adapter = evaluator(quality_gate=recording_gate, engine_cleanup=lambda engine: cleanups.append(engine.optimizations.max_batch_size))
    baseline = adapter.baseline(TASK)
    candidate = adapter.evaluate(TASK, CANDIDATE)
    repeated = adapter.evaluate(TASK, CANDIDATE)
    assert baseline.status == candidate.status == repeated.status == 'ok'
    assert len({baseline.trial_id, candidate.trial_id, repeated.trial_id}) == 3
    assert baseline.input_digest == candidate.input_digest
    assert calls == [(None, True), (CANDIDATE, False), (CANDIDATE, False)]
    assert cleanups == [1, 2, 2]
    evidence = candidate.to_dict()['evidence']
    assert evidence['engine_config']['optimizations']['max_batch_size'] == 2
    assert evidence['effective_config']['actual_batch_sizes'] == [2]
    assert evidence['warmups'] == 0 and evidence['request_samples'] == 1
    assert 'initialization excluded' in evidence['latency_boundary']
    assert 'first-use compilation included' in evidence['latency_boundary']
    assert candidate.peak_allocated_bytes == 2 * 3 * 8 * 8 * 3 * 4
    assert 'not GPU allocation' in evidence['memory_boundary']
    assert evidence['vista_gpu_bridge_validated'] is False
    assert evidence['quality_preservation_claim'] is False
    assert [row['event'] for row in adapter.journal] == ['started', 'finished'] * 3
    json.dumps(adapter.journal, allow_nan=False)


def test_real_output_difference_fails_gate_and_preserves_reference_incumbent():
    def change_output(result):
        result.frames['future-0'][0, 0, 0, 0] = .25

    adapter = evaluator(engine_factory=mutate_result(change_output))
    env = BudgetedTuningEnv([TASK], [CANDIDATE], adapter, trial_budget=1, allow_synthetic=True)
    _, baseline_info = env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 0 and terminated and not truncated
    assert info['measurement']['status'] == 'incorrect'
    assert info['measurement']['evidence']['observed_synthetic_output_nbytes'] > 0
    assert info['measurement']['evidence']['observed_engine_peak_allocated_bytes'] == 0
    assert info['incumbent_candidate_id'] is None
    assert info['incumbent_cost'] == baseline_info['incumbent_cost'] == 1.
    assert info['trials_used'] == 1
    env.close()


def test_candidate_mapper_connects_external_payload_to_engine_settings():
    mapped = CandidateSpec('external-batch', {'batch': 2})
    adapter = evaluator(candidate_config_mapper=lambda task, candidate: {'optimizations': {'max_batch_size': candidate.config['batch']}})
    adapter.baseline(TASK)
    measured = adapter.evaluate(TASK, mapped)
    assert measured.candidate_id == 'external-batch'
    assert measured.evidence['effective_config']['requested_batch_size'] == 2


@pytest.mark.parametrize('mutator,match', [
    (lambda r: r.candidate_ids.reverse(), 'identity'),
    (lambda r: r.seeds.reverse(), 'identity'),
    (lambda r: r.effective_config.update(steps=19), 'effective configuration'),
    (lambda r: r.effective_config.update(candidate_count=1), 'effective configuration'),
    (lambda r: r.effective_config['optimizations'].update(max_batch_size=1), 'effective configuration'),
    (lambda r: r.frames.pop('future-0'), 'output tensors'),
    (lambda r: r.frames.update({'future-0': np.zeros((1, 8, 8, 3), np.float32)}), 'output tensors'),
    (lambda r: r.frames['future-0'].fill(np.nan), 'output tensors'),
    (lambda r: r.frames['future-0'].fill(1.1), 'output tensors'),
])
def test_result_identity_and_output_contract_fail_closed(mutator, match):
    adapter = evaluator(engine_factory=mutate_result(mutator))
    adapter.baseline(TASK)
    with pytest.raises(ValueError, match=match):
        adapter.evaluate(TASK, CANDIDATE)
    assert adapter.journal[-1]['event'] == 'failed'
    assert adapter.journal[-2]['event'] == 'started'


@pytest.mark.parametrize('mutator', [
    lambda r: setattr(r, 'status', 'partial'),
    lambda r: r.failures.append({'status': 'out_of_memory'}),
    lambda r: r.cache_stats.update(oom_attempts=[{'retried': True}]),
])
def test_execution_failure_and_recovered_oom_do_not_receive_quality_or_reward(mutator):
    calls = []

    def recording_gate(*args):
        calls.append(args[1])
        return gate(*args)

    adapter = evaluator(engine_factory=mutate_result(mutator), quality_gate=recording_gate)
    adapter.baseline(TASK)
    result = adapter.evaluate(TASK, CANDIDATE)
    assert result.status == 'error' and not result.correct
    assert result.latency_ms is None and result.peak_allocated_bytes is None
    assert calls == [None]
    assert result.evidence['quality_gate_not_run'] is True
    assert result.evidence['observed_synthetic_output_nbytes'] > 0


@pytest.mark.parametrize('role', ['result', 'reference'])
def test_quality_callback_cannot_change_accepted_outputs(role):
    def bad_gate(task, candidate, result, reference):
        if candidate is not None:
            target = result if role == 'result' else reference
            target.frames['future-0'].fill(0)
        return QualityDecision(True, {'policy': 'deliberately invalid test callback'})

    adapter = evaluator(quality_gate=bad_gate)
    adapter.baseline(TASK)
    with pytest.raises(ValueError, match='callback mutated'):
        adapter.evaluate(TASK, CANDIDATE)
    # The callback only saw a defensive copy; the accepted reference survives.
    adapter.quality_gate = gate
    assert adapter.evaluate(TASK, CANDIDATE).status == 'ok'


def test_baseline_quality_is_mandatory_and_rejected_baseline_cannot_authorize_trials():
    with pytest.raises(ValueError, match='mandatory'):
        evaluator(quality_gate=None)
    adapter = evaluator(quality_gate=lambda *args: QualityDecision(False, {'policy': 'reject fixture'}))
    assert adapter.baseline(TASK).status == 'incorrect'
    with pytest.raises(RuntimeError, match='accepted reference'):
        adapter.evaluate(TASK, CANDIDATE)
    adapter.quality_gate = lambda *args: True
    with pytest.raises(ValueError, match='not a bare boolean'):
        adapter.baseline(TASK)


def test_cleanup_failure_prevents_success_and_new_failed_reset_discards_old_reference():
    cleanup_calls = []
    adapter = evaluator(engine_cleanup=lambda e: cleanup_calls.append(1))
    adapter.baseline(TASK)

    def failed_cleanup(engine):
        raise RuntimeError('fixture cleanup failed')

    adapter.cleanup = failed_cleanup
    with pytest.raises(RuntimeError, match='cleanup failed'):
        adapter.baseline(TASK)
    assert adapter.journal[-1]['event'] == 'failed'
    assert len([e for e in adapter.journal if e['event'] == 'finished']) == 1
    with pytest.raises(RuntimeError, match='accepted reference'):
        adapter.evaluate(TASK, CANDIDATE)


def test_engine_exception_is_journaled_and_cleanup_runs():
    calls = []

    def failed_factory(task, config):
        engine = factory(task, config)
        engine.prepare_context = lambda frames: (_ for _ in ()).throw(RuntimeError('fixture inference failed'))
        return engine

    adapter = evaluator(engine_factory=failed_factory, engine_cleanup=lambda engine: calls.append(1))
    with pytest.raises(RuntimeError, match='inference failed'):
        adapter.baseline(TASK)
    assert calls == [1]
    assert [e['event'] for e in adapter.journal] == ['started', 'failed']


def test_execution_and_cleanup_failure_both_survive_with_attempt_bindings():
    def broken_factory(task, config):
        engine = factory(task, config)
        engine.prepare_context = lambda frames: (_ for _ in ()).throw(RuntimeError('execution failed'))
        return engine

    def broken_cleanup(engine):
        raise RuntimeError('cleanup failed')

    adapter = evaluator(engine_factory=broken_factory, engine_cleanup=broken_cleanup)
    with pytest.raises(RuntimeError, match='execution failed'):
        adapter.baseline(TASK)
    event = adapter.journal[-1]
    assert event['error']['message'] == 'execution failed'
    assert event['evidence']['cleanup_error']['message'] == 'cleanup failed'
    assert len(event['input_digest']) == 64
    assert event['evidence']['mapped_config'] == REFERENCE
    assert event['evidence']['engine_initialization_seconds'] >= 0


@pytest.mark.parametrize('identity_key', ['checkpoint_sha256', 'upstream_commit', 'inference_config_sha256', 'unknown_model_identity'])
def test_actual_model_content_cannot_change_under_same_declared_config(identity_key):
    def changing_factory(task, config):
        engine = factory(task, config)
        engine.backend.identity = dict(engine.backend.identity)
        engine.backend.identity[identity_key] = 'reference' if config['optimizations']['max_batch_size'] == 1 else 'changed'
        return engine

    adapter = evaluator(engine_factory=changing_factory)
    adapter.baseline(TASK)
    with pytest.raises(ValueError, match='model-content identity changed'):
        adapter.evaluate(TASK, CANDIDATE)


def test_execution_identity_may_change_without_permitting_model_identity_drift():
    def changing_factory(task, config):
        engine = factory(task, config)
        engine.backend.identity = dict(engine.backend.identity)
        engine.backend.identity['candidate_execution'] = str(config['optimizations']['max_batch_size'])
        return engine

    adapter = evaluator(engine_factory=changing_factory)
    adapter.baseline(TASK)
    assert adapter.evaluate(TASK, CANDIDATE).status == 'ok'


@pytest.mark.parametrize('drift', ['checkpoint', 'unspecified_generation_default', 'runtime', 'declared_model'])
def test_harness_rejects_actual_reference_contract_drift_across_fresh_factories(drift):
    factory_count = 0

    def make_env():
        nonlocal factory_count
        generation = factory_count
        factory_count += 1

        def varying_engine(task, config):
            if drift == 'unspecified_generation_default':
                config['generation']['guidance_scale'] = 2.5 + generation
            engine = factory(task, config)
            engine.backend.identity = dict(engine.backend.identity)
            if drift == 'checkpoint':
                engine.backend.identity['checkpoint_sha256'] = str(generation)
            if drift == 'declared_model':
                engine.model_config['fixture_model_revision'] = generation
            return engine

        adapter = evaluator(engine_factory=varying_engine,
                            runtime_evidence={'generation': generation if drift == 'runtime' else 0})
        return BudgetedTuningEnv([TASK], [CANDIDATE], adapter, trial_budget=1, allow_synthetic=True)

    with pytest.raises(EvaluationError, match='input') as failure:
        evaluate_policy(make_env, grid_policy, [0], repeats=2)
    report = failure.value.report
    assert report['status'] == 'aborted'
    assert len(report['episodes']) == 2
    assert report['episodes'][0]['status'] == 'completed'
    assert report['episodes'][1]['status'] == 'failed'


def test_input_or_task_drift_rejected_before_candidate_engine_construction():
    calls = []
    adapter = evaluator(engine_factory=lambda task, config: (calls.append(config) or factory(task, config)))
    adapter.baseline(TASK)
    adapter.request_loader = lambda task: InferenceRequest(np.zeros((1, 3, 8, 8)), (np.zeros(8), np.ones(8)), (12, 13))
    with pytest.raises(ValueError, match='inputs or task identity changed'):
        adapter.evaluate(TASK, CANDIDATE)
    assert len(calls) == 1
    adapter.request_loader = request
    with pytest.raises(ValueError, match='inputs or task identity changed'):
        adapter.evaluate(TaskSpec(TASK.task_id, TASK.features, {'changed': True}), CANDIDATE)


def test_semantic_changes_cannot_be_disguised_as_execution_tuning():
    adapter = evaluator()
    adapter.baseline(TASK)
    with pytest.raises(ValueError, match='generation semantics'):
        adapter.evaluate(TASK, CandidateSpec('steps-1', {'generation': {'steps': 1}}))
    assert adapter.evaluate(TASK, CandidateSpec('decode-1', {'generation': {'decode_chunk_size': 1}})).status == 'ok'


def test_engine_factory_must_apply_mapping_and_preserve_unspecified_settings():
    adapter = evaluator(engine_factory=lambda task, config: factory(task, REFERENCE))
    adapter.baseline(TASK)
    with pytest.raises(ValueError, match='ignored or changed'):
        adapter.evaluate(TASK, CANDIDATE)

    def drifting_factory(task, config):
        if config['optimizations']['max_batch_size'] == 2:
            config['optimizations']['conditioning_cache_bytes'] = 100
        return factory(task, config)

    adapter = evaluator(engine_factory=drifting_factory)
    adapter.baseline(TASK)
    with pytest.raises(ValueError, match='unspecified configuration'):
        adapter.evaluate(TASK, CANDIDATE)


def test_declared_cuda_cannot_silently_use_the_cpu_fixture():
    with pytest.raises(ValueError, match='runtime evidence'):
        evaluator(backend='cuda')
    adapter = evaluator(backend='cuda', runtime_evidence={'fixture': 'deliberate mismatch'})
    with pytest.raises(ValueError, match='not a CUDA backend'):
        adapter.baseline(TASK)
    assert adapter.journal[-1]['event'] == 'failed'


def test_requests_and_policy_evidence_are_deeply_immutable_copies():
    frames, action = np.zeros((1, 3, 8, 8)), np.zeros(8)
    item = InferenceRequest(frames, [action], [4])
    frames.fill(1); action.fill(1)
    assert not item.observed_frames.any() and not item.trajectories[0].any()
    for array in (item.observed_frames, item.trajectories[0]):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    evidence = {'policy': 'example', 'nested': [1]}
    decision = QualityDecision(True, evidence)
    evidence['nested'][0] = 2
    assert decision.evidence['nested'][0] == 1


@pytest.mark.parametrize('kwargs', [
    {'observed_frames': np.zeros((1, 8, 8, 3))},
    {'observed_frames': np.full((1, 3, 8, 8), np.nan)},
    {'observed_frames': np.full((1, 3, 8, 8), 2.)},
    {'trajectories': [np.zeros(7)]},
    {'seeds': [True]},
    {'output_ids': ['duplicate', 'duplicate'], 'trajectories': [np.zeros(8)] * 2, 'seeds': [1, 2]},
])
def test_invalid_requests_rejected(kwargs):
    values = {'observed_frames': np.zeros((1, 3, 8, 8)), 'trajectories': [np.zeros(8)], 'seeds': [1]}
    with pytest.raises(ValueError):
        InferenceRequest(**(values | kwargs))


def test_close_releases_references_preserves_detached_journal_and_sink():
    sink = []
    adapter = evaluator(event_sink=sink.append)
    adapter.baseline(TASK)
    assert sink == adapter.journal
    copied = adapter.journal
    copied.clear(); sink[0]['event'] = 'tampered'
    adapter.close()
    assert adapter.journal[0]['event'] == 'started'
    with pytest.raises(RuntimeError, match='accepted reference'):
        adapter.evaluate(TASK, CANDIDATE)
    assert adapter.baseline(TASK).status == 'ok'


def test_runnable_quickstart_emits_serializable_cpu_only_report():
    example = Path(__file__).parents[1] / 'examples' / 'inference_env_quickstart.py'
    result = subprocess.run([sys.executable, str(example)], check=True, text=True, capture_output=True)
    report = json.loads(result.stdout)
    assert 'no Vista, GPU performance or quality evidence' in report['example_scope']
    assert report['episodes'][0]['final_info']['backend'] == 'synthetic'
    assert len(report['episodes']) == 1
