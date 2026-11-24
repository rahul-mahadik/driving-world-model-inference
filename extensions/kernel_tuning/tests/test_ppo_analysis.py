"""Analysis contracts using explicit fixtures, never measured GPU evidence."""
from copy import deepcopy

import pytest

from drivebench_tuning.native_autotune import PROTOCOL
from drivebench_tuning.ppo_analysis import family_interval, live_comparison, replay_comparison
from drivebench_tuning.suite import load_suite
from test_native_audit import make_recorded_task


@pytest.fixture(scope='module')
def suite():
    return load_suite('operator-generalization-v2')


def test_bootstrap_resamples_six_shape_families_not_eighteen_dtypes(suite):
    full, representative, family_values = {}, {}, {}
    for index in suite.splits['test']:
        task = suite.tasks[index]
        workload = task.to_dict()['metadata']['workload']
        family = (workload['operator'], tuple(workload['shape']))
        if family not in family_values:
            family_values[family] = len(family_values)
            representative[task.task_id] = family_values[family]
        full[task.task_id] = family_values[family]
    all_dtypes = family_interval(full, suite)
    one_per_family = family_interval(representative, suite)
    assert all_dtypes['shape_families'] == 6
    assert all_dtypes['tasks'] == 18
    assert one_per_family['tasks'] == 6
    # Copying each family's value to all three dtypes must not shrink its CI
    # as though those copies were eighteen independently sampled shapes.
    for key in ('estimate', 'lower_95', 'upper_95'):
        assert all_dtypes[key] == pytest.approx(one_per_family[key])


@pytest.mark.parametrize('geometric', [False, True])
def test_seeded_family_bootstrap_does_not_depend_on_task_insertion_order(suite, geometric):
    values = {suite.tasks[index].task_id: (position + 1) ** 2 / 17
              for position, index in enumerate(suite.splits['test'])}
    reversed_values = dict(reversed(list(values.items())))
    # Different report/set iteration orders must produce exactly the same
    # seeded interval, including when unequal dtype values share a family.
    assert family_interval(values, suite, geometric=geometric) == family_interval(
        reversed_values, suite, geometric=geometric)


@pytest.fixture
def live_documents(suite):
    documents = []
    for repeat in range(3):
        results = []
        for index in suite.splits['test']:
            row, cohort, _ = make_recorded_task(suite, index, prefix=f'fixture-{repeat}-{index}')
            for role, latency in [('eager', 3.), ('compiled', 1.5), ('native', 2.), ('ppo', 1.)]:
                # Independent confirmation outcomes preserve the original known
                # paired ratios; the complete selection trajectories stay intact.
                measurement = row['confirmation']['measurements'][role]
                measurement['latency_ms'] = latency
                measurement['evidence']['cuda_event_ms'] = [latency] * 30
            results.append(row)
        documents.append({'repeat_index': repeat, 'cohort_id': cohort,
                          'suite_sha256': suite.sha256, 'protocol': deepcopy(PROTOCOL), 'results': results})
    return documents


def test_live_summary_keeps_confirmation_latency_separate_from_regret(suite, live_documents):
    result = live_comparison(live_documents, suite)
    assert result['counts']['confirmation_passed'] == 216
    assert result['counts']['native_config_trials'] == 1296
    assert result['counts']['ppo_candidate_queries'] == 432
    assert result['live_regret_reported'] is False
    assert result['equal_search_budget'] is False
    assert result['ppo_speedup_ratios']['native']['estimate'] == pytest.approx(2.)
    assert result['ppo_speedup_ratios']['native_with_fallback']['estimate'] == pytest.approx(1.5)


def test_confirmation_summary_rejects_median_not_supported_by_samples(suite, live_documents):
    row = live_documents[0]['results'][0]['confirmation']['measurements']['native']
    row['latency_ms'] = .001  # A favorable scalar cannot override its raw samples.
    with pytest.raises(ValueError, match='sample median'):
        live_comparison(live_documents, suite)


@pytest.mark.parametrize('mutation,match', [
    ('missing_repeat', 'three distinct'),
    ('cross_cohort', 'One physical'),
    ('reused_search_trial', 'trial ID missing or reused'),
    ('reused_confirmation_trial', 'trial ID missing or reused'),
    ('fallback_selected_after_confirmation', 'fallback selected after confirmation'),
])
def test_live_comparison_rejects_incomplete_or_nonindependent_evidence(suite, live_documents, mutation, match):
    first = live_documents[0]['results'][0]
    if mutation == 'missing_repeat':
        live_documents.pop()
    elif mutation == 'cross_cohort':
        live_documents[2]['cohort_id'] = 'a-different-physical-gpu'
    elif mutation == 'reused_search_trial':
        first['confirmation']['measurements']['ppo']['trial_id'] = first['selectors']['ppo']['turns'][0]['info']['measurement']['trial_id']
    elif mutation == 'reused_confirmation_trial':
        first['confirmation']['measurements']['ppo']['trial_id'] = first['confirmation']['measurements']['native']['trial_id']
    else:
        first['selectors']['native']['reference_fallback']['selected_before_confirmation'] = False
    with pytest.raises(ValueError, match=match):
        live_comparison(live_documents, suite)


def _replay_report(suite, name, seed_effect):
    episodes = []
    for task_position, index in enumerate(suite.splits['test']):
        task_id = suite.tasks[index].task_id
        # Shared task/repeat effects are substantial but cancel in paired means.
        for repeat in range(20):
            regret = seed_effect + task_position * .001 + repeat * .0001
            episodes.append({'task_id': task_id, 'repeat': repeat, 'status': 'completed',
                'steps': [{}] * 8, 'table_analysis': {'curve': [{'table_regret': regret}] * 8},
                'budget_analysis': {'repeated_action_turns': 0, 'failed_or_rejected_turns': 0}})
    return {'policy_name': name, 'status': 'completed', 'split': 'test', 'suite_sha256': suite.sha256,
            'backend': 'measured_replay', 'fresh_gpu_measurements': 0, 'repeats': 20, 'seed': 2101,
            'protocol': {'trial_budget': 8}, 'table_sha256': 'fixture-table-only', 'episodes': episodes,
            'rl_training_performed': name.startswith('ppo-seed-')}


@pytest.fixture
def replay_reports(suite):
    return [_replay_report(suite, f'ppo-seed-{seed}', .05 + (seed - 3101) * .01)
            for seed in range(3101, 3106)] + [_replay_report(suite, 'random', .1)]


def test_replay_averages_all_five_checkpoints_then_pairs_workload_means(suite, replay_reports):
    result = replay_comparison(replay_reports, suite)
    paired = result['paired_comparisons'][0]['ppo_minus_baseline_regret']
    assert result['ppo_training_seeds'] == 5
    assert result['fresh_gpu_measurements'] == 0
    # Average PPO seed effect is .07; random is .10. Selecting the best PPO
    # seed would incorrectly produce -.05 instead of the correct -.03.
    assert paired['estimate'] == pytest.approx(-.03)
    assert paired['lower_95'] == pytest.approx(-.03)
    assert paired['upper_95'] == pytest.approx(-.03)
    assert paired['shape_families'] == 6


def test_replay_refuses_missing_training_seed(suite, replay_reports):
    replay_reports.pop(4)
    with pytest.raises(ValueError, match='All five'):
        replay_comparison(replay_reports, suite)


def test_replay_refuses_duplicated_draw_instead_of_twenty_fresh_indices(suite, replay_reports):
    replay_reports[0]['episodes'][1]['repeat'] = 0
    with pytest.raises(ValueError, match='twenty replay draws'):
        replay_comparison(replay_reports, suite)
