"""Offline plot validation using explicitly watermarked CPU fixtures only."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from drivebench_tuning import table_collection as tc
from drivebench_tuning.adapters import OperatorEvaluator
from drivebench_tuning.benchmark import EvidenceJournal
from drivebench_tuning.confirmation import confirm, selection_pairs
from drivebench_tuning.evaluation import PolicyActionRejected
from drivebench_tuning.replay import MeasuredTable
from drivebench_tuning.replay_cli import selection_manifest, write_json
from drivebench_tuning.replay_policy import fixed_policy, run_replay_policy
from test_table_collection import FixtureRunner

pytest.importorskip('matplotlib')
script = Path(__file__).parents[3] / 'scripts' / 'plot_operator_search.py'
spec = importlib.util.spec_from_file_location('operator_plot_fixture', script)
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, 'source_hashes', lambda: {'fixture.py': 'f' * 64})
    directories = []
    for index in range(3):
        monkeypatch.setattr(tc, '_PROCESS_INSTANCE_ID', f'CPU-plot-test-{index}')
        path = tmp_path / f'repeat-{index}'
        assert tc.collect(path, repeat_index=index, repetitions=3, runner_factory=FixtureRunner) == 0
        directories.append(path)
    path = tmp_path / 'table.json'
    tc.merge(directories, path)
    table = MeasuredTable.load(path)
    report = run_replay_policy(table, fixed_policy(), name='CPU-fixture', repeats=2)
    report_path = tmp_path / 'policy.json'
    write_json(report_path, report)
    selections = selection_manifest(table, [report])
    confirmation = tmp_path / 'confirmation-0'
    journal = EvidenceJournal(confirmation)
    runner = FixtureRunner(journal, repetitions=3, warmups=5, max_seconds=600)
    selection_sha = hashlib.sha256(json.dumps(selections).encode()).hexdigest()
    journal.write('plan.json', {'repeat_index': 0, 'repetitions': 3, 'selection_document': selections,
                  'selection_sha256': selection_sha, 'independent_of_policy_selection': True,
                  'selection_updates_permitted': False})
    results = confirm(OperatorEvaluator(runner), selection_pairs(selections, table.suite),
                      repeat_index=0, seed=2901, expected_cohort_id=table.cohort_id,
                      expected_input_digests=selections['expected_input_digests'], journal=journal)
    journal.write('result.json', {'backend': 'cuda', 'repeat_index': 0, 'cohort_id': table.cohort_id,
                  'suite_sha256': table.suite.sha256, 'selection_sha256': selection_sha, 'results': results})
    journal.write('completed.json', {'completed': True, 'selected_pairs': len(results), 'cuda_trials': len(results) * 3})
    return table, path, report_path, confirmation


def test_plotter_refuses_fixture_unless_explicitly_watermarked(inputs, tmp_path):
    _, path, report, confirmation = inputs
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='fixture'):
        plots.main(['--table', str(path), '--output', str(output)])
    assert not output.exists()


def test_exported_figures_are_traceable_and_fixture_watermarked(inputs, tmp_path):
    _, path, report, confirmation = inputs
    output = tmp_path / 'fixture-figures'
    assert plots.main(['--table', str(path), '--policy-reports', str(report), '--confirmation-dirs', str(confirmation),
                       '--output', str(output), '--allow-test-fixtures']) == 0
    receipt = json.loads((output / 'analysis.json').read_text())
    assert receipt['test_fixture'] and receipt['new_gpu_measurements'] == 0
    assert receipt['table_process_repeats'] == 3
    assert len(receipt['figure_sha256']) == 6
    for name, sha in receipt['figure_sha256'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == sha
    for path in output.glob('*.svg'):
        assert 'CPU TEST FIXTURE' in path.read_text()
        assert 'CPU-CONTRACT-FIXTURE' not in path.read_text()


def test_changed_replay_outcome_and_duplicate_confirmation_rejected(inputs, tmp_path):
    table, _, report_path, confirmation = inputs
    report = json.loads(report_path.read_text())
    report['episodes'][0]['steps'][0]['info']['measurement']['latency_ms'] = 999
    changed = tmp_path / 'changed-report.json'
    write_json(changed, report)
    with pytest.raises(ValueError, match='recorded source'):
        plots.policy_curves([changed], table)
    with pytest.raises(ValueError, match='duplicated'):
        plots.confirmation_points([confirmation, confirmation], table)


def test_failures_mask_latency_cells_and_repeats_remain_distinct(inputs):
    table, _, report_path, confirmation = inputs
    matrix = plots.latency_matrix(table)
    failed = [cell for row in matrix for cell in row if cell['candidate_id'] == 'triton-b128-w8']
    assert len(failed) == 12 and all(cell['failed_repeats'] == 3 for cell in failed)
    assert all(cell['median_paired_latency_ratio'] is None for cell in failed)
    curves, _ = plots.policy_curves([report_path], table)
    assert curves[0]['episodes_per_task'] == 2 and curves[0]['task_count'] == 4
    points, _ = plots.confirmation_points([confirmation], table)
    assert len(points) == 4 and all(p['process_repeat'] == 0 for p in points)


def test_matched_comparison_counts_only_first_repeats_without_changing_sources(inputs, tmp_path):
    table, _, _, _ = inputs
    def different_outcomes(observation, info):
        if info['evaluation_episode_index'] % 2:
            raise PolicyActionRejected('CPU-fixture-refusal')
        return 2  # Recorded fixture candidate always fails.
    report = run_replay_policy(table, different_outcomes, name='CPU-two-repeat-fixture', repeats=2)
    path = tmp_path / 'two-repeats.json'
    write_json(path, report)
    original_bytes = path.read_bytes()
    complete, _ = plots.policy_curves([path], table)
    matched, _ = plots.policy_curves([path], table, comparison_repeats=1)
    assert complete[0]['episodes_per_task'] == complete[0]['available_episodes_per_task'] == 2
    assert complete[0]['failed_or_incorrect_trials'] == complete[0]['rejected_policy_turns'] == 16
    assert matched[0]['episodes_per_task'] == 1 and matched[0]['available_episodes_per_task'] == 2
    assert matched[0]['total_policy_episodes'] == 4 and matched[0]['available_total_policy_episodes'] == 8
    assert matched[0]['failed_or_incorrect_trials'] == 16 and matched[0]['rejected_policy_turns'] == 0
    assert path.read_bytes() == original_bytes


def test_matched_comparison_still_validates_excluded_episodes(inputs, tmp_path):
    table, _, report_path, _ = inputs
    report = json.loads(report_path.read_text())
    next(e for e in report['episodes'] if e['repeat'] == 1)['steps'][0]['info']['measurement']['latency_ms'] = 999
    path = tmp_path / 'corrupted-excluded-episode.json'
    write_json(path, report)
    with pytest.raises(ValueError, match='recorded source'):
        plots.policy_curves([path], table, comparison_repeats=1)


def test_matched_comparison_rejects_excessive_or_invalid_repeat_limit(inputs, tmp_path):
    table, _, report_path, _ = inputs
    short = run_replay_policy(table, fixed_policy(), name='CPU-one-repeat-fixture', repeats=1)
    path = tmp_path / 'one-repeat.json'
    write_json(path, short)
    with pytest.raises(ValueError, match='exceed'):
        plots.policy_curves([report_path, path], table, comparison_repeats=2)
    for count in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            plots.policy_curves([report_path], table, comparison_repeats=count)


def test_matched_cli_labels_plotted_and_available_counts(inputs, tmp_path):
    _, table_path, report_path, _ = inputs
    output = tmp_path / 'matched-fixture-plots'
    assert plots.main(['--table', str(table_path), '--policy-reports', str(report_path),
                       '--comparison-repeats', '1', '--output', str(output), '--allow-test-fixtures']) == 0
    receipt = json.loads((output / 'analysis.json').read_text())
    assert receipt['comparison_repeats'] == 1
    assert receipt['policy_curves'][0]['episodes_per_task'] == 1
    assert '1 plotted / 2 available per task' in (output / 'policy-regret-vs-budget.svg').read_text()
