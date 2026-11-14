import importlib.util
import json
from pathlib import Path

import pytest


def example():
    path = Path(__file__).parents[1] / 'examples' / 'compare_search_policies.py'
    spec = importlib.util.spec_from_file_location('comparison_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_preserves_equal_budget_and_balanced_order(tmp_path):
    report = example().run(tmp_path / 'run')
    assert report['synthetic'] and report['status'] == 'completed'
    assert not report['selected_config_confirmation_performed']
    assert not report['policy_superiority_established']
    for counts in report['counts_by_policy'].values():
        assert counts == {'baseline_calls': 4, 'candidate_calls': 16, 'failed_or_incorrect_trials': 0}
    plan = json.loads((tmp_path / 'run' / 'plan.json').read_text())
    for task in {item['task_index'] for item in plan['schedule']}:
        rows = [r for r in plan['schedule'] if r['task_index'] == task]
        assert [r['policy'] for r in rows] == ['grid', 'random', 'random', 'grid']
        assert rows[0]['episode_seed'] == rows[1]['episode_seed']
        assert rows[2]['episode_seed'] == rows[3]['episode_seed']
    assert len(list((tmp_path / 'run').glob('episode-*.json'))) == 8
    with pytest.raises(FileExistsError):
        example().run(tmp_path / 'run')


def test_comparison_requires_balanced_count_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match='even repeat'):
        example().run(tmp_path / 'unstarted', repeats=3)
    assert not (tmp_path / 'unstarted').exists()


def test_comparison_retains_partial_episode_and_has_no_completion(tmp_path, monkeypatch):
    module = example()
    class Interrupted(RuntimeError):
        report = {'status': 'aborted', 'evidence': 'partial fixture'}
    def fail(*args, **kwargs):
        raise Interrupted('intentional fixture failure')
    monkeypatch.setattr(module, 'evaluate_policy', fail)
    with pytest.raises(Interrupted):
        module.run(tmp_path / 'failed')
    failure = json.loads((tmp_path / 'failed' / 'failed.json').read_text())
    assert failure['partial_episode'] == Interrupted.report
    assert not (tmp_path / 'failed' / 'completed.json').exists()
