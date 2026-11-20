import pytest

from drivebench_tuning.confirmation import confirm, selection_pairs, selection_input_digests
from drivebench_tuning.search import TrialResult
from drivebench_tuning.suite import load_suite


class Journal:
    def __init__(self):
        self.rows = []

    def append(self, row):
        self.rows.append(row)


class Evaluator:
    def __init__(self, *, cohort='c' * 64, wrong_candidate=False, incorrect=False):
        self.count = 0
        self.cohort = cohort
        self.wrong_candidate = wrong_candidate
        self.incorrect = incorrect

    def run(self, task, candidate):
        self.count += 1
        bad = self.incorrect and candidate is not None
        return TrialResult(str(self.count), task.task_id,
                           'unknown' if self.wrong_candidate else candidate.candidate_id if candidate else None,
                           'cuda', self.cohort, 'a' * 64,
                           'incorrect' if bad else 'ok', not bad,
                           None if bad else 2., None if bad else 12, {'contract_fixture': True})

    def baseline(self, task):
        return self.run(task, None)

    def evaluate(self, task, candidate):
        return self.run(task, candidate)


def pairs():
    suite = load_suite()
    return suite, selection_pairs({'suite_sha256': suite.sha256, 'selections': [
        {'task_id': suite.tasks[0].task_id, 'candidate_id': 'triton-b128-w4'},
        {'task_id': suite.tasks[0].task_id, 'candidate_id': 'triton-b128-w4'}]}, suite)


def test_fixed_selection_deduplicates_but_preserves_fresh_roles():
    suite, selected = pairs()
    assert len(selected) == 1
    journal = Journal()
    result = confirm(Evaluator(), selected, repeat_index=0, seed=1,
                     expected_cohort_id='c' * 64,
                     expected_input_digests={selected[0][0].task_id: 'a' * 64}, journal=journal)
    assert len(journal.rows) == 3
    assert len({r['measurement']['trial_id'] for r in journal.rows}) == 3
    assert result[0]['candidate_id'] == 'triton-b128-w4'
    assert result[0]['descriptive_speedup_vs'] == {'eager': 1., 'compiled': 1.}


@pytest.mark.parametrize('kwargs', [{'cohort': 'd' * 64}, {'wrong_candidate': True}])
def test_rejects_wrong_device_or_candidate(kwargs):
    _, selected = pairs()
    with pytest.raises(ValueError, match='differs'):
        confirm(Evaluator(**kwargs), selected, repeat_index=0, seed=1,
                expected_cohort_id='c' * 64,
                     expected_input_digests={selected[0][0].task_id: 'a' * 64}, journal=Journal())


def test_failed_correctness_does_not_get_speedup():
    _, selected = pairs()
    result = confirm(Evaluator(incorrect=True), selected, repeat_index=0, seed=1,
                     expected_cohort_id='c' * 64,
                     expected_input_digests={selected[0][0].task_id: 'a' * 64}, journal=Journal())
    assert result[0]['descriptive_speedup_vs'] == {'eager': None, 'compiled': None}


def test_rejects_mismatched_suite_and_unknown_selections():
    suite, _ = pairs()
    with pytest.raises(ValueError, match='suite'):
        selection_pairs({'suite_sha256': 'a' * 64, 'selections': []}, suite)
    with pytest.raises(ValueError, match='Unknown'):
        selection_pairs({'suite_sha256': suite.sha256, 'selections': [
            {'task_id': suite.tasks[0].task_id, 'candidate_id': 'fabricated'}]}, suite)


def test_consistent_but_different_fresh_inputs_are_rejected():
    _, selected = pairs()
    with pytest.raises(ValueError, match='original measured task'):
        confirm(Evaluator(), selected, repeat_index=0, seed=1, expected_cohort_id='c' * 64,
                expected_input_digests={selected[0][0].task_id: 'b' * 64}, journal=Journal())


@pytest.mark.parametrize('digests', [None, {}, {'train-geglu-1': 'invalid'}, {'train-geglu-1': 'i' * 64}])
def test_original_input_binding_required_before_execution(digests):
    _, selected = pairs()
    evaluator = Evaluator()
    with pytest.raises(ValueError, match='input digest|SHA256'):
        confirm(evaluator, selected, repeat_index=0, seed=1, expected_cohort_id='c' * 64,
                expected_input_digests=digests, journal=Journal())
    assert evaluator.count == 0
