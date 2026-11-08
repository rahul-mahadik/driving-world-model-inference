from dataclasses import replace
import pytest
from drivebench_tuning.adapters import OperatorEvaluator, SyntheticEvaluator
from drivebench_tuning.contract import Measurement, Workload
from drivebench_tuning.suite import load_suite


def test_fixture_is_always_explicitly_synthetic_and_fresh():
    suite=load_suite()
    e=SyntheticEvaluator()
    a=e.baseline(suite.tasks[0]); b=e.baseline(suite.tasks[0])
    assert a.backend == b.backend == 'synthetic'
    assert a.trial_id != b.trial_id and a.input_digest == b.input_digest
    assert a.evidence['invented_metrics']


class Runner:
    def baseline(self, w):
        return Measurement(w.key(),None,'ok',1.,100,True,'cuda',
                           {'trial_id':'fresh','cohort_id':'one','input_sha256':['a'*64], 'implementation':'eager'})
    def compiled_baseline(self,w):
        result=self.baseline(w)
        return replace(result,evidence={**result.evidence,'implementation':'compiled'})
    def evaluate(self,w,c):
        result=self.baseline(w)
        return replace(result,config=c,evidence={**result.evidence,'implementation':'triton'})
    def close(self): pass


def test_adapter_preserves_identity_and_selects_correct_implementation():
    suite=load_suite(); e=OperatorEvaluator(Runner()); t=suite.tasks[0]
    assert e.baseline(t).candidate_id is None
    for c in suite.candidates:
        result=e.evaluate(t,c)
        assert result.task_id == t.task_id and result.candidate_id == c.candidate_id
        assert result.evidence['implementation'] == c.config['implementation']


def test_adapter_rejects_mislabeled_candidate():
    class Wrong(Runner):
        def compiled_baseline(self,w): return self.baseline(w)
    suite=load_suite()
    with pytest.raises(ValueError,match='another candidate'):
        OperatorEvaluator(Wrong()).evaluate(suite.tasks[0],suite.candidates[0])
