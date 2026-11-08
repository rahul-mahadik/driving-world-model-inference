from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from drivebench_tuning.benchmark import CudaEvaluator, EvidenceJournal, percentile
from drivebench_tuning.contract import KernelConfig, Measurement, Workload, DEFAULT_CONFIGS
from drivebench_tuning.study import comparison, load_workloads, run_sweep


def test_journal_preserves_partial_attempt_and_refuses_overwrite(tmp_path):
    journal = EvidenceJournal(tmp_path/'run')
    journal.write('plan.json', {'scope': 'unit test'})
    with pytest.raises(FileExistsError):
        journal.write('plan.json', {})
    with pytest.raises(FileExistsError):
        EvidenceJournal(tmp_path/'run')
    journal.append({'event': 'started', 'trial_id': 'partial'})
    journal.append({'event': 'failed', 'trial_id': 'partial'})
    rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert [r['sequence'] for r in rows] == [1,2]
    assert [r['event'] for r in rows] == ['started','failed']
    with pytest.raises(ValueError):
        journal.write('../bad.json', {})


def test_workload_binding_includes_settings_seed_and_dtype():
    w = Workload('geglu',(2,256))
    assert w.key() == Workload('geglu',[2,256]).key()
    assert w.key() != replace(w,seed=1).key()
    assert w.key() != replace(w,dtype='float32').key()
    assert w.key() != replace(w,eps=1e-4).key()


@pytest.mark.parametrize('kwargs', [dict(shape=(2**30,2)),dict(shape=(2,3)),dict(dtype='int8'),dict(seed=-1)])
def test_bad_workload_rejected_before_allocation(kwargs):
    arguments = {'operator':'geglu','shape':(2,256),**kwargs}
    with pytest.raises(ValueError):
        Workload(**arguments)


def test_supported_configs_match_kernel_contract():
    from drivebench_tuning.kernels import SUPPORTED_BLOCK_SIZES,SUPPORTED_NUM_WARPS
    assert all(c.block_size in SUPPORTED_BLOCK_SIZES and c.num_warps in SUPPORTED_NUM_WARPS for c in DEFAULT_CONFIGS)
    with pytest.raises(ValueError):
        KernelConfig(256,16)


def test_percentile_linear_interpolation():
    assert percentile([1,2,4],.5) == 2
    assert percentile([1,2,4],.95) == pytest.approx(3.8)


def row(**kwargs):
    return Measurement(**{'workload_key':'workload','config':None,'status':'ok','latency_ms':2.,
        'peak_allocated_bytes':4096,'correct':True,'backend':'cuda',
        'evidence':{'cohort_id':'test cohort','input_sha256':['test hash']},**kwargs})


def test_comparison_requires_real_cohort_correctness_and_inputs():
    baseline=row()
    candidate=row(config=KernelConfig(256,4),latency_ms=1.,peak_allocated_bytes=2048)
    assert comparison(baseline,candidate)['memory_reduced']
    assert comparison(baseline,candidate)['latency_reduction_percent']==50
    for bad in [replace(candidate,backend='synthetic'),replace(candidate,workload_key='different'),
                replace(candidate,evidence={}),replace(candidate,evidence={'cohort_id':'different','input_sha256':['test hash']}),
                replace(candidate,evidence={'cohort_id':'test cohort','input_sha256':['different']})]:
        with pytest.raises(ValueError): comparison(baseline,bad)
    assert not comparison(baseline,replace(candidate,status='incorrect',correct=False))['comparable']


def test_no_cuda_fallback(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError,match='NVIDIA CUDA GPU'):
        CudaEvaluator(EvidenceJournal(tmp_path/'run'))


def test_cli_records_setup_failure_without_success_artifact(tmp_path, monkeypatch):
    import torch
    from drivebench_tuning.cli import main
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    config=tmp_path/'workloads.json'
    config.write_text(json.dumps({'workloads':[{'operator':'geglu','shape':[2,256]}]}))
    output=tmp_path/'failed-run'
    assert main(['benchmark','--workloads',str(config),'--output',str(output)]) == 1
    assert json.loads((output/'failed.json').read_text())['completed'] is False
    assert not (output/'completed.json').exists()
    assert (output/'plan.json').is_file()


def test_checked_in_workload_documents_are_valid():
    for path in (Path(__file__).parents[1]/'configs').glob('*.json'):
        doc, workloads=load_workloads(path)
        assert doc['provenance']
        assert workloads


def test_sweep_confirmation_is_fresh_and_can_reject_apparent_memory_win(tmp_path):
    # Deterministic fixture for orchestration bookkeeping only, never written as
    # an executed CUDA study. Production CudaEvaluator has no fixture backend.
    class Fixture:
        cohort_id='unit-test-fixture'
        def __init__(self):
            self.journal=EvidenceJournal(tmp_path/'test')
            self.count=0
        def measure(self,w,c,implementation):
            self.count+=1
            # Three roles x two rounds, then independent confirmation worsens.
            memory=1000 if implementation!='triton' or self.count>6 else 500
            return row(workload_key=w.key(),config=c,latency_ms=1. if c else 2.,peak_allocated_bytes=memory,
                       evidence={'cohort_id':self.cohort_id,'input_sha256':['fixture'],'trial_id':str(self.count),
                                 'fixture_not_hardware':True})
    evaluator=Fixture()
    result=run_sweep(evaluator,[Workload('geglu',(2,256))],configs=[KernelConfig(256,4)],rounds=2)
    assert evaluator.count==12
    assert not result['confirmations'][0]['memory_reduction_validated']
    assert not result['whole_model_kernel_benefit_validated']
