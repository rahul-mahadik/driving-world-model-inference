"""CPU publication fixtures; invented timings are never scientific evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from drivebench_tuning.adapters import OperatorEvaluator
from drivebench_tuning.benchmark import EvidenceJournal
from drivebench_tuning.confirmation import confirm, selection_pairs
from drivebench_tuning.generalization import digest, train_cost_model, RidgeSelector, validate_cost_model
from drivebench_tuning.generalization_analysis import freeze_modal_selections
from drivebench_tuning.generalization_analysis import transfer_analysis
from drivebench_tuning import table_collection as tc
from drivebench_tuning.replay import MeasuredTable
from drivebench_tuning.replay_cli import write_json
from drivebench_tuning.replay_policy import fixed_policy, run_replay_policy
from test_export_operator_table import raw, FixtureRunner, exporter, GPU_ID
from test_export_related_operator_results import related

_transfer_spec = importlib.util.spec_from_file_location('transfer_export_fixture',
    Path(__file__).resolve().parents[3] / 'scripts/export_operator_transfer.py')
transfer_export = importlib.util.module_from_spec(_transfer_spec)
_transfer_spec.loader.exec_module(transfer_export)


@pytest.fixture
def study(raw, tmp_path):
    raw_path, runs = raw
    public_dir = tmp_path / 'public-table'
    exporter.export(raw_path, runs, public_dir)
    table = MeasuredTable.load(raw_path)
    model = train_cost_model(table)
    model_path = tmp_path / 'model.json'; write_json(model_path, model)
    reports = [run_replay_policy(table, policy, name=name, repeats=2, trials=8)
               for name, policy in [('fixed', fixed_policy()),
                    ('ridge', RidgeSelector(model, suite=table.suite))]]
    paths = []
    for i, report in enumerate(reports):
        path = tmp_path / f'policy-{i}.json'; write_json(path, report); paths.append(path)
    selection = freeze_modal_selections(table, reports)
    selection_path = tmp_path / 'selection.json'; write_json(selection_path, selection)
    confirmations = []
    for i in range(2):
        directory = tmp_path / f'confirm-{i}'; journal = EvidenceJournal(directory)
        runner = FixtureRunner(journal, repetitions=3, warmups=5, max_seconds=600)
        selection_sha = exporter.sha(selection_path)
        journal.write('plan.json', {'repeat_index': i, 'repetitions': 3,
            'selection_document': selection, 'selection_sha256': selection_sha,
            'independent_of_policy_selection': True, 'selection_updates_permitted': False})
        results = confirm(OperatorEvaluator(runner), selection_pairs(selection, table.suite),
            repeat_index=i, seed=2901, expected_cohort_id=table.cohort_id,
            expected_input_digests=selection['expected_input_digests'], journal=journal)
        journal.write('result.json', {'backend': 'cuda', 'repeat_index': i,
            'cohort_id': table.cohort_id, 'suite_sha256': table.suite.sha256,
            'selection_sha256': selection_sha, 'results': results})
        journal.write('completed.json', {'completed': True, 'selected_pairs': len(results),
                                         'cuda_trials': len(results)*3})
        confirmations.append(directory)
    return dict(source_table=raw_path, public_table=public_dir/'table.json',
                trained_model=model_path, selections=selection_path,
                policy_reports=paths, confirmation_dirs=confirmations)


def test_model_modal_and_confirmation_roundtrip_rebinds_canonical_hashes(study, tmp_path):
    original_model = json.loads(study['trained_model'].read_text())
    original_selection = json.loads(study['selections'].read_text())
    sources = [study['source_table'], study['trained_model'], study['selections'], *study['policy_reports']]
    before = {p: p.read_bytes() for p in sources}
    output = tmp_path / 'derived'
    receipt = related.export_related(output=output, **study)
    public_table = MeasuredTable.load(study['public_table'])
    model = json.loads((output/'trained-model.json').read_text())
    validate_cost_model(model, public_table.suite, expected_table_sha256=public_table.sha256,
                        expected_cohort_id=public_table.cohort_id)
    for name in ('mean', 'scale', 'coefficients', 'intercept'):
        assert model[name] == original_model[name]
    assert model['training']['records'] == original_model['training']['records']
    reports = [json.loads((output/name).read_text()) for name in receipt['policy_report_artifacts']]
    ridge = reports[1]
    assert ridge['policy_configuration']['model_sha256'] == digest(model)
    assert digest(model) != digest(original_model)
    assert ridge['training']['externally_supplied_metadata']['training_table_sha256'] == public_table.sha256
    selected_path = output / receipt['frozen_modal_selection_artifact']
    selection = json.loads(selected_path.read_text())
    related.validate_modal(selection, public_table, reports)
    hashes = {r['policy_name']: digest(r) for r in reports}
    assert all(d['source_report_sha256'] == hashes[d['policy']] for d in selection['modal_decisions'])
    assert selection['frozen_at'] == original_selection['frozen_at']
    for name in receipt['confirmation_artifacts']:
        plan = json.loads((output/name/'plan.json').read_text())
        assert plan['selection_document'] == selection
        assert plan['selection_sha256'] == exporter.sha(selected_path)
    assert {p: p.read_bytes() for p in sources} == before
    assert receipt['trained_model_binding']['training_rerun'] is False
    assert receipt['modal_decisions_reconstructed_from_source_and_derived_reports'] is True
    text = '\n'.join(p.read_text() for p in output.rglob('*') if p.is_file())
    assert GPU_ID not in text and '/workspace/private' not in text


@pytest.mark.parametrize('mutation', ['model_weight', 'model_provenance', 'report_model_hash',
                                     'report_training_alias', 'modal_vote', 'modal_report_hash'])
def test_mismatched_model_or_modal_binding_fails_before_public_output(study, tmp_path, mutation):
    if mutation.startswith('model_'):
        path = study['trained_model']; doc = json.loads(path.read_text())
        if mutation == 'model_weight': doc['coefficients'][0] += 0.01
        else: doc['training']['training_cohort_id'] = 'f'*64
    elif mutation.startswith('report_'):
        path = study['policy_reports'][1]; doc = json.loads(path.read_text())
        if mutation == 'report_model_hash': doc['policy_configuration']['model_sha256'] = 'f'*64
        else: doc['training']['externally_supplied_metadata']['training_table_sha256'] = 'f'*64
    else:
        path = study['selections']; doc = json.loads(path.read_text())
        if mutation == 'modal_vote': doc['modal_decisions'][0]['votes'][0]['count'] += 1
        else: doc['modal_decisions'][0]['source_report_sha256'] = 'f'*64
    path.write_text(json.dumps(doc))
    output = tmp_path/'refused'
    with pytest.raises(ValueError): related.export_related(output=output, **study)
    assert not output.exists()


@pytest.fixture
def transfer_study(study, tmp_path, monkeypatch):
    class TargetRunner(FixtureRunner):
        def __init__(self, journal, *, repetitions, warmups, max_seconds):
            self.journal, self.repetitions, self.warmups = journal, repetitions, warmups
            self.runtime = {'gpu_uuid': '22345678-1234-5678-9abc-123456789abc',
                'source_hashes': tc.source_hashes(), 'gpu_name': 'Second CPU fixture only',
                'torch': 'fixture', 'driver_inventory': [
                    'GPU-22345678-1234-5678-9abc-123456789abc, 580.178.04']}
            self.cohort_id = tc._digest(self.runtime)
            journal.write('runtime.json', {**self.runtime, 'cohort_id': self.cohort_id})

    runs = []
    for index in range(2):
        monkeypatch.setattr(tc, '_PROCESS_INSTANCE_ID', f'target-cpu-fixture-{index}')
        directory = tmp_path / f'target-repeat-{index}'
        assert tc.collect(directory, repeat_index=index, planned_repeats=2, repetitions=3,
            task_splits=['test'], runner_factory=TargetRunner) == 0
        runs.append(directory)
    target_path = tmp_path/'target.json'
    tc.merge(runs, target_path)
    target_public_dir = tmp_path/'public-target'
    exporter.export(target_path, runs, target_public_dir)
    related_dir = tmp_path/'related'
    related.export_related(output=related_dir, **study)
    analysis = transfer_analysis(MeasuredTable.load(study['source_table']),
        MeasuredTable.load(target_path), json.loads(study['selections'].read_text()))
    analysis_path = tmp_path/'raw-analysis.json'; write_json(analysis_path, analysis)
    return dict(source_table=study['source_table'], target_table=target_path,
        public_source_table=study['public_table'], public_target_table=target_public_dir/'table.json',
        selections=study['selections'], raw_analysis=analysis_path, related_export=related_dir)


def test_transfer_export_preserves_test_subset_and_all_scientific_values(transfer_study, tmp_path):
    original = json.loads(transfer_study['raw_analysis'].read_text())
    sources = [p for p in transfer_study.values() if p.is_file()]
    before = {p: p.read_bytes() for p in sources}
    output = tmp_path/'public-transfer'
    receipt = transfer_export.export_transfer(output=output, **transfer_study)
    derived = json.loads((output/'analysis.json').read_text())
    for key in ('records', 'summary', 'suite_sha256', 'source_gpu', 'target_gpu'):
        assert derived[key] == original[key]
    assert derived['training_on_target'] is False and derived['selection_updated_on_target'] is False
    target = MeasuredTable.load(transfer_study['public_target_table'])
    assert target._document['protocol']['task_splits'] == ['test']
    assert len(target.measured_tasks) == len(target.suite.splits['test'])
    assert derived['target_table_sha256'] == target.sha256
    assert derived['source_table_sha256'] == MeasuredTable.load(transfer_study['public_source_table']).sha256
    assert {p: p.read_bytes() for p in sources} == before
    assert receipt['new_gpu_measurements'] == 0
    assert receipt['numeric_fields_and_trial_ids_unchanged'] is True
    text = '\n'.join(p.read_text() for p in output.rglob('*') if p.is_file())
    assert GPU_ID not in text and '/workspace/private' not in text


@pytest.mark.parametrize('mutation', ['analysis', 'selection', 'policy'])
def test_transfer_export_refuses_modified_analysis_or_public_bindings(transfer_study, tmp_path, mutation):
    if mutation == 'analysis':
        path = transfer_study['raw_analysis']
        doc = json.loads(path.read_text())
        doc['records'][0]['target_oracle_cost'] += 0.1
    else:
        directory = transfer_study['related_export']
        receipt = json.loads((directory/'export-receipt.json').read_text())
        if mutation == 'selection':
            path = directory/receipt['frozen_modal_selection_artifact']
            doc = json.loads(path.read_text())
            doc['modal_decisions'][0]['votes'][0]['count'] += 1
        else:
            path = directory/receipt['policy_report_artifacts'][0]
            doc = json.loads(path.read_text())
            doc['episodes'][0]['reward_sum'] += 1
    path.write_text(json.dumps(doc))
    output = tmp_path/'refused-transfer'
    with pytest.raises(ValueError):
        transfer_export.export_transfer(output=output, **transfer_study)
    assert not output.exists()
