"""CPU fixtures for public binding conversion; no model/API/GPU calls occur."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from drivebench_tuning.adapters import OperatorEvaluator
from drivebench_tuning.benchmark import EvidenceJournal
from drivebench_tuning.confirmation import confirm, selection_pairs
from drivebench_tuning.replay import MeasuredTable
from drivebench_tuning.replay_cli import selection_manifest, write_json
from drivebench_tuning.replay_policy import fixed_policy, run_replay_policy
from test_export_operator_table import raw, FixtureRunner, exporter, GPU_ID, PRIVATE_ERROR

SCRIPT = Path(__file__).resolve().parents[3] / 'scripts/export_related_operator_results.py'
spec = importlib.util.spec_from_file_location('related_operator_public_export', SCRIPT)
related = importlib.util.module_from_spec(spec)
spec.loader.exec_module(related)


@pytest.fixture
def studies(raw, tmp_path):
    raw_table, raw_directories = raw
    public_dir = tmp_path / 'public-table'
    exporter.export(raw_table, raw_directories, public_dir)
    table = MeasuredTable.load(raw_table)
    report = run_replay_policy(table, fixed_policy(), name='CPU-fixture-policy', repeats=2)
    report['policy_api_receipts'] = [{'response_id': 'resp_private_fixture_123', 'usage': {'input_tokens': 47},
                                    'audit': {'error': {'type': 'FixtureError', 'message': PRIVATE_ERROR}}}]
    report_path = tmp_path / 'raw-policy.json'
    write_json(report_path, report)
    selections = selection_manifest(table, [report])
    # The original input is deliberately not in the exporter's serialization.
    selection_bytes = json.dumps(selections, indent=1).encode()
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    (tmp_path / 'raw-selection.json').write_bytes(selection_bytes)
    directories = []
    for index in range(2):
        directory = tmp_path / f'raw-confirmation-{index}'
        journal = EvidenceJournal(directory)
        runner = FixtureRunner(journal, repetitions=3, warmups=5, max_seconds=600)
        journal.write('plan.json', {'repeat_index': index, 'repetitions': 3,
                      'selection_document': selections, 'selection_sha256': selection_sha,
                      'independent_of_policy_selection': True, 'selection_updates_permitted': False})
        results = confirm(OperatorEvaluator(runner), selection_pairs(selections, table.suite),
                          repeat_index=index, seed=2901, expected_cohort_id=table.cohort_id,
                          expected_input_digests=selections['expected_input_digests'], journal=journal)
        journal.write('result.json', {'backend': 'cuda', 'repeat_index': index,
                      'cohort_id': table.cohort_id, 'suite_sha256': table.suite.sha256,
                      'selection_sha256': selection_sha, 'results': results})
        journal.write('completed.json', {'completed': True, 'selected_pairs': len(results),
                                         'cuda_trials': len(results) * 3})
        directories.append(directory)
    return raw_table, public_dir / 'table.json', report_path, directories


def test_complete_related_roundtrip_preserves_values_and_rebinds_selection(studies, tmp_path):
    source_table, public_table, policy, directories = studies
    sources = [source_table, public_table, policy] + [p for d in directories for p in d.iterdir()]
    before = {p: p.read_bytes() for p in sources}
    target = tmp_path / 'public-related'
    receipt = related.export_related(source_table, public_table, target,
                                     policy_reports=[policy], confirmation_dirs=directories)
    assert receipt['measurements_rerun'] is False and receipt['policy_calls_rerun'] is False
    assert receipt['numeric_fields_and_trial_ids_unchanged'] is True
    assert receipt['source_artifact_hashes_unchanged'] is True
    assert receipt['source_and_derived_plot_validation_passed'] is True
    assert receipt['policy_report_count'] == 1 and receipt['confirmation_repeat_count'] == 2
    assert len(receipt['selection_bindings']) == 1
    assert receipt['redactions'] == {'error_messages': 1, 'provider_response_ids': 1}
    raw = MeasuredTable.load(source_table)
    public = MeasuredTable.load(public_table)
    curves, _ = related.plots.policy_curves([policy], raw)
    public_policies = [target / p for p in receipt['policy_report_artifacts']]
    public_confirmations = [target / p for p in receipt['confirmation_artifacts']]
    assert related.plots.policy_curves(public_policies, public)[0] == curves
    assert related.plots.confirmation_points(directories, raw)[0] == related.plots.confirmation_points(public_confirmations, public)[0]
    binding = receipt['selection_bindings'][0]
    selection = target / binding['artifact']
    assert exporter.sha(selection) == binding['derived_selection_sha256']
    assert binding['source_selection_sha256_as_recorded'] != binding['derived_selection_sha256']
    for directory in public_confirmations:
        plan = json.loads((directory / 'plan.json').read_text())
        result = json.loads((directory / 'result.json').read_text())
        assert plan['selection_sha256'] == result['selection_sha256'] == exporter.sha(selection)
        assert related.canonical_bytes(plan['selection_document']) == selection.read_bytes()
        assert plan['selection_document']['table_sha256'] == exporter.sha(public_table)
    assert {p: p.read_bytes() for p in sources} == before
    for entry in receipt['artifacts']:
        assert exporter.sha(target / entry['artifact']) == entry['derived_sha256']
    text = '\n'.join(p.read_text() for p in target.rglob('*') if p.is_file())
    for private in [GPU_ID, 'resp_private_fixture_123', '/workspace/private', '10.1.2.3', 'pod-123456']:
        assert private not in text


def test_tampered_policy_outcome_is_rejected_before_export(studies, tmp_path):
    source, public, policy, directories = studies
    report = json.loads(policy.read_text())
    report['episodes'][0]['steps'][0]['info']['measurement']['latency_ms'] = 999.0
    policy.write_text(json.dumps(report))
    target = tmp_path / 'refused'
    with pytest.raises(ValueError, match='recorded source'):
        related.export_related(source, public, target, policy_reports=[policy], confirmation_dirs=directories)
    assert not target.exists()


def test_tampered_confirmation_selection_binding_rejected_before_export(studies, tmp_path):
    source, public, policy, directories = studies
    path = directories[0] / 'result.json'
    report = json.loads(path.read_text())
    report['selection_sha256'] = 'f' * 64
    path.write_text(json.dumps(report))
    target = tmp_path / 'refused'
    with pytest.raises(ValueError, match='selection binding'):
        related.export_related(source, public, target, policy_reports=[policy], confirmation_dirs=directories)
    assert not target.exists()


def test_public_table_must_be_derived_from_supplied_raw_table(studies, tmp_path):
    source, public, policy, directories = studies
    document = json.loads(public.read_text())
    document['public_export']['source_table_sha256'] = 'f' * 64
    public.write_text(json.dumps(document))
    target = tmp_path / 'refused'
    with pytest.raises(ValueError, match='bound to this raw table'):
        related.export_related(source, public, target, policy_reports=[policy], confirmation_dirs=directories)
    assert not target.exists()


def test_numeric_and_trial_identity_guard_detects_changes():
    source = {'trials': [{'trial_id': 'kept', 'latency_ms': 1.2, 'correct': True, 'count': 2}]}
    with pytest.raises(ValueError, match='numerical field or measurement identifier'):
        related.unchanged(source, {'trials': [{'trial_id': 'changed', 'latency_ms': 1.2, 'correct': True, 'count': 2}]})
    with pytest.raises(ValueError, match='numerical field or measurement identifier'):
        related.unchanged(source, {'trials': [{'trial_id': 'kept', 'latency_ms': 1.3, 'correct': True, 'count': 2}]})
