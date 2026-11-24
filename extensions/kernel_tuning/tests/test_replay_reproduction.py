"""A new CPU replay must preserve the measured dataset's older provenance."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json

import pytest

from drivebench_tuning import generalization_cli as cli
from drivebench_tuning.generalization import digest
from test_measured_replay import table


def test_replay_cli_accepts_later_source_without_relabeling_collection(table, tmp_path, monkeypatch):
    pytest.importorskip('optuna')
    path = tmp_path / 'table.json'
    measured_bytes = path.read_bytes()
    historical_hashes = deepcopy(table._document['protocol']['source_hashes'])
    monkeypatch.setattr(cli, 'source_hashes', lambda: historical_hashes)
    historical = cli.make_protocol(table.suite)
    historical['frozen_at'] = '2000-01-01T00:00:00+00:00'
    historical_before = deepcopy(historical)

    # A later package includes unrelated registration and changed policy code.
    # The historical collection still records its actual earlier source.
    current = {'replay.py': 'a' * 64, 'registration.py': 'b' * 64}
    monkeypatch.setattr(cli, 'source_hashes', lambda: current)
    with pytest.raises(ValueError, match='installed source'):
        cli.run_comparison(table, historical, tmp_path / 'old-run-must-fail')
    assert not (tmp_path / 'old-run-must-fail').exists()

    output = tmp_path / 'new-replay'
    assert cli.main(['replay', '--suite', table.suite.document['suite_id'],
                     '--table', str(path), '--output', str(output)]) == 0
    protocol = json.loads((output / 'protocol.json').read_text())
    assert protocol['kind'] == 'operator_generalization_replay_protocol'
    assert protocol['source_hashes'] == current
    binding = protocol['replay_binding']
    assert binding['table_sha256'] == hashlib.sha256(measured_bytes).hexdigest()
    assert binding['collection_source_hashes'] == historical_hashes
    assert binding['collection_protocol_sha256'] == digest(table._document['protocol'])
    assert binding['cohort_id'] == table.cohort_id
    assert all(datetime.fromisoformat(protocol['frozen_at']) > datetime.fromisoformat(r['at'])
               for r in table._document['runs'])
    for name in protocol['cpu_policies']:
        report = json.loads((output / f'{name}.json').read_text())
        assert report['status'] == 'completed'
        assert report['replay_protocol_sha256'] == digest(protocol)
        assert 'frozen_study_protocol_sha256' not in report
        assert report['fresh_gpu_measurements'] == 0
        assert report['table_sha256'] == binding['table_sha256']
        assert len(report['episodes']) == len(table.suite.splits['test']) * 20
        assert all(len(episode['steps']) == 8 for episode in report['episodes'])
    assert historical == historical_before
    assert path.read_bytes() == measured_bytes
    assert table._document['protocol']['source_hashes'] == historical_hashes
    # A repeated invocation cannot overwrite its earlier replay binding/results.
    with pytest.raises(FileExistsError):
        cli.run_replay_comparison(table, output)


@pytest.mark.parametrize('field', ['table_sha256', 'cohort_id', 'collection_protocol_sha256',
                                  'collection_source_hashes'])
def test_replay_binding_rejects_changed_measurement_provenance(table, monkeypatch, field):
    monkeypatch.setattr(cli, 'source_hashes', lambda: {'current.py': 'c' * 64})
    protocol = cli.make_replay_protocol(table)
    protocol['frozen_at'] = '2026-11-24T00:00:00+00:00'
    assert cli.validate_replay_protocol(protocol, table) == protocol
    protocol['replay_binding'][field] = {} if field.endswith('hashes') else '0' * 64
    with pytest.raises(ValueError, match='recorded table'):
        cli.validate_replay_protocol(protocol, table)


def test_replay_binding_rejects_source_drift_after_its_own_freeze(table, monkeypatch):
    monkeypatch.setattr(cli, 'source_hashes', lambda: {'current.py': 'c' * 64})
    protocol = cli.make_replay_protocol(table)
    protocol['frozen_at'] = '2026-11-24T00:00:00+00:00'
    monkeypatch.setattr(cli, 'source_hashes', lambda: {'current.py': 'd' * 64})
    with pytest.raises(ValueError, match='installed source'):
        cli.validate_replay_protocol(protocol, table)


def test_replay_rejects_incompatible_measurement_policy_before_training(table, tmp_path, monkeypatch):
    def no_training(*args, **kwargs):
        raise AssertionError('Training started before collection-policy validation')
    monkeypatch.setattr(cli, 'train_cost_model', no_training)
    del table._document['protocol']['compiler_state_policy']
    output = tmp_path / 'invalid-replay'
    with pytest.raises(ValueError, match='compiler-state policy'):
        cli.run_replay_comparison(table, output)
    assert not output.exists()
