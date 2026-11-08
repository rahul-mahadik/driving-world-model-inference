import copy
import json
from pathlib import Path
import pytest
from drivebench_tuning.suite import load_suite


def test_packaged_suite_has_disjoint_shapes_and_compiled_control():
    suite = load_suite()
    assert len(suite.tasks) == 12 and len(suite.candidates) == 9
    assert [len(suite.splits[x]) for x in ('train','validation','test')] == [6,2,4]
    assert suite.document['default_trial_budget'] < len(suite.candidates)
    assert suite.candidates[0].to_dict()['config'] == {'implementation':'compiled'}
    assert len({len(t.features) for t in suite.tasks}) == 1
    assert len(suite.sha256) == 64


def test_seeds_do_not_make_heldout_shapes(tmp_path):
    doc = copy.deepcopy(load_suite().document)
    duplicate = copy.deepcopy(doc['tasks'][0])
    duplicate['id'] = 'new-seed-test'
    duplicate['split'] = 'test'
    duplicate['workload']['seed'] += 10
    duplicate['workload']['dtype'] = 'bfloat16'
    duplicate['workload']['num_groups'] = 16  # unused in GEGLU; cannot create a new shape group
    doc['tasks'].append(duplicate)
    path = tmp_path/'bad.json'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='overlaps'):
        load_suite(path)


@pytest.mark.parametrize('change', ['duplicate_id', 'duplicate_config', 'exhaustive_budget', 'nonfinite_weight'])
def test_invalid_suite_rejected(tmp_path, change):
    doc = copy.deepcopy(load_suite().document)
    if change == 'duplicate_id':
        doc['tasks'][1]['id'] = doc['tasks'][0]['id']
    elif change == 'duplicate_config':
        doc['candidates'][1]['config'] = doc['candidates'][0]['config']
    elif change == 'exhaustive_budget':
        doc['default_trial_budget'] = len(doc['candidates'])
    else:
        doc['default_memory_weight'] = float('nan')
    path = tmp_path/'bad.json'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_suite(path)


def test_suite_documents_are_detached_from_frozen_protocol():
    suite = load_suite()
    suite.document["tasks"][0]["workload"]["shape"][0] = 999
    suite.splits["test"] = (0,)
    assert suite.document["tasks"][0]["workload"]["shape"][0] != 999
    assert 0 not in suite.splits["test"]


@pytest.mark.parametrize("payload", ['{"schema_version":1,"x":NaN}', '{"schema_version":1,"x":1e999}', '{"schema_version":1,"schema_version":1}', '[]', '{"schema_version":true}'])
def test_malformed_json_rejected(tmp_path, payload):
    p = tmp_path / "malformed.json"
    p.write_text(payload)
    with pytest.raises(ValueError):
        load_suite(p)
