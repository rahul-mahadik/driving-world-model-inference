import json
import pytest
from drivebench_tuning.search_cli import main


def test_synthetic_cli_pipeline_and_no_overwrite(tmp_path):
    out=tmp_path/'run'
    assert main(['run','--backend','synthetic','--output',str(out),'--repeats','1']) == 0
    plan=json.loads((out/'plan.json').read_text())
    done=json.loads((out/'completed.json').read_text())
    assert done['synthetic'] and done['completed'] and not done['rl_improvement_validated']
    assert plan['effective_trial_budget'] == 4
    assert (out/'result.json').is_file() and (out/'environment-journals.json').is_file()
    assert len((out/'trials.jsonl').read_text().splitlines()) == 10
    with pytest.raises(FileExistsError):
        main(['run','--backend','synthetic','--output',str(out)])


def test_export_portable_suite_and_no_overwrite(tmp_path):
    path=tmp_path/'suite.json'
    assert main(['export-suite','--output',str(path)]) == 0
    assert json.loads(path.read_text())['suite_id'] == 'operator-search-v1'
    with pytest.raises(FileExistsError): main(['export-suite','--output',str(path)])


def test_reject_exhaustive_search_before_creating_run(tmp_path):
    out=tmp_path/'run'
    with pytest.raises(ValueError,match='smaller'):
        main(['run','--backend','synthetic','--trials','9','--output',str(out)])
    assert not out.exists()
