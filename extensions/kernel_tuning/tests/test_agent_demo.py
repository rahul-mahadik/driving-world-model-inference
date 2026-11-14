import json
from drivebench_tuning.agent_demo import run_demo


def test_complete_scripted_cpu_episode_with_real_arrays(tmp_path):
    report=run_demo(tmp_path/'demo')
    assert report['final_state']['status']=='completed'
    assert report['agent_loop']['turns_used']==4
    assert report['quality_calibration_status']=='ready'
    assert not report['real_quality_calibrated']
    assert not report['real_gpu_validation'] and not report['llm_demonstration']
    final=json.loads((tmp_path/'demo'/'final-quality.json').read_text())
    assert final['status']=='pass' and final['record_count']==8
    assert not final['quality_preserving_claim']
    assert len(list((tmp_path/'demo'/'session'/'agent-actions').glob('*-finished.json')))==4
    assert list((tmp_path/'demo').rglob('*.npy'))
