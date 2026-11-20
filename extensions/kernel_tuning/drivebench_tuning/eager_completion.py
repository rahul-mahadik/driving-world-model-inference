#!/usr/bin/env python3
"""Verify one eager restart with the existing strict verifier; no inference."""
import argparse
import json
from pathlib import Path
from drivebench.config import file_sha256
from drivebench_tuning.restart_measurement import read_restart


def verify(plan_path, expected_sha, run_path, output):
    plan_path, run_path, output = map(Path, (plan_path, run_path, output))
    if file_sha256(plan_path) != expected_sha:
        raise ValueError('Plan hash differs')
    plan = json.loads(plan_path.read_text())
    if (plan.get('mode') != 'feasibility' or plan.get('warm_requests_per_process') != 1
            or len(plan.get('scene_ids', [])) != 1):
        raise ValueError('Exactly one full-workload feasibility scene/request required')
    config = plan.get('configs', {}).get('modern_eager', {})
    expected = {'steps': 50, 'total_frames': 25, 'conditioning_frames': 1,
                'height': 576, 'width': 1024, 'decode_chunk_size': 14, 'dtype': 'fp16'}
    if (config.get('model', {}).get('backend') != 'modern_vista'
            or any(config.get('generation', {}).get(key) != value for key, value in expected.items())
            or config.get('optimizations', {}).get('compile_denoiser') is not False):
        raise ValueError('Fixed full-resolution 50-step modern eager workload required')
    eager = [job for job in plan.get('schedule', []) if job.get('role') == 'modern_eager']
    if len(eager) != 1 or not isinstance(eager[0].get('run_id'), str):
        raise ValueError('Exactly one declared eager restart is required')
    if output.exists():
        raise FileExistsError('Never overwrite terminal evidence')
    row = read_restart(run_path, plan, expected_sha)
    if row.get('role') != 'modern_eager' or row.get('run_id') != eager[0]['run_id']:
        raise ValueError('Only completed eager evidence is admitted')
    result = {'status': 'single_eager_full_workload_execution_feasibility_only',
              'plan_sha256': expected_sha, 'hardware': plan['hardware'], 'row': row,
              'executed_roles': ['modern_eager'], 'compiled_execution_requested': False,
              'profile_execution_requested': False, 'comparison_complete': False,
              'comparison_plan_note': 'Existing plan retains both role definitions for source compatibility; only eager was admitted and verified.',
              'quality_validated': False, 'quality_preserving': False,
              'headline_eligible': False, 'noise_estimates': False,
              'speedup': None, 'new_inference_performed_by_verifier': False}
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args(argv); print(json.dumps(verify(a.plan,a.plan_sha256,a.run,a.output),allow_nan=False))


if __name__ == '__main__':
    main()
