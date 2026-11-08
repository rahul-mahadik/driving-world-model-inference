"""Offline per-frame localization of the two preserved first-denoiser inputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
root = Path(__file__).resolve().parents[2]
archive = root/'outputs/a100-repeatability-archive-20261027'
rows = json.loads((archive/'remote-inventory.json').read_text())
inputs = {}
for index in range(2):
    relative = f'outputs/a100-repeatability-v1/capture-{index}.npz'
    row = next(r for r in rows if r['path']==relative)
    path = archive/'files'/relative
    assert path.stat().st_size == row['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    with np.load(path,allow_pickle=False) as value: inputs[index]={k:value[k].copy() for k in value.files}
a,b = inputs[0],inputs[1]
assert a.keys() == b.keys()
assert a['x'].shape == b['x'].shape == (50,4,72,128)
per_frame=[]
for i in range(50):
    delta=a['x'][i].astype(np.float64)-b['x'][i].astype(np.float64)
    per_frame.append({'index':i,'different_values':int(np.count_nonzero(delta)), 'max_absolute_error':float(np.abs(delta).max()),'rmse':float(np.sqrt(np.mean(delta*delta)))})
result={'type':'offline_a100_first_input_frame_localization_v1','created_at':datetime.now(timezone.utc).isoformat(), 'source_artifacts':{r['path']:r['sha256'] for r in rows if r['path'].endswith(('capture-0.npz','capture-1.npz'))}, 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'per_frame_x':per_frame,'differing_frame_indices':[r['index'] for r in per_frame if r['different_values']], 'captured_half_0_equals_half_1':{str(i):bool(np.array_equal(inputs[i]['x'][:25],inputs[i]['x'][25:])) for i in range(2)},'other_inputs_and_rng_exact':{k:bool(np.array_equal(a[k],b[k])) for k in a if k!='x'},'gpu_execution':False,'numerical_acceptance':False,'causal_kernel_identification':False}
out=root/'outputs/a100-repeatability-analysis-v1';out.mkdir(exist_ok=False)
(out/'per-frame-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('per_frame_x','source_artifacts')},indent=2))
