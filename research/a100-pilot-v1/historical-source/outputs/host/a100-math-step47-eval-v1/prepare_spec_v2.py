#!/usr/bin/env python3
"""Prepare an immutable local comparison spec only from actual completed evidence."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('step47_comparison_contract', HERE / 'evaluate.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
BUILDER = 'outputs/host/a100-math-step47-v1/prepare_specs_v2.py'
BUILDER_SHA = '5abd5177e6e48974db055d65df23b711a994d939e6d305aeb928ab0ce035ad1c'
CPU_SPEC = 'outputs/host/a100-math-quality-eval-v1/feature2-compare-01-spec.json'
CPU_SPEC_SHA = '229d5d01ef9e6e56f59b650707163011a1f580238edef1671c17f33e312f1f84'
BASE_ARCHIVE = 'outputs/a100-math-development-eager50-evidence-20261028'
GT_ARCHIVE = 'outputs/a100-math-development-gt-evidence-20261028'
BASE_RESULT = 'outputs/a100-math-development-v1/runs/eager50'
BASE_HOST = 'outputs/host/a100-math-development-v1/run-01-host.json'
BASE_SPEC = 'outputs/host/a100-math-development-v1/run-01-spec.json'
BASE_SPEC_SHA = 'caa01d0b5bc004867f63925087e7c0f57023b98bf5505cfb3956123a7af469d6'
BASE_DONE_SHA = 'ffd613d8fb96e7214a55d4c484361cab5cf6650503587fc64355dd9003263209'
GT_REL = 'outputs/host/a100-math-development-eval-v1'
GT_RESULT = 'outputs/a100-math-development-eval-v1/eager50-recorded-future'
GT_SPEC_SHA = '24b703bcc52a5119a62aab85114b838c912699e52662df67966cf30f9a28486b'
GT_PRODUCER_SHA = '9bd233865f8ef9e12800a97a19e8c219f424edaf4edceb52dbb9d618026a5957'
ARCHIVE_PRODUCERS = {
    BASE_ARCHIVE: '5fc5c126c7122573d186283949f615fdb3577b4daed06735991fe0cc2d8b1395',
    GT_ARCHIVE: 'a8edb727c4a802e83f3a22cfb2f4d8b8ef033dd12795579f77d270027e472d5d',
}


def relative(value):
    if not isinstance(value, str) or any(c in value for c in ('\n', '\r', '\x00', '\\')):
        raise ValueError('Unsafe input path')
    if value.startswith('/workspace/'): value = value[len('/workspace/'):]
    p = PurePosixPath(value)
    if not value or value == '.' or p.is_absolute() or '..' in p.parts or str(p) != value:
        raise ValueError('Noncanonical or escaping input path')
    return value


def add(pins, path, checksum):
    path = relative(path)
    if (not isinstance(checksum, str) or len(checksum) != 64
            or any(c not in '0123456789abcdef' for c in checksum)
            or path in pins and pins[path] != checksum):
        raise ValueError('Invalid or conflicting immutable pin')
    pins[path] = checksum


def inventory(path):
    if Path(path).is_symlink(): raise ValueError('Inventory symlink forbidden')
    rows = json.loads(Path(path).read_text()); pins, sizes = {}, {}
    if not isinstance(rows, list) or not rows: raise ValueError('Actual remote inventory required')
    for row in rows:
        name = relative(row['path'])
        if name in pins or type(row.get('bytes')) is not int or row['bytes'] < 0:
            raise ValueError('Invalid or duplicate inventory row')
        add(pins, name, row['sha256']); sizes[name] = row['bytes']
    return pins, sizes


def copied(folder, pins, sizes, name):
    name = relative(name); path = folder / 'files' / name
    if (name not in pins or path.is_symlink() or not path.resolve().is_relative_to((folder / 'files').resolve())
            or path.stat().st_size != sizes[name] or m.sha(path) != pins[name]):
        raise ValueError('Actual metadata copy does not match remote inventory: ' + name)
    return path


def verified_archive(relative_folder):
    folder = ROOT / relative_folder
    if folder.is_symlink() or any(p.is_symlink() for p in folder.rglob('*')):
        raise ValueError('Verified archive cannot contain symlinks')
    inv = folder / 'remote-inventory.json'; receipt_path = folder / 'verification.json'
    receipt = m.read(receipt_path); pins, sizes = inventory(inv)
    if (receipt.get('passed') is not True or receipt.get('remote_before_after_equal') is not True
            or receipt.get('all_local_files_sha256_verified') is not True
            or receipt.get('remote_deletion_performed') is not False
            or receipt.get('producer_sha256') != ARCHIVE_PRODUCERS[relative_folder]
            or receipt.get('remote_inventory_sha256') != m.sha(inv)
            or receipt.get('files') != len(pins) or receipt.get('bytes') != sum(sizes.values())):
        raise ValueError('Full completed archive verification required; metadata-only receipt is insufficient')
    for name in pins: copied(folder, pins, sizes, name)
    return folder, pins, sizes, {'archive': relative_folder,
        'verification_sha256': m.sha(receipt_path), 'inventory_sha256': m.sha(inv),
        'files': len(pins), 'bytes': sum(sizes.values())}


def successful_host(q, host, spec, spec_sha, done):
    from drivebench.config import digest
    stamps = [q.epoch(host['started_at']), q.epoch(done['started_at']), q.epoch(done['finished_at']), q.epoch(host['finished_at'])]
    if (host.get('status') != 'completed' or host.get('exit_code') != 0
            or host.get('specification_sha256') != spec_sha or host.get('launcher_sha256') != q.LAUNCHER_SHA
            or host.get('source_identity') != m.SOURCE or digest(host.get('source_files')) != m.SOURCE
            or host.get('input_file_hashes') != spec.get('expected_file_hashes')
            or host.get('source_unchanged') is not True or host.get('inputs_unchanged') is not True
            or host.get('client_stop_error') is not None or not q.cleanup_complete(host.get('cleanup', []))
            or host.get('allocation_deadline_epoch') != m.DEADLINE
            or host.get('runtime_identity') != done.get('platform_gate', {}).get('runtime_identity')
            or stamps != sorted(stamps) or stamps[1] >= stamps[2] or stamps[-1] >= m.DEADLINE):
        raise ValueError('Actual successful host provenance and cleanup required')


def complete_inventory(pins, sizes, output, done, count):
    artifacts = done.get('result', {}).get('artifact_sha256') if count else done.get('artifact_sha256')
    if not isinstance(artifacts, dict) or not artifacts: raise ValueError('Completed artifact inventory required')
    wanted = {}
    for p, h in artifacts.items():
        if p.startswith('/'): raise ValueError('Artifact names must be relative to output')
        add(wanted, output + '/' + relative(p), h)
    actual = {p: h for p, h in pins.items() if p.startswith(output + '/') and p != output + '/completed.json'}
    if wanted != actual: raise ValueError('Remote result inventory differs from final completed inventory')
    arrays = [p for p in wanted if p.endswith('.npy')]
    if len(arrays) != count or any(sizes[p] != 176947328 for p in arrays):
        raise ValueError('Complete retained array count and sizes required')


def prerequisites(q):
    bf, bp, bs, br = verified_archive(BASE_ARCHIVE)
    gf, gp, gs, gr = verified_archive(GT_ARCHIVE)
    baseline = m.read(copied(bf, bp, bs, BASE_RESULT + '/completed.json'))
    if bp.get(BASE_RESULT + '/completed.json') != BASE_DONE_SHA or bp.get(BASE_SPEC) != BASE_SPEC_SHA:
        raise ValueError('Original completed32-baseline identity changed')
    complete_inventory(bp, bs, BASE_RESULT, baseline, 32)
    successful_host(q, m.read(copied(bf, bp, bs, BASE_HOST)), m.read(copied(bf, bp, bs, BASE_SPEC)), BASE_SPEC_SHA, baseline)
    done = m.read(copied(gf, gp, gs, GT_RESULT + '/completed.json'))
    host = m.read(copied(gf, gp, gs, GT_REL + '/run-01-host.json'))
    spec = m.read(copied(gf, gp, gs, GT_REL + '/run-01-spec.json'))
    if gp.get(GT_REL + '/run-01-spec.json') != GT_SPEC_SHA: raise ValueError('Reviewed32-GT specification differs')
    if (done.get('type') != 'a100_math_development_evaluation_v1' or done.get('status') != 'completed'
            or done.get('producer_sha256') != GT_PRODUCER_SHA or done.get('source_identity') != m.SOURCE
            or done.get('allocation_deadline_epoch') != m.DEADLINE or done.get('inputs_and_source_unchanged') is not True
            or done.get('platform_restored') is not True or done.get('cuda_initialized_before') is not False
            or done.get('cuda_initialized_after') is not False
            or any(done.get(k) is not False for k in ('quality_preserving','headline_eligible','h100_reference_eligible','tolerances_calibrated'))):
        raise ValueError('Actual completed CPU32-baseline GT evidence required')
    complete_inventory(gp, gs, GT_RESULT, done, 0); successful_host(q, host, spec, GT_SPEC_SHA, done)
    result = m.read(copied(gf, gp, gs, GT_RESULT + '/recorded-future.json'))
    if (result.get('pair_count') != 32 or result.get('scene_count') != 32
            or result.get('future_frames_per_pair') != 24 or result.get('quality_preserving') is not False):
        raise ValueError('Complete32-scene recorded-future diagnostics required')
    pins = {}
    for values in (bp, gp, spec['expected_file_hashes'], done['input_sha256']):
        for p, h in values.items(): add(pins, p, h)
    proof = {'type':'a100_step47_comparison_operator_prerequisites_v1',
        'source_identity':m.SOURCE,'allocation_deadline_epoch':m.DEADLINE,
        'baseline32_full_backup_verified':True,'baseline32_gt_completed':True,
        'automatic_launch_authorized':False,'quality_acceptance_claimed':False,
        'baseline_backup':br,'baseline_gt_backup':gr,
        'remote_artifact_sha256':dict(sorted(pins.items())),
        'fresh_provider_guard_budget_and_idle_checks_required_before_launch':True}
    return proof



RELATIVE_ADMISSION_ALIASES = frozenset({
    'outputs/host/a100-pilot-01/configs/baseline.yaml',
    'outputs/host/verify-a100-pilot-v1.py',
    'scripts/probe_candidate_memory.py',
    'scripts/probe_compiler_operator.py',
})


def actual_admission_inputs(admission, expected):
    """Validate exact normalized pins, then preserve the actual remote alias map.

    The frozen runtime recorded four relative aliases alongside their identical
    /workspace keys. Recreating only absolute keys loses real provenance.
    """
    values = admission.get('input_sha256')
    if not isinstance(values, dict): raise ValueError('Actual admission input map required')
    normalized = {}
    for path, checksum in values.items():
        if not isinstance(path, str): raise ValueError('Actual admission input path must be a string')
        if not path.startswith('/workspace/'):
            if path not in RELATIVE_ADMISSION_ALIASES or values.get('/workspace/' + path) != checksum:
                raise ValueError('Only known identical relative aliases of absolute admission pins are allowed')
        add(normalized, path, checksum)
    if normalized != expected['input_sha256']:
        raise ValueError('Actual normalized admission pins differ from exact frozen provenance')
    return copy.deepcopy(values)

def build(generation_archive):
    folder = Path(generation_archive)
    if folder.is_symlink(): raise ValueError('Actual47 archive path cannot be a symlink')
    pins, sizes = inventory(folder / 'remote-inventory.json')
    required_paths = (m.ADMISSION, m.GEN_SPEC, m.GEN_HOST, m.STEP_RESULT + '/completed.json')
    paths = [copied(folder, pins, sizes, p) for p in required_paths]
    admission, spec, host, done = [m.read(p) for p in paths]
    frozen = m.load(BUILDER, BUILDER_SHA, 'comparison_step47_spec_builder')
    s, _, expected = frozen.build_admission(); q = s.quality()
    expected_spec = frozen.run_spec(paths[0], now=q.epoch(done['started_at']))
    if spec != expected_spec: raise ValueError('Actual47 specification differs from frozen original builder')
    complete_inventory(pins, sizes, m.STEP_RESULT, done, 4)
    for p in pins:
        if not p.endswith('.npy'): copied(folder, pins, sizes, p)
    base_done = m.read(ROOT / 'outputs/a100-math-quality-eager50-evidence-20261028/files/outputs/a100-math-quality-v1/runs/eager50/completed.json')
    plan = NS(admitted=expected['platform_gate'],strict=expected['strict_prerequisite'],
        inputs=actual_admission_inputs(admission, expected),
        bound=expected['time_bound'],hypothesis=expected['hypothesis'],
        context=NS(runtime=NS(declaration=lambda _:base_done['algorithm_policy']),base=None))
    m.generation_documents(s,q,plan,admission,done,spec,host,
        admission_sha=m.sha(paths[0]),spec_sha=m.sha(paths[1]),actual=done['result'])
    proof = prerequisites(q)
    if m.sha(ROOT / CPU_SPEC) != CPU_SPEC_SHA: raise ValueError('Successful CPU comparison template changed')
    cpu = m.read(ROOT / CPU_SPEC)
    merged = {}
    for values in (spec['expected_file_hashes'],pins,cpu['expected_file_hashes'],proof['remote_artifact_sha256']):
        for p,h in values.items(): add(merged,p,h)
    for p,h in {m.REL+'/evaluate.py':m.sha(HERE/'evaluate.py'),m.STEP:m.STEP_SHA,m.EVAL:m.EVAL_SHA}.items(): add(merged,p,h)
    # Hash the exact bytes save() will publish, avoiding a fabricated prerequisite pin.
    payload=(json.dumps(proof,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    import hashlib
    add(merged,m.OPERATOR,hashlib.sha256(payload).hexdigest())
    result=copy.deepcopy(spec)
    result.update(metadata=m.REL+'/compare-01-host.json',container_name='drivebench-a100-math-step47-compare-01',
        fresh_outputs=[m.DEST],timeout_seconds=m.OUTER,expected_file_hashes=dict(sorted(merged.items())),
        python_arguments=[m.REL+'/evaluate.py','--expected-sha256',m.sha(HERE/'evaluate.py'),'--output','/workspace/'+m.DEST],
        scope=m.SCOPE+' Fixed1800s outer/1680s inner plus1200s archive reserve. Fresh idle/provider/guard/disk admission required; no automatic launch.')
    return proof,result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generation-archive',type=Path,required=True)
    args=p.parse_args();proof,result=build(args.generation_archive)
    # Both outputs are exclusive. Preserve a lone prerequisite receipt after interruption.
    for path,value in ((HERE/'operator-prerequisites-01.json',proof),(HERE/'compare-01-spec.json',result)):
        if path.exists():
            if path.is_symlink() or m.read(path)!=value:raise ValueError('Existing prepared file differs; preserve it')
        else:m.save(path,value)
    print(json.dumps({'prepared':True,'launched':False,'specification_sha256':m.sha(HERE/'compare-01-spec.json')}))
