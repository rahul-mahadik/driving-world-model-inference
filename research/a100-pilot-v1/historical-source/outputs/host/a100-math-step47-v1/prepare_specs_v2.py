#!/usr/bin/env python3
"""Local-only immutable specification preparation. Never deploys or launches."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HELPER_SHA = 'a42fde8397fce3d98387f36a959e78d9e8ba0e61c9974be4c10439f505575d75'
CONFIG_SHA = '1d25f08f352d32123ef56679d2e4cdf8dcdb007278342de8483dd510584a4c9c'
INVENTORY_SHAS = {
    'eager50': '0e447bf86bbd91887518f187d4b3b3a18fcb0242784133b58da4507c5e3c9c4a',
    'eager40': 'e569c64d2e125b39b4322d4e4c6b2faa32b4fd46484633da90f33a596251acc8',
    'feature2': '2dcebcecfb31ae073d30457049b85c2b3aa5027783cdc64a9a81c33c5b6653fb',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def relative(name, *, require_remote=False):
    if not isinstance(name, str) or any(c in name for c in ('\n', '\r', '\x00', '\\')):
        raise ValueError('Unsafe pin path')
    if name.startswith('/workspace/'):
        name = name[len('/workspace/'):]
    elif require_remote or name.startswith('/'):
        raise ValueError('Admission pins must be actual /workspace paths')
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name or name == '.':
        raise ValueError('Noncanonical or escaping pin path')
    return name


def add(pins, name, digest):
    name = relative(name)
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid SHA256 pin')
    if name in pins and pins[name] != digest:
        raise ValueError('Conflicting immutable input pin: ' + name)
    pins[name] = digest


def merged(*maps):
    out = {}
    for values in maps:
        if not isinstance(values, dict): raise ValueError('Expected pin dictionary')
        for p, h in values.items(): add(out, p, h)
    return out


def frozen_helper():
    path = HERE / 'step47.py'
    if path.is_symlink() or sha(path) != HELPER_SHA or sha(HERE / 'configs/eager47.json') != CONFIG_SHA:
        raise ValueError('Frozen prepared helper/config changed')
    spec = importlib.util.spec_from_file_location('step47_spec_frozen_helper', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def archive(name, expected_inventory):
    folder = ROOT / ('outputs/a100-math-quality-' + name + '-evidence-20261028')
    inv = folder / 'remote-inventory.json'; receipt = read(folder / 'verification.json')
    if (inv.is_symlink() or sha(inv) != expected_inventory or receipt.get('passed') is not True
            or receipt.get('errors', []) != [] or receipt.get('remote_inventory_sha256') != expected_inventory):
        raise ValueError('Prior verified archive receipt or sealed inventory differs')
    rows = read(inv); inventory = {}; sizes = {}
    if not isinstance(rows, list) or not rows: raise ValueError('Empty prior inventory')
    for row in rows:
        path = relative(row['path'])
        if path in inventory or type(row['size']) is not int or row['size'] < 0:
            raise ValueError('Duplicate or invalid inventory entry')
        add(inventory, path, row['sha256']); sizes[path] = row['size']
    if receipt.get('files') != len(rows) or receipt.get('bytes') != sum(sizes.values()):
        raise ValueError('Verified inventory coverage differs')
    def artifact(path):
        path = relative(path); target = folder / 'files' / path
        if (path not in inventory or target.is_symlink() or target.stat().st_size != sizes[path]
                or sha(target) != inventory[path]):
            raise ValueError('Used archived metadata does not match verified inventory')
        return target
    return inventory, artifact


def build_admission():
    m = frozen_helper(); q = m.quality()
    host_pins, exact_inputs, items, template, old_baseline = {}, {}, {}, None, None
    for name, spec_sha in m.SPEC_PINS.items():
        spec_name = q.REL + '/' + name + '-run-01-spec.json'
        spec_path = ROOT / spec_name
        if sha(spec_path) != spec_sha: raise ValueError('Pinned prior generation specification changed')
        spec = read(spec_path)
        if template is None: template = copy.deepcopy(spec)
        if any(spec.get(k) != template.get(k) for k in
               ('root', 'allocation_deadline_epoch', 'expected_source_identity', 'environment_verification', 'environment_lock')):
            raise ValueError('Preserved specifications have different runtime envelopes')
        inventory, artifact = archive(name, INVENTORY_SHAS[name])
        if sha(artifact(spec_name)) != spec_sha:
            raise ValueError('Local and archived generation specifications differ')
        out = q.RESULT + '/runs/' + name
        done = read(artifact(out + '/completed.json'))
        host = read(artifact(spec['metadata']))
        if (done.get('status') != 'completed' or done.get('variant') != name
                or host.get('status') != 'completed' or host.get('exit_code') != 0
                or host.get('specification_sha256') != spec_sha
                or not q.cleanup_complete(host.get('cleanup', []))):
            raise ValueError('Prior completed generation and cleanup required')
        if old_baseline is None: old_baseline = done
        if any(done.get(k) != old_baseline.get(k) for k in ('platform_gate', 'strict_prerequisite')):
            raise ValueError('Prior generation platform/strict provenance differs')
        rows = [json.loads(line) for line in artifact(out + '/measurements.jsonl').read_text().splitlines() if line.strip()]
        items[name] = {'run': type('Rows', (), {'measurements': rows})(), 'baseline': {'result': done['result']}}
        host_pins = merged(host_pins, spec['expected_file_hashes'], inventory)
        # Reconstruct exactly step47.prepare's generation() bindings, while the
        # outer specification also preserves unused host logs/cleanup receipts.
        exact_inputs = merged(exact_inputs, spec['expected_file_hashes'], done['input_sha256'],
                              done['strict_prerequisite']['artifact_sha256'])
        for path, digest in done['result']['artifact_sha256'].items():
            add(exact_inputs, out + '/' + relative(path), digest)
        for path in (out + '/completed.json', spec_name, spec['metadata']):
            add(exact_inputs, path, inventory[path])
    _, _, observed = q.smoke_inputs(read(HERE / 'configs/eager47.json'))
    exact_inputs = merged(exact_inputs, {str(Path(p).relative_to(ROOT)): h for p, h in observed.items()})
    for path, digest in {m.REL + '/step47.py': HELPER_SHA,
                         m.REL + '/configs/eager47.json': CONFIG_SHA,
                         m.QUALITY: m.QUALITY_SHA, m.EVALUATOR: m.EVALUATOR_SHA}.items():
        if sha(ROOT / path) != digest: raise ValueError('Required frozen file changed')
        add(host_pins, path, digest); add(exact_inputs, path, digest)
    host_pins = merged(host_pins, exact_inputs)
    if (template['allocation_deadline_epoch'] != m.DEADLINE or template['expected_source_identity'] != q.SOURCE
            or host_pins.get(q.LAUNCHER) != q.LAUNCHER_SHA or sha(ROOT / q.LAUNCHER) != q.LAUNCHER_SHA):
        raise ValueError('Frozen source, deadline or host launcher differs')
    template.update(metadata=m.REL + '/admit-01-host.json', container_name='drivebench-a100-math-step47-admit-01',
        fresh_outputs=[m.REL + '/admission-01.json'], timeout_seconds=900,
        expected_file_hashes=dict(sorted(host_pins.items())),
        python_arguments=[m.REL + '/step47.py', 'admit', '--expected-sha256', HELPER_SHA,
                          '--output', '/workspace/' + m.REL + '/admission-01.json'],
        scope=m.SCOPE + ' Read-only admission only; no automatic launch. Parent must first verify completed32-scene '
              'baseline/backup/evaluation and fresh provider budget plus unchanged guard. No deadline extension.')
    expected = {'platform_gate': old_baseline['platform_gate'], 'strict_prerequisite': old_baseline['strict_prerequisite'],
        'input_sha256': dict(sorted(exact_inputs.items())), 'time_bound': m.estimate(q, items),
        'hypothesis': m.hypothesis(q, items), 'automatic_launch_authorized': False, 'scope': m.SCOPE}
    return m, template, expected


def run_spec(admission_copy, *, now=None):
    m, template, expected = build_admission(); q = m.quality()
    admission_copy = Path(admission_copy)
    if admission_copy.is_symlink() or not admission_copy.is_file(): raise ValueError('Actual admission copy required')
    admission = read(admission_copy)
    now = time.time() if now is None else now
    if (admission.get('type') != 'a100_math_step47_admission_v1' or admission.get('producer_sha256') != HELPER_SHA
            or admission.get('source_identity') != q.SOURCE or admission.get('allocation_deadline_epoch') != m.DEADLINE
            or not 0 <= now - q.epoch(admission.get('created_at')) <= 300):
        raise ValueError('Fresh actual admission with fixed source, helper and deadline required')
    supplied = {}
    if not isinstance(admission.get('input_sha256'), dict): raise ValueError('Admission input map required')
    allowed_relative_aliases = {
        'outputs/host/a100-pilot-01/configs/baseline.yaml',
        'outputs/host/verify-a100-pilot-v1.py',
        'scripts/probe_candidate_memory.py',
        'scripts/probe_compiler_operator.py',
    }
    for path, digest in admission['input_sha256'].items():
        if not path.startswith('/workspace/'):
            if (path not in allowed_relative_aliases
                    or admission['input_sha256'].get('/workspace/' + path) != digest):
                raise ValueError('Only known consistent relative aliases of absolute pins are allowed')
        name = relative(path)
        add(supplied, name, digest)  # Consistent aliases merge; conflicting pins still fail.
    if supplied != expected['input_sha256']:
        raise ValueError('Admission input pins differ from exact frozen provenance')
    if any(admission.get(k) != v for k, v in expected.items() if k != 'input_sha256'):
        raise ValueError('Actual admission proof, hypothesis, scope or bound differs')
    disk = admission.get('disk', {})
    if (not isinstance(disk, dict) or type(disk.get('observed_free_bytes')) is not int
            or disk != m.admission_limits(now=now, free_bytes=disk['observed_free_bytes'])):
        raise ValueError('Actual remote disk observation differs; never substitute local free space')
    result = copy.deepcopy(template)
    add(result['expected_file_hashes'], m.REL + '/admission-01.json', sha(admission_copy))
    result.update(metadata=m.REL + '/run-01-host.json', container_name='drivebench-a100-math-step47-run-01',
        fresh_outputs=[m.RESULT], timeout_seconds=m.OUTER,
        python_arguments=[m.REL + '/step47.py', 'run', '--expected-sha256', HELPER_SHA,
            '--admission', '/workspace/' + m.REL + '/admission-01.json', '--output', '/workspace/' + m.RESULT],
        scope=m.SCOPE + ' Fixed1800s outer/1680s inner plus1200s evaluation/archive/cleanup reserve. '
              'Parent must freshly verify provider budget and unchanged armed guard. No automatic follow-up.')
    return result


def save_new(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists():
        if path.is_symlink() or path.read_text() != text: raise ValueError('Existing prepared specification differs; preserve it')
    else:
        with path.open('x') as f: f.write(text)
    print(json.dumps({'path': str(path), 'sha256': sha(path), 'pins': len(value['expected_file_hashes']), 'launched': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission-copy', type=Path)
    args = parser.parse_args()
    if args.admission_copy:
        save_new(HERE / 'run-01-spec.json', run_spec(args.admission_copy))
    else:
        save_new(HERE / 'admit-01-spec.json', build_admission()[1])
