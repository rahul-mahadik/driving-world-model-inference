"""Private v2 launcher: bound runtime proofs and absolute allocation deadlines."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import re
import signal
import subprocess
import sys
import time
import uuid


LAUNCHER = 'outputs/host/run-bounded-job-v2.py'
LABEL = 'drivebench.bounded-job'
MAX_SECONDS = 96 * 60 * 60
CLEANUP_MARGIN_SECONDS = 240
IMAGE = 'sha256:d2ab425962b505ae4bf21175451d889a66c12bf394ce4763479be9e3f7776022'
ENVIRONMENT = {'CUDA_MODULE_LOADING': 'LAZY', 'CUDA_CACHE_MAXSIZE': '2147483648',
               'CUDA_CACHE_PATH': '/workspace/outputs/model-cache/cuda',
               'HF_HOME': '/workspace/outputs/model-cache/huggingface',
               'TORCH_HOME': '/workspace/outputs/model-cache/torch',
               'XDG_CACHE_HOME': '/workspace/outputs/model-cache/xdg', 'PYTHONUNBUFFERED': '1'}
SPEC_FIELDS = {'root', 'metadata', 'container_name', 'fresh_outputs', 'timeout_seconds',
               'allocation_deadline_epoch', 'expected_source_identity', 'expected_file_hashes',
               'python_arguments', 'scope', 'environment_verification', 'environment_lock'}


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(root):
    return {str(p.relative_to(root / 'drivebench')): sha(p)
            for p in sorted((root / 'drivebench').rglob('*.py'))}


def project_path(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Project paths must be nonempty relative paths without traversal')
    result = root / value
    if not result.resolve().is_relative_to(root):
        raise ValueError('Project path escapes root through a symlink')
    return result


def file_hashes(root, expected):
    return {name: sha(project_path(root, name)) for name in expected}


def validate_deadline(spec, now):
    duration = spec['timeout_seconds']
    if type(duration) is not int or not 1 <= duration <= MAX_SECONDS:
        raise ValueError('timeout_seconds must be a positive integer <=345600; zero disables timeout')
    deadline = spec['allocation_deadline_epoch']
    if (type(deadline) not in (int, float) or not math.isfinite(deadline)
            or now + duration + CLEANUP_MARGIN_SECONDS >= deadline):
        raise ValueError('Insufficient allocation time for bounded job and cleanup margin')


def environment_identity(environment):
    """Pinned reference_identity layout, without importing GPU/project packages."""
    return {'gpu': environment.get('gpu'), 'torch': environment.get('torch'),
            'cuda_runtime_environment': environment.get('cuda_runtime_environment'),
            'xformers_build': environment.get('xformers_build'),
            'python': environment.get('python'), 'packages': environment.get('packages'),
            'tools': {k: v.get('stdout') for k, v in environment.get('tools', {}).items()}}


def validate_environment(root, spec):
    proof = json.loads(project_path(root, spec['environment_verification']).read_text())
    lock = json.loads(project_path(root, spec['environment_lock']).read_text())
    if not isinstance(proof, dict) or proof.get('passed') is not True:
        raise ValueError('Environment verification is not passing')
    environment = proof.get('environment')
    if (not isinstance(environment, dict) or not isinstance(environment.get('gpu'), dict)
            or not environment['gpu'].get('uuid') or not environment.get('torch')
            or not environment.get('python') or not environment.get('packages')
            or not isinstance(environment.get('tools'), dict) or not environment['tools']
            or any(not isinstance(v, dict) or not v.get('stdout') for v in environment['tools'].values())):
        raise ValueError('Environment proof lacks complete GPU/runtime/tool identity')
    identity = environment_identity(environment)
    if not isinstance(lock, dict) or lock != identity:
        raise ValueError('Environment lock differs from explicitly bound proof')
    return identity


def validate_spec(spec, now):
    if not isinstance(spec, dict) or set(spec) - SPEC_FIELDS:
        raise ValueError('Unknown specification fields; Docker image and environment are fixed')
    if any(key not in spec for key in SPEC_FIELDS - {'fresh_outputs'}):
        raise ValueError('Missing required specification field, including explicit environment proof/lock')
    root = Path(spec['root'])
    if not root.is_absolute() or not root.is_dir():
        raise ValueError('Root must be an existing absolute project directory')
    root = root.resolve()
    validate_deadline(spec, now)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', spec['container_name']):
        raise ValueError('Invalid exact container name')
    if not re.fullmatch(r'[0-9a-f]{64}', spec['expected_source_identity']):
        raise ValueError('Missing reviewed package source identity')
    args = spec['python_arguments']
    if not isinstance(args, list) or not args or any(not isinstance(a, str) or '\0' in a for a in args):
        raise ValueError('python_arguments must be a nonempty argv list')
    expected = spec.get('expected_file_hashes')
    if not isinstance(expected, dict) or not expected:
        raise ValueError('Bind input helpers, inherited configs, and environment proof/lock hashes')
    for key in ('environment_verification', 'environment_lock'):
        if project_path(root, spec[key]).suffix != '.json':
            raise ValueError('Environment proof and lock must be explicitly bound JSON paths')
    if spec['environment_verification'] == spec['environment_lock']:
        raise ValueError('Environment proof and lock must be distinct files')
    required = {spec['environment_verification'], spec['environment_lock'], LAUNCHER}
    if args[0].endswith('.py'):
        required.add(args[0])
    for index, value in enumerate(args[:-1]):
        if value in {'--config', '--base-config'}:
            required.add(args[index + 1])
    if not required.issubset(expected):
        raise ValueError('Input hash declaration omits a required direct input')
    for name, digest in expected.items():
        project_path(root, name)
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Invalid SHA256 in input hash declaration')
    actual = file_hashes(root, expected)
    if actual != expected:
        raise ValueError('Reviewed input file hashes differ')
    if expected[LAUNCHER] != sha(Path(__file__)):
        raise ValueError('Executing v2 launcher differs from its reviewed self binding')
    validate_environment(root, spec)
    meta = project_path(root, spec['metadata'])
    if not meta.is_relative_to(root / 'outputs/host') or meta.suffix != '.json':
        raise ValueError('Metadata must be a JSON file under outputs/host')
    if not isinstance(spec.get('fresh_outputs', []), list):
        raise ValueError('fresh_outputs must be a list of relative paths')
    if not isinstance(spec['scope'], str) or not spec['scope']:
        raise ValueError('Scope must describe the reviewed job')
    for name in spec.get('fresh_outputs', []):
        path = project_path(root, name)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Output already exists: {name}')
    return root, meta, actual


def inspect_container(name):
    result = subprocess.run(['docker', 'container', 'inspect', name], capture_output=True, text=True, timeout=15)
    if result.returncode:
        if 'No such container' in result.stderr or 'No such object' in result.stderr:
            return None
        raise RuntimeError('Cannot verify Docker container identity: ' + result.stderr[-1000:])
    values = json.loads(result.stdout)
    if len(values) != 1:
        raise RuntimeError('Unexpected Docker identity response')
    return values[0]


def cleanup_container(name, specification, nonce):
    found = inspect_container(name)
    if found is None:
        return {'status': 'absent'}
    labels = found.get('Config', {}).get('Labels') or {}
    identifier = found.get('Id', '')
    if (found.get('Name') != '/' + name or labels.get(LABEL) != specification
            or labels.get(LABEL + '-nonce') != nonce or not re.fullmatch(r'[0-9a-f]{64}', identifier)):
        raise RuntimeError('Refusing cleanup: exact container identity/labels do not match this job')
    result = subprocess.run(['docker', 'container', 'rm', '--force', identifier], capture_output=True, text=True, timeout=25)
    if result.returncode or inspect_container(name) is not None:
        raise RuntimeError('Exact container cleanup did not verify disappearance')
    return {'status': 'removed', 'container_id': identifier, 'volumes_deleted': False}


def main():
    spec_path = Path(sys.argv[1]).resolve()
    specification = sha(spec_path)
    spec = json.loads(spec_path.read_text())
    root, meta, inputs = validate_spec(spec, time.time())
    log = meta.with_suffix('.log')
    lock = (root / 'outputs/host/gpu-job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if meta.exists() or log.exists():
        raise FileExistsError('Use a fresh attempt; previous evidence must remain intact')
    before = sources(root)
    identity = hashlib.sha256(json.dumps(before, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if identity != spec['expected_source_identity']:
        raise RuntimeError('Source identity differs from reviewed job specification')
    runtime_identity = validate_environment(root, spec)
    if inspect_container(spec['container_name']) is not None:
        raise RuntimeError('Container name already exists; preserve and inspect the earlier job')
    # An interrupted host process may leave a container after releasing flock.
    active = subprocess.run(['docker', 'ps', '--quiet', '--filter', 'label=' + LABEL],
                            capture_output=True, text=True, timeout=15)
    if active.returncode or active.stdout.strip():
        raise RuntimeError('Another labeled GPU job is active or Docker state is unavailable')
    nonce = uuid.uuid4().hex
    args = ['docker', 'run', '--rm', '--name', spec['container_name'],
            '--label', LABEL + '=' + specification, '--label', LABEL + '-nonce=' + nonce,
            '--gpus', '"device=0"', '--shm-size=8g', '--mount', f'type=bind,source={root},target=/workspace', '--workdir', '/workspace']
    for key, value in ENVIRONMENT.items():
        args += ['--env', key + '=' + value]
    args += [IMAGE,
             'timeout', '--signal=TERM', '--kill-after=30s', str(spec['timeout_seconds']) + 's',
             '/workspace/outputs/venvs/vista-upstream-cu118-sm90/bin/python', '-u', *spec['python_arguments']]
    data = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
            'argv': args, 'source_identity': identity, 'source_files': before, 'input_file_hashes': inputs,
            'specification_sha256': specification, 'launcher_sha256': sha(Path(__file__)),
            'launcher_version': 2, 'allocation_deadline_epoch': spec['allocation_deadline_epoch'],
            'timeout_seconds': spec['timeout_seconds'], 'cleanup_margin_seconds': CLEANUP_MARGIN_SECONDS,
            'environment_verification': spec['environment_verification'], 'environment_lock': spec['environment_lock'],
            'runtime_identity': runtime_identity, 'container_nonce': nonce, 'scope': spec['scope']}
    atomic(meta, data)
    caught = []
    previous = {sig: signal.signal(sig, lambda signum, frame: caught.append(signum)) for sig in (signal.SIGTERM, signal.SIGINT)}
    process = None
    code = 1
    try:
        with log.open('x') as stream:
            if caught:
                raise InterruptedError('Host launcher interrupted before Docker start')
            validate_deadline(spec, time.time())
            if file_hashes(root, inputs) != inputs or sha(spec_path) != specification:
                raise RuntimeError('Reviewed inputs/specification changed before Docker start')
            process = subprocess.Popen(args, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            started = time.monotonic()
            while process.poll() is None:
                if caught:
                    raise InterruptedError('Host launcher received signal ' + str(caught[0]))
                if time.monotonic() - started >= spec['timeout_seconds'] + 45:
                    raise TimeoutError('Host deadline exceeded inner timeout plus kill margin')
                if time.time() >= spec['allocation_deadline_epoch'] - CLEANUP_MARGIN_SECONDS:
                    raise TimeoutError('Absolute allocation deadline cleanup margin reached')
                time.sleep(0.25)
            if caught:
                raise InterruptedError('Host launcher interrupted during Docker completion')
            code = process.returncode
            data['status'] = 'completed'
    except BaseException as error:
        code = 128 + caught[0] if caught else 1
        data.update(status='interrupted' if caught else 'failed', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        # Check the exact labeled container even after normal Docker exit.
        # This also handles a creation/host-interruption race with bounded retries.
        cleanup = []
        for attempt in range(3):
            try:
                result = cleanup_container(spec['container_name'], specification, nonce)
                cleanup.append(result)
                if result['status'] == 'removed' or (attempt >= 1 and result['status'] == 'absent'):
                    break
            except Exception as error:
                cleanup.append({'status': 'failed', 'type': type(error).__name__, 'message': str(error)})
            time.sleep(0.5)
        data['cleanup'] = cleanup
        if cleanup[-1]['status'] == 'failed':
            code = 1
            data['status'] = 'cleanup_failed'
        try:
            data['source_unchanged'] = sources(root) == before
            data['inputs_unchanged'] = file_hashes(root, inputs) == inputs and sha(spec_path) == specification
        except Exception as error:
            data.update(source_unchanged=False, inputs_unchanged=False, drift_error=str(error))
        if not data['source_unchanged'] or not data['inputs_unchanged']:
            code = 1
            data['status'] = 'source_or_input_drift'
        data.update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
        atomic(meta, data)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        lock.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
