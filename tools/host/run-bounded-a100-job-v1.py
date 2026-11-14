"""Private A100 diagnostic launcher: same bounded lifecycle with explicit sm80 interpreter and cache paths."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import os
import hashlib
import json
import math
import re
import signal
import subprocess
import sys
import time
import uuid


LAUNCHER = 'outputs/host/run-bounded-a100-job-v1.py'
LABEL = 'drivebench.bounded-job'
MAX_SECONDS = 96 * 60 * 60
CLEANUP_MARGIN_SECONDS = 240
IMAGE = 'sha256:d2ab425962b505ae4bf21175451d889a66c12bf394ce4763479be9e3f7776022'
ENVIRONMENT = {'CUDA_MODULE_LOADING': 'LAZY', 'CUDA_CACHE_MAXSIZE': '2147483648',
               'CUDA_CACHE_PATH': '/workspace/outputs/model-cache/a100-cuda',
               'HF_HOME': '/workspace/outputs/model-cache/huggingface',
               'TORCH_HOME': '/workspace/outputs/model-cache/torch',
               'XDG_CACHE_HOME': '/workspace/outputs/model-cache/a100-xdg', 'PYTHONUNBUFFERED': '1',
               'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
SPEC_FIELDS = {'root', 'metadata', 'container_name', 'fresh_outputs', 'timeout_seconds',
               'allocation_deadline_epoch', 'expected_source_identity', 'expected_file_hashes',
               'python_arguments', 'scope', 'environment_verification', 'environment_lock'}


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


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
    if (not isinstance(proof, dict) or proof.get('passed') is not True
            or proof.get('verification_type')!='explicit_a100_pilot_environment_v1'
            or proof.get('target_contract')!='single_nvidia_a100_sxm4_80gb_pilot_v1'
            or proof.get('allocation_deadline_epoch')!=spec['allocation_deadline_epoch']):
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
        raise ValueError('Executing A100 diagnostic launcher differs from its reviewed self binding')
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


def inspect_container(name, timeout=15):
    result = subprocess.run(['docker', 'container', 'inspect', name], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        if 'No such container' in result.stderr or 'No such object' in result.stderr:
            return None
        raise RuntimeError('Cannot verify Docker container identity: ' + result.stderr[-1000:])
    values = json.loads(result.stdout)
    if len(values) != 1:
        raise RuntimeError('Unexpected Docker identity response')
    return values[0]


def _remaining(deadline, maximum):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Bounded exact-container cleanup expired')
    return min(maximum,remaining)


def cleanup_container(name, specification, nonce, *, expected_id=None, deadline=None, before_remove=None):
    deadline=time.monotonic()+60 if deadline is None else deadline
    found = inspect_container(name,timeout=_remaining(deadline,15))
    if found is None:
        return {'status': 'absent', 'volumes_deleted': False}
    labels = found.get('Config', {}).get('Labels') or {}
    identifier = found.get('Id', '')
    if (found.get('Name') != '/' + name or found.get('Image')!=IMAGE or labels.get(LABEL) != specification
            or labels.get(LABEL + '-nonce') != nonce or not re.fullmatch(r'[0-9a-f]{64}', identifier)
            or (expected_id is not None and identifier!=expected_id)):
        raise RuntimeError('Refusing cleanup: exact container name/image/ID/labels do not match this job')
    if before_remove is not None:before_remove(identifier)
    result = subprocess.run(['docker', 'container', 'rm', '--force', identifier], capture_output=True,
                            text=True,timeout=_remaining(deadline,25))
    if result.returncode or inspect_container(name,timeout=_remaining(deadline,15)) is not None:
        raise RuntimeError('Exact container cleanup did not verify disappearance')
    return {'status': 'removed', 'container_id': identifier, 'volumes_deleted': False}


def cleanup_retries(name,specification,nonce,*,expected_id=None,before_remove=None):
    deadline=time.monotonic()+60;results=[];absent=0
    for attempt in range(3):
        try:
            result=cleanup_container(name,specification,nonce,expected_id=expected_id,
                                     deadline=deadline,before_remove=before_remove)
            results.append(result)
            if result['status']=='removed':break
            absent+=1
            if absent>=2:break
        except Exception as error:
            absent=0
            results.append({'status':'failed','type':type(error).__name__,'message':str(error)})
        if time.monotonic()+.5>=deadline:break
        time.sleep(.5)
    if results[-1]['status']=='absent' and absent<2:
        results.append({'status':'failed','message':'Two separated absence observations were not completed'})
    return results


def cleanup_from_spec(spec_path):
    """ExecStopPost works after expiry or source drift; only its own recorded container may be removed."""
    spec_path=Path(spec_path).resolve();specification=sha(spec_path);spec=json.loads(spec_path.read_text())
    root=Path(spec['root']).resolve();meta=project_path(root,spec['metadata'])
    if not meta.is_relative_to(root/'outputs/host') or meta.suffix!='.json':raise ValueError('Invalid metadata location')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',spec['container_name']):raise ValueError('Invalid container name')
    if not meta.exists():
        if inspect_container(spec['container_name']) is not None:raise ValueError('No job metadata; refusing unknown container cleanup')
        return 0
    data=json.loads(meta.read_text());nonce=data.get('container_nonce','');expected_id=data.get('container_id')
    if (data.get('specification_sha256')!=specification or data.get('launcher_sha256')!=sha(Path(__file__))
            or spec.get('expected_file_hashes',{}).get(LAUNCHER)!=sha(Path(__file__))
            or data.get('input_file_hashes')!=spec.get('expected_file_hashes')
            or data.get('container_name')!=spec['container_name'] or data.get('container_image')!=IMAGE
            or not re.fullmatch(r'[0-9a-f]{32}',nonce)
            or (expected_id is not None and not re.fullmatch(r'[0-9a-f]{64}',expected_id))):
        raise ValueError('Cleanup specification/launcher/metadata/nonce/ID binding differs')
    receipt=meta.with_name(meta.stem+'.cleanup-'+uuid.uuid4().hex+'.json')
    value={'type':'a100_diagnostic_poststop_cleanup_v1','started_at':datetime.now(timezone.utc).isoformat(),
           'specification_sha256':specification,'metadata_sha256':sha(meta),'container_name':spec['container_name'],
           'container_nonce':nonce,'expected_container_id':expected_id,'launcher_sha256':sha(Path(__file__)),
           'status':'running','volumes_deleted':False}
    atomic(receipt,value)
    def bind_actual(identifier):
        if value.get('observed_container_id') not in (None,identifier):raise ValueError('Container ID changed during cleanup')
        value['observed_container_id']=identifier;atomic(receipt,value)
    results=cleanup_retries(spec['container_name'],specification,nonce,expected_id=expected_id,before_remove=bind_actual)
    code=0 if results[-1]['status'] in ('absent','removed') else 1
    value.update(status='completed' if code==0 else 'failed',finished_at=datetime.now(timezone.utc).isoformat(),
                 cleanup=results,exit_code=code);atomic(receipt,value)
    return code


def stop_client(process):
    if process is None:return None
    try:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
    except Exception as error:return {'type':type(error).__name__,'message':str(error)}
    return None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('specification',nargs='?');parser.add_argument('--cleanup',metavar='SPEC')
    options=parser.parse_args(argv)
    if bool(options.cleanup)==bool(options.specification):parser.error('Provide one SPEC or --cleanup SPEC')
    if options.cleanup:return cleanup_from_spec(options.cleanup)
    spec_path = Path(options.specification).resolve()
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
            '--gpus', '"device=0"', '--network=none', '--shm-size=8g', '--mount', f'type=bind,source={root},target=/workspace', '--workdir', '/workspace']
    for key, value in ENVIRONMENT.items():
        args += ['--env', key + '=' + value]
    args += [IMAGE,
             'timeout', '--signal=TERM', '--kill-after=30s', str(spec['timeout_seconds']) + 's',
             '/workspace/outputs/venvs/vista-a100-cu118/bin/python', '-u', *spec['python_arguments']]
    data = {'status': 'running', 'container_name':spec['container_name'],'container_image':IMAGE,'container_id':None, 'started_at': datetime.now(timezone.utc).isoformat(),
            'argv': args, 'source_identity': identity, 'source_files': before, 'input_file_hashes': inputs,
            'specification_sha256': specification, 'launcher_sha256': sha(Path(__file__)),
            'launcher_version': 'a100-diagnostic-v1', 'allocation_deadline_epoch': spec['allocation_deadline_epoch'],
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
                if data['container_id'] is None:
                    found=inspect_container(spec['container_name'])
                    if found is not None:
                        labels=found.get('Config',{}).get('Labels') or {}
                        if found.get('Image')!=IMAGE or labels.get(LABEL)!=specification or labels.get(LABEL+'-nonce')!=nonce:
                            raise RuntimeError('Running container identity differs')
                        data['container_id']=found.get('Id');atomic(meta,data)
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
            data['status'] = 'completed' if code == 0 else 'failed'
    except BaseException as error:
        code = 128 + caught[0] if caught else 1
        data.update(status='interrupted' if caught else 'failed', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        client_stop_error=stop_client(process)
        data['client_stop_error']=client_stop_error
        if client_stop_error:code=1;data['status']='failed'
        def bind_cleanup(identifier):
            if data.get('container_id') not in (None,identifier):raise ValueError('Container ID changed before cleanup')
            data['container_id']=identifier;atomic(meta,data)
        cleanup=cleanup_retries(spec['container_name'],specification,nonce,
                                expected_id=data.get('container_id'),before_remove=bind_cleanup)
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
