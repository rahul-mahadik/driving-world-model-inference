import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('launcher', Path(__file__).resolve().parents[3] / 'tools/host/run-bounded-portable-v1.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
_real_inspect_image = launcher.inspect_image
PROOF = "outputs/environment-next/verification.json"
LOCK = "outputs/environment-next/reference-lock.json"


@pytest.fixture(autouse=True)
def installed_image(monkeypatch):
    monkeypatch.setattr(launcher,'inspect_image',lambda image:None)

def _rebind_deadline(root,value):
    proof=json.loads((root/PROOF).read_text());proof['allocation_deadline_epoch']=value['allocation_deadline_epoch']
    (root/PROOF).write_text(json.dumps(proof));value['expected_file_hashes'][PROOF]=launcher.sha(root/PROOF)

@pytest.fixture
def job(tmp_path):
    (tmp_path/'outputs/host').mkdir(parents=True)
    names = [PROOF, LOCK, launcher.LAUNCHER, 'scripts/probe.py', 'configs/probe.yaml']
    for name in names:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}')
    (tmp_path / launcher.LAUNCHER).write_bytes(Path(launcher.__file__).read_bytes())
    environment = {'gpu': {'uuid': 'GPU-CPU-FIXTURE'}, 'torch': {'version': 'fixture'}, 'python': 'fixture',
                   'packages': {'torch': 'fixture'}, 'tools': {'ncu': {'stdout': 'fixture'}}}
    runtime = {'image_id':'sha256:'+'a'*64,'python_executable':'/opt/vista/bin/python','environment_proof_kind':'fixture-environment-v1'}
    (tmp_path / PROOF).write_text(json.dumps({'passed': True, 'environment': environment, 'launch_runtime':runtime, 'verification_type':runtime['environment_proof_kind'], 'allocation_deadline_epoch':100000000000}))
    (tmp_path / LOCK).write_text(json.dumps(launcher.environment_identity(environment)))
    (tmp_path / 'drivebench').mkdir()
    (tmp_path / 'drivebench/a.py').write_text('value = 1\n')
    value = {'runtime':runtime, 'root': str(tmp_path), 'metadata': 'outputs/host/job.json', 'container_name': 'test-only',
             'timeout_seconds': 600, 'allocation_deadline_epoch': 100000000000,
             'environment_verification': PROOF, 'environment_lock': LOCK,
             'expected_source_identity': hashlib.sha256(json.dumps(launcher.sources(tmp_path), sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
             'python_arguments': ['scripts/probe.py', '--config', 'configs/probe.yaml'],
             'fresh_outputs': ['outputs/new-attempt'], 'scope': 'test',
             'expected_file_hashes': {n: launcher.sha(tmp_path / n) for n in names}}
    return tmp_path, value


@pytest.mark.parametrize('duration', [0, -1, True, 1.5, 345601])
def test_rejects_unbounded_time(job, duration):
    _, value = job
    value['timeout_seconds'] = duration
    with pytest.raises(ValueError, match='positive integer'):
        launcher.validate_spec(value, 0)


def test_deadline_cleanup_margin(job):
    _, value = job
    value['allocation_deadline_epoch'] = 600 + launcher.CLEANUP_MARGIN_SECONDS
    with pytest.raises(ValueError, match='cleanup margin'):
        launcher.validate_spec(value, 0)


@pytest.mark.parametrize('hours', [16, 30])
def test_planned_jobs_fit_explicit_84_hour_allocation(job, hours):
    root, value = job
    value['timeout_seconds'] = hours * 3600
    value['allocation_deadline_epoch'] = 84 * 3600
    _rebind_deadline(root,value)
    assert launcher.validate_spec(value, 0)[0] == root


def test_96_hour_maximum_is_not_permission_to_overrun_allocation(job):
    root, value = job
    value['timeout_seconds'] = launcher.MAX_SECONDS
    value['allocation_deadline_epoch'] = launcher.MAX_SECONDS + launcher.CLEANUP_MARGIN_SECONDS + 1
    _rebind_deadline(root,value)
    assert launcher.validate_spec(value, 0)[0] == root
    value['allocation_deadline_epoch'] = 84 * 3600
    with pytest.raises(ValueError, match='cleanup margin'):
        launcher.validate_spec(value, 0)


@pytest.mark.parametrize('deadline', [True, None, float('inf'), float('nan'), 0])
def test_absolute_deadline_is_mandatory_finite_and_future(job, deadline):
    _, value = job
    value['allocation_deadline_epoch'] = deadline
    with pytest.raises(ValueError, match='cleanup margin'):
        launcher.validate_spec(value, 0)


def test_proof_and_lock_paths_are_explicit_and_match(job):
    root, value = job
    value.pop('environment_verification')
    with pytest.raises(ValueError, match='explicit environment'):
        launcher.validate_spec(value, 0)
    value['environment_verification'] = PROOF
    lock = json.loads((root / LOCK).read_text()); lock['gpu']['uuid'] = 'GPU-DIFFERENT'
    (root / LOCK).write_text(json.dumps(lock))
    value['expected_file_hashes'][LOCK] = launcher.sha(root / LOCK)
    with pytest.raises(ValueError, match='lock differs'):
        launcher.validate_spec(value, 0)


@pytest.mark.parametrize('fault', ['failed', 'no_uuid', 'traversal', 'absolute', 'unbound', 'same'])
def test_reject_invalid_environment_evidence(job, fault):
    root, value = job
    proof = json.loads((root / PROOF).read_text())
    if fault == 'failed': proof['passed'] = False
    if fault == 'no_uuid': proof['environment']['gpu'].pop('uuid')
    if fault in {'failed', 'no_uuid'}:
        (root / PROOF).write_text(json.dumps(proof))
        value['expected_file_hashes'][PROOF] = launcher.sha(root / PROOF)
    elif fault == 'traversal': value['environment_verification'] = '../proof.json'
    elif fault == 'absolute': value['environment_verification'] = str(root / PROOF)
    elif fault == 'unbound': value['expected_file_hashes'].pop(PROOF)
    elif fault == 'same': value['environment_lock'] = PROOF
    with pytest.raises(ValueError): launcher.validate_spec(value, 0)


@pytest.mark.parametrize('key', ['env', 'environment', 'docker_image', 'docker_arguments'])
def test_no_arbitrary_environment_or_container_overrides(job, key):
    _, value = job
    value[key] = {'TOKEN': 'FABRICATED_TEST_VALUE'}
    with pytest.raises(ValueError, match='forbidden'):
        launcher.validate_spec(value, 0)


def test_v2_binds_executed_launcher_not_historical_v1(job):
    root, value = job
    value['expected_file_hashes']['outputs/host/run-bounded-job.py'] = value['expected_file_hashes'].pop(launcher.LAUNCHER)
    with pytest.raises(ValueError, match='omits'):
        launcher.validate_spec(value, 0)
    value['expected_file_hashes'].pop('outputs/host/run-bounded-job.py')
    (root / launcher.LAUNCHER).write_text('other launcher')
    value['expected_file_hashes'][launcher.LAUNCHER] = launcher.sha(root / launcher.LAUNCHER)
    with pytest.raises(ValueError, match='self binding'):
        launcher.validate_spec(value, 0)


def test_input_hash_and_required_dependency(job):
    root, value = job
    assert launcher.validate_spec(value, 0)[0] == root
    (root / 'scripts/probe.py').write_text('changed')
    with pytest.raises(ValueError, match='hashes differ'):
        launcher.validate_spec(value, 0)
    del value['expected_file_hashes'][LOCK]
    with pytest.raises(ValueError, match='omits'):
        launcher.validate_spec(value, 0)


def test_paths_no_traversal_or_symlink(job, tmp_path):
    root, value = job
    value['metadata'] = '../bad.json'
    with pytest.raises(ValueError, match='traversal'):
        launcher.validate_spec(value, 0)
    outside = root.parent / 'outside-launcher-test'
    outside.mkdir(exist_ok=True)
    (root / 'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        launcher.project_path(root, 'escape/a.json')


def test_cleanup_never_removes_mismatched_container(monkeypatch):
    monkeypatch.setattr(launcher, 'inspect_container', lambda name: {'Id':'a'*64, 'Name':'/test-only', 'Config':{'Labels':{launcher.LABEL:'another-job'}}})
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('must not delete mismatched identity'))
    with pytest.raises(RuntimeError, match='Refusing cleanup'):
        launcher.cleanup_container('test-only','spec','nonce')


def test_cleanup_exact_id_preserves_volumes(monkeypatch):
    found = {'Id':'a'*64, 'Name':'/test-only', 'Config':{'Labels':{launcher.LABEL:'spec', launcher.LABEL+'-nonce':'nonce'}}}
    states = iter([found, None])
    monkeypatch.setattr(launcher, 'inspect_container', lambda name: next(states))
    calls = []
    def run(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher.subprocess, 'run', run)
    assert launcher.cleanup_container('test-only','spec','nonce')['status'] == 'removed'
    assert calls == [['docker','container','rm','--force','a'*64]]


@pytest.mark.parametrize('drift', [None, 'source', 'input'])
def test_main_success_and_nonzero_drift(job, monkeypatch, drift):
    root, value = job
    jobfile = root / 'spec.json'; jobfile.write_text(json.dumps(value))
    monkeypatch.setattr(launcher.sys,'argv',['launcher',str(jobfile)])
    monkeypatch.setattr(launcher,'inspect_container',lambda name:None)
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='',stderr=''))
    monkeypatch.setattr(launcher.time,'sleep',lambda s:None)
    class Process:
        returncode = 0
        def __init__(self,*a,**kw):
            self.argv = a[0]
            if drift == 'source': (root/'drivebench/a.py').write_text('changed')
            if drift == 'input': (root/'scripts/probe.py').write_text('changed')
        def poll(self): return self.returncode
    monkeypatch.setattr(launcher.subprocess,'Popen',Process)
    assert launcher.main() == (0 if drift is None else 1)
    data=json.loads((root/'outputs/host/job.json').read_text())
    assert data['status'] == ('completed' if drift is None else 'source_or_input_drift')
    assert data['input_file_hashes'] == value['expected_file_hashes']
    assert len(data['cleanup']) == 2
    assert value['runtime']['image_id'] in data['argv'] and data['launcher_version'] == 'portable-v1'
    assert '--network=none' in data['argv'] and '--pull=never' in data['argv']
    assert value['runtime']['python_executable'] in data['argv']
    assert {data['argv'][i + 1] for i, v in enumerate(data['argv']) if v == '--env'} == {
        k + '=' + v for k, v in launcher.ENVIRONMENT.items()}
    assert data['environment_verification'] == PROOF and data['environment_lock'] == LOCK


def test_gpu_lock_excludes_concurrent_launcher(job, monkeypatch):
    root, value = job
    jobfile = root / 'spec.json'; jobfile.write_text(json.dumps(value))
    monkeypatch.setattr(launcher.sys, 'argv', ['launcher', str(jobfile)])
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **kw: pytest.fail('lock must prevent launch'))
    with (root / 'outputs/host/gpu-job.lock').open('a') as held:
        launcher.fcntl.flock(held, launcher.fcntl.LOCK_EX | launcher.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): launcher.main()


def test_cleanup_failure_is_nonzero_even_after_successful_process(job, monkeypatch):
    root, value = job
    jobfile = root / 'spec.json'; jobfile.write_text(json.dumps(value))
    monkeypatch.setattr(launcher.sys, 'argv', ['launcher', str(jobfile)])
    monkeypatch.setattr(launcher, 'inspect_container', lambda name: None)
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''))
    monkeypatch.setattr(launcher.time, 'sleep', lambda s: None)
    class Process:
        returncode = 0
        def __init__(self, *a, **kw): pass
        def poll(self): return self.returncode
    monkeypatch.setattr(launcher.subprocess, 'Popen', Process)
    def cleanup(*a): raise RuntimeError('Cannot verify owned container removal')
    monkeypatch.setattr(launcher, 'cleanup_container', cleanup)
    assert launcher.main() == 1
    data = json.loads((root / 'outputs/host/job.json').read_text())
    assert data['status'] == 'cleanup_failed' and len(data['cleanup']) == 3


def test_host_timeout_cleans_only_owned_container(job, monkeypatch):
    root,value=job
    jobfile=root/'spec.json';jobfile.write_text(json.dumps(value))
    monkeypatch.setattr(launcher.sys,'argv',['launcher',str(jobfile)])
    monkeypatch.setattr(launcher,'inspect_container',lambda name:None)
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='',stderr=''))
    ticks=iter([0,1000]);monkeypatch.setattr(launcher.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(launcher.time,'sleep',lambda s:None)
    events=[]
    class Process:
        returncode=None
        def __init__(self,*a,**kw):pass
        def poll(self):return self.returncode
        def terminate(self):events.append('terminate');self.returncode=-15
        def wait(self,**kw):return self.returncode
    monkeypatch.setattr(launcher.subprocess,'Popen',Process)
    cleanup=[]
    monkeypatch.setattr(launcher,'cleanup_container',lambda *a:cleanup.append(a) or {'status':'removed'})
    assert launcher.main()==1
    assert events==['terminate']
    assert cleanup[0][0]=='test-only'
    data=json.loads((root/'outputs/host/job.json').read_text())
    assert data['error']['type']=='TimeoutError'


def test_absolute_deadline_guard_stops_and_cleans_owned_container(job, monkeypatch):
    root, value = job
    value['allocation_deadline_epoch'] = 1000
    _rebind_deadline(root,value)
    jobfile = root / 'spec.json'; jobfile.write_text(json.dumps(value))
    monkeypatch.setattr(launcher.sys, 'argv', ['launcher', str(jobfile)])
    monkeypatch.setattr(launcher, 'inspect_container', lambda name: None)
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''))
    ticks = iter([0, 0, 760])
    monkeypatch.setattr(launcher.time, 'time', lambda: next(ticks))
    monkeypatch.setattr(launcher.time, 'monotonic', lambda: 0)
    monkeypatch.setattr(launcher.time, 'sleep', lambda s: None)
    class Process:
        returncode = None
        def __init__(self, *a, **kw): pass
        def poll(self): return self.returncode
        def terminate(self): self.returncode = -15
        def wait(self, **kw): return self.returncode
    monkeypatch.setattr(launcher.subprocess, 'Popen', Process)
    cleanup = []
    monkeypatch.setattr(launcher, 'cleanup_container', lambda *a: cleanup.append(a) or {'status': 'removed'})
    assert launcher.main() == 1
    data = json.loads((root / 'outputs/host/job.json').read_text())
    assert 'Absolute allocation deadline' in data['error']['message']
    assert len(cleanup) == 1


def test_deadline_rechecked_immediately_before_launch(job, monkeypatch):
    root, value = job
    value['allocation_deadline_epoch'] = 1000
    _rebind_deadline(root,value)
    jobfile = root / 'spec.json'; jobfile.write_text(json.dumps(value))
    monkeypatch.setattr(launcher.sys, 'argv', ['launcher', str(jobfile)])
    monkeypatch.setattr(launcher, 'inspect_container', lambda name: None)
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''))
    ticks = iter([0, 200]); monkeypatch.setattr(launcher.time, 'time', lambda: next(ticks))
    monkeypatch.setattr(launcher.time, 'sleep', lambda s: None)
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **kw: pytest.fail('must not launch after margin is lost'))
    assert launcher.main() == 1
    data = json.loads((root / 'outputs/host/job.json').read_text())
    assert 'cleanup margin' in data['error']['message']


def test_existing_container_refused_before_launch(job, monkeypatch):
    root,value=job;jobfile=root/'spec.json';jobfile.write_text(json.dumps(value))
    monkeypatch.setattr(launcher.sys,'argv',['launcher',str(jobfile)])
    monkeypatch.setattr(launcher,'inspect_container',lambda name:{'Name':'/test-only'})
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not launch'))
    with pytest.raises(RuntimeError,match='already exists'):launcher.main()


def test_sigterm_records_interruption_and_cleans(job, monkeypatch):
    root,value=job;jobfile=root/'spec.json';jobfile.write_text(json.dumps(value))
    monkeypatch.setattr(launcher.sys,'argv',['launcher',str(jobfile)])
    monkeypatch.setattr(launcher,'inspect_container',lambda name:None)
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='',stderr=''))
    monkeypatch.setattr(launcher.time,'sleep',lambda s:None)
    handlers={}
    monkeypatch.setattr(launcher.signal,'signal',lambda sig,handler:handlers.setdefault(sig,handler))
    class Process:
        returncode=None
        def __init__(self,*a,**kw):handlers[launcher.signal.SIGTERM](launcher.signal.SIGTERM,None)
        def poll(self):return self.returncode
        def terminate(self):self.returncode=-15
        def wait(self,**kw):return self.returncode
    monkeypatch.setattr(launcher.subprocess,'Popen',Process)
    cleanup=[]
    monkeypatch.setattr(launcher,'cleanup_container',lambda *a:cleanup.append(a) or {'status':'removed'})
    assert launcher.main()==143
    assert len(cleanup)==1
    data=json.loads((root/'outputs/host/job.json').read_text())
    assert data['status']=='interrupted'


@pytest.mark.parametrize('field,bad', [
    ('image_id','vendor/image:latest'),('image_id','--privileged'),
    ('python_executable','python'),('python_executable','/workspace/../bin/python'),
    ('python_executable','/opt//python'),('environment_proof_kind','')])
def test_mutable_or_ambiguous_runtime_rejected(job, field, bad):
    _,value=job;value['runtime'][field]=bad
    with pytest.raises(ValueError): launcher.validate_spec(value,0)


@pytest.mark.parametrize('field,bad', [('image_id','sha256:'+'b'*64),
    ('python_executable','/other/python'),('environment_proof_kind','other-kind')])
def test_runtime_changes_require_fresh_matching_proof(job, field, bad):
    _,value=job;value['runtime'][field]=bad
    with pytest.raises(ValueError,match='bind this launch runtime'):launcher.validate_spec(value,0)


def test_previous_allocation_proof_cannot_be_rebound(job):
    _,value=job;value['allocation_deadline_epoch']+=3600
    with pytest.raises(ValueError,match='immutable allocation deadline'):launcher.validate_spec(value,0)


@pytest.mark.parametrize('status,rows',[(1,[]),(0,[]),(0,[{'Id':'sha256:'+'b'*64}])])
def test_missing_or_wrong_installed_image_fails_closed(monkeypatch,status,rows):
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(returncode=status,stdout=json.dumps(rows),stderr='')
    monkeypatch.setattr(launcher.subprocess,'run',run)
    with pytest.raises(RuntimeError):_real_inspect_image('sha256:'+'a'*64)
    assert calls==[['docker','image','inspect','sha256:'+'a'*64]]


def test_installed_immutable_image_accepted_without_pull(monkeypatch):
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(returncode=0,stdout=json.dumps([{'Id':'sha256:'+'a'*64}]),stderr='')
    monkeypatch.setattr(launcher.subprocess,'run',run)
    _real_inspect_image('sha256:'+'a'*64)
    assert len(calls)==1
