"""Public runtime artifacts and real read-only Git behavior; no GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT/'scripts/with_modern_vista_env.sh'


def test_exact_dependency_inventory_covers_lock_and_bundled_wheel():
    manifest = json.loads((ROOT/'environments/modern-vista-cu130-linux.wheels.json').read_text())
    lines = [line for line in (ROOT/'environments/modern-vista-cu130-linux.lock.txt').read_text().splitlines()
             if line and not line.startswith('#')]
    expected = {f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}" for row in manifest['packages']}
    assert len(lines) == len(expected) == manifest['wheel_count'] == 84
    assert set(lines) == expected
    assert sum(row['bytes'] for row in manifest['packages']) == manifest['total_bytes']
    for row in manifest['packages']:
        assert Path(row['filename']).name == row['filename']
        assert re.fullmatch('[a-f0-9]{64}', row['sha256'])
        origin = row['origin']
        if origin['kind'] == 'bundled_pure_python_wheel':
            path = ROOT/origin['path']
            assert path.stat().st_size == row['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
            assert (ROOT/origin['license']).is_file()
        else:
            assert origin['url'].startswith(('https://files.pythonhosted.org/', 'https://download-r2.pytorch.org/'))


@pytest.mark.skipif(shutil.which('git') is None, reason='Git required for checkout validation')
def test_wrapper_preserves_stale_git_index_and_rejects_changed_sources(tmp_path):
    repo = tmp_path/'repo'
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    tracked = repo/'tracked.txt'
    tracked.write_text('clean source\n')
    subprocess.run(['git', '-C', str(repo), 'add', 'tracked.txt'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=CPU test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fixture'], check=True)
    revision = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    # Deliberately stale cached stat information normally permits index refresh.
    stat = tracked.stat()
    os.utime(tracked, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))
    index = repo/'.git/index'
    before = index.read_bytes()
    code = '''import os,sys
from drivebench.models import upstream
assert os.environ['GIT_OPTIONAL_LOCKS']=='0'
assert os.environ['HF_HUB_OFFLINE']=='1'
assert os.environ['TRANSFORMERS_OFFLINE']=='1'
assert os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']=='1'
assert os.environ['PYTHONDONTWRITEBYTECODE']=='1'
upstream.VISTA_COMMIT=sys.argv[2]
upstream.verify_checkout(sys.argv[1], expected_commit=sys.argv[2])
'''
    env = {**os.environ, 'PYTHONPATH':str(ROOT), 'GIT_OPTIONAL_LOCKS':'1', 'HF_HUB_OFFLINE':'0'}
    args = ['bash',str(WRAPPER),sys.executable,'-c',code,str(repo),revision]
    subprocess.run(args, check=True, env=env, capture_output=True, text=True)
    assert index.read_bytes() == before
    tracked.write_text('modified source\n')
    failed = subprocess.run(args, env=env, capture_output=True, text=True)
    assert failed.returncode != 0 and 'must be pristine' in failed.stderr
    assert index.read_bytes() == before
