#!/usr/bin/env python3
"""Prospective fixed-asset fetch after a fresh counter-free modern proof.

The external bounded process launcher still enforces the phase deadline. This
helper preserves partial bytes and never modifies the historical downloader.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import urllib.request

PROOF_KIND = 'modern_declared_gpu_counter_free_v2'
ASSETS = {
    'vista': {
        'url': 'https://huggingface.co/OpenDriveLab/Vista/resolve/a92d28c7b3da931ec959d587153dc45fffe570ef/vista.safetensors',
        'sha256': '1b267bb753aa5c5e4d363f6e4c864dd4e343996dedc18b5ab20af6d20be34b98',
        'bytes': 10053461572,
        'relative_path': 'checkpoints/vista.safetensors',
    },
    'openclip': {
        'url': 'https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_pytorch_model.bin',
        'sha256': '9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4',
        'bytes': 3944692325,
        'relative_path': 'hf/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_pytorch_model.bin',
    },
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def current_gpu_identity():
    if platform.system() != 'Linux':raise RuntimeError('Linux CUDA host required')
    r=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=20)
    rows=[line.split(',') for line in r.stdout.splitlines() if line.strip()]
    if len(rows)!=1 or len(rows[0])!=2:raise RuntimeError('Exactly one GPU required')
    return tuple(v.strip() for v in rows[0])


def require_proof(path, expected_sha):
    from drivebench.verify_environment import has_cuda_kernel_trace
    from drivebench.benchmarks.environment import reference_identity
    from drivebench_tuning.modern_verify_environment import validate_policy
    if sha(path)!=expected_sha:raise RuntimeError('Fresh proof file hash differs')
    r=json.loads(Path(path).read_text()); p=r.get('profiling_proof',{})
    validate_policy(r.get('hardware_policy'))
    if (r.get('proof_kind')!=PROOF_KIND or r.get('passed') is not True
            or r.get('dedicated_attestation') is not True or r.get('failures')
            or r.get('legacy_h100_proof') is not False
            or p.get('modern_runtime_passed') is not True
            or p.get('torch_cuda_execution_passed') is not True
            or p.get('nsys_trace_collected') is not True
            or p.get('hardware_counters_collected') is not False
            or p.get('hardware_counters_required') is not False
            or p.get('hardware_counter_status')!='not_requested_counter_free_protocol'):
        raise RuntimeError('Complete fresh counter-free modern proof required')
    if (p.get('nsys',{}).get('returncode')!=0 or p.get('nsys_stats',{}).get('returncode')!=0
            or not has_cuda_kernel_trace(p.get('nsys_stats',{}).get('stdout',''),kernel='drivebench_smoke')):
        raise RuntimeError('Positive expected-kernel CUDA activity trace required')
    report=Path(path).parent/'smoke-systems.nsys-rep'
    if not report.is_file() or report.stat().st_size==0:raise RuntimeError('Missing CUDA activity report')
    before,after=r.get('environment',{}),r.get('environment_after',{})
    if reference_identity(before)!=reference_identity(after):raise RuntimeError('Preflight runtime changed')
    devices=before.get('gpus',[])
    if len(devices)!=1:raise RuntimeError('One recorded GPU required')
    expected=devices[0].get('uuid'),devices[0].get('driver_version')
    if not all(expected) or current_gpu_identity()!=expected:raise RuntimeError('Proof belongs to another GPU/driver')
    return r


def fetch(asset, root, verification, proof_sha256, deadline_unix):
    if asset not in ASSETS:raise ValueError('Unknown asset')
    if type(deadline_unix) not in (int,float) or not math.isfinite(deadline_unix) or not time.time()<deadline_unix:raise TimeoutError('Expired asset phase')
    mono_deadline=time.monotonic()+deadline_unix-time.time()
    def remaining():
        left=min(deadline_unix-time.time(),mono_deadline-time.monotonic())
        if left<=0:raise TimeoutError('Asset phase deadline expired')
        return left
    require_proof(verification,proof_sha256); remaining()
    spec=ASSETS[asset]; dest=Path(root)/spec['relative_path'];dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        raise FileExistsError('Asset destination exists; retain it and choose a fresh attempt root')
    partial=dest.with_suffix(dest.suffix+'.part')
    if partial.exists():raise FileExistsError('Partial asset exists; preserve it and use a fresh attempt root')
    h=hashlib.sha256(); size=0
    request=urllib.request.Request(spec['url'],headers={'User-Agent':'drivebench-fixed-asset/2'})
    with urllib.request.urlopen(request,timeout=min(15,remaining())) as response,partial.open('xb') as f:
        length=response.headers.get('Content-Length')
        if length is not None and int(length)!=spec['bytes']:raise RuntimeError('Asset length header differs')
        while True:
            remaining();block=response.read(8*1024*1024)
            if not block:break
            size+=len(block)
            if size>spec['bytes']:raise RuntimeError('Asset exceeded pinned size')
            f.write(block);h.update(block)
        f.flush();os.fsync(f.fileno())
    remaining()
    if size!=spec['bytes'] or h.hexdigest()!=spec['sha256']:raise RuntimeError('Asset byte/hash mismatch; partial retained')
    require_proof(verification,proof_sha256);remaining()
    os.link(partial,dest)
    # Preserve the original verified partial as a hard link: no evidence deletion.
    receipt={'asset':asset,'status':'downloaded_verified','url':spec['url'],'bytes':size,'sha256':h.hexdigest(),'path':str(dest.resolve()),'verification_sha256':proof_sha256,'deadline_unix':deadline_unix,'partial_hard_link_preserved':True,'model_loaded':False}
    with dest.with_suffix(dest.suffix+'.receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    if asset=='openclip':
        ref=Path(root)/'hf/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/refs/main';ref.parent.mkdir(parents=True,exist_ok=True)
        with ref.open('x') as f:f.write('1c2b8495b28150b8a4922ee1c8edee224c284c0c')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset',choices=ASSETS,required=True);p.add_argument('--root',required=True);p.add_argument('--verification',required=True);p.add_argument('--proof-sha256',required=True);p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--failure-receipt',required=True)
    a=p.parse_args()
    if Path(a.failure_receipt).exists():raise FileExistsError('Refuse to overwrite failed attempt')
    try:print(json.dumps(fetch(a.asset,a.root,a.verification,a.proof_sha256,a.deadline_unix),indent=2))
    except BaseException as error:
        with Path(a.failure_receipt).open('x') as f:json.dump({'status':'failed','asset':a.asset,'error_type':type(error).__name__,'deadline_unix':a.deadline_unix,'partial_bytes_preserved':True},f,indent=2);f.write('\n')
        raise

if __name__=='__main__':main()
