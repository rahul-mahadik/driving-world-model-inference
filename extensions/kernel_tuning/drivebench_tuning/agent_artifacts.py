"""Trusted, bounded exports and evidence inventories; no inference or scheduling.

Exports consume pinned final metric receipts, preserve original arrays, and keep
the blind answer key in a private directory. Rendering is not visual review.
Inventory covers exactly the operator-declared new study trees and extra files,
including failed/partial attempts. Its output must be outside those input trees.
A successful output includes its completion receipt within maximum_output_bytes.
This is a checked bound, not an OS quota: failures preserve partials and may
exceed it by the last attempted bounded artifact plus the failure receipt.
An external worker timeout and host/container cleanup remain caller obligations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

from . import agent_evidence as evidence, agent_metrics_worker as metrics_worker
from . import agent_scoring, vista_agent, vista_batch
from .agent_session import _bytes, _write_new

COMMON = {'schema_version', 'kind', 'synthetic', 'created_at', 'deadline_unix',
          'minimum_free_bytes', 'maximum_output_bytes', 'output', 'sources'}
EXTRA = {'export': {'final_plan', 'quality_policy', 'review_plan', 'comparisons'},
         'inventory': {'roots', 'extra_files', 'required_pins', 'maximum_input_bytes'}}
POLICIES = ('llm', 'random', 'grid')


def _now(): return datetime.now(timezone.utc).isoformat()
def _write(path, value): _write_new(Path(path), value)


def _write_completion(output, receipt, maximum_output_bytes):
    # Match _write_new's exact UTF-8 serialization and trailing newline.
    total=sum(p.stat().st_size for p in _files(output))+len(_bytes(receipt))+1
    if total>maximum_output_bytes:
        raise ValueError('completion receipt would exceed artifact output byte bound')
    _write(output/'completed.json',receipt)


def required_source_paths():
    return metrics_worker.required_source_paths()


def _files(root):
    paths = list(root.rglob('*'))
    if any(p.is_symlink() for p in paths): raise ValueError('evidence symlinks forbidden')
    if any(not p.is_file() and not p.is_dir() for p in paths): raise ValueError('nonregular evidence forbidden')
    return sorted(p for p in paths if p.is_file())


def _hash(path, check):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('regular evidence file required')
    before = path.stat(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            check(scan_output=False); digest.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('evidence changed while hashing')
    return {'bytes': after.st_size, 'sha256': digest.hexdigest()}


def _metric_bundle(pin, *, policy, final_pin, quality, synthetic, check):
    complete = metrics_worker._pin(pin); directory = Path(pin['path']).parent
    if Path(pin['path']).name != 'completed.json': raise ValueError('final metric completion required')
    expected = {'schema_version':1, 'status':'ok', 'kind':'final', 'synthetic':synthetic,
                'real_lpips':not synthetic, 'quality_preserving_claim':False, 'external_worker_exit_verified':False}
    for key, value in expected.items(): evidence._same(complete.get(key), value, 'metric completion '+key)
    actual = {p.relative_to(directory).as_posix():_hash(p, check)['sha256'] for p in _files(directory) if p != Path(pin['path'])}
    evidence._same(complete.get('artifacts'), actual, 'metric artifact inventory')
    # The worker preserves the parsed spec in canonical JSON; its receipt's
    # spec_sha256 pins the original operator file, whose formatting can differ.
    spec, _ = evidence._read(directory/'spec.json', actual['spec.json'])
    evidence._same(spec['final_plan'], final_pin, 'export final plan pin')
    evidence._same(metrics_worker._pin(spec['quality_policy']), quality.to_dict(), 'export quality policy')
    metrics_worker._check_sources(spec['sources'])
    refs = metrics_worker._batches(spec['references']); candidates = metrics_worker._batches(spec['candidates'])
    if any(b.to_dict()['role'] != policy+'_selected' for b in candidates): raise ValueError('wrong final comparison role')
    final = metrics_worker._pin(final_pin)
    metrics_worker._study_bindings(spec['study_bindings'], refs+candidates, final)
    evidence._same(complete['cohort_id'], final['cohort_id'], 'export cohort')
    identity = quality.to_dict()['plan']['identity']
    evidence._same(complete['identity_sha256'], quality.identity_sha256, 'metric identity')
    paired, _ = evidence._read(directory/'paired-records.json', actual['paired-records.json'])
    values = {}
    for row in paired['quality_rows']:
        key = row['reference_output_sha256'], row['output_sha256']
        if key in values: evidence._same(values[key], row['metrics'], 'identical-array metric values')
        values[key] = row['metrics']
    def recorded(a, b, **pins):
        check()
        return values[pins['reference_sha256'], pins['candidate_sha256']]
    rebuilt = evidence.pair_rows(refs, candidates, identity=identity, candidate_sha256=spec['candidate_sha256'],
        measure_pair=recorded, metric_names=metrics_worker.METRICS, final_plan_path=final_pin['path'], expected_final_plan_sha256=final_pin['sha256'])
    evidence._same(rebuilt, paired, 'recorded metric-to-array provenance reconstruction')
    assessment = agent_scoring.assess_proxy_quality(quality, paired['quality_rows'], split='final', selection_started_at=spec['selection_started_at'])
    score = agent_scoring.score_heldout_timing(quality, paired['timing_rows'], quality_assessment=assessment, selection_started_at=spec['selection_started_at'])
    evidence._same(evidence._read(directory/'quality-assessment.json')[0], assessment, 'quality receipt replay')
    evidence._same(evidence._read(directory/'timing-score.json')[0], score, 'timing receipt replay')
    evidence._same(complete['result'], {'quality_status':assessment['status'], 'score':score,
        'quality_policy_sha256':quality.sha256, 'candidate_sha256':spec['candidate_sha256']}, 'final result replay')
    return refs, candidates, spec


def _export(spec, output, check):
    from drivebench.evaluation.review import export_review
    final = metrics_worker._pin(spec['final_plan'])
    quality = agent_scoring.QualityPolicy.from_dict(metrics_worker._pin(spec['quality_policy']))
    review = metrics_worker._pin(spec['review_plan'])
    vista_batch._keys(review, {'schema_version','created_at','cells','review_seed','maximum_cases'}, 'review plan')
    if review['schema_version'] != 1 or type(review['review_seed']) is not int or type(review['maximum_cases']) is not int:
        raise ValueError('invalid review plan')
    if not 1 <= review['maximum_cases'] <= 24 or not isinstance(review['cells'], list) or not review['cells']:
        raise ValueError('bounded explicit review cells required')
    if evidence._time(review['created_at']) >= final['all_selections_locked_at']:
        raise ValueError('review selection must precede the final selection barrier')
    for row in review['cells']: vista_batch._keys(row, {'scene_id','seed','replicate'}, 'review cell')
    cells = {evidence._cell(row) for row in review['cells']}
    if len({(scene,seed) for scene,seed,rep in cells}) != len(cells):
        raise ValueError('review plan permits one replicate per scene and seed')
    if len(cells) != len(review['cells']) or len(cells)*3 > review['maximum_cases']:
        raise ValueError('duplicate or excessive review cases')
    plan = quality.to_dict()['plan']
    if any(scene not in plan['final_scenes'] or seed not in plan['seeds'] or rep not in plan['replicates'] for scene,seed,rep in cells):
        raise ValueError('review cell outside frozen final population')
    if not isinstance(spec['comparisons'], list) or len(spec['comparisons']) != 3:
        raise ValueError('all three final comparisons required')
    for item in spec['comparisons']: vista_batch._keys(item, {'policy','metric_completion'}, 'comparison')
    if {item['policy'] for item in spec['comparisons']} != set(POLICIES): raise ValueError('duplicate or missing comparison')
    pairs = []; batches = {}; pinned_files = {}; common_refs = None
    for item in sorted(spec['comparisons'], key=lambda item:item['policy']):
        check(); policy = item['policy']
        refs, candidates, metric = _metric_bundle(item['metric_completion'], policy=policy,
            final_pin=spec['final_plan'], quality=quality, synthetic=spec['synthetic'], check=check)
        ref_pins = sorted(metric['references'], key=lambda row:row['directory'])
        if common_refs is not None: evidence._same(ref_pins, common_refs, 'shared final references')
        common_refs = ref_pins
        left = {evidence._cell(row):row for b in refs for row in b.to_dict()['rows']}
        right = {evidence._cell(row):row for b in candidates for row in b.to_dict()['rows']}
        settings = refs[0].to_dict()['spec']['settings']['generation']
        expected_shape = tuple(settings[k] for k in ('total_frames','height','width'))+(3,)
        if settings['conditioning_frames'] != 1: raise ValueError('one conditioning frame required')
        for cell in sorted(cells):
            a,b = left[cell],right[cell]
            for row in (a,b):
                raw = np.load(row['raw_path'], mmap_mode='r', allow_pickle=False)
                if (raw.dtype != np.float32 or raw.shape != expected_shape or not np.isfinite(raw).all() or
                        raw.min() < 0 or raw.max() > 1): raise ValueError('full-shape float32 RGB output required')
                pinned_files[row['raw_path']] = row['output_sha256']
            pairs.append({'request_id':vista_agent.canonical(list(cell)), 'scene_id':cell[0], 'seed':cell[1],
                'repeat':cell[2], 'candidate_id':'future', 'run':policy+'_selected', 'reference_run':'reference',
                'action_kind':'recorded', 'trajectory_sha256':a['conditioning_sha256'], 'conditioning_frames':1,
                'reference_path':a['raw_path'], 'candidate_path':b['raw_path']})
        for batch in refs+candidates: batches[batch.to_dict()['batch_id']] = batch
    private = output/'private'; private.mkdir(mode=0o700); rendering = private/'rendering'
    rendering.mkdir(mode=0o700); public = output/'public'; public.mkdir()
    check(); rendered = export_review(pairs, rendering, limit=len(cells), seed=review['review_seed']); check()
    key_path = Path(rendered['private_key']); key = evidence._read(key_path)[0]
    if len(key['cases']) != len(pairs): raise ValueError('core renderer dropped declared review cases')
    # Only non-key render artifacts move into the public review directory.
    for path in sorted(rendering.iterdir()):
        if path != key_path: path.rename(public/path.name)
    native = public/'native'; native.mkdir(); links=[]
    for case in key['cases']:
        check(); arrays=[np.load(case[name],mmap_mode='r',allow_pickle=False)[1:] for name in ('reference_path','candidate_path')]
        left,right = arrays if case['A']=='reference' else list(reversed(arrays))
        case_dir=native/case['case_id'];case_dir.mkdir(); frame_links=[]
        with Image.open(public/(case['case_id']+'.gif')) as preview:
            if preview.n_frames != len(left): raise ValueError('motion preview dropped a future frame')
        for index,(a,b) in enumerate(zip(left,right),1):
            check(); height,width,_=a.shape
            canvas=Image.new('RGB',(2*width,height+24),'#111827'); draw=ImageDraw.Draw(canvas)
            for panel,(label,frame) in enumerate((('A',a),('B',b))):
                draw.text((panel*width+8,6),f'{label} | Future frame {index}/{len(left)}',fill='white')
                canvas.paste(Image.fromarray(np.rint(frame*255).astype(np.uint8)),(panel*width,24))
            path=case_dir/f'frame-{index:04d}.png';canvas.save(path,compress_level=1)
            with Image.open(path) as image:
                pixels=np.asarray(image)
                if image.size!=(2*width,height+24) or any(not np.array_equal(pixels[24:,j*width:(j+1)*width],np.rint(frame*255).astype(np.uint8)) for j,frame in enumerate((a,b))):
                    raise ValueError('native display-pixel verification failed')
            frame_links.append(f'<a href="native/{case["case_id"]}/{path.name}">Frame {index}</a>')
        links.append(f'<h2>{case["case_id"]}</h2><p>'+', '.join(frame_links)+'</p>')
    with (public/'index.html').open('a') as index:
        index.write('<h1>Native ordered future frames</h1><p>PNG preserves rounded 8-bit display pixels at native resolution, not original float32 values. Exports are not a completed visual review.</p>'+''.join(links))
        index.flush(); os.fsync(index.fileno())
    for batch in batches.values(): check();batch.recheck()
    for item in spec['comparisons']: metrics_worker._pin(item['metric_completion'])
    for name in ('quality_policy','final_plan','review_plan'): metrics_worker._pin(spec[name])
    for path,sha in pinned_files.items():
        if _hash(Path(path),check)['sha256'] != sha: raise ValueError('export source array changed')
    artifacts={p.relative_to(public).as_posix():_hash(p,check) for p in _files(public)}
    _write(private/'provenance.json', {'review_plan_sha256':spec['review_plan']['sha256'], 'final_plan_sha256':spec['final_plan']['sha256'],
        'metric_completion_pins':spec['comparisons'], 'source_arrays':pinned_files, 'public_artifacts':artifacts,
        'private_key_sha256':vista_agent.file_hash(key_path), 'future_frames_per_case':len(left),
        'renderer':'drivebench.evaluation.review.export_review plus native display-pixel export',
        'metrics_recomputed':False, 'visual_review_performed':False, 'quality_preserving_claim':False})
    return {'cases':len(pairs), 'public_index':'public/index.html', 'public_review_csv':'public/review.csv',
            'private_key':'private/rendering/private_key.json', 'visual_review_performed':False,
            'review_mapping_disclosure':'Keep private until reviewer observations are separately sealed.',
            'native_display_pixels_verified':True, 'float32_arrays_preserved':True, 'metric_values_recomputed':False}


def _inventory(spec, output, check):
    roots=spec['roots']; extras=spec['extra_files']; maximum=spec['maximum_input_bytes']
    if not isinstance(roots,dict) or not roots or not isinstance(extras,list) or type(maximum) is not int or maximum<=0:
        raise ValueError('bounded explicit inventory scope required')
    resolved=[]
    for label,value in roots.items():
        path=Path(value)
        if (not isinstance(label,str) or not label or '/' in label or not path.is_absolute() or path.is_symlink() or not path.is_dir() or
                output.resolve().is_relative_to(path.resolve()) or path.resolve().is_relative_to(output.resolve())):
            raise ValueError('inventory output must be separate from declared input trees')
        resolved.append(path.resolve())
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(resolved) for b in resolved[i+1:]):
        raise ValueError('inventory roots overlap')
    if not isinstance(spec['required_pins'],list) or not spec['required_pins']: raise ValueError('required evidence pins missing')
    def paths():
        result={}
        for label,root in roots.items():
            for p in _files(Path(root)):result[f'{label}/{p.relative_to(root).as_posix()}']=p
        for index,name in enumerate(extras):
            p=Path(name)
            if not p.is_absolute() or p.is_symlink() or not p.is_file() or p.resolve().is_relative_to(output.resolve()):
                raise ValueError('invalid additional evidence file')
            if p.resolve() in {v.resolve() for v in result.values()}:raise ValueError('duplicate inventory file')
            result[f'extra-{index:04d}/{p.name}']=p
        if sum(p.stat().st_size for p in result.values())>maximum:raise ValueError('inventory exceeds declared byte bound')
        return result
    def verify_required(items):
        allowed={p.resolve() for p in items.values()}
        for pin in spec['required_pins']:
            vista_batch._keys(pin,{'path','sha256'},'required artifact pin')
            if Path(pin['path']).resolve() not in allowed:raise ValueError('required artifact outside inventory scope')
            if _hash(Path(pin['path']),check)['sha256']!=pin['sha256']:raise ValueError('required inventory artifact changed')
    before_paths=paths();verify_required(before_paths); first={};journal=output/'inventory-progress';journal.mkdir()
    output_bytes=sum(p.stat().st_size for p in _files(output))
    for index,(key,path) in enumerate(sorted(before_paths.items())):
        record={'path':str(path),**_hash(path,check)};first[key]=record
        progress=journal/f'{index:06d}.json';_write(progress,{'key':key,**record});output_bytes+=progress.stat().st_size
        if output_bytes>spec['maximum_output_bytes']:raise ValueError('artifact output byte bound exceeded')
    _write(output/'inventory.json',{'files':first,'root_scope':roots,'extra_file_scope':extras})
    after_paths=paths()
    if before_paths!=after_paths:raise ValueError('inventory file set changed during verification')
    for key,path in sorted(after_paths.items()):
        if {'path':str(path),**_hash(path,check)}!=first[key]:raise ValueError('inventory file changed during verification')
    verify_required(after_paths)
    _write(output/'inventory-verified.json',{'passed':True,'files':len(first),'bytes':sum(v['bytes'] for v in first.values()),
        'inventory_sha256':vista_agent.file_hash(output/'inventory.json'),'two_identical_hash_passes':True,
        'scope':'Every regular file in declared new-study roots plus explicit extras; includes failures/partials.',
        'planned_study_completed_claim':False,'provider_or_process_cleanup_verified':False})
    return {'files':len(first),'bytes':sum(v['bytes'] for v in first.values()),'two_identical_hash_passes':True,
            'planned_study_completed_claim':False,'provider_or_process_cleanup_verified':False}


def run_spec(path, *, expected_sha256):
    path=Path(path).absolute();spec,_=evidence._read(path,expected_sha256)
    if spec.get('kind') not in EXTRA:raise ValueError('unknown artifact operation')
    vista_batch._keys(spec,COMMON|EXTRA[spec['kind']],'artifact spec')
    if spec['schema_version']!=1 or type(spec['synthetic']) is not bool:raise ValueError('invalid artifact schema')
    output=Path(spec['output'])
    if not output.is_absolute() or output.is_symlink():raise ValueError('absolute fresh output required')
    output.mkdir(parents=True,exist_ok=False);_write(output/'spec.json',spec)
    previous=time.time()
    def check(*, scan_output=True):
        nonlocal previous
        now=time.time()
        if type(spec['deadline_unix']) not in (int,float) or not math.isfinite(spec['deadline_unix']) or now<previous or now>=spec['deadline_unix']:
            raise ValueError('artifact deadline expired or clock moved backwards')
        previous=now
        for name in ('minimum_free_bytes','maximum_output_bytes'):
            if type(spec[name]) is not int or spec[name]<0:raise ValueError('nonnegative byte bound required')
        if shutil.disk_usage(output).free<spec['minimum_free_bytes']:raise ValueError('artifact storage reserve exhausted')
        if scan_output and sum(p.stat().st_size for p in _files(output))>spec['maximum_output_bytes']:raise ValueError('artifact output byte bound exceeded')
    try:
        check();metrics_worker._check_sources(spec['sources'])
        if evidence._time(spec['created_at'])>time.time():raise ValueError('artifact spec created in future')
        result=(_export if spec['kind']=='export' else _inventory)(spec,output,check)
        check();metrics_worker._check_sources(spec['sources']);evidence._read(path,expected_sha256)
        receipt={'schema_version':1,'status':'ok','kind':spec['kind'],'spec_sha256':expected_sha256,'synthetic':spec['synthetic'],
            'completed_at':_now(),'result':result,'quality_preserving_claim':False,'external_worker_exit_verified':False,
            'artifacts':{p.relative_to(output).as_posix():_hash(p,check) for p in _files(output)}}
        check();_write_completion(output,receipt,spec['maximum_output_bytes']);return receipt
    except BaseException as error:
        _write(output/'failure.json',{'status':'failed','error_type':type(error).__name__,'error':str(error),'at':_now(),
            'spec_sha256':expected_sha256,'partial_artifacts_preserved':True,'quality_preserving_claim':False})
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True);parser.add_argument('--spec-sha256',required=True)
    args=parser.parse_args(argv);receipt=run_spec(args.spec,expected_sha256=args.spec_sha256)
    # Do not expose the private mapping, paths, hashes or final-input identifiers.
    print(json.dumps({'status':receipt['status'],'kind':receipt['kind'],'visual_review_performed':False}));return 0


if __name__=='__main__':raise SystemExit(main())
