#!/usr/bin/env python3
"""CPU-only blind review export from retained strict-proof NPZs.

Export is not review or qualification. Source proofs remain unchanged. Private
NPY copies retain exact float32 values; public PNGs use the established display
conversion and ordered atlases use its downscaled, palette-quantized previews.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drivebench.config import file_sha256
from drivebench.evaluation.review import export_review
from scripts import qualify_compiler_numerics as qualifier


def native_helper():
    path = Path(__file__).with_name('run_quality_bounded.py')
    spec = importlib.util.spec_from_file_location('proof_native_export', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def check(condition, message):
    if not condition:
        raise ValueError(message)


def bound(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':file_sha256(path)}


def export(proof_path, plan_path, output, *, minimum_free_bytes=1024**3):
    proof_path,plan_path,output=map(lambda p:Path(p).resolve(),(proof_path,plan_path,output))
    check(not output.exists(),'Use a fresh review export directory')
    check(type(minimum_free_bytes) is int and minimum_free_bytes >= 0,'Invalid disk reserve')
    check(qualifier.source_identity()==qualifier.SOURCE,'Frozen package changed')
    proof,plan=qualifier.read(proof_path),qualifier.read(plan_path)
    proof_hash,plan_hash=file_sha256(proof_path),file_sha256(plan_path)
    check(plan.get('type')=='compiler_norm_localized_review_plan_v1' and plan.get('status')=='unrated_private_export_plan'
          and plan.get('qualified') is False and plan.get('rule')==qualifier.REVIEW_RULE,'Require unrated qualifier review plan')
    check(plan.get('proof_sha256')==proof_hash and plan.get('artifact_sha256',{}).get(str(proof_path))==proof_hash
          and plan['artifact_sha256'].get(str(Path(qualifier.__file__).resolve()))==file_sha256(qualifier.__file__),
          'Plan/proof/qualifier binding changed')
    check(proof.get('schema_version')==2 and proof.get('mode')=='all' and type(proof.get('passed')) is bool
          and proof.get('atol')==0 and proof.get('rtol')==0 and proof.get('implementation_sha256')==qualifier.SOURCE,
          'Preserve original all-mode strict proof')
    ids=proof['input_provenance']['candidate_ids'];generation=proof['generation']
    check(2 <= len(ids) <= 4 and len(set(ids))==len(ids),'Bounded export supports two to four proof candidates')
    check(generation['total_frames']==25 and generation['conditioning_frames']==1,'Require one observed and24 generated frames')
    selected={k:v for k,v in plan['cases'].items() if k.startswith('selected-proof:')}
    check(set(selected)=={'selected-proof:'+key for key in ids},'Selected-proof plan coverage differs')
    source_bindings={str(proof_path):proof_hash,str(plan_path):plan_hash,str(Path(qualifier.__file__).resolve()):file_sha256(qualifier.__file__)}
    sources={}
    for label in ('upstream','refactored'):
        run=proof['runs'][label];path=Path(run['outputs']['normal']).resolve();expected=run['outputs_sha256']['normal']
        check(not path.is_symlink() and path.is_file() and file_sha256(path)==expected,'Saved NPZ missing or changed')
        check(plan['artifact_sha256'].get(str(path))==expected,'NPZ absent from review-plan provenance')
        source_bindings[str(path)]=expected;sources[label]=path
    nearest=output.parent
    while not nearest.exists():nearest=nearest.parent
    # Exact private NPY copies plus conservative native RGB output/PNG allowance.
    array_bytes=len(ids)*2*25*generation['height']*generation['width']*3*4
    native_bytes=len(ids)*24*(generation['height']+28)*generation['width']*2*3
    check(shutil.disk_usage(nearest).free>=minimum_free_bytes+array_bytes+native_bytes*2+16*1024**2,
          'Insufficient disk reserve for complete blind export')
    output.mkdir(parents=True);private=output/'private';private.mkdir(mode=0o700)
    array_dir=private/'arrays';array_dir.mkdir(mode=0o700)
    helper=native_helper();source_bindings[str(Path(helper.__file__).resolve())]=file_sha256(helper.__file__)
    source_bindings[str(Path(__file__).resolve())]=file_sha256(__file__)
    saved={};registry={}
    for label,path in sources.items():
        values=qualifier.arrays(path,ids,generation)
        for i,key in enumerate(ids):
            target=array_dir/f'{label}-{i:03}.npy'
            descriptor=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(descriptor,'wb') as stream:np.save(stream,values[key],allow_pickle=False)
            saved[label,key]=target;registry[str(target)]=file_sha256(target)
        del values
    pairs=[]
    for i,key in enumerate(ids):
        case=selected['selected-proof:'+key]
        check(case.get('candidate_id')==key and case.get('request_id')==proof['input_provenance']['request_id']
              and case.get('reference_output_sha256')==source_bindings[str(sources['upstream'])]
              and case.get('candidate_output_sha256')==source_bindings[str(sources['refactored'])], 'Review plan candidate/source differs')
        left,right=[np.load(saved[label,key],mmap_mode='r',allow_pickle=False) for label in ('upstream','refactored')]
        errors=qualifier.rgb_errors(left,right,1)
        frames={'maximum_absolute_error':errors['max_abs_location_THWC'][0],
                'maximum_frame_rmse':max(errors['per_frame'],key=lambda x:x['rmse'])['frame']}
        check(frames==case.get('native_frame_indices'),'Native highlight indices differ from recomputed arrays')
        pairs.append({'request_id':proof['input_provenance']['request_id'],'scene_id':proof['input_provenance']['scene_id'],
                      'repeat':0,'candidate_id':key,'seed':proof['input_provenance']['seeds'][i],
                      'run':'compiled_candidate','reference_run':'upstream_reference','action_kind':'diagnostic',
                      'trajectory_sha256':proof['input_provenance']['actions_sha256'][i],'conditioning_frames':1,
                      'reference_path':str(saved['upstream',key]),'candidate_path':str(saved['refactored',key])})
    preview=private/'preview';preview.mkdir(mode=0o700)
    review=export_review(pairs,preview,limit=len(pairs),seed=20261027)
    metric=private/'export-input.json';receipt=private/'export-input-receipt.json'
    helper.private_json(metric,{'runs':[{'complete_pairs':True}],'review':review,'scope':'Display export only; no evaluator or acceptance claim'})
    helper.private_json(receipt,{'evaluation_sha256':file_sha256(metric),
                                'review_artifact_sha256':{str(p):file_sha256(p) for p in preview.rglob('*') if p.is_file()}})
    bundle=helper.export_comparison(metric,receipt,registry,private/'native',minimum_free_bytes=minimum_free_bytes)
    public=output/'public';shutil.copytree(bundle/'public',public)
    (public/'index.html').rename(public/'native-index.html')
    # Read only the new private mapping inside this exporter. Never display it.
    mappings=qualifier.read(review['private_key'])['cases'];links=[];public_cases=[];private_cases=[]
    for item in mappings:
        opaque=item['case_id'];original='selected-proof:'+item['candidate_id'];case=selected[original]
        with Image.open(preview/f'{opaque}.gif') as animation:
            check(animation.n_frames==24,'Ordered preview lost a future frame')
            w,h=animation.size;atlas=Image.new('RGB',(4*w,6*h),'#111827')
            for index in range(24):
                animation.seek(index);atlas.paste(animation.convert('RGB'),((index%4)*w,(index//4)*h))
        atlas_path=public/f'{opaque}-all24.png';atlas.save(atlas_path)
        highlights={name:{'future_frame_1based':frame,'file':f'{opaque}-frame-{frame:04}.png'}
                    for name,frame in case['native_frame_indices'].items()}
        assets=[bound(atlas_path),*[bound(public/v['file']) for v in highlights.values()]]
        public_cases.append({'case_id':opaque,'future_frames':24,'ordered_atlas':atlas_path.name,'highlights':highlights})
        private_cases.append({'qualification_case_id':original,'opaque_case_id':opaque,**case,
                              'derived_reference_npy':bound(saved['upstream',item['candidate_id']]),
                              'derived_candidate_npy':bound(saved['refactored',item['candidate_id']]),'assets':assets})
        highlight_links=' | '.join(f'<a href="{v["file"]}">{name}: frame {v["future_frame_1based"]}</a>' for name,v in highlights.items())
        links.append(f'<article><h2>{opaque}</h2><p>{highlight_links}</p><img src="{atlas_path.name}" alt="All24 ordered A/B future frames"></article>')
    (public/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Blinded compiler proof review</title>'
        '<h1>Unrated A/B proof review</h1><p>All24 ordered generated frames; read left-to-right, then top-to-bottom. '
        'Atlases are downscaled and palette-quantized. Open both highlighted native PNG pairs for spatial ambiguity. '
        'Static viewing is not realtime playback. No inspection, acceptance, safety or action-adherence claim has been made.</p>'
        '<p><a href="native-index.html">All native frame pairs</a> | <a href="review.csv">Blank review form</a></p>'+''.join(links)+'</html>')
    (public/'highlights.json').write_text(json.dumps({'type':'unrated_blind_proof_export_v1','cases':public_cases,'reviewed':False},indent=2)+'\n')
    files={str(p.relative_to(public)):file_sha256(p) for p in public.rglob('*') if p.is_file()}
    check(all(file_sha256(p)==h for p,h in {**source_bindings,**registry}.items()),'Source changed during export')
    check(qualifier.source_identity()==qualifier.SOURCE,'Frozen package changed during export')
    helper.private_json(private/'export-receipt.json',{'type':'compiler_norm_proof_review_export_v1','created_at':datetime.now(timezone.utc).isoformat(),
        'source_sha256':source_bindings,'policy_sha256':plan['policy_sha256'],'proof_sha256':proof_hash,'review_plan_sha256':plan_hash,
        'private_mapping':bound(review['private_key']),'native_receipt':bound(bundle/'private-receipt.json'),
        'cases':private_cases,'public_artifact_sha256':files,'reviewed':False,'qualified':False,
        'inspection_declarations_created':False,'originals_modified':False,
        'conversion':'Exact private float32 NPY values; native PNGs use round(value*255)uint8; ordered atlases use established GIF preview.',
        'scope':'Export only. Original NPZs and exact output hashes remain authoritative.'})
    return {'type':'unrated_blind_proof_export_v1','public_index':str(public/'index.html'),'case_count':len(mappings),
            'public_artifact_sha256':files,'reviewed':False,'qualified':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('proof','review-plan','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--minimum-free-bytes',type=int,default=1024**3)
    args=parser.parse_args(argv)
    result=export(args.proof,args.review_plan,args.output,minimum_free_bytes=args.minimum_free_bytes)
    print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
