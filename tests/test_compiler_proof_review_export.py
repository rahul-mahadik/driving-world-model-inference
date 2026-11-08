"""Synthetic CPU display export contracts; these fixtures are not GPU evidence."""
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import export_compiler_proof_review as bridge
from scripts import qualify_compiler_numerics as q
from drivebench.config import file_sha256


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def fixture(tmp):
    ids=['private-request:c00','private-request:c01'];g={'total_frames':25,'conditioning_frames':1,'height':8,'width':8}
    a=np.broadcast_to(np.linspace(.05,.8,25,dtype=np.float32)[:,None,None,None],(25,8,8,3)).copy()
    data={key:a.copy() for key in ids};new=copy.deepcopy(data)
    new[ids[0]][7,2,3,1]+=.08;new[ids[0]][11]+=.01
    new[ids[1]][9,1,1,2]+=.12;new[ids[1]][14]+=.01
    runs={}
    for name,values in [('upstream',data),('refactored',new)]:
        path=tmp/f'{name}.npz';np.savez(path,**values)
        runs[name]={'outputs':{'normal':str(path)},'outputs_sha256':{'normal':file_sha256(path)}}
    proof={'schema_version':2,'mode':'all','passed':False,'atol':0,'rtol':0,'implementation_sha256':q.SOURCE,
           'generation':g,'runs':runs,'input_provenance':{'request_id':'private-request','scene_id':'private-scene','candidate_ids':ids,
                                                      'seeds':[23,24],'actions_sha256':['a'*64,'b'*64]}}
    path=put(tmp/'proof.json',proof);cases={}
    for key in ids:
        error=q.rgb_errors(data[key],new[key],1)
        cases['selected-proof:'+key]={'candidate_id':key,'request_id':'private-request',
            'reference_output_sha256':runs['upstream']['outputs_sha256']['normal'],
            'candidate_output_sha256':runs['refactored']['outputs_sha256']['normal'],
            'native_frame_indices':{'maximum_absolute_error':error['max_abs_location_THWC'][0],
                                   'maximum_frame_rmse':max(error['per_frame'],key=lambda x:x['rmse'])['frame']}}
    bindings={str(path):file_sha256(path),str(Path(q.__file__).resolve()):file_sha256(q.__file__)}
    bindings.update({v['outputs']['normal']:v['outputs_sha256']['normal'] for v in runs.values()})
    plan={'type':'compiler_norm_localized_review_plan_v1','status':'unrated_private_export_plan','qualified':False,'rule':q.REVIEW_RULE,
          'proof_sha256':file_sha256(path),'policy_sha256':'c'*64,'artifact_sha256':bindings,'cases':cases}
    pp=put(tmp/'private-plan.json',plan)
    return path,pp,tmp/'export',proof,plan,data,new


def run(f):return bridge.export(f[0],f[1],f[2],minimum_free_bytes=0)


def test_exact_arrays_native_frames_atlases_and_unrated_public_output(tmp_path):
    f=fixture(tmp_path);proof_hash=file_sha256(f[0]);plan_hash=file_sha256(f[1]);result=run(f)
    assert result['case_count']==2 and result['reviewed'] is False and result['qualified'] is False
    private=f[2]/'private';public=f[2]/'public'
    receipt=json.loads((private/'export-receipt.json').read_text())
    assert private.stat().st_mode & 0o777==0o700
    assert (private/'export-receipt.json').stat().st_mode & 0o777==0o600
    mapping=json.loads(Path(receipt['private_mapping']['path']).read_text())['cases']
    for item in mapping:
        key=item['candidate_id'];a,b=(f[6][key],f[5][key]) if item['A']==item['run'] else (f[5][key],f[6][key])
        for frame in range(1,25):
            with Image.open(public/f'{item["case_id"]}-frame-{frame:04}.png') as image:
                pixels=np.asarray(image)
                assert image.size==(16,36)
                assert np.array_equal(pixels[28:,:8],np.rint(a[frame]*255).astype(np.uint8))
                assert np.array_equal(pixels[28:,8:],np.rint(b[frame]*255).astype(np.uint8))
        assert np.array_equal(np.load(item['reference_path']),f[5][key])
        assert np.array_equal(np.load(item['candidate_path']),f[6][key])
        assert Path(item['reference_path']).stat().st_mode & 0o777==0o600
        with Image.open(public/f'{item["case_id"]}-all24.png') as image:assert image.size==(1920,1608)
    rows=list(csv.DictReader((public/'review.csv').open()))
    assert len(rows)==2 and all(all(value=='' for key,value in row.items() if key!='case_id') for row in rows)
    texts='\n'.join(p.read_text() for p in public.iterdir() if p.suffix in ('.html','.json','.csv'))
    assert all(secret not in texts for secret in ('private-request','private-scene','compiled_candidate','upstream_reference',str(tmp_path)))
    assert file_sha256(f[0])==proof_hash and file_sha256(f[1])==plan_hash
    assert all(file_sha256(public/name)==h for name,h in receipt['public_artifact_sha256'].items())
    assert receipt['inspection_declarations_created'] is False
    with pytest.raises(ValueError,match='fresh'):run(f)


@pytest.mark.parametrize('fault',['proof','npz','plan_coverage','highlight','array_hash','helper','range','dtype','shape','insufficient_space'])
def test_fail_closed_sources_plan_and_disk(tmp_path,monkeypatch,fault):
    f=fixture(tmp_path);proof,plan=f[3],f[4]
    if fault=='proof':proof['atol']=.5;put(f[0],proof)
    elif fault=='npz':Path(proof['runs']['refactored']['outputs']['normal']).write_bytes(b'changed')
    elif fault=='plan_coverage':plan['cases'].pop(next(iter(plan['cases'])))
    elif fault=='highlight':plan['cases'][next(iter(plan['cases']))]['native_frame_indices']['maximum_frame_rmse']=1
    elif fault=='array_hash':plan['cases'][next(iter(plan['cases']))]['candidate_output_sha256']='0'*64
    elif fault=='helper':plan['artifact_sha256'][str(Path(q.__file__).resolve())]='0'*64
    elif fault=='insufficient_space':monkeypatch.setattr(bridge.shutil,'disk_usage',lambda _:SimpleNamespace(free=1))
    else:
        data=copy.deepcopy(f[6]);key=next(iter(data))
        if fault=='range':data[key][0,0,0,0]=1.1
        elif fault=='dtype':data[key]=data[key].astype(np.float64)
        else:data[key]=data[key][:-1]
        path=Path(proof['runs']['refactored']['outputs']['normal']);np.savez(path,**data)
        h=file_sha256(path);proof['runs']['refactored']['outputs_sha256']['normal']=h;put(f[0],proof)
        plan['proof_sha256']=file_sha256(f[0]);plan['artifact_sha256'][str(f[0])]=file_sha256(f[0]);plan['artifact_sha256'][str(path)]=h
        for case in plan['cases'].values():case['candidate_output_sha256']=h
    put(f[1],plan)
    with pytest.raises((ValueError,RuntimeError)):run(f)
    assert not (f[2]/'private/export-receipt.json').exists()


def test_no_acceptance_when_source_changes_during_export(tmp_path,monkeypatch):
    f=fixture(tmp_path);original=bridge.export_review
    def changed(*args,**kwargs):
        result=original(*args,**kwargs);f[0].write_text(f[0].read_text()+' ');return result
    monkeypatch.setattr(bridge,'export_review',changed)
    with pytest.raises(ValueError,match='changed during'):run(f)
    assert not (f[2]/'private/export-receipt.json').exists()
