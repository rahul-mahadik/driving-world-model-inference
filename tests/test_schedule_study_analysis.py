"""Synthetic software fixtures only; no GPU measurements or study findings."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('schedule_analysis',HERE.parent/'scripts/analyze_schedule_study.py')
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


class SyntheticStudy:
    """Deliberately fixture-labeled fake timings for analysis contract tests."""
    def __init__(self,root):
        self.root=root;self.study=root/'fixture-study';self.study.mkdir()
        self.manifest=root/'fixture-source-manifest.json'
        self.frozen=root/'fixture-frozen-source'
        self.plan=root/'fixture-analysis-plan.json'
        code_hashes={}
        for name in analysis.SOURCE_MODULES:
            path=self.frozen/'extensions/kernel_tuning/drivebench_tuning'/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('# Synthetic software fixture; not GPU source\n')
            code_hashes[name]=analysis.file_hash(path)
        tasks=[]
        for i in range(6):
            op='geglu' if i<3 else 'groupnorm_silu'
            tasks.append(dict(task_id=f'test-{op}-f{i:02d}-float16',metadata=dict(split='test',
                workload=dict(operator=op,shape=[2,8+i],dtype='float16',num_groups=2,eps=1e-5,seed=100+i))))
        self.protocol=dict(schema_version=1,study_id='sequential-schedules-live-v1',fixture_only=True,
            policies=list(analysis.POLICIES),seeds=[7101,7102,7103],episode_seconds=90,tasks=tasks,
            source_sha256=code_hashes,episodes=[])
        self.rows=[]
        for seed in self.protocol['seeds']:
            for task in tasks:
                for position,policy in enumerate(analysis.POLICIES):
                    identity=dict(episode_id=f'{len(self.rows):03d}',task_id=task['task_id'],seed=seed,
                                  policy=policy,order_position=position)
                    self.protocol['episodes'].append(identity)
                    initial=self.measurement(1.)
                    reference=self.measurement(10.)
                    confirmed=self.measurement(10.*(1-.2*position))
                    row={**identity,'status':'finished','successful':True,'environment_completed':True,
                        'workload_key':analysis.canonical_hash(task['metadata']['workload']),
                        'stop_reason':'fixed_reference_confirmed' if position==0 else 'confirmation_reserve',
                        'episode_budget_seconds':90,'elapsed_seconds':10.+position,
                        'exploration_queries':position*10,'failed_queries':0,'invalid_actions':0,'restore_actions':position,
                        'baseline':initial,'paired_final_reference':reference,'confirmation':confirmed,
                        'baseline_median_ms':1.,'paired_final_reference_median_ms':10.,
                        'confirmed_median_ms':10.*(1-.2*position),'confirmed_latency_ratio':1-.2*position,
                        'failure_adjusted_speed_score':.2*position,'selected_reference':'compiled',
                        'selected_schedule':None,'environment_return':None if position==0 else .123,
                        'cache_directory':'/private/DO-NOT-EXPORT-cache',
                        'evidence_directory':'/private/DO-NOT-EXPORT-evidence',
                        'error':{'message':'DO-NOT-EXPORT-rawstderr provider-account-secret'}}
                    schedule={key:value for key,value in zip(analysis.SCHEDULE_FIELDS,
                        (True,256,4,256,4,'chunked',256,4))}
                    if position:row['selected_schedule']=schedule
                    for key,value in [('baseline',initial),('paired_final_reference',reference),('confirmation',confirmed)]:
                        value.update(workload_key=row['workload_key'],schedule=schedule if position and key=='confirmation' else None)
                        value['evidence'].update(trial_id=identity['episode_id']+'-'+key)
                        if key=='baseline':value['evidence']['selected_reference']='compiled'
                        elif key=='paired_final_reference' or not position:
                            value['evidence']['fixed_reference_confirmation']='compiled'
                    self.rows.append(row)
        self.save()

    @staticmethod
    def measurement(value):
        return dict(backend='fixture',status='ok',correct=True,raw_samples_ms=[value]*7,
            latency_ms=value,
            elapsed_seconds=.5,peak_allocated_bytes=1024,
            evidence=dict(runtime={'gpu_uuid':'DO-NOT-EXPORT-GPU-UUID','provider_id':'DO-NOT-EXPORT-account',
                                   'gpu_name':'synthetic fixture'},cohort_id='DO-NOT-EXPORT-cohort',
                          input_sha256=['b'*64],compile_and_first_call_seconds=.1,sampled_index=0))

    def save(self):
        protocol_sha=analysis.canonical_hash(self.protocol)
        self.summary=dict(expected_episodes=54,attempted_episodes=54,missing_episode_ids=[],failure=None,
            protocol_sha256=protocol_sha,all_attempts_preserved=True)
        self.completion=dict(schema_version=1,status='complete',failure=None,expected_episodes=54,
            attempted_episodes=54,all_attempts_preserved=True,protocol_sha256=protocol_sha,
            summary_sha256=analysis.canonical_hash(self.summary),gpu_validation_claim=False,
            all_episodes_successful=all(row['successful'] for row in self.rows))
        for name,value in [('protocol.json',self.protocol),('summary.json',self.summary),('completed.json',self.completion)]:
            dump(self.study/name,value)
        dump(self.frozen/'runtime-inputs/study-protocol.json',self.protocol)
        dump(self.plan,dict(schema_version=1,study_id='sequential-schedules-live-v1',protocol_sha256=protocol_sha,
                            registered_before_gpu_measurement=True,fixture_only=True))
        (self.study/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        for row in self.rows:
            dump(self.study/'episodes'/row['episode_id']/'result.json',row)
            workload=next(task['metadata']['workload'] for task in self.protocol['tasks'] if task['task_id']==row['task_id'])
            events=[]
            for key in ('baseline','paired_final_reference','confirmation'):
                value=row.get(key)
                if value is None:continue
                role=row['selected_reference'] if key=='paired_final_reference' or (key=='confirmation' and row['policy']==analysis.POLICIES[0]) else None
                events.append(dict(event='query_started',trial_id=value['evidence']['trial_id'],workload=workload,
                                   repetitions=7,schedule=value['schedule'],reference_role=role))
                events.append(dict(event='query_finished',measurement=value))
            journal=self.study/'episodes'/row['episode_id']/'evaluator/queries.jsonl'
            journal.parent.mkdir(parents=True,exist_ok=True)
            journal.write_text(''.join(json.dumps(event)+'\n' for event in events))
        files=[dict(path='extensions/kernel_tuning/drivebench_tuning/'+key,sha256=value)
               for key,value in self.protocol['source_sha256'].items()]
        files.append(dict(path='runtime-inputs/study-protocol.json',sha256=analysis.file_hash(self.study/'protocol.json')))
        dump(self.manifest,dict(files=files))

    def run(self,name='export',**kwargs):
        return analysis.export(self.study,self.root/name,self.manifest,self.frozen,self.plan,fixture_only=True,replicates=100,**kwargs)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.fixture=SyntheticStudy(self.root)

    def test_paired_final_denominator_and_privacy_allowlist(self):
        before={p.relative_to(self.fixture.study).as_posix():analysis.file_hash(p) for p in self.fixture.study.rglob('*') if p.is_file()}
        result=self.fixture.run()
        self.assertFalse(result['scientific_results'])
        self.assertEqual(result['evidence_kind'],'synthetic-software-test-fixture')
        self.assertAlmostEqual(result['policies']['random_walk']['mean_primary_score'],.2)
        self.assertAlmostEqual(result['policies']['greedy_local']['mean_primary_score'],.4)
        for path in (self.root/'export').iterdir():
            text=path.read_text()
            self.assertNotIn('DO-NOT-EXPORT',text,path.name)
            self.assertNotIn('provider-account-secret',text,path.name)
            self.assertNotIn('/private/',text,path.name)
        after={p.relative_to(self.fixture.study).as_posix():analysis.file_hash(p) for p in self.fixture.study.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        provenance=json.loads((self.root/'export/provenance.json').read_text())
        self.assertEqual(len(provenance['consumed_files']),125)
        self.assertEqual(len(provenance['scientific_code_sha256_verified_against_frozen_payload']),10)

    def test_failures_are_retained_and_penalized(self):
        row=self.fixture.rows[1]
        row.update(status='failed',successful=False,stop_reason='episode_exception',confirmation=None,
                   confirmed_median_ms=None,confirmed_latency_ratio=None,failure_adjusted_speed_score=-1.,failed_queries=1)
        self.fixture.save();result=self.fixture.run()
        random=result['policies']['random_walk']
        self.assertEqual(random['attempts'],18);self.assertEqual(random['confirmed'],17)
        self.assertAlmostEqual(random['mean_primary_score'],(17*.2-1)/18)
        self.assertAlmostEqual(random['median_confirmed_latency_ratio_successes_only'],.8)
        exported=json.loads((self.root/'export/episodes.json').read_text())['episodes']
        self.assertEqual(len(exported),54);self.assertEqual(exported[1]['primary_score'],-1.)

    def test_paired_hierarchical_resampling_and_reproducibility(self):
        _,tasks,rows,_,_=analysis.validate_and_sanitize(analysis.Inputs(self.fixture.study),self.fixture.manifest,self.fixture.frozen,fixture_only=True)
        for row in rows:
            task_index=sorted(tasks).index(row['task_id'])
            row['primary_score']+=task_index*.03+(row['seed']-7101)*.01
        first=analysis.aggregate(rows,tasks,[7101,7102,7103],replicates=200,bootstrap_seed=20261125)
        second=analysis.aggregate(rows,tasks,[7101,7102,7103],replicates=200,bootstrap_seed=20261125)
        self.assertEqual(first,second)
        self.assertGreater(first['policies']['random_walk']['primary_score_ci95'][1]-first['policies']['random_walk']['primary_score_ci95'][0],0)
        # Correlated task/seed effects cancel exactly only with paired draws.
        for pair in first['paired_comparisons']:
            self.assertAlmostEqual(pair['paired_difference_ci95'][0],pair['mean_score_difference'])
            self.assertAlmostEqual(pair['paired_difference_ci95'][1],pair['mean_score_difference'])

    def test_wrong_denominator_score_rejected(self):
        self.fixture.rows[1]['failure_adjusted_speed_score']=-1.
        self.fixture.save()
        with self.assertRaisesRegex(ValueError,'primary score'):self.fixture.run()

    def test_changed_manifest_or_protocol_rejected(self):
        manifest=json.loads(self.fixture.manifest.read_text());manifest['files'][0]['sha256']='c'*64
        dump(self.fixture.manifest,manifest)
        with self.assertRaisesRegex(ValueError,'code hash'):self.fixture.run()

    def test_dropped_or_reordered_attempt_rejected(self):
        rows=copy.deepcopy(self.fixture.rows)
        self.fixture.rows=self.fixture.rows[:-1];self.fixture.save()
        with self.assertRaisesRegex(ValueError,'54 attempts'):self.fixture.run()
        self.fixture.rows=rows
        self.fixture.rows[0],self.fixture.rows[1]=self.fixture.rows[1],self.fixture.rows[0]
        self.fixture.save()
        with self.assertRaisesRegex(ValueError,'identity or order'):self.fixture.run()

    def test_cross_gpu_pooling_rejected(self):
        for field in ('baseline','paired_final_reference','confirmation'):
            self.fixture.rows[-1][field]['evidence']['cohort_id']='second-test-only-cohort'
        self.fixture.save()
        with self.assertRaisesRegex(ValueError,'cohorts cannot be pooled'):self.fixture.run()

    def test_source_episode_mismatch_rejected(self):
        row=copy.deepcopy(self.fixture.rows[0]);row['elapsed_seconds']+=1
        dump(self.fixture.study/'episodes/000/result.json',row)
        with self.assertRaisesRegex(ValueError,'append-only'):self.fixture.run()

    def test_replayed_query_identity_rejected(self):
        row=self.fixture.rows[1]
        row['confirmation']['evidence']['trial_id']=row['paired_final_reference']['evidence']['trial_id']
        self.fixture.save()
        with self.assertRaisesRegex(ValueError,'Repeated query identity'):self.fixture.run()

    def test_wrong_workload_schedule_or_reference_binding_rejected(self):
        original=copy.deepcopy(self.fixture.rows[1])
        for mutation,expected in [('workload','workload key'),('schedule','selected schedule'),('role','reference role')]:
            row=copy.deepcopy(original)
            if mutation=='workload':row['confirmation']['workload_key']='c'*64
            elif mutation=='schedule':row['confirmation']['schedule']={**row['confirmation']['schedule'],'block_size':512}
            else:row['paired_final_reference']['evidence']['fixed_reference_confirmation']='eager'
            self.fixture.rows[1]=row;self.fixture.save()
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,expected):self.fixture.run(name='export-'+mutation)

    def test_incomplete_samples_or_non_sample_observation_rejected(self):
        original=copy.deepcopy(self.fixture.rows[1])
        for mutation in ('samples','latency','index'):
            row=copy.deepcopy(original)
            if mutation=='samples':row['confirmation']['raw_samples_ms'].pop()
            elif mutation=='latency':row['confirmation']['latency_ms']=999.
            else:row['confirmation']['evidence']['sampled_index']=99
            self.fixture.rows[1]=row;self.fixture.save()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.fixture.run()

    def test_changed_query_journal_rejected(self):
        journal=self.fixture.study/'episodes/001/evaluator/queries.jsonl'
        events=[json.loads(line) for line in journal.read_text().splitlines()]
        events[-1]['measurement']['elapsed_seconds']+=1
        journal.write_text(''.join(json.dumps(event)+'\n' for event in events))
        with self.assertRaisesRegex(ValueError,'evaluator journal'):self.fixture.run()

    def test_changed_actual_frozen_code_rejected(self):
        path=self.fixture.frozen/'extensions/kernel_tuning/drivebench_tuning/schedule.py'
        path.write_text('# Changed synthetic software fixture\n')
        with self.assertRaisesRegex(ValueError,'Actual frozen scientific code'):self.fixture.run()

    def test_wrong_analysis_plan_protocol_binding_rejected(self):
        plan=json.loads(self.fixture.plan.read_text());plan['protocol_sha256']='c'*64
        dump(self.fixture.plan,plan)
        with self.assertRaisesRegex(ValueError,'Analysis plan does not bind'):self.fixture.run()

    def test_production_runtime_identity_and_code_hashes_are_verified(self):
        # Pure metadata contract fixture; no real GPU or scientific export.
        sources={name:'a'*64 for name in analysis.SOURCE_MODULES}
        runtime={'gpu_uuid':'GPU-software-test-only','source_hashes':sources.copy()}
        identity={'runtime':runtime,'cohort_id':hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest()}
        analysis.verify_runtime_identity(identity,sources,fixture_only=False)
        for mutation,expected in [('uuid','physical GPU identity'),('cohort','cohort digest'),('source','scientific code')]:
            changed=copy.deepcopy(identity)
            if mutation=='uuid':changed['runtime']['gpu_uuid']=''
            elif mutation=='cohort':changed['cohort_id']='0'*64
            else:
                changed['runtime']['source_hashes']['schedule.py']='b'*64
                changed['cohort_id']=hashlib.sha256(json.dumps(changed['runtime'],sort_keys=True).encode()).hexdigest()
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,expected):
                analysis.verify_runtime_identity(changed,sources,fixture_only=False)

    def test_fixture_cannot_be_exported_as_gpu_evidence(self):
        with self.assertRaisesRegex(ValueError,'fixture-only'):
            analysis.export(self.fixture.study,self.root/'export',self.fixture.manifest,self.fixture.frozen,self.fixture.plan,replicates=100)

    def test_existing_output_and_evidence_destination_rejected(self):
        self.fixture.run()
        with self.assertRaisesRegex(ValueError,'new directory'):self.fixture.run()
        with self.assertRaisesRegex(ValueError,'outside source evidence'):
            analysis.export(self.fixture.study,self.fixture.study/'export',self.fixture.manifest,self.fixture.frozen,self.fixture.plan,fixture_only=True,replicates=100)

    def test_all_failed_attempts_with_bound_baselines_remain_exportable(self):
        for row in self.fixture.rows:
            row.update(status='failed',successful=False,stop_reason='episode_exception',
                paired_final_reference=None,confirmation=None,
                paired_final_reference_median_ms=None,confirmed_median_ms=None,confirmed_latency_ratio=None,
                failure_adjusted_speed_score=-1.)
        self.fixture.save();result=self.fixture.run()
        self.assertEqual(result['scope']['verified_hardware_runtime_cohorts'],1)
        for policy in result['policies'].values():
            self.assertEqual(policy['confirmed'],0);self.assertEqual(policy['mean_primary_score'],-1.)

    def test_no_runtime_cohort_cannot_be_exported_as_complete_measured_study(self):
        for row in self.fixture.rows:
            row.update(status='failed',successful=False,stop_reason='episode_exception',baseline=None,
                paired_final_reference=None,confirmation=None,baseline_median_ms=None,
                paired_final_reference_median_ms=None,confirmed_median_ms=None,confirmed_latency_ratio=None,
                failure_adjusted_speed_score=-1.,selected_reference=None)
        self.fixture.save()
        with self.assertRaisesRegex(ValueError,'No verified runtime cohort'):self.fixture.run()


if __name__=='__main__':unittest.main()
