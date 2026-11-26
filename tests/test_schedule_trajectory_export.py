"""Synthetic trajectory fixtures; no GPU measurements or scientific findings."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

exporter = load('trajectory_export', HERE.parent/'scripts/export_schedule_trajectories.py')
fixtures = load('trajectory_analysis_fixtures', HERE/'test_schedule_study_analysis_v2.py')


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value)+'\n' for value in values))


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = fixtures.SyntheticStudy(self.root)
        for row in self.fixture.rows:
            if row['policy'] != exporter.analysis.POLICIES[0]:
                row['environment_return'] = .123
        self.fixture.save()
        for row in self.fixture.rows:
            episode = self.fixture.study/'episodes'/row['episode_id']
            query_path = episode/'evaluator/queries.jsonl'
            queries = [json.loads(line) for line in query_path.read_text().splitlines()]
            for event in queries:
                if event['event'] == 'query_started':
                    event.update(timeout_seconds=15., warmups=2, device=0,
                        cache_directory='/private/DO-NOT-EXPORT-cache', worker_directory='/private/DO-NOT-EXPORT-worker')
            queries.insert(0, dict(event='created', sampling_seed=row['seed'], repetitions=7, warmups=2, device=0))
            queries.append(dict(event='closed', cleanup_seconds=.01))
            jsonl(query_path, queries)
            workload = next(t['metadata']['workload'] for t in self.fixture.protocol['tasks'] if t['task_id'] == row['task_id'])
            schedule = row['selected_schedule'] or dict(zip(exporter.analysis.SCHEDULE_FIELDS, (True,256,4,256,4,'chunked',256,4)))
            events = [dict(event='reset_started', workload=workload, budget_seconds=90, initial_schedule=schedule),
                      dict(event='baseline', measurement=row['baseline'], elapsed_seconds=.5)]
            if row['policy'] != exporter.analysis.POLICIES[0]:
                events.extend([
                    dict(event='trial_started', action='submit', schedule=schedule, remaining_seconds=89.5),
                    dict(event='trial_returned', measurement=row['confirmation']),
                    dict(event='step', action='submit', schedule_before=schedule, schedule_after=schedule,
                        measurement=row['confirmation'], outcome='ok', reward=.123, submission_score=.2,
                        reward_components=dict(speed_score=.2, elapsed_penalty=.077, invalid_action_penalty=0., failed_measurement_penalty=0.),
                        elapsed_seconds=1., terminated=True, truncated=False)])
            jsonl(self.environment_path(row['episode_id']), events)
        self.fixture.run('analysis')

    def environment_path(self, episode='001'):
        return self.fixture.study/'episodes'/episode/'environment/episode-PRIVATE-DO-NOT-EXPORT/environment.jsonl'

    def queries_path(self, episode='001'):
        return self.fixture.study/'episodes'/episode/'evaluator/queries.jsonl'

    def read_lines(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def run_export(self):
        return exporter.export(self.fixture.study, self.root/'analysis', self.root/'trajectories',
            self.fixture.manifest, self.fixture.frozen, self.fixture.plan, fixture_only=True)

    def inspect_episode(self, episode='001', seen=None):
        spec = next(s for s in self.fixture.protocol['episodes'] if s['episode_id'] == episode)
        workload = next(t['metadata']['workload'] for t in self.fixture.protocol['tasks'] if t['task_id'] == spec['task_id'])
        return exporter.episode_records(exporter.analysis.Inputs(self.fixture.study), spec, workload,
            self.fixture.protocol['source_sha256'], fixture_only=True, seen_trials=seen)

    def test_all_parent_events_samples_and_costs_preserved_without_secrets(self):
        before = {str(p):exporter.analysis.file_hash(p) for p in self.fixture.study.rglob('*') if p.is_file()}
        receipt = self.run_export()
        raw = gzip.decompress((self.root/'trajectories/trajectories.jsonl.gz').read_bytes())
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        self.assertEqual(len(receipt['episodes']),54)
        self.assertEqual(receipt['events'],648)
        self.assertEqual(receipt['streams'],{'evaluator':432, 'environment':216})
        self.assertFalse(receipt['scientific_results'])
        self.assertFalse(receipt['statistical_aggregates'])
        self.assertEqual(sum(x['queries_started'] for x in receipt['episodes']),162)
        self.assertEqual(sum(x['queries_finished'] for x in receipt['episodes']),162)
        completion = next(row for row in rows if row['episode_id']=='001' and row['event']=='query_finished')
        self.assertEqual(completion['measurement']['raw_samples_ms'],[1.]*7)
        self.assertEqual(completion['measurement']['sampled_latency_ms'],1.)
        step = next(row for row in rows if row['episode_id']=='001' and row['event']=='step')
        self.assertEqual(step['elapsed_since_previous_step_seconds'],1.)
        for text in (raw.decode(),json.dumps(receipt)):
            for secret in ('DO-NOT-EXPORT','/private/','GPU-UUID','provider-account','episode-PRIVATE'):
                self.assertNotIn(secret,text)
        after = {str(p):exporter.analysis.file_hash(p) for p in self.fixture.study.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_public_episode_results_must_match_validation(self):
        path=self.root/'analysis/episodes.json'
        value=json.loads(path.read_text());value['episodes'][1]['seed']+=1
        fixtures.dump(path,value)
        with self.assertRaisesRegex(ValueError,'Public episodes differ'):self.run_export()

    def test_unknown_event_rejected(self):
        path=self.environment_path();events=self.read_lines(path);events.append(dict(event='unknown'))
        jsonl(path,events)
        with self.assertRaisesRegex(ValueError,'Unknown environment'):self.inspect_episode()

    def test_sampling_seed_must_match_protocol(self):
        path=self.queries_path();events=self.read_lines(path);events[0]['sampling_seed']+=1;jsonl(path,events)
        with self.assertRaisesRegex(ValueError,'sampling seed'):self.inspect_episode()

    def test_query_workload_identity_rejected(self):
        path=self.queries_path();events=self.read_lines(path);events[1]['workload']['seed']+=1;jsonl(path,events)
        with self.assertRaisesRegex(ValueError,'workload/sample'):self.inspect_episode()

    def test_environment_measurement_must_match_evaluator(self):
        path=self.environment_path();events=self.read_lines(path);events[3]['measurement']['elapsed_seconds']+=1;jsonl(path,events)
        with self.assertRaisesRegex(ValueError,'differs from evaluator'):self.inspect_episode()

    def test_replayed_trial_across_episodes_rejected(self):
        seen=set();self.inspect_episode('001',seen)
        with self.assertRaisesRegex(ValueError,'Repeated evaluator trial'):self.inspect_episode('001',seen)

    def test_schedule_chain_and_action_names_validated(self):
        path=self.environment_path();original=self.read_lines(path)
        for field,value in (('action','unknown'),('schedule_before',{**original[-1]['schedule_before'],'block_size':512})):
            events=copy.deepcopy(original);events[-1][field]=value;jsonl(path,events)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'step state/action'):self.inspect_episode()

    def test_reward_components_and_episode_return_checked(self):
        path=self.environment_path();events=self.read_lines(path);events[-1]['reward']+=.01;jsonl(path,events)
        with self.assertRaisesRegex(ValueError,'reward components'):self.inspect_episode()
        events[-1]['reward_components']['speed_score']+=.01;events[-1]['submission_score']+=.01;jsonl(path,events)
        with self.assertRaisesRegex(ValueError,'episode return'):self.inspect_episode()

    def test_intermediate_runtime_identity_rejected(self):
        path=self.queries_path();events=self.read_lines(path)
        events[4]['measurement']['evidence']['cohort_id']='changed';jsonl(path,events)
        with self.assertRaisesRegex(ValueError,'runtime/input identity'):self.inspect_episode()

    def test_unknown_exception_classes_never_export_their_text(self):
        self.assertEqual(exporter.error_type('RuntimeError: DO-NOT-EXPORT /private/path'),'RuntimeError')
        self.assertEqual(exporter.error_type({'type':'DO-NOT-EXPORT','message':'/private/path'}),'OtherError')

    def test_failed_queries_keep_error_type_and_empty_timings(self):
        path=self.queries_path();queries=self.read_lines(path)
        value=queries[-2]['measurement']
        value.update(status='timeout',correct=False,latency_ms=None,raw_samples_ms=[],peak_allocated_bytes=None)
        value['evidence'].update(error=dict(type='TimeoutError',message='DO-NOT-EXPORT'),worker_reaped=True)
        jsonl(path,queries)
        path=self.environment_path();events=self.read_lines(path)
        events[3]['measurement']=copy.deepcopy(value);events[4]['measurement']=copy.deepcopy(value)
        events[4].update(outcome='timeout',reward=-.1,submission_score=0.,reward_components=dict(speed_score=0.,elapsed_penalty=0.,invalid_action_penalty=0.,failed_measurement_penalty=.1))
        jsonl(path,events)
        result_path=self.fixture.study/'episodes/001/result.json';result=json.loads(result_path.read_text())
        result.update(status='failed',environment_return=-.1);fixtures.dump(result_path,result)
        records,counts=self.inspect_episode()
        final=next(row['measurement'] for row in reversed(records) if row.get('measurement'))
        self.assertEqual(final['error_type'],'TimeoutError');self.assertEqual(final['raw_samples_ms'],[])
        self.assertEqual(counts['failed_queries'],1)
        self.assertNotIn('DO-NOT-EXPORT',json.dumps(records))

    def test_unfinished_query_requires_failed_or_interrupted_episode(self):
        path=self.queries_path();queries=self.read_lines(path);queries=queries[:-2];jsonl(path,queries)
        with self.assertRaisesRegex(ValueError,'Incomplete query'):self.inspect_episode()
        path=self.environment_path();events=self.read_lines(path);events=events[:3];jsonl(path,events)
        result_path=self.fixture.study/'episodes/001/result.json';result=json.loads(result_path.read_text())
        result.update(status='interrupted',environment_return=0.);fixtures.dump(result_path,result)
        records,counts=self.inspect_episode()
        self.assertEqual(counts['queries_without_recorded_completion'],1)
        self.assertEqual(sum(row['event']=='query_started' for row in records),3)

    def test_duplicate_environment_journal_rejected(self):
        source=self.environment_path();other=source.parent.parent/'another/environment.jsonl'
        other.parent.mkdir();other.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError,'exactly one environment'):self.inspect_episode()

    def test_v1_protocol_cannot_pass_production_or_fixture_entrypoint(self):
        self.fixture.protocol['study_id']='sequential-schedules-live-v1';self.fixture.save()
        with self.assertRaisesRegex(ValueError,'Unknown study protocol'):self.run_export()


    def test_failure_before_journal_creation_is_preserved_explicitly(self):
        self.queries_path().unlink();self.environment_path().unlink()
        path=self.fixture.study/'episodes/001/result.json';result=json.loads(path.read_text())
        result.update(status='failed',baseline=None,paired_final_reference=None,confirmation=None,environment_return=None)
        fixtures.dump(path,result)
        records,counts=self.inspect_episode()
        self.assertEqual(records,[])
        self.assertTrue(counts['missing_evaluator_journal'])
        self.assertTrue(counts['missing_environment_journal'])
        self.assertEqual(counts['queries_started'],0)

    def test_correctness_failure_metrics_are_retained(self):
        value=copy.deepcopy(self.fixture.rows[1]['confirmation'])
        value.update(status='incorrect',correct=False,latency_ms=None,raw_samples_ms=[])
        value['evidence'].pop('sampled_index')
        value['evidence']['correctness']=dict(passed=False,finite=True,shape_dtype_match=True,
            rtol=.002,atol=.002,max_abs_error=.3,rmse=.1,private_note='DO-NOT-EXPORT')
        public=exporter.measurement(value,fixture_only=True)
        self.assertFalse(public['correctness']['passed'])
        self.assertEqual(public['correctness']['max_abs_error'],.3)
        self.assertEqual(public['median_ms'],None)
        self.assertNotIn('DO-NOT-EXPORT',json.dumps(public))


    def test_failed_sampled_index_rejects_private_payload_and_invalid_types(self):
        value=copy.deepcopy(self.fixture.rows[1]['confirmation'])
        value.update(status='timeout',correct=False,latency_ms=None,raw_samples_ms=[])
        for index in ({'private_note':'/private/DO-NOT-EXPORT'}, 'DO-NOT-EXPORT', True, -1, 7, .5):
            value['evidence']['sampled_index']=index
            with self.subTest(index=index), self.assertRaisesRegex(ValueError,'Sampled index'):
                exporter.measurement(value,fixture_only=True)
        for index in (None,0,6):
            value['evidence']['sampled_index']=index
            public=exporter.measurement(value,fixture_only=True)
            self.assertEqual(public['sampled_index'],index)
            self.assertIsNone(public['sampled_latency_ms'])

    def test_returned_then_rejected_trial_preserves_return_and_error(self):
        path=self.environment_path();events=self.read_lines(path)
        events.insert(-1,dict(event='trial_error',error='ValueError: /private/DO-NOT-EXPORT'))
        step=events[-1]
        step.update(outcome='evaluator_error',measurement=None,reward=-.1,submission_score=0.,
            reward_components=dict(speed_score=0.,elapsed_penalty=0.,invalid_action_penalty=0.,failed_measurement_penalty=.1))
        jsonl(path,events)
        result_path=self.fixture.study/'episodes/001/result.json';result=json.loads(result_path.read_text())
        result.update(status='failed',environment_return=-.1);fixtures.dump(result_path,result)
        records,_=self.inspect_episode()
        returned=[row for row in records if row['event']=='trial_returned']
        errors=[row for row in records if row['event']=='trial_error']
        self.assertEqual(len(returned),1);self.assertEqual(len(errors),1)
        self.assertEqual(errors[0]['error_type'],'ValueError')
        self.assertIsNone(records[-1]['measurement'])
        self.assertEqual(records[-1]['outcome'],'evaluator_error')
        self.assertNotIn('DO-NOT-EXPORT',json.dumps(records))
        events[-1]['measurement']=events[-3]['measurement'];jsonl(path,events)
        with self.assertRaisesRegex(ValueError,'Rejected trial'):self.inspect_episode()

    def test_null_step_without_trial_error_is_rejected(self):
        path=self.environment_path();events=self.read_lines(path)
        events[-1].update(outcome='evaluator_error',measurement=None);jsonl(path,events)
        with self.assertRaisesRegex(ValueError,'does not match its trial'):self.inspect_episode()

    def test_final_reference_cannot_replace_environment_initial_baseline(self):
        path=self.environment_path();events=self.read_lines(path)
        events[1]['measurement']=copy.deepcopy(self.fixture.rows[1]['paired_final_reference']);jsonl(path,events)
        with self.assertRaisesRegex(ValueError,'initial reference-selection'):self.inspect_episode()

    def test_successful_reset_requires_environment_baseline_event(self):
        path=self.environment_path();events=self.read_lines(path);events.pop(1);jsonl(path,events)
        with self.assertRaisesRegex(ValueError,'missing its environment baseline'):self.inspect_episode()

    def test_failed_reset_may_omit_unaccepted_baseline_event(self):
        path=self.environment_path();events=self.read_lines(path)
        jsonl(path,[events[0],dict(event='reset_failed',error='RuntimeError: DO-NOT-EXPORT')])
        queries=self.read_lines(self.queries_path());jsonl(self.queries_path(),queries[:3]+[queries[-1]])
        result_path=self.fixture.study/'episodes/001/result.json';result=json.loads(result_path.read_text())
        result.update(status='failed',baseline=None,paired_final_reference=None,confirmation=None,environment_return=None)
        fixtures.dump(result_path,result)
        records,counts=self.inspect_episode()
        self.assertEqual(counts['environment_events'],2)
        self.assertEqual(records[-1]['error_type'],'RuntimeError')


if __name__ == '__main__':
    unittest.main()
