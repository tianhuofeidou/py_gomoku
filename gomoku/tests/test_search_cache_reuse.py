import copy
import hashlib
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from gomoku.tools import teacher_train as V,teacher_tune as U,tune_params as T
from gomoku.tests.test_teacher_train import FakeTeacher
from gomoku.tests.test_teacher_tune import position,sample_games


class SearchReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.source=self.root/'source';self.out=self.root/'out'
        self.settings=dict(t0=80,node_limit=50000,search_seconds=15,margin=.08,cost_ratio=1.1,time_ratio=1.25)
        self.params=T.defaults();self.pos=position()
        self.result=dict(rows=[dict(r=0,c=0,state=0,score=1.,constant=1.,terms=[])],
                         move=[0,0],nodes=1,leaves=0,seconds=.01,reason='none')

    def origin(self,record=True,new=False,settings=None):
        settings=dict(settings or self.settings)
        dataset=dict(schema=1,positions=[self.pos])
        identity=dict(schema=V.SCHEMA,source=T.digest('legacy-optimizer'),base_source=U.source_fingerprint(),
                      python=sys.version,settings=settings,dataset=T.digest(dataset),seed=1)
        if new:identity['search_source']=V.search_fingerprint()
        U.write_json(self.source/'checkpoint.json',dict(identity=identity,status='complete'))
        U.write_json(self.source/'positions.json',dataset)
        effective={k:v for k,v in self.params.items() if not record or k not in T.LEAF_KEYS}
        key=T.digest([identity.get('search_source',identity['source']),settings,self.pos,effective,record])
        path=self.source/('trees' if record else 'probes')/(key+'.json')
        U.write_json(path,dict(key=key,result=self.result,payload_sha256=T.digest(self.result)))
        return path

    def corpus(self,settings=None):
        return V.Corpus(self.out,[self.pos],FakeTeacher('fixture',self.out),
                        settings or self.settings,2,time.monotonic()+60,reuse_search=[self.source])

    def edit_identity(self,**changes):
        data=U.read_json(self.source/'checkpoint.json');data['identity'].update(changes)
        U.write_json(self.source/'checkpoint.json',data)

    def test_legacy_namespace_reuses_across_optimizer_identity_and_is_read_only(self):
        self.origin();before={p.relative_to(self.source):p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        with mock.patch.object(V,'fingerprint',return_value=T.digest('different-optimizer')):
            corpus=self.corpus()
            with mock.patch.object(U,'isolated_search',side_effect=AssertionError('must reuse')):
                self.assertEqual(corpus.cache.tree(self.pos,self.params),self.result)
        self.assertEqual(corpus.search_reuse.stats,dict(trees=1,probes=0))
        self.assertEqual(before,{p.relative_to(self.source):p.read_bytes() for p in self.source.rglob('*') if p.is_file()})

    def test_new_search_namespace_reuses_and_receipts_resume_without_checkpoint_stats(self):
        self.origin(new=True);corpus=self.corpus()
        with mock.patch.object(U,'isolated_search',side_effect=AssertionError('must reuse')):
            corpus.cache.tree(self.pos,self.params)
            resumed=self.corpus();self.assertEqual(resumed.search_reuse.stats['trees'],1)
            resumed.cache.tree(self.pos,self.params)
            self.assertEqual(resumed.search_reuse.stats['trees'],1)
        data=U.read_json(next((self.out/'trees').glob('*.json')))
        self.assertIn('search_cache_reuse',data)

    def test_base_code_or_old_memory_context_is_rejected(self):
        self.origin();self.edit_identity(base_source=T.digest('old-memory-context'))
        with self.assertRaisesRegex(ValueError,'基础源码/上下文'):self.corpus()

    def test_python_version_is_rejected(self):
        self.origin();self.edit_identity(python='3.13 wrong runtime')
        with self.assertRaisesRegex(ValueError,'Python'):self.corpus()

    def test_each_actual_search_budget_difference_is_rejected(self):
        self.origin()
        for key in ('t0','node_limit','search_seconds'):
            settings=dict(self.settings);settings[key]+=1
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'预算'):self.corpus(settings)

    def test_acceptance_threshold_change_does_not_change_actual_search(self):
        self.origin();settings=dict(self.settings,margin=.09,cost_ratio=1.2,time_ratio=1.3)
        corpus=self.corpus(settings)
        with mock.patch.object(U,'isolated_search',side_effect=AssertionError('must reuse')):
            self.assertEqual(corpus.cache.tree(self.pos,self.params),self.result)
        target=next((self.out/'trees').glob('*.json'))
        expected=T.digest([corpus.search_source,settings,self.pos,
                           {k:v for k,v in self.params.items() if k not in T.LEAF_KEYS},True])
        self.assertEqual(target.stem,expected)

    def test_unknown_namespace_and_dataset_corruption_are_rejected(self):
        self.origin(new=True);self.edit_identity(search_source=T.digest('wrong-search'))
        with self.assertRaisesRegex(ValueError,'命名空间'):self.corpus()
        self.origin();U.write_json(self.source/'positions.json',dict(positions=[]))
        with self.assertRaisesRegex(ValueError,'局面指纹'):self.corpus()

    def test_incomplete_source_and_nested_destination_are_rejected(self):
        self.origin();data=U.read_json(self.source/'checkpoint.json');data['status']='paused'
        U.write_json(self.source/'checkpoint.json',data)
        with self.assertRaisesRegex(ValueError,'已完成'):self.corpus()
        self.origin();self.out=self.source/'nested'
        with self.assertRaisesRegex(ValueError,'目录之外'):self.corpus()

    def test_leaf_changes_reuse_full_tree_but_not_probe(self):
        self.origin();self.origin(record=False)
        changed=dict(self.params,my_0=self.params['my_0']*1.01);T.validate(changed,self.params)
        corpus=self.corpus()
        with mock.patch.object(U,'isolated_search',return_value=dict(self.result,nodes=2)) as search:
            self.assertEqual(corpus.cache.tree(self.pos,changed,True)['nodes'],1)
            search.assert_not_called()
            self.assertEqual(corpus.cache.tree(self.pos,changed,False)['nodes'],2)
            self.assertEqual(search.call_count,1)

    def test_front_parameter_and_record_flag_misses_are_new_searches(self):
        self.origin();corpus=self.corpus()
        with mock.patch.object(U,'isolated_search',return_value=self.result) as search:
            corpus.cache.tree(self.pos,dict(self.params,type_5=self.params['type_5']+1),True)
            corpus.cache.tree(self.pos,self.params,False)
            self.assertEqual(search.call_count,2)
        self.assertEqual(corpus.search_reuse.stats,dict(trees=0,probes=0))

    def test_complete_position_metadata_and_history_must_match(self):
        self.origin();corpus=self.corpus()
        changed=copy.deepcopy(self.pos);changed['id']='different-position'
        with mock.patch.object(U,'isolated_search',return_value=self.result) as search:
            corpus.cache.tree(changed,self.params,True)
            changed=copy.deepcopy(self.pos);changed['moves']=changed['moves'][:-1]
            corpus.cache.tree(changed,self.params,True)
            self.assertEqual(search.call_count,2)
        self.assertEqual(corpus.search_reuse.stats['trees'],0)

    def test_corrupt_source_key_or_payload_fails_instead_of_falling_back(self):
        path=self.origin()
        for field,value in [('key','wrong'),('result',dict(nodes=999)),('payload_sha256','wrong')]:
            with self.subTest(field=field):
                path=self.origin();data=U.read_json(path);data[field]=value;U.write_json(path,data)
                corpus=self.corpus()
                with mock.patch.object(U,'isolated_search',side_effect=AssertionError('no fallback')):
                    with self.assertRaisesRegex(ValueError,'指纹'):corpus.cache.tree(self.pos,self.params)

    def test_corrupt_target_is_not_silently_replaced(self):
        self.origin();corpus=self.corpus()
        key=T.digest([corpus.search_source,self.settings,self.pos,
                      {k:v for k,v in self.params.items() if k not in T.LEAF_KEYS},True])
        path=self.out/'trees'/(key+'.json');U.write_json(path,dict(key=key,result=self.result,payload_sha256='wrong'))
        with self.assertRaisesRegex(ValueError,'指纹'):corpus.cache.tree(self.pos,self.params)
        self.assertEqual(U.read_json(path)['payload_sha256'],'wrong')

    def test_origin_identity_drift_is_detected_after_entry_was_imported(self):
        self.origin();corpus=self.corpus();corpus.cache.tree(self.pos,self.params)
        self.edit_identity(seed=999)
        with self.assertRaisesRegex(ValueError,'身份发生变化'):corpus.check_deadline()

    def test_corrupt_resume_receipt_is_refused(self):
        self.origin();corpus=self.corpus();corpus.cache.tree(self.pos,self.params)
        path=next((self.out/'trees').glob('*.json'));data=U.read_json(path)
        data['search_cache_reuse']['search_source']=T.digest('wrong-source')
        U.write_json(path,data)
        with self.assertRaisesRegex(ValueError,'回执'):self.corpus()

    def test_parallel_duplicate_hits_commit_one_receipt(self):
        from concurrent.futures import ThreadPoolExecutor
        self.origin();corpus=self.corpus()
        with mock.patch.object(U,'isolated_search',side_effect=AssertionError('must reuse')):
            with ThreadPoolExecutor(max_workers=4) as jobs:
                values=list(jobs.map(lambda _:corpus.cache.tree(self.pos,self.params),range(8)))
        self.assertTrue(all(v==self.result for v in values))
        self.assertEqual(corpus.search_reuse.stats['trees'],1)

    def test_serial_calibration_is_not_reused_from_another_experiment(self):
        self.pos.update(split='validation');self.origin();corpus=self.corpus()
        old_key=T.digest([T.digest('legacy-optimizer'),self.settings,self.pos,self.params,'serial-calibration-v1'])
        U.write_json(self.source/'calibration'/(old_key+'.json'),dict(result=self.result,sha256=T.digest(self.result)))
        with mock.patch.object(U,'isolated_search',return_value=self.result) as search:
            cost=corpus.calibrate(self.params,'validation')
            self.assertEqual(search.call_count,1);self.assertEqual(cost['nodes'],1)

    def test_real_cli_reuses_searches_but_runs_new_serial_measurements(self):
        games=self.root/'games.json';U.write_json(games,sample_games())
        args=['run','--games',str(games),'--teacher','fixture','--positions','12',
              '--groups','leaf','--rounds','1','--epochs','1','--workers','2','--max-seconds','120']
        with mock.patch.object(U,'RapfiTeacher',FakeTeacher):
            self.assertEqual(V.main(args+['--out',str(self.source)]),0)
            before={p.relative_to(self.source):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob('*') if p.is_file()}
            native=U.isolated_search
            with mock.patch.object(U,'isolated_search',wraps=native) as searches:
                target=args+['--out',str(self.out),'--reuse-search-cache',str(self.source),
                             '--reuse-labels',str(self.source/'labels')]
                query_count=len(FakeTeacher.calls)
                self.assertEqual(V.main(target),0)
                self.assertEqual(len(FakeTeacher.calls),query_count)
                checkpoint=U.read_json(self.out/'checkpoint.json')
                positions=U.read_json(self.out/'positions.json')['positions']
                nval=sum(p['split']=='validation' for p in positions)
                distinct=len({T.digest(checkpoint[k]) for k in ('baseline','params')})
                self.assertEqual(searches.call_count,nval*distinct)
                self.assertGreater(checkpoint['search_cache_reuse']['trees'],0)
                self.assertGreater(checkpoint['search_cache_reuse']['probes'],0)
            with mock.patch.object(U,'isolated_search',side_effect=AssertionError('complete resume')):
                self.assertEqual(V.main(target+['--resume']),0)
            self.assertEqual(before,{p.relative_to(self.source):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob('*') if p.is_file()})


if __name__=='__main__':unittest.main()

# 本轮发布默认改为8/3/120；以下旧机制回归显式固定旧口径。
from gomoku.tests._legacy_defaults import enter_legacy as _enter_legacy, leave_legacy as _leave_legacy

def setUpModule():
    _enter_legacy()

def tearDownModule():
    _leave_legacy()