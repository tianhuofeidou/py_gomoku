"""新版训练反例：公共参考、收益交换、留出组、并发预算与中断恢复。"""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from gomoku.tools import teacher_train as V, teacher_tune as U, tune_params as T
from gomoku.tools import leaf_fit as F, training_jobs as J
from gomoku.tests.test_teacher_tune import MOVES, position, sample_games


def fit_cases(count=4):
    cases=[]
    for i in range(count):
        a,b=[0]*24+[1.0],[0]*24+[1.0]
        a[0],b[4]=1,1
        rows=[dict(r=1,c=1,state=0,constant=0,terms=[[1,a]]),
              dict(r=1,c=2,state=0,constant=0,terms=[[1,b]])]
        cases.append(dict(position=dict(id=str(i),group=str(i)),tree=dict(rows=rows),
                          utilities={'1,1':-1.0,'1,2':1.0}))
    return cases


class FakeTeacher:
    calls=[]
    def __init__(self, exe, out, nodes=200000, milliseconds=2000, deadline=None,max_queries=5000):
        self.identity=dict(fixture='parallel-labels',nodes=nodes,milliseconds=milliseconds)
        self.cache=Path(out)/'labels'
        self.cache.mkdir(parents=True,exist_ok=True)
        self.max_queries=max_queries
        self.deadline=deadline or float('inf')
        self.queries=0

    def _query(self,moves):
        key=T.digest([self.identity,moves])
        path=self.cache/(key+'.json')
        if path.exists():
            data=U.read_json(path)
            data.pop('payload_sha256')
            return data
        self.calls.append(key)
        board,_=U.board_of(moves)
        move=next([r,c] for r in range(15) for c in range(15) if not board[r][c])
        data=dict(key=key,identity=self.identity,move=move,
                  evaluation=dict(utility=((moves[-1][0]+moves[-1][1])%5)/10),seconds=0)
        data['payload_sha256']=T.digest(data)
        U.write_json(path,data)
        return data


class FitterTests(unittest.TestCase):
    def test_score_coordinate_preserves_other_groups_and_checks_cost_before_labels(self):
        from types import SimpleNamespace
        import argparse
        base=T.defaults()
        for key in T.GROUPS['score']:
            for item in F.coordinate_proposals(key,base,base,'score'):
                T.validate(item['params'],base)
                self.assertTrue(all(item['params'][k]==base[k] for k in base if k not in T.GROUPS['score']))
        self.assertEqual(set(F.coordinate_order(1,'score')),set(T.GROUPS['score']))
        with self.assertRaises(ValueError):
            F.coordinate_order(1,'gear')
        with self.assertRaises(ValueError):
            list(F.coordinate_proposals('my_0',base,base,'score'))
        self.assertEqual(V.score_axes('neighbor,type_5'),['neighbor','type_5'])
        for value in ('','neighbor,neighbor','my_0','unknown'):
            with self.assertRaises(argparse.ArgumentTypeError):
                V.score_axes(value)

        class Front:
            deadline=float('inf')
            def __init__(self,stop=None):
                self.calls=0;self.stop=stop;self.ready=False;self.events=[]
            def check_deadline(self):
                self.calls+=1
                if self.calls==self.stop:
                    raise U.BudgetStop('fixture-front-pause')
            def compute_guard(self,params,*args):
                self.events.append(('guard',params['neighbor']))
                return params['neighbor']>=base['neighbor'],1
            def prepare(self,parameters,split):
                self.assert_parameters=parameters
                self.ready=True
                self.events.append(('prepare',parameters[-1]['neighbor']))
            def metrics(self,params,*args):
                self.events.append(('metrics',params['neighbor']))
                return dict(regret=.5+(.1 if self.ready else 0)-params['neighbor']/10000,
                    family_regret=.2,selection_regret=.2,worst_fold_regret=.3,tail_regret=.5,severe=1,new_bad=0)
            def verify(self,*args):
                self.events.append(('verify',args[0]['neighbor']))
                return dict(nodes=1)

        initial=dict(params=dict(base),initial=dict(base),history=[],baseline_train_compute=dict(nodes=1))
        continuous=copy.deepcopy(initial);interrupted=copy.deepcopy(initial)
        full,partial=Front(),Front(4)
        with tempfile.TemporaryDirectory() as directory,mock.patch.object(F,'coordinate_order',return_value=['type_5','neighbor']):
            args=SimpleNamespace(seed=1,score_keys=['neighbor'])
            V.coordinate_phase(full,continuous,0,args,directory,base,'score')
            with self.assertRaises(U.BudgetStop):
                V.coordinate_phase(partial,interrupted,0,args,directory,base,'score')
            saved=U.read_json(Path(directory)/'checkpoint.json')
            partial.stop=None
            V.coordinate_phase(partial,saved,0,args,directory,base,'score')
            self.assertEqual(saved,continuous)
        rejected=[e for e in continuous['history'] if e['direction']==-1]
        self.assertTrue(all(e['metrics'] is None and not e['accepted'] for e in rejected))
        self.assertTrue(all(value>=base['neighbor'] for event,value in full.events if event=='prepare'))
        first=next(e for e in continuous['history'] if e['accepted'])
        self.assertAlmostEqual(first['previous']['regret'],.6-base['neighbor']/10000)
        self.assertEqual([event for event,value in full.events[:5]],['guard','guard','prepare','metrics','metrics'])
        self.assertGreater(continuous['params']['neighbor'],base['neighbor'])
        self.assertTrue(all(continuous['params'][k]==base[k] for k in T.LEAF_KEYS))

    def test_coordinate_grid_bounds_and_resume_after_accepted_proposal(self):
        from types import SimpleNamespace
        base=T.defaults()
        proposals=list(F.coordinate_proposals('my_0',base,base))
        self.assertEqual({p['direction'] for p in proposals},{-1,1})
        self.assertEqual(F.coordinate_order(9),F.coordinate_order(9))
        self.assertEqual(set(F.coordinate_order(9)),set(T.LEAF_KEYS))
        for item in proposals:
            T.validate(item['params'],base)
            self.assertTrue(all(item['params'][k]==base[k] for k in base if k not in T.LEAF_KEYS))
        with self.assertRaises(ValueError):
            list(F.coordinate_proposals('type_5',base,base))

        class Replay:
            def __init__(self,stop=None):
                self.calls=0;self.stop=stop
            def check_deadline(self):
                self.calls+=1
                if self.calls==self.stop:
                    raise U.BudgetStop('fixture-coordinate-pause')
            def metrics(self,params,*args):
                return dict(regret=.5-params['my_0']/10000,family_regret=.2,selection_regret=.2,
                    worst_fold_regret=.3,tail_regret=.5,severe=1,new_bad=0)
            def verify(self,*args):return dict(nodes=1)
            def compute_guard(self,*args):return True,1

        initial=dict(params=dict(base),history=[],baseline_train_compute=dict(nodes=1))
        continuous=copy.deepcopy(initial);interrupted=copy.deepcopy(initial)
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(seed=1)
            V.coordinate_phase(Replay(),continuous,0,args,directory,base)
            with self.assertRaises(U.BudgetStop):
                V.coordinate_phase(Replay(4),interrupted,0,args,directory,base)
            saved=U.read_json(Path(directory)/'checkpoint.json')
            V.coordinate_phase(Replay(),saved,0,args,directory,base)
            self.assertEqual(saved,continuous)

    def test_rejected_epoch_restores_anchor_and_adam_but_keeps_work_count(self):
        cases,base=fit_cases(5),T.defaults()
        fitter=F.LeafFit(cases,base,base,batch_size=2,backtrack=True)
        fitter.epoch()
        fitter.feedback(True)
        anchor=copy.deepcopy(fitter.state)
        fitter.epoch()
        self.assertNotEqual(fitter.state['params'],anchor['params'])
        fitter.feedback(False)
        for key in ('params','m','v','adam_step'):
            self.assertEqual(fitter.state[key],anchor[key])
        self.assertGreater(fitter.state['updates'],anchor['updates'])
        self.assertEqual(fitter.state['epoch'],2)
        self.assertEqual(fitter.state['rate'],.0075)
        self.assertEqual(fitter.state['backtracks'],1)
        resumed=F.LeafFit(cases,base,base,batch_size=2,backtrack=True,state=copy.deepcopy(fitter.state))
        fitter.epoch();resumed.epoch()
        fitter.feedback(False);resumed.feedback(False)
        self.assertEqual(fitter.state,resumed.state)
        for rate in (0,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                F.LeafFit(cases,base,base,learning_rate=rate)

    def test_batches_visit_all_examples_and_resume_adam_exactly(self):
        cases,base=fit_cases(5),T.defaults()
        continuous=F.LeafFit(cases,base,base,batch_size=2)
        info=continuous.epoch()
        self.assertEqual((info['positions'],info['updates']),(5,3))
        interrupted=F.LeafFit(cases,base,base,batch_size=2)
        saved=[]
        def stop(state):
            saved.append(copy.deepcopy(state))
            raise RuntimeError('interrupted')
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            interrupted.epoch(stop)
        resumed=F.LeafFit(cases,base,base,batch_size=2,state=saved[0])
        resumed.epoch()
        self.assertEqual(resumed.state,continuous.state)
        with self.assertRaisesRegex(ValueError,'配置变化'):
            F.LeafFit(cases,base,base,batch_size=3,state=saved[0])

    def test_batch_gradient_finite_difference_and_real_ranking_improves(self):
        cases,base=fit_cases(2),T.defaults()
        examples=F.examples(cases,.08)
        _,grad=F.batch_gradient(examples,base,base)
        eps=1e-6
        for k in (0,4,12):
            name=T.LEAF_KEYS[k]
            left,right=dict(base),dict(base)
            left[name]-=eps*base[name]
            right[name]+=eps*base[name]
            numerical=(F.batch_gradient(examples,right,base)[0]-F.batch_gradient(examples,left,base)[0])/(2*eps)
            self.assertAlmostEqual(grad[k],numerical,places=6)
        fitter=F.LeafFit(cases,base,base,batch_size=2)
        for _ in range(30):
            fitter.epoch()
        self.assertEqual(T.choose(cases[0]['tree']['rows'],fitter.state['params'])['c'],2)
        self.assertEqual(F.examples([dict(cases[0],tree=dict(rows=[dict(
            cases[0]['tree']['rows'][0],state=1),cases[0]['tree']['rows'][1]]))],.08),[])

    def test_new_bad_is_diagnostic_but_holdout_and_tail_remain_hard_gates(self):
        current=dict(regret=.2,family_regret=.2,selection_regret=.2,worst_fold_regret=.3,
                     tail_regret=.5,severe=1,new_bad=0)
        better=dict(current,regret=.15,family_regret=.15)
        self.assertEqual(V.quality_reasons(better,current),[])
        self.assertEqual(V.quality_reasons(dict(better,new_bad=3),current),[])
        self.assertIn('训练内留出组差距增加',V.quality_reasons(dict(better,selection_regret=.21),current))
        self.assertIn('严重误选数增加',V.quality_reasons(dict(better,severe=2),current))
        self.assertIn('最差十分之一局面的平均差距增加',V.quality_reasons(dict(better,tail_regret=.6),current))
        checkpoint=dict(params={'weight':1})
        self.assertEqual(V.keep_candidate(checkpoint,{'weight':2},better,current),[])
        self.assertTrue(V.keep_candidate(checkpoint,{'weight':3},current,better))
        self.assertEqual(checkpoint['params'],{'weight':2})


class ParallelTests(unittest.TestCase):
    def test_dedup_and_global_budget_are_not_multiplied_by_workers(self):
        FakeTeacher.calls=[]
        with tempfile.TemporaryDirectory() as directory:
            teacher=FakeTeacher('fixture',directory,max_queries=2)
            labels=J.Labels(teacher,J.Jobs(4))
            histories=[MOVES[:i] for i in (2,3,4)]
            with self.assertRaises(U.BudgetStop):
                labels.prefetch(histories+histories)
            self.assertEqual(len(FakeTeacher.calls),2)
            self.assertEqual(len(list(teacher.cache.glob('*.json'))),2)
            teacher.max_queries=3
            labels.prefetch(histories*3)
            self.assertEqual(len(FakeTeacher.calls),3)
            labels.prefetch(histories)
            self.assertEqual(len(FakeTeacher.calls),3)
            path=next(teacher.cache.glob('*.json'))
            data=U.read_json(path)
            data['evaluation']['utility']=.999
            U.write_json(path,data)
            with self.assertRaisesRegex(ValueError,'缓存指纹'):
                labels.prefetch(histories)

    def test_parallel_real_search_uses_processes_and_matches_serial(self):
        params=T.defaults()
        positions=[position(),position(MOVES+[[5,8,1]])]
        settings=dict(t0=80,node_limit=50000,search_seconds=15)
        function=lambda p:U.isolated_search(p,params,settings,True,time.monotonic()+60)
        serial=J.Jobs(1).map(function,positions)
        parallel=J.Jobs(2).map(function,positions)
        for a,b in zip(serial,parallel):
            self.assertEqual(a['move'],b['move'])
            self.assertEqual(a['rows'],b['rows'])
            self.assertEqual(a['nodes'],b['nodes'])
        self.assertEqual(T.defaults(),params)


class CorpusTests(unittest.TestCase):
    def test_union_reference_includes_better_candidate_and_rebases_both(self):
        with tempfile.TemporaryDirectory() as directory:
            pos=position()
            teacher=FakeTeacher('fixture',directory)
            settings=dict(t0=80,node_limit=50000,search_seconds=15,margin=.08)
            corpus=V.Corpus(directory,[pos],teacher,settings,2,time.monotonic()+60)
            base=T.defaults()
            new=dict(base,type_5=base['type_5']+1)
            def tree(p,params,record=True):
                rows=[dict(r=1,c=1,state=0,constant=10,terms=[])]
                if params['type_5']!=base['type_5']:
                    rows.append(dict(r=1,c=2,state=0,constant=20,terms=[]))
                return dict(rows=rows)
            corpus.cache.tree=tree
            corpus.labels.utility=lambda p,move:{(0,0):-.8,(1,1):.2,(1,2):.9}[tuple(move)]
            corpus.prepare([base],'train')
            first=corpus.metrics(base,'train',base)
            self.assertEqual(first['regret'],0)
            corpus.prepare([base,new],'train')
            case=corpus.cases(base,'train')[0]
            self.assertEqual(case['reference'],.9)
            self.assertAlmostEqual(corpus.metrics(base,'train',base)['regret'],.7)
            self.assertEqual(corpus.metrics(new,'train',base)['regret'],0)
            self.assertEqual(corpus.metrics(base,'train',base)['pool_sha256'],
                             corpus.metrics(new,'train',base)['pool_sha256'])
            with self.assertRaisesRegex(ValueError,'三个训练棋局家族'):
                corpus.fit_cases(base)
            old_pool=copy.deepcopy(corpus.pool)
            original_prefetch=corpus.labels.prefetch
            def fail(histories):
                raise T.SearchLimit('fixture-limited')
            corpus.labels.prefetch=fail
            with self.assertRaises(T.SearchLimit):
                corpus.prepare([new],'train')
            self.assertEqual(corpus.pool,old_pool)
            corpus.labels.prefetch=original_prefetch

    def test_cli_budget_resume_holdout_and_export_real_search(self):
        with tempfile.TemporaryDirectory() as directory:
            games=Path(directory)/'games.json'
            out=Path(directory)/'out'
            U.write_json(games,sample_games())
            args=['run','--games',str(games),'--teacher','fixture','--out',str(out),
                  '--positions','12','--groups','leaf','--rounds','1','--epochs','2',
                  '--batch-size','2','--workers','2','--max-seconds','120']
            with mock.patch.object(U,'RapfiTeacher',FakeTeacher):
                with mock.patch.object(V.Corpus,'prepare',side_effect=U.BudgetStop('fixture-pause')):
                    self.assertEqual(V.main(args),2)
                self.assertEqual(V.main(args+['--resume']),0)
                checkpoint=U.read_json(out/'checkpoint.json')
                self.assertEqual(checkpoint['status'],'complete')
                positions=U.read_json(out/'positions.json')['positions']
                teacher=FakeTeacher('fixture',out)
                for pos in positions:
                    if pos['split']=='test':
                        self.assertFalse((out/'labels'/(T.digest([teacher.identity,pos['moves']])+'.json')).exists())
                best=U.read_json(out/'best.json')
                self.assertEqual(best['source'],U.source_fingerprint())
                result=subprocess.run([sys.executable,'-B','-m','gomoku.tools.teacher_tune','protocol',
                    '--params',str(out/'best.json')],input='START 15\nBEGIN\nEND\n',
                    text=True,capture_output=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(result.stdout.splitlines(),['OK','7,7'])
                self.assertEqual(V.main(args+['--resume']),0)
                self.assertEqual(V.main(args+['--resume','--batch-size','3']),1)
                warm_args=args.copy()
                warm_args[warm_args.index('--out')+1]=str(Path(directory)/'warm')
                warm_args+=['--initial-params',str(out/'best.json')]
                self.assertEqual(V.main(warm_args),1)
                self.assertEqual(V.main(warm_args+['--validation-role','seen','--leaf-policy','backtrack']),0)
                warm=U.read_json(Path(directory)/'warm'/'checkpoint.json')
                self.assertEqual(warm['initial'],best['params'])
                self.assertEqual(warm['identity']['validation_role'],'seen')
                coordinate_args=warm_args.copy()
                coordinate_args[coordinate_args.index('--out')+1]=str(Path(directory)/'coordinate')
                self.assertEqual(V.main(coordinate_args+['--validation-role','seen','--leaf-policy','coordinate']),0)
                coordinate=U.read_json(Path(directory)/'coordinate/checkpoint.json')
                self.assertTrue(any(e['group']=='leaf-coordinate' for e in coordinate['history']))
                front_args=warm_args.copy()
                front_args[front_args.index('--out')+1]=str(Path(directory)/'front')
                front_args[front_args.index('--groups')+1]='score'
                self.assertEqual(V.main(front_args+['--validation-role','seen','--score-policy','coordinate','--score-keys','neighbor']),0)
                front=U.read_json(Path(directory)/'front/checkpoint.json')
                self.assertEqual(front['schedule'],['score'])
                self.assertTrue(all(e['group']=='score-coordinate' for e in front['history']))
                self.assertTrue(all(e['coordinate_key']=='neighbor' for e in front['history']))
                self.assertEqual(V.main(front_args+['--validation-role','seen','--score-policy','coordinate','--score-keys','neighbor','--resume','--seed','2']),1)


if __name__=='__main__':
    unittest.main()

# 本轮发布默认改为8/3/120；以下旧机制回归显式固定旧口径。
from gomoku.tests._legacy_defaults import enter_legacy as _enter_legacy, leave_legacy as _leave_legacy

def setUpModule():
    _enter_legacy()

def tearDownModule():
    _leave_legacy()