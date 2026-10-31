"""有界叶子分界搜索：粗网格漏窗、数据隔离、精确续跑和评分预算反例。"""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from gomoku.tools import leaf_fit as F, tune_params as T, teacher_train as V, teacher_tune as U


def island_cases():
    base=T.defaults()
    feature=[0]*24+[1.0];feature[0]=1
    variable=dict(r=1,c=2,state=0,constant=0,terms=[[1,feature]])
    score=lambda alpha:T.replay_row(variable,dict(base,my_0=base['my_0']*(1+alpha)))[0]
    rows=[dict(r=1,c=1,state=0,constant=score(.4),terms=[]),variable,
          dict(r=1,c=3,state=0,constant=-score(.6),terms=[[2,feature]])]
    return [dict(position=dict(id=str(i),group=str(i)),tree=dict(rows=copy.deepcopy(rows)),
                 utilities={'1,1':0,'1,2':1,'1,3':0}) for i in range(3)]


def drive(fitter):
    proposals=[]
    while (p:=fitter.next_proposal()) is not None:
        fitter.reserve()
        signature=fitter.ranking.signature(p['params'])
        fitter.observe(signature)
        proposals.append(p)
    return proposals


class BoundaryTests(unittest.TestCase):
    def test_parameter_scope_restores_memory_after_failure(self):
        params=T.defaults()
        with mock.patch.object(T.Engine,'_memory_bonus',return_value={(4,1):30}) as original:
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                with T.parameter_scope(params):
                    self.assertEqual(T.Engine()._memory_bonus(),{})
                    raise RuntimeError('interrupted')
            self.assertIs(T.Engine._memory_bonus,original)
            self.assertEqual(T.Engine()._memory_bonus(),{(4,1):30})

    def test_native_search_ignores_local_memory_for_both_colors(self):
        moves=[[2,0,1],[1,0,2],[2,1,1],[1,1,2],[3,2,1],[6,5,2],
               [3,3,1],[3,4,2],[2,3,1],[2,4,2],[4,4,1],[4,3,2],[5,2,1],[4,2,2]]
        params=T.defaults()
        for count in (13,14):
            position=dict(moves=moves[:count],player=count%2+1)
            results=[]
            for adjustment in (-50,50):
                with mock.patch.object(T.Engine,'_memory_bonus',return_value={(4,1):adjustment}) as memory:
                    result=T.search_position(position,params,t0=80,record=True)
                    memory.assert_not_called()
                result.pop('seconds');results.append(result)
            self.assertEqual(results[0],results[1])

    def test_old_grid_misses_island_and_boundary_finds_it_in_original_bounds(self):
        cases,base=island_cases(),T.defaults()
        for p in F.coordinate_proposals('my_0',base,base):
            self.assertNotEqual((T.choose(cases[0]['tree']['rows'],p['params'])['r'],
                                 T.choose(cases[0]['tree']['rows'],p['params'])['c']),(1,2))
        with mock.patch.object(F,'coordinate_order',return_value=['my_0']):
            fitter=F.BoundaryFit(cases,base,base,budget=80)
            proposals=drive(fitter)
        self.assertTrue(any(T.choose(cases[0]['tree']['rows'],p['params'])['c']==2 for p in proposals))
        self.assertTrue(any(p['kind']=='refine' for p in proposals))
        for p in proposals:
            T.validate(p['params'],base)
            self.assertTrue(all(p['params'][k]==base[k] for k in base if k not in T.LEAF_KEYS))
        self.assertLessEqual(fitter.state['attempts'],80)

    def test_saved_pending_resume_is_exact_and_reserves_retry_budget(self):
        cases,base=island_cases(),T.defaults()
        with mock.patch.object(F,'coordinate_order',return_value=['my_0']):
            original=F.BoundaryFit(cases,base,base,budget=40)
            original.next_proposal();original.reserve()
            saved=json.loads(json.dumps(original.state))
            resumed=F.BoundaryFit(cases,base,base,budget=40,state=saved)
            self.assertEqual(original.next_proposal(),resumed.next_proposal())
            # Both runs account for the reserved attempt before a simulated interruption.
            original.reserve();resumed.reserve()
            for fit in (original,resumed):fit.observe(fit.ranking.signature(fit.next_proposal()['params']))
            self.assertEqual(drive(original),drive(resumed))
            self.assertEqual(original.state,resumed.state)
            self.assertEqual(original.state['attempts'],original.state['completed']+1)

    def test_budget_one_cannot_reset_after_interrupted_reservation(self):
        cases,base=island_cases(),T.defaults()
        fit=F.BoundaryFit(cases,base,base,budget=1)
        fit.next_proposal();fit.reserve()
        resumed=F.BoundaryFit(cases,base,base,budget=1,state=copy.deepcopy(fit.state))
        self.assertIsNone(resumed.next_proposal())
        self.assertEqual(resumed.state['stop_reason'],'evaluation-budget')
        self.assertEqual(resumed.state['attempts'],1)
        self.assertEqual(resumed.state['completed'],0)

    def test_resume_rejects_budget_seed_input_or_anchor_drift(self):
        cases,base=island_cases(),T.defaults()
        fit=F.BoundaryFit(cases,base,base,seed=2,budget=10)
        fit.next_proposal();saved=copy.deepcopy(fit.state)
        for kwargs in (dict(seed=3,budget=10),dict(seed=2,budget=11)):
            with self.assertRaises(ValueError):F.BoundaryFit(cases,base,base,state=saved,**kwargs)
        changed=copy.deepcopy(cases);changed[0]['tree']['rows'][0]['constant']+=1
        with self.assertRaises(ValueError):F.BoundaryFit(changed,base,base,seed=2,budget=10,state=saved)
        with self.assertRaises(ValueError):F.BoundaryFit(cases,dict(base,my_0=base['my_0']*.99),base,seed=2,budget=10,state=saved)

    def test_corrupt_resume_cannot_reset_budget_or_change_non_leaf(self):
        cases,base=island_cases(),T.defaults()
        fit=F.BoundaryFit(cases,base,base,budget=10)
        fit.next_proposal();fit.reserve();saved=copy.deepcopy(fit.state)
        for field,value in [('attempts',-1),('completed',20),('budget',999)]:
            bad=copy.deepcopy(saved);bad[field]=value
            with self.assertRaises(ValueError):F.BoundaryFit(cases,base,base,budget=10,state=bad)
        bad=copy.deepcopy(saved);bad['pending']['params']['neighbor']+=1
        with self.assertRaises(ValueError):F.BoundaryFit(cases,base,base,budget=10,state=bad)

    def test_signature_matches_formal_ties_and_state_priority(self):
        cases,base=island_cases(),T.defaults()
        cases[0]['tree']['rows'][0]['state']=2
        rank=F.CachedRanking(cases)
        for alpha in (.01,.2,.45,.8,2):
            params=dict(base,my_0=base['my_0']*(1+alpha))
            expected=[(T.choose(c['tree']['rows'],params)['r'],T.choose(c['tree']['rows'],params)['c']) for c in cases]
            self.assertEqual(rank.signature(params),expected)

    def test_boundary_proposals_do_not_depend_on_teacher_utilities(self):
        cases,base=island_cases(),T.defaults()
        changed=copy.deepcopy(cases)
        for c in changed:c['utilities']={key:-v for key,v in c['utilities'].items()}
        a,b=F.BoundaryFit(cases,base,base,budget=30),F.BoundaryFit(changed,base,base,budget=30)
        self.assertEqual(drive(a),drive(b))

    def test_phase_resume_after_accept_and_after_finished_commit_window(self):
        base,cases=T.defaults(),island_cases()
        class Corpus:
            fold={('train',str(i)):i%3 for i in range(3)}
            def __init__(self,pause=None):self.calls=0;self.pause=pause;self.verifications=0
            def fit_cases(self,params):return cases[1:]
            def cases(self,params,split):return cases
            def check_deadline(self):pass
            def metrics(self,params,*args):
                self.calls+=1
                if self.calls==self.pause:raise U.BudgetStop('pause-after-reservation')
                chosen=T.choose(cases[0]['tree']['rows'],params)
                return dict(regret=.5 if chosen['c']==2 else 1.0,
                    family_regret=.1,selection_regret=.1,worst_fold_regret=.1,tail_regret=.1,severe=0)
            def verify(self,*args):self.verifications+=1;return dict(nodes=1)
            def compute_guard(self,*args):return True,1
        args=SimpleNamespace(seed=1,leaf_evaluations=128)
        original=dict(params=dict(base),history=[],baseline_train_compute=dict(nodes=1))
        full,partial=copy.deepcopy(original),copy.deepcopy(original)
        with tempfile.TemporaryDirectory() as directory,mock.patch.object(F,'coordinate_order',return_value=['my_0']):
            V.boundary_phase(Corpus(),full,0,args,directory,base)
            with self.assertRaises(U.BudgetStop):V.boundary_phase(Corpus(12),partial,0,args,directory,base)
            saved=U.read_json(Path(directory)/'checkpoint.json')
            V.boundary_phase(Corpus(),saved,0,args,directory,base)
            self.assertEqual(saved['params'],full['params'])
            self.assertTrue(any(h['accepted'] for h in saved['history']))
            strip=lambda history:[{k:v for k,v in h.items() if k!='evaluation_attempt'} for h in history]
            self.assertEqual(strip(saved['history']),strip(full['history']))
            self.assertEqual(saved['boundary']['attempts'],full['boundary']['attempts']+1)
            repeat=Corpus();before=copy.deepcopy(saved)
            V.boundary_phase(repeat,saved,0,args,directory,base)
            self.assertEqual(saved,before)
            self.assertEqual(repeat.verifications,0)

    def test_real_cli_boundary_budget_resume_and_changed_budget_refusal(self):
        from gomoku.tests.test_teacher_train import FakeTeacher
        from gomoku.tests.test_teacher_tune import sample_games
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);games=path/'games.json';out=path/'out'
            U.write_json(games,sample_games())
            args=['run','--games',str(games),'--teacher','fixture','--out',str(out),
                  '--positions','12','--groups','leaf','--rounds','1','--workers','2',
                  '--leaf-policy','boundary','--leaf-evaluations','32','--max-seconds','120']
            with mock.patch.object(U,'RapfiTeacher',FakeTeacher):
                with mock.patch.object(V.Corpus,'metrics',side_effect=U.BudgetStop('pause-before-proposal')):
                    self.assertEqual(V.main(args),2)
                self.assertEqual(V.main(args+['--resume']),0)
                checkpoint=U.read_json(out/'checkpoint.json')
                self.assertEqual(checkpoint['status'],'complete')
                self.assertTrue(any(e['group']=='leaf-boundary' for e in checkpoint['history']))
                self.assertLessEqual(checkpoint['boundary_runs'][0]['attempts'],32)
                self.assertIsNone(checkpoint['boundary'])
                changed=list(args);changed[changed.index('--leaf-evaluations')+1]='33'
                self.assertEqual(V.main(changed+['--resume']),1)
                teacher=FakeTeacher('fixture',out)
                for pos in U.read_json(out/'positions.json')['positions']:
                    if pos['split']=='test':
                        self.assertFalse((out/'labels'/(T.digest([teacher.identity,pos['moves']])+'.json')).exists())

    def test_warm_boundary_rejects_old_gain_and_keeps_initial_export(self):
        from gomoku.tests.test_teacher_train import FakeTeacher
        from gomoku.tests.test_teacher_tune import sample_games
        base=T.defaults();initial=T.project(dict(base,my_0=base['my_0']*1.1),base)
        candidate=T.project(dict(base,my_0=base['my_0']*1.05),base)
        class Corpus:
            def __init__(self,*args):pass
            def prepare(self,*args):pass
            def check_deadline(self):pass
            def verify(self,*args):return dict(nodes=1)
            def calibrate(self,*args):return dict(nodes=1,seconds=1,positions=1,p95_seconds=1)
            def metrics(self,params,*args):
                regret=1-(params['my_0']/base['my_0']-1)
                return dict(regret=regret,family_regret=.1,selection_regret=.1,worst_fold_regret=.1,
                    tail_regret=.1,severe=0,new_bad=0,attainable_regret=.1,best_move_absent=0,
                    state_blocked=0,positions=1)
        def stage(corpus,checkpoint,*args):checkpoint['params']=dict(candidate)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);games=path/'games.json';warm=path/'initial.json';out=path/'out'
            U.write_json(games,sample_games());U.write_json(warm,dict(source=U.source_fingerprint(),params=initial))
            args=['run','--games',str(games),'--teacher','fixture','--out',str(out),'--positions','12',
                  '--groups','leaf','--rounds','1','--leaf-policy','boundary','--initial-params',str(warm),
                  '--validation-role','seen','--workers','1']
            with mock.patch.object(U,'RapfiTeacher',FakeTeacher),mock.patch.object(V,'Corpus',Corpus),mock.patch.object(V,'boundary_phase',stage):
                self.assertEqual(V.main(args),0)
            self.assertEqual(U.read_json(out/'best.json')['params'],initial)
            exported=U.read_json(out/'candidate.json')
            self.assertEqual(exported['params'],candidate)
            self.assertFalse(exported['recommended'])
            self.assertFalse(exported['improved_initial'])


if __name__=='__main__':unittest.main()
