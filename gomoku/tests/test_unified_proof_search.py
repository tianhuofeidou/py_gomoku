"""统一方案：独立四层真值、硬态定分早停、温度分类和防点来源排序。"""
import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from gomoku.core.engine import Engine
from gomoku.core.deep_search import DeepSearch
from gomoku.core.utils import BLACK, WHITE, empty_board


class UnifiedProofSearchTest(unittest.TestCase):
    ROOTS = {(7,7):100., (7,9):100., (9,9):100.}
    REPLIES = [(6,6,100.),(6,8,100.)]
    CHOICES = [(8,6,100.),(8,8,100.)]
    ENDS = [(5,5,100.),(5,7,100.)]

    def fixture(self, player=BLACK, all_lose=False, first_wins=False):
        e=Engine()
        d=e.deep_search
        d.T0=0.
        board=empty_board()
        steps=[]
        class Observer:
            def record(self,kind,node=None,**data):
                if kind=='temperature':
                    steps.append(tuple(data.items()))
        d._debug_observer=Observer()
        visited=[]
        def count(b):
            return sum(cell!=0 for row in b for cell in row)
        def cands(b,turn,n=5,plies=0):
            depth=count(b)
            return ([(*p,w) for p,w in self.ROOTS.items()] if depth==0 else
                    self.REPLIES if depth==1 else self.CHOICES if depth==2 else
                    self.ENDS if depth==3 else [])
        def tactical(b,turn):
            return 'my-vcf',{(r,c):w for r,c,w in cands(b,turn)}
        def won(b,r,c,turn):
            visited.append((count(b),r,c,turn))
            losing_root=all_lose or (not first_wins and b[7][7]==player)
            return ((turn!=player and count(b)==4 and (r,c)==(5,5)) if losing_root
                    else turn==player and count(b)==3 and (r,c)==(8,6))
        def leaf(scorer,b,branch,status,main,contexts=None):
            return -37. if status=='lose' else 19.  # 硬态必须覆盖这个故意不等于±1000的旧值。
        stack=ExitStack()
        stack.enter_context(patch.object(d,'_candidate_points',side_effect=cands))
        stack.enter_context(patch.object(d,'_tactical_result',side_effect=tactical))
        stack.enter_context(patch.object(d,'_check_win',side_effect=won))
        stack.enter_context(patch.object(d,'_active_threat_points',return_value=frozenset()))
        stack.enter_context(patch.object(DeepSearch,'_leaf_score_impl',new=leaf))
        self.addCleanup(stack.close)
        return e,board,visited,steps

    def snapshot(self,e,board):
        s=e.search
        return copy.deepcopy((board,e.moves,s.sb,s.sw,s._direction_levels,s.zero_history,s.zero_active,s.gear))

    def test_four_layers_existential_and_universal_proof_both_colors(self):
        for player in (BLACK,WHITE):
            e,board,visited,steps=self.fixture(player)
            before=self.snapshot(e,board)
            rows=e.deep_search.rank_candidates(board,self.ROOTS,player)
            self.assertEqual([(r['r'],r['c'],r['state'],r['score']) for r in rows],
                             [(7,9,1,1000.),(7,7,-1,-1000.)])
            self.assertTrue(all(r['score_complete'] for r in rows))
            self.assertEqual(len(steps),11)
            self.assertFalse(any((r,c)==(5,7) for _depth,r,c,_p in visited))
            self.assertFalse(any((r,c)==(9,9) for _depth,r,c,_p in visited))
            self.assertEqual(self.snapshot(e,board),before)
            root=e.deep_search._tree_root
            losing=root['children'][(7,7,player)]
            self.assertNotIn((6,8,3-player),losing['children'])
            self.assertEqual(losing['move_result']['score'],-1000.)

    def test_first_proven_win_stops_roots_even_complete_scores_request(self):
        for complete in (False,True):
            e,board,visited,steps=self.fixture(first_wins=True)
            with patch.object(e.deep_search,'_rank_candidates_pass',wraps=e.deep_search._rank_candidates_pass) as run:
                rows=e.deep_search.rank_candidates(board,self.ROOTS,BLACK,complete_scores=complete)
            self.assertEqual(run.call_count,1)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['state'],1)
            self.assertEqual(rows[0]['score'],1000.)
            self.assertFalse(any((r,c)==(7,9) for _depth,r,c,_p in visited))

    def test_all_hard_losses_have_fixed_score_and_stable_legal_fallback(self):
        e,board,visited,steps=self.fixture(all_lose=True)
        rows=e.deep_search.rank_candidates(board,self.ROOTS,BLACK)
        self.assertEqual([(r['r'],r['c']) for r in rows],list(self.ROOTS))
        self.assertTrue(all(r['state']==-1 and r['score']==-1000. and r['score_complete'] for r in rows))
        self.assertEqual(len(steps),18)

    def test_hard_cache_and_contribution_do_not_revisit_descendants(self):
        e,board,visited,steps=self.fixture(first_wins=True)
        d=e.deep_search
        rows=d.rank_candidates(board,self.ROOTS,BLACK)
        before=len(visited)
        with patch.object(d,'_recursive_impl',side_effect=AssertionError('重推硬态')), \
             patch.object(d,'_walk_cached_edges',side_effect=AssertionError('遍历硬态后代')):
            cached=d.cached_turn(board,BLACK)[2]
        self.assertEqual(rows,cached)
        self.assertEqual(len(visited),before)
        self.assertEqual(d.root_contribution_report['total'],1000.)
        self.assertEqual(d.root_contribution_report['leaf_count'],1)
        self.assertEqual(d._tree_root['contribution_unit'],1000.)

    def test_real_reroot_preserves_hard_proof_without_continuation(self):
        for player in (BLACK,WHITE):
            e,board,visited,steps=self.fixture(player,first_wins=True)
            d=e.deep_search
            d.rank_candidates(board,self.ROOTS,player)
            for point,turn in (((7,7),player),((6,6),3-player)):
                child=d._tree_root['children'][(*point,turn)]
                board[point[0]][point[1]]=turn
                e.on_move(board,*point,turn)
                self.assertIs(d._tree_root,child)
            before=self.snapshot(e,board)
            n=len(steps)
            candidate_calls=d._candidate_points.call_count
            with patch.object(d,'_walk_cached_edges',side_effect=AssertionError('重推硬态')):
                result=e.analyze_turn(board,player)
            self.assertEqual(d._candidate_points.call_count,candidate_calls)
            row=result['part4_result']['ranked'][0]
            self.assertEqual((row['state'],row['score']),(1,1000.))
            self.assertEqual(len(steps),n)
            self.assertEqual(self.snapshot(e,board),before)
            self.assertEqual(d.root_contribution_report['leaf_count'],1)
            self.assertEqual(d.last_temperature_kind,'continuation')
            self.assertIn(player,e._searched_sides)

    def test_hard_score_perspectives_and_unknown_value_unchanged(self):
        d=Engine().deep_search
        for mover in (BLACK,WHITE):
            for main in (BLACK,WHITE):
                for state in (-1,1):
                    self.assertEqual(d._hard_score(state,mover,main,123.),
                                     state*1000.*(1 if mover==main else -1))
                for state in (-2,0,2):
                    self.assertEqual(d._hard_score(state,mover,main,123.),123.)
        with self.assertRaises(ValueError):
            d._hard_score(1,None,BLACK,0.)

    def test_unknown_reply_prevents_all_losing_claim(self):
        d=Engine().deep_search
        parent=d._new_tree_node(bytes(225)); parent['mover']=BLACK
        parent['edges']=((1,1,WHITE,100.,0.,1.),(1,2,WHITE,100.,0.,1.))
        child=d._new_tree_node(bytes(225)); child['mover']=WHITE
        d._store_cached_move(child,3.,-1,0,1,1,BLACK)
        parent['children'][(1,1,WHITE)]=child
        value=d._aggregate_cached_move(parent,BLACK)
        self.assertNotIn(value['state'],(-1,1))
        self.assertFalse(value['complete'])

    def test_hard_child_is_a_single_weighted_scoring_leaf(self):
        d=Engine().deep_search
        parent=d._new_tree_node(bytes(225)); parent['mover']=BLACK
        parent['edges']=((1,1,WHITE,100.,0.,1.),(1,2,WHITE,100.,0.,1.))
        for point,state,score in (((1,1),-1,12.),((1,2),0,200.)):
            child=d._new_tree_node(bytes(225)); child['mover']=WHITE
            d._store_cached_move(child,score,state,0,0,1,BLACK)
            parent['children'][(*point,WHITE)]=child
        value=d._aggregate_cached_move(parent,BLACK)
        self.assertEqual(value['state'],2)
        self.assertEqual(value['score'],600.)
        self.assertEqual(parent['contribution_unit'],600.)

    def test_normal_order_keeps_values_and_opponent_uses_own_scores(self):
        d=Engine().deep_search
        points=[(1,1,30.),(1,2,80.),(1,3,80.),(1,4,20.)]
        with patch.object(d,'_tactical_result',return_value=(None,{})):
            for player in (BLACK,WHITE):
                self.assertEqual(d._ordered_candidates(empty_board(),player,points),
                                 [points[1],points[2],points[0],points[3]])

    def test_defense_center_then_gap_then_endpoints_with_unchanged_weights(self):
        for player in (BLACK,WHITE):
            e=Engine(); s=e.search; board=empty_board(); board[7][9]=player
            before=copy.deepcopy(board)
            record=dict(parent='LIVE3',dir=(0,1),l_edge=(3,1),r_edge=(3,1),gap_location=1)
            with patch.object(s.pattern,'analyze_point',return_value=[record]), \
                 patch.object(s,'_eval',side_effect=lambda b,r,c,p:4 if (r,c)==(7,7) else 5):
                priority={}
                points=s._defense_candidates(board,7,7,player=player,_priorities=priority)
            self.assertEqual(list(points),[(7,7),(7,8),(7,4),(7,10)])
            self.assertEqual(list(points.values()),[100.,80.,80.,80.])
            self.assertEqual([priority[p] for p in points],[0,1,2,2])
            self.assertEqual(board,before)

    def test_duplicate_defense_source_retains_highest_priority(self):
        s=Engine().search; priority={}
        s._ordered_defenses({(1,1):80.,(1,2):80.},{(1,1):2,(1,2):1},priority)
        s._ordered_defenses({(1,1):100.},{(1,1):0},priority)
        self.assertEqual(priority,{(1,1):0,(1,2):1})

    def test_opening_middle_import_and_new_game_budgets(self):
        e=Engine(); d=e.deep_search; board=empty_board(); seen=[]
        def rank(b,pts,p,**kwargs):
            seen.append((p,d._initial_temperature(),d.T0,d.last_temperature_kind))
            return []
        with patch.object(d,'rank_candidates',side_effect=rank):
            e._rank_search(board,{},BLACK,False)
            d.clear_search_cache()
            e._rank_search(board,{},BLACK,False)
            e._rank_search(board,{},WHITE,False)
            e.reset([(7,7,BLACK)])
            e._rank_search(board,{},BLACK,False)
            e.reset()
            e._rank_search(board,{},BLACK,False)
        self.assertEqual([r[1] for r in seen],[160.,160.,160.,160.,160.])
        self.assertEqual([r[3] for r in seen],['opening_cold','middle_cold','opening_cold','middle_cold','opening_cold'])
        self.assertTrue(all(r[2]==160. for r in seen))
        self.assertIsNone(d._cold_temperature)

    def test_explicit_cold_override_does_not_change_continuation_credit(self):
        d=Engine().deep_search
        d._cold_temperature=80.
        d._begin_search_cache(empty_board(),False)
        self.assertEqual(d._initial_temperature(),80.)
        self.assertEqual(d._cache_credit,160.)
        value,blocked=d._continuation_temperature(dict(state=0,path_weight=1.,depth=0,
                                    temperature=-150.,score=0.,extension_state=(0,0,0)))
        self.assertFalse(blocked)
        self.assertEqual(value,18.)


if __name__=='__main__':
    unittest.main()

# 本轮发布默认改为8/3/120；以下旧机制回归显式固定旧口径。
from gomoku.tests._legacy_defaults import enter_legacy as _enter_legacy, leave_legacy as _leave_legacy

def setUpModule():
    _enter_legacy()

def tearDownModule():
    _leave_legacy()