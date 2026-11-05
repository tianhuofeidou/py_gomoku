"""最低根深度实验：正常扣温、负温继续、按实际落子层数而非递归帧。"""
import copy
import unittest
from unittest.mock import patch

from gomoku.core.engine import Engine
from gomoku.core.deep_search import DeepSearch
from gomoku.core.utils import empty_board,BLACK,WHITE


class Observer:
    def __init__(self):
        self.steps=[]
        self.leaves=[]
    def record(self,kind,node=None,**data):
        if kind=='temperature':
            self.steps.append(data)
        if kind=='leaf':
            self.leaves.append(dict(data))


def state(engine,board):
    s=engine.search
    return copy.deepcopy((board,engine.moves,s.sb,s.sw,s._direction_levels,s.gear,s.gear_progress,
                          s.zero_history,s.zero_active,s._gear_active))


class MinSearchDepthTests(unittest.TestCase):
    def chain(self,player,limit,win_at=None,run_cached=False):
        engine=Engine()
        d=engine.deep_search
        d.T0=1.
        # 将本用例恢复信用固定为0，证明是根最低深度生效而非续温奖励。
        d.CONTINUE_BASE=0.
        board=empty_board()
        observer=Observer()
        d._debug_observer=observer
        points=[(0,0),(2,2),(4,4),(6,6),(8,8),(10,10),(12,12)]
        def count(b):
            return sum(cell!=0 for row in b for cell in row)
        def candidates(b,side,n=5,plies=0):
            depth=count(b)
            return [(*points[depth],25.)] if depth<len(points) else []
        before=state(engine,board)
        with patch.object(DeepSearch,'DT_M',200.),patch.object(DeepSearch,'MIN_SEARCH_PLIES',limit), \
                patch.object(d,'_candidate_points',side_effect=candidates), \
                patch.object(d,'_pattern_temperature_weight',return_value=25.), \
                patch.object(d,'_tactical_result',return_value=(None,{})), \
                patch.object(d,'_check_win',side_effect=lambda b,*args: count(b)==win_at):
            rows=d.rank_candidates(board,{points[0]:25.},player)
            self.assertEqual(state(engine,board),before)
            if run_cached:
                first_steps=len(observer.steps)
                root=d._tree_root
                again=d.rank_candidates(board,{points[0]:25.},player)
                self.assertEqual(rows,again)
                self.assertEqual(len(observer.steps),first_steps,'同盘面不能重领最低三层')
                self.assertIs(d._tree_root,root)
                for (r,c),side in zip(points[:2],(player,3-player)):
                    child=d._tree_root['children'][(r,c,side)]
                    board[r][c]=side
                    engine.on_move(board,r,c,side)
                    self.assertIs(d._tree_root,child)
                committed=state(engine,board)
                observer.steps.clear()
                observer.leaves.clear()
                d.cached_turn(board,player)
                self.assertEqual(state(engine,board),committed)
                # 旧叶子位于新根第1层，只补到整个新根第3层，不是叶子后再补3手。
                node=d._tree_root
                depth=0
                while node['children']:
                    self.assertEqual(len(node['children']),1)
                    node=next(iter(node['children'].values()))
                    depth+=1
                self.assertEqual(depth,3)
                self.assertEqual(node['frontier']['depth'],3)
                self.assertTrue(observer.leaves)
        return engine,d,observer,rows

    def test_disabled_and_three_plies_both_colors(self):
        self.assertEqual(DeepSearch.MIN_SEARCH_PLIES,3)
        self.assertEqual(DeepSearch.DT_M,200.)
        for player in (BLACK,WHITE):
            for limit,depth in ((0,1),(3,3)):
                with self.subTest(player=player,limit=limit):
                    engine,d,observer,rows=self.chain(player,limit)
                    self.assertEqual(len(observer.steps),depth)
                    self.assertTrue(rows)
                    self.assertEqual([round(s['temperature'],4) for s in observer.steps],
                                     [round(1.-115.5*i,4) for i in range(1,depth+1)])
                    self.assertTrue(all(not s['tactical'] for s in observer.steps))
                    self.assertTrue(all(s['tactical_count']==0 for s in observer.steps))
                    node=d._tree_root
                    for _ in range(depth):
                        node=next(iter(node['children'].values()))
                    self.assertEqual(node['frontier']['depth'],depth)
                    self.assertAlmostEqual(node['frontier']['temperature'],1.-115.5*depth)

    def test_actual_terminal_can_stop_before_three(self):
        for player in (BLACK,WHITE):
            _,_,observer,rows=self.chain(player,3,win_at=2)
            self.assertEqual(len(observer.steps),2)
            self.assertEqual(rows[0]['state'],-1)

    def test_hot_read_and_real_reroot_do_not_grant_three_per_leaf(self):
        for player in (BLACK,WHITE):
            self.chain(player,3,run_cached=True)

    def test_remaining_depth_is_in_result_key_and_policy_invalidates_tree(self):
        d=Engine().deep_search
        b=empty_board()
        d._begin_search_cache(b,False)
        old=d._tree_root
        try:
            with patch.object(DeepSearch,'MIN_SEARCH_PLIES',0):
                d._begin_search_cache(b,False)
                self.assertIsNot(d._tree_root,old)
                old=d._tree_root
            with patch.object(DeepSearch,'MIN_SEARCH_PLIES',3):
                self.assertNotEqual(d._result_key(BLACK,BLACK,-1,1,depth=1),
                                    d._result_key(BLACK,BLACK,-1,1,depth=2))
                self.assertEqual(d._result_key(BLACK,BLACK,-1,1,depth=3),
                                 d._result_key(BLACK,BLACK,-1,1,depth=8))
                d._begin_search_cache(b,False)
                self.assertIsNot(d._tree_root,old)
        finally:
            d._reuse_active=False

    def test_only_temperature_gate_is_waived(self):
        d=Engine().deep_search
        b=empty_board()
        with patch.object(DeepSearch,'MIN_SEARCH_PLIES',3), \
                patch.object(d,'_tactical_result',return_value=(None,{})):
            self.assertFalse(d._temperature_exhausted(b,BLACK,-20,0,depth=2))
            self.assertTrue(d._temperature_exhausted(b,BLACK,-20,0,depth=3))
            self.assertTrue(d._temperature_exhausted(b,BLACK,-20,0,blocked=True,depth=1))
            with patch.object(d,'TACTICAL_EXTENSION_PLIES',8):
                self.assertTrue(d._temperature_exhausted(b,BLACK,-20,0,extension_count=8,depth=1))
            self.assertEqual(d._prune_branches([(1.,0,0,100.),(.001,0,1,1.)]),[(1.,0,0,100.)])
            with patch.object(d,'_tactical_result',return_value=('opp-vcf',{})):
                self.assertFalse(d._temperature_exhausted(b,BLACK,-20,0,depth=1))


if __name__=='__main__':
    unittest.main(verbosity=2)
