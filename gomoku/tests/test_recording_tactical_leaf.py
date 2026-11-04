"""战术截断的静态应手评分也必须记录叶子依赖，不能冻结为常量。"""
import copy
import unittest
from unittest.mock import patch

from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import Search
from gomoku.core.utils import BLACK,WHITE,empty_board
from gomoku.tools import tune_params as T


class RecordingTacticalLeafTest(unittest.TestCase):
    def test_tactical_probe_replays_changed_leaf_weights_and_perspective(self):
        base=T.defaults()
        changed=dict(base,**{k:base[k]*1.16 for k in T.LEAF_KEYS})
        for turn in (BLACK,WHITE):
            for main in (BLACK,WHITE):
                with self.subTest(turn=turn,main=main):
                    board=empty_board()
                    for r,c,p in ((7,7,BLACK),(7,6,WHITE),(6,6,BLACK),(6,7,WHITE)):
                        board[r][c]=p
                    search=Search(PatternAnalyzer());search.rebuild_all(board)
                    before=copy.deepcopy((board,search.sb,search.sw,search._direction_levels,
                                          search.gear,search.zero_history,search.zero_active))
                    with T.parameter_scope(base,80):
                        recorder=T.RecordingSearch(search,record=True)
                        with patch.object(recorder,'_tactical_result',return_value=('opp-d33',{(0,0):100.})):
                            value=recorder._tactical_budget_score(board,turn,main)
                        self.assertIsInstance(value,T.Expression)
                        constant,terms=T.flatten(value)
                        self.assertTrue(terms)
                        self.assertEqual(all(w<0 for w,_f in terms),turn==main)
                        row=dict(constant=constant,terms=terms)
                        self.assertAlmostEqual(T.replay_row(row,base)[0],float(value),places=7)
                    with T.parameter_scope(changed,80):
                        fresh=T.RecordingSearch(search,record=False)
                        with patch.object(fresh,'_tactical_result',return_value=('opp-d33',{(0,0):100.})):
                            actual=fresh._tactical_budget_score(board,turn,main)
                        self.assertAlmostEqual(T.replay_row(row,changed)[0],actual,places=7)
                    self.assertEqual((board,search.sb,search.sw,search._direction_levels,
                                      search.gear,search.zero_history,search.zero_active),before)
