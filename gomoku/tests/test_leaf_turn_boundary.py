# -*- coding: utf-8 -*-
"""耗尽温度只续零耗温战术；在当前盘面按根方评分，实际成五优先。"""
import copy
import types
import unittest
from unittest.mock import patch

from gomoku.core.engine import Engine
from gomoku.core.deep_search import DeepSearch
from gomoku.core.utils import BLACK, WHITE, empty_board
from gomoku.tests.test_four_with_outer_gap import TPTC_PREFIX, point


def capture_leaves(deep):
    records = []
    original = deep._leaf_score_impl

    def observe(self, board, branch, status, player, ctx_branch=None):
        value = original(board, branch, status, player, ctx_branch)
        last_player = branch[-1][2] if branch else None
        next_player = 3 - last_player if last_player is not None else player
        records.append(dict(status=status, last_player=last_player,
            path=[(r, c, p) for r, c, p, _w in branch], score=value,
            score_player=player, next_player=next_player,
            next_reason=self._tactical_result(board, next_player)[0],
            kill_active=self._has_kill_points(board)))
        return value

    deep._leaf_score_impl = types.MethodType(observe, deep)
    return records


class LeafTurnBoundaryTest(unittest.TestCase):
    def setUp(self):
        # 验证耗尽边界本身；根最低深度豁免由test_min_search_depth覆盖。
        setting = patch.object(DeepSearch, 'MIN_SEARCH_PLIES', 0)
        setting.start()
        self.addCleanup(setting.stop)

    def test_root_budget_used_by_own_move_stops_before_opponent(self):
        for player in (BLACK, WHITE):
            with self.subTest(player=player):
                engine = Engine()
                engine.deep_search.T0 = 2
                board = empty_board()
                leaves = capture_leaves(engine.deep_search)
                engine.deep_search.rank_candidates(board, {(7, 7): 100}, player,
                                                   complete_scores=True)
                self.assertEqual(len(leaves), 1)
                self.assertEqual(leaves[0]['status'], 'stale')
                self.assertEqual(leaves[0]['path'], [(7, 7, player)])
                self.assertEqual(board, empty_board())

    def test_tptc_stops_on_current_board_without_ordinary_extra_move(self):
        for defender in (WHITE, BLACK):
            for complete in (False, True):
                with self.subTest(defender=defender, complete=complete):
                    engine = Engine()
                    # 夹具目标是“战术应手后耗尽”；普通最小扣温为3，入口1确保触达。
                    engine.deep_search.T0 = 1
                    board = empty_board()
                    for i, label in enumerate(TPTC_PREFIX):
                        r, c = point(label)
                        player = 3 - defender if i % 2 == 0 else defender
                        board[r][c] = player
                        engine.on_move(board, r, c, player)
                    before = copy.deepcopy((board, engine.moves, engine.search.sb, engine.search.sw,
                        engine.search._direction_levels, engine.search.gear,
                        engine.search.zero_history, engine.search.zero_active))
                    leaves = capture_leaves(engine.deep_search)
                    engine.deep_search.rank_candidates(board, {point('H5'): 100}, defender,
                                                       complete_scores=complete)
                    stale = [r for r in leaves if r['status'] == 'stale']
                    self.assertTrue(stale)
                    self.assertTrue(all(r['score_player'] == defender for r in stale))
                    self.assertTrue(all(r['next_reason'] is None for r in stale))
                    # 当前输入确实覆盖对方刚落完即停止，不能再强补本方普通一步。
                    self.assertTrue(any(r['last_player'] == 3 - defender for r in stale))
                    stopped = [r for r in stale if [m[:2] for m in r['path']]
                               == [point('H5'), point('H4')]]
                    self.assertTrue(stopped)
                    self.assertTrue(all(r['score'] > -920 for r in stale))
                    self.assertEqual((board, engine.moves, engine.search.sb, engine.search.sw,
                        engine.search._direction_levels, engine.search.gear,
                        engine.search.zero_history, engine.search.zero_active), before)
                    self.assertIsNone(engine.deep_search._root_pass)

    def test_tptc_forced_g2_block_is_free_even_with_no_attacking_subclass(self):
        for defender in (WHITE, BLACK):
            for complete in (False, True):
                with self.subTest(defender=defender, complete=complete):
                    engine = Engine()
                    deep = engine.deep_search
                    deep.T0 = -5
                    board = empty_board()
                    for i, token in enumerate(list(TPTC_PREFIX) + ['H5', 'G3']):
                        r, c = point(token)
                        side = 3 - defender if i % 2 == 0 else defender
                        board[r][c] = side
                        engine.on_move(board, r, c, side)
                    # 额外两步的实际行棋方明确固定，避免前缀长度隐含视角。
                    self.assertEqual(board[point('H5')[0]][point('H5')[1]], defender)
                    self.assertEqual(board[point('G3')[0]][point('G3')[1]], 3 - defender)
                    reason, points = engine.search.tactical_candidates(board, defender)
                    self.assertEqual(reason, 'opp-five')
                    self.assertEqual(points, {point('G2'): 100.0})
                    state = engine.search.sb if defender == BLACK else engine.search.sw
                    self.assertEqual(state[point('G2')[0]][point('G2')[1]], 17)
                    before = copy.deepcopy((board, engine.moves, engine.search.sb, engine.search.sw,
                        engine.search._direction_levels, engine.search.gear,
                        engine.search.zero_history, engine.search.zero_active))
                    leaves = capture_leaves(deep)
                    deep.rank_candidates(board, points, defender, complete_scores=complete)
                    self.assertEqual(len(leaves), 1)
                    self.assertEqual(leaves[0]['path'], [(*point('G2'), defender)])
                    self.assertEqual(leaves[0]['status'], 'stale')
                    self.assertEqual(leaves[0]['score_player'], defender)
                    self.assertIsNone(leaves[0]['next_reason'])
                    self.assertEqual((board, engine.moves, engine.search.sb, engine.search.sw,
                        engine.search._direction_levels, engine.search.gear,
                        engine.search.zero_history, engine.search.zero_active), before)
                    self.assertIsNone(deep._root_pass)

    def test_opponent_actual_five_is_terminal_without_extra_reply(self):
        for defender in (WHITE, BLACK):
            with self.subTest(defender=defender):
                engine = Engine()
                engine.deep_search.T0 = 4
                board = empty_board()
                for c in range(5, 9):
                    board[7][c] = 3 - defender
                for r in range(3):
                    board[r][0] = defender
                engine.search.rebuild_all(board)
                leaves = capture_leaves(engine.deep_search)
                result = engine.deep_search.rank_candidates(board, {(14, 14): 100}, defender,
                                                           complete_scores=True)
                self.assertEqual(result[0]['state_name'], '必败')
                self.assertTrue(leaves)
                self.assertTrue(all(r['status'] == 'lose' and len(r['path']) == 2
                                    and r['last_player'] == 3 - defender for r in leaves))


if __name__ == '__main__':
    unittest.main()
