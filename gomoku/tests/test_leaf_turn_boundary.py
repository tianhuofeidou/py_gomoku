# -*- coding: utf-8 -*-
"""普通叶子只在根候选方刚落子后评估；实际成五优先于截断补手。"""
import copy
import types
import unittest

from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK, WHITE, empty_board
from gomoku.tests.test_four_with_outer_gap import TPTC_PREFIX, point


def capture_leaves(deep):
    records = []
    original = deep._leaf_score_impl

    def observe(self, board, branch, status, player, ctx_branch=None):
        value = original(board, branch, status, player, ctx_branch)
        records.append(dict(status=status, last_player=branch[-1][2],
            path=[(r, c, p) for r, c, p, _w in branch], score=value,
            kill_active=self._has_kill_points(board)))
        return value

    deep._leaf_score_impl = types.MethodType(observe, deep)
    return records


class LeafTurnBoundaryTest(unittest.TestCase):
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

    def test_tptc_forced_block_is_played_before_leaf_and_state_restored(self):
        for defender in (WHITE, BLACK):
            for complete in (False, True):
                with self.subTest(defender=defender, complete=complete):
                    engine = Engine()
                    engine.deep_search.T0 = 160
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
                    self.assertTrue(all(r['last_player'] == defender for r in stale))
                    self.assertTrue(all(not r['kill_active'] for r in stale))
                    forcing = [r for r in stale if [m[:2] for m in r['path'][:2]]
                               == [point('H5'), point('G3')]]
                    self.assertTrue(forcing)
                    self.assertTrue(all([m[:2] for m in r['path'][:3]]
                                        == [point('H5'), point('G3'), point('G2')] for r in forcing))
                    # 普通步不能靠后续新威胁重开免温链；G2 堵住后允许立即截断。
                    self.assertTrue(all(len(r['path']) == 3 for r in forcing))
                    self.assertTrue(all(r['score'] > -920 for r in forcing))
                    self.assertEqual((board, engine.moves, engine.search.sb, engine.search.sw,
                        engine.search._direction_levels, engine.search.gear,
                        engine.search.zero_history, engine.search.zero_active), before)
                    self.assertIsNone(engine.deep_search._root_pass)

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
