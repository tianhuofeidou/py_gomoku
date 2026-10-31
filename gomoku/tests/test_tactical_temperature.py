# -*- coding: utf-8 -*-
"""当前战术候选免耗温；活二落成活三不能追溯免温或在负温度重开搜索。"""
import copy
import types
import unittest

from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK, WHITE, empty_board
from gomoku.tests.test_leaf_turn_boundary import capture_leaves


def snapshot(engine, board):
    search = engine.search
    return copy.deepcopy((board, engine.moves, search.sb, search.sw,
                          search._direction_levels, search.gear,
                          search.zero_history, search.zero_active))


class TacticalTemperatureTest(unittest.TestCase):
    def test_live_four_is_resolved_even_at_zero_or_negative_temperature(self):
        for player in (BLACK, WHITE):
            for temperature in (0, -5):
                for complete in (False, True):
                    with self.subTest(player=player, temperature=temperature, complete=complete):
                        engine = Engine()
                        engine.deep_search.T0 = temperature
                        board = empty_board()
                        for c in (5, 6, 7):
                            board[7][c] = player
                        engine.search.rebuild_all(board)
                        before = snapshot(engine, board)
                        leaves = capture_leaves(engine.deep_search)
                        rows = engine.deep_search.rank_candidates(
                            board, {(7, 8): 60}, player, complete_scores=complete)
                        self.assertEqual(rows[0]['state_name'], '必胜')
                        self.assertTrue(leaves)
                        self.assertTrue(all(row['status'] == 'win' and len(row['path']) == 3
                                            for row in leaves))
                        self.assertEqual(snapshot(engine, board), before)

    def test_two_stones_to_live_three_still_costs_and_cannot_restart_at_zero(self):
        for player in (BLACK, WHITE):
            for complete in (False, True):
                with self.subTest(player=player, complete=complete):
                    engine = Engine()
                    deep = engine.deep_search
                    deep.T0 = 0
                    board = empty_board()
                    board[7][5] = board[7][6] = player
                    engine.search.rebuild_all(board)
                    before = snapshot(engine, board)
                    self.assertFalse(deep._tactical_points(board, player))
                    self.assertEqual((engine.search.sb if player == BLACK else engine.search.sw)[7][7] > 4, True)
                    original_candidates = deep._candidate_points

                    def candidates(self, b, side, n=5, plies=0):
                        self_case.assertEqual(plies, 0, '普通步不能以落子后的新威胁重开搜索')
                        return original_candidates(b, side, n, plies)

                    self_case = self
                    deep._candidate_points = types.MethodType(candidates, deep)
                    costs = []
                    original_cost = deep._temperature_cost

                    def cost(weight, is_tactical):
                        value = original_cost(weight, is_tactical)
                        costs.append(value)
                        return value

                    deep._temperature_cost = cost
                    leaves = capture_leaves(deep)
                    rows = deep.rank_candidates(board, {(7, 7): 60}, player,
                                                complete_scores=complete)
                    self.assertEqual(rows[0]['state_name'], '无')
                    self.assertEqual(len(leaves), 1)
                    self.assertEqual(leaves[0]['path'], [(7, 7, player)])
                    self.assertTrue(leaves[0]['kill_active'], '落子后确实产生新的活四候选')
                    self.assertEqual(costs, [deep._dt(60)])
                    self.assertEqual(snapshot(engine, board), before)

    def test_unrelated_root_is_not_free_even_when_board_has_tactical_points(self):
        for player in (BLACK, WHITE):
            engine = Engine()
            deep = engine.deep_search
            deep.T0 = 0
            board = empty_board()
            for c in (5, 6, 7):
                board[7][c] = player
            engine.search.rebuild_all(board)
            self.assertTrue(deep._tactical_points(board, player))
            self.assertNotIn((14, 14), deep._tactical_points(board, player))
            before = snapshot(engine, board)
            leaves = capture_leaves(deep)
            deep.rank_candidates(board, {(14, 14): 100}, player, complete_scores=True)
            self.assertEqual(len(leaves), 1)
            self.assertEqual(leaves[0]['path'], [(14, 14, player)])
            self.assertEqual(snapshot(engine, board), before)

    def test_weight_80_defense_is_current_tactical_move_and_is_free(self):
        from gomoku.tests.test_four_with_outer_gap import TPTC_PREFIX, point
        for player in (BLACK, WHITE):
            engine = Engine()
            deep = engine.deep_search
            deep.T0 = 0
            board = empty_board()
            for i, label in enumerate(TPTC_PREFIX):
                r, c = point(label)
                side = 3 - player if i % 2 == 0 else player
                board[r][c] = side
                engine.on_move(board, r, c, side)
            points = deep._tactical_points(board, player)
            self.assertEqual(points[point('D1')], 80)
            before = snapshot(engine, board)
            costs = []
            original_cost = deep._temperature_cost

            def cost(weight, is_tactical):
                value = original_cost(weight, is_tactical)
                costs.append((weight, is_tactical, value))
                return value

            deep._temperature_cost = cost
            deep.rank_candidates(board, {point('D1'): 80}, player, complete_scores=True)
            self.assertEqual(costs[0], (80, True, 0))
            self.assertEqual(snapshot(engine, board), before)

    def test_trigger_uses_empty_point_subclasses_not_weight_or_legacy_fatal(self):
        engine = Engine()
        deep = engine.deep_search
        board = empty_board()
        for player in (BLACK, WHITE):
            for sub in range(18):
                engine.search.rebuild_all(board)
                state = engine.search.sb if player == BLACK else engine.search.sw
                state[4][4] = sub
                self.assertEqual(deep._has_kill_points(board), sub <= 4)
                board[4][4] = BLACK
                self.assertFalse(deep._has_kill_points(board))
                board[4][4] = 0

    def test_cost_depends_only_on_current_tactical_identity(self):
        deep = Engine().deep_search
        for weight in (1, 60, 80, 100):
            self.assertEqual(deep._temperature_cost(weight, True), 0)
            self.assertEqual(deep._temperature_cost(weight, False), deep._dt(weight))


if __name__ == '__main__':
    unittest.main()
