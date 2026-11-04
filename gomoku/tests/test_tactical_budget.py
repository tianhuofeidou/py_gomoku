# -*- coding: utf-8 -*-
"""实验梯度耗温：额度、行棋视角、完整估值和路径缓存必须彼此一致。"""
import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from gomoku.core.engine import Engine
from gomoku.core.deep_search import DeepSearch
from gomoku.core.utils import BLACK, WHITE, empty_board
from gomoku.tests import test_cold_result_reuse as cold_fixture


def snapshot(engine, board):
    s = engine.search
    return copy.deepcopy((board, engine.moves, s.sb, s.sw, s._direction_levels,
                          s.zero_history, s.zero_active, s.gear))


class TacticalBudgetTest(unittest.TestCase):
    def test_gradient_boundaries_and_ordinary_steps(self):
        d = Engine().deep_search
        for free in (8, 16, 24):
            d.TACTICAL_FREE_PLIES = free
            for before, cost in ((0, 0.), (free-1, 0.), (free, 2.),
                                 (free+7, 2.), (free+8, 4.),
                                 (free+15, 4.), (free+16, 8.), (free+50, 8.)):
                self.assertEqual(d._path_temperature_cost(100., True, before), cost)
            self.assertEqual(d._path_temperature_cost(60., False, free), d._dt(60.))
            self.assertEqual(d._advance_tactical_count(free, False), free)
            self.assertEqual(d._advance_tactical_count(free, True), free+1)
        d.TACTICAL_FREE_PLIES = None
        self.assertEqual(d._path_temperature_cost(100., True, 999), 0.)
        self.assertEqual(d._advance_tactical_count(999, True), 0)

    def test_budget_keys_and_signature_are_isolated(self):
        d = Engine().deep_search
        b = empty_board()
        d.TACTICAL_FREE_PLIES = 9
        signature = d._reuse_signature()
        d.TACTICAL_FREE_PLIES = 10
        self.assertNotEqual(signature, d._reuse_signature())
        self.assertNotEqual(d._result_key(BLACK, BLACK, 0, 1., 8),
                            d._result_key(BLACK, BLACK, 0, 1., 9))
        d._cold_results = {}
        self.assertNotEqual(d._cold_result_key(b, BLACK, WHITE, BLACK, 0., 1., 2, 8),
                            d._cold_result_key(b, BLACK, WHITE, BLACK, 0., 1., 2, 9))

    def run_chain(self, player, free, temperature=0., multiple_roots=False, repeat=False, warm=False):
        engine = Engine()
        d = engine.deep_search
        d.T0 = temperature
        d.TACTICAL_FREE_PLIES = free
        # 本文件显式启用旧负温梯度截断；显式8手延伸另由 extension 测试覆盖。
        d.TACTICAL_EXTENSION_PLIES = None
        board = empty_board()
        points = [(2, 2), (9, 9), (3, 5), (10, 3), (6, 10), (12, 11)]
        second = (2, 3)

        def tactical(b, turn):
            count = sum(v != 0 for row in b for v in row)
            if count >= len(points):
                return None, {}
            candidates = {points[count]: 100.}
            if multiple_roots and count == 0:
                candidates[second] = 100.
            return 'my-vcf', candidates

        def candidates(b, turn, n=5, plies=0):
            return [(r, c, w) for (r, c), w in tactical(b, turn)[1].items()]

        before = snapshot(engine, board)
        with ExitStack() as stack:
            stack.enter_context(patch.object(d, '_tactical_result', side_effect=tactical))
            stack.enter_context(patch.object(d, '_candidate_points', side_effect=candidates))
            stack.enter_context(patch.object(d, '_check_win', return_value=False))
            stack.enter_context(patch.object(d, '_stale_result_score', return_value=123.))
            roots = {points[0]: 100.}
            if multiple_roots:
                roots[second] = 100.
            rows = d.rank_candidates(board, roots, player, complete_scores=True)
            self.assertEqual(before, snapshot(engine, board))
            if repeat:
                again = d.cached_turn(board, player)
                self.assertIsNotNone(again)
                self.assertEqual(again[2][0]['score'], rows[0]['score'])
                self.assertEqual(before, snapshot(engine, board))
            if warm:
                for point, turn in ((points[0], player), (points[1], 3-player)):
                    board[point[0]][point[1]] = turn
                    engine.on_move(board, *point, turn)
                self.assertEqual(d._tree_root['tactical_count'], 2)
                before_warm = snapshot(engine, board)
                costs = []
                original_cost = d._path_temperature_cost

                def cost(weight, tactical_step, count, *temperature):
                    result = original_cost(weight, tactical_step, count, *temperature)
                    costs.append((tactical_step, count, result))
                    return result

                with patch.object(d, '_path_temperature_cost', side_effect=cost):
                    result = d.cached_turn(board, player)
                self.assertIsNotNone(result)
                self.assertEqual(before_warm, snapshot(engine, board))
                self.assertTrue(costs)
                self.assertTrue(all(count >= 2 for tactical_step, count, result in costs))
                self.assertTrue(all(result > 0 for tactical_step, count, result in costs
                                    if tactical_step))
        return engine, board, rows, points

    def test_zero_temperature_free_chain_stops_and_is_complete_estimate(self):
        for player in (BLACK, WHITE):
            with self.subTest(player=player):
                engine, board, rows, path = self.run_chain(player, 2)
                d = engine.deep_search
                node = d._tree_root['children'][(*path[0], player)]['children'][(*path[1], 3-player)]
                self.assertEqual(node['tactical_count'], 2)
                self.assertEqual(node['frontier']['tactical_count'], 2)
                self.assertEqual(node['frontier']['leaf_end_reason'], 'tactical_budget')
                self.assertEqual(node['frontier']['to_move'], player)
                self.assertIsNone(node['edges'])
                self.assertEqual(rows[0]['state'], 0)
                self.assertTrue(rows[0]['score_complete'])
                self.assertEqual(rows[0]['score'], -123.)
                self.assertEqual(d.cache_stats['tactical_cutoffs'], 1)
                self.assertEqual(d.cache_stats['tactical_probe_moves'], 1)
                self.assertTrue(node['move_result']['complete'])

    def test_siblings_start_at_same_count_not_global_counter(self):
        engine, board, rows, path = self.run_chain(BLACK, 1, multiple_roots=True)
        self.assertEqual(len(rows), 2)
        for node in engine.deep_search._tree_root['children'].values():
            self.assertEqual(node['tactical_count'], 1)
            self.assertEqual(node['frontier']['tactical_count'], 1)
        self.assertEqual(rows[0]['score'], rows[1]['score'])

    def test_cached_root_keeps_spent_free_quota(self):
        engine, board, rows, path = self.run_chain(BLACK, 2)
        for point, turn in ((path[0], BLACK), (path[1], WHITE)):
            board[point[0]][point[1]] = turn
            engine.on_move(board, *point, turn)
        d = engine.deep_search
        self.assertEqual(d._tree_root['tactical_count'], 2)
        self.assertEqual(d._tree_root['frontier']['tactical_count'], 2)
        # 借助保存的计数可知第3个战术步收费，真实换根不能重新获得免费段。
        self.assertEqual(d._path_temperature_cost(100., True,
                         d._tactical_count([], d._tree_root)), 2.)

    def test_warm_continuation_charges_after_saved_quota(self):
        for player in (BLACK, WHITE):
            with self.subTest(player=player):
                self.run_chain(player, 2, temperature=4., warm=True)

    def test_leaf_value_cache_distinguishes_tactical_count(self):
        d = Engine().deep_search
        d.TACTICAL_FREE_PLIES = 2
        d._reuse_active = True
        board = empty_board()
        with patch.object(d, '_leaf_score_impl', side_effect=[11., 22.]) as score:
            first = d._leaf_score(board, [], 'tactical_budget:1', BLACK,
                                  [{'tactical_count': 2}])
            second = d._leaf_score(board, [], 'tactical_budget:1', BLACK,
                                   [{'tactical_count': 3}])
            again = d._leaf_score(board, [], 'tactical_budget:1', BLACK,
                                  [{'tactical_count': 2}])
            self.assertEqual((first, second, again), (11., 22., 11.))
            self.assertEqual(score.call_count, 2)

    def test_no_free_quota_root_can_be_reanalysed(self):
        engine, board, rows, path = self.run_chain(BLACK, 0, repeat=True)
        d = engine.deep_search
        self.assertEqual(d._tree_root['tactical_count'], 0)
        self.assertTrue(d._tree_root['move_result']['complete'])

    def test_defense_score_uses_real_reply_and_original_weights(self):
        for turn in (BLACK, WHITE):
            with self.subTest(turn=turn):
                engine = Engine()
                d = engine.deep_search
                d.TACTICAL_FREE_PLIES = 8
                board = empty_board()
                before = snapshot(engine, board)

                def value(b, next_turn):
                    self.assertEqual(next_turn, 3-turn)
                    return -200. if b[7][7] == turn else 100.

                with patch.object(d, '_tactical_result', return_value=(
                        'opp-vcf', {(7, 7): 75., (7, 8): 25.})), \
                     patch.object(d, '_check_win', return_value=False), \
                     patch.object(d, '_stale_result_score', side_effect=value):
                    score = d._tactical_budget_score(board, turn, turn)
                self.assertEqual(score, 125.)  # 防守不强制为负；没有额外反四奖励。
                self.assertEqual(d.cache_stats['tactical_probe_moves'], 2)
                self.assertEqual(before, snapshot(engine, board))

    def test_next_player_wins_when_both_have_five_points(self):
        for player in (BLACK, WHITE):
            engine = Engine()
            d = engine.deep_search
            d.TACTICAL_FREE_PLIES = 0
            board = empty_board()
            for col in (4, 5, 6, 7):
                board[4][col] = player
                board[10][col] = 3-player
            engine.search.rebuild_all(board)
            before = snapshot(engine, board)
            self.assertEqual(d._tactical_budget_score(board, 3-player, player), -d.WIN_SCORE)
            self.assertEqual(before, snapshot(engine, board))

    def test_direct_five_is_not_hidden_by_exhausted_budget(self):
        for player in (BLACK, WHITE):
            engine = Engine()
            d = engine.deep_search
            d.TACTICAL_FREE_PLIES = 0
            d.T0 = 0.
            board = empty_board()
            for col in (4, 5, 6, 7):
                board[7][col] = player
            engine.search.rebuild_all(board)
            before = snapshot(engine, board)
            rows = d.rank_candidates(board, {(7, 8): 100.}, player)
            self.assertEqual(rows[0]['state'], d.STATE_WIN)
            self.assertEqual(rows[0]['score'], d.WIN_SCORE)
            self.assertEqual(before, snapshot(engine, board))

    def test_probe_exception_restores_board_and_search(self):
        engine = Engine()
        d = engine.deep_search
        board = empty_board()
        before = snapshot(engine, board)
        with patch.object(d, '_tactical_result', return_value=('opp-vcf', {(7, 7): 100.})), \
             patch.object(d, '_stale_result_score', side_effect=RuntimeError('probe')):
            with self.assertRaisesRegex(RuntimeError, 'probe'):
                d._tactical_budget_score(board, BLACK, BLACK)
        self.assertEqual(before, snapshot(engine, board))

    def test_cold_cache_enabled_policy_still_has_exact_scores(self):
        case = cold_fixture.ColdResultReuseTest()
        with patch.object(DeepSearch, 'TACTICAL_FREE_PLIES', 8):
            for player in (BLACK, WHITE):
                reference = case.run_tree(False, player, real_leaf=True)
                actual = case.run_tree(True, player, real_leaf=True)
                self.assertEqual(actual[0], reference[0])
                self.assertGreater(actual[2], 0)


if __name__ == '__main__':
    unittest.main()
