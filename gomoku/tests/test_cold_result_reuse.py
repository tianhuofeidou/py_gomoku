# -*- coding: utf-8 -*-
"""单次冷搜索完整结果复用：原加权分、五态和路径状态均须保留。"""
import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK, WHITE, empty_board


class ColdResultReuseTest(unittest.TestCase):
    def run_tree(self, enabled=True, player=BLACK, rotated=False, terminal=False, warm=False, complete=True, real_leaf=False):
        engine = Engine()
        deep = engine.deep_search
        deep.T0 = 20.
        # 合成图固定每步3温度；本测试比较缓存前后，真实棋型扣温另有定向回归。
        deep.MIN_SEARCH_PLIES = 0
        board = empty_board()
        flip = lambda point: (14-point[0], 14-point[1]) if rotated else point
        a, b, e, g, h = map(flip, [(7, 7), (5, 5), (5, 9), (3, 3), (3, 11)])
        c, d, f = map(flip, [(8, 6), (8, 8), (10, 7)])
        calls = []

        def snapshot():
            search = engine.search
            return copy.deepcopy((board, engine.moves, search.sb, search.sw,
                search._direction_levels, search.zero_history, search.zero_active, search.gear))

        def candidates(position, turn, n=5, plies=0):
            calls.append((deep._position_key(position), turn))
            count = sum(row.count(turn) for row in position)
            if turn == player:
                if count == 0:
                    return [(*a, 100.)]
                if count in (1, 2):
                    return [(*point, 100.) for point in (b, e) if not position[point[0]][point[1]]]
                if count == 3:
                    return [(*g, 100.), (*h, 80.)]
            elif count < 3:
                point = (c, d, f)[count]
                return [(*point, 100.)]
            return []

        def leaf(scorer, position, branch, status, main_player, contexts=None):
            if status == 'win':
                return scorer.WIN_SCORE
            if status == 'lose':
                return -scorer.WIN_SCORE
            scorer.leaf_eval_stats['total'] += 1
            scorer.leaf_eval_stats['heuristic'] += 1
            scorer.leaf_eval_stats['reason'] = 'fixture'
            return 120. if position[g[0]][g[1]] == main_player else -80.

        def won(position, r, col, turn):
            return terminal and turn == player and (r, col) == g

        before = snapshot()
        with ExitStack() as stack:
            stack.enter_context(patch.object(deep, '_pattern_temperature_weight', return_value=100.))
            stack.enter_context(patch.object(engine.search, '_record_gear_threats', return_value=None))
            stack.enter_context(patch.object(deep, '_candidate_points', side_effect=candidates))
            stack.enter_context(patch.object(deep, '_tactical_result', return_value=(None, {})))
            stack.enter_context(patch.object(deep, '_check_win', side_effect=won))
            if not real_leaf:
                stack.enter_context(patch.object(DeepSearch, '_leaf_score_impl', new=leaf))
            if not enabled:
                stack.enter_context(patch.object(deep, '_cold_result_key', return_value=None, create=True))
            rows = deep.rank_candidates(board, {a: 100.}, player, complete_scores=complete)
            self.assertEqual(before, snapshot())
            cold_calls = len(calls)
            cold_hits = deep.cache_stats.get('cold_result_hits', 0)
            self.assertIsNone(getattr(deep, '_cold_results', None))
            root = deep._tree_root
            if terminal:
                # 首条路径已经证明硬胜，换序转置路径不应再展开。
                proof = root
                for point, turn in ((a, player), (c, 3-player), (b, player),
                                    (d, 3-player), (e, player), (f, 3-player), (g, player)):
                    proof = proof['children'][(*point, turn)]
                self.assertEqual(proof['move_result']['score'], 1000.)
                self.assertNotIn((*h, player), root['children'][(*a, player)]['children'][(*c, 3-player)]['children'][(*b, player)]['children'][(*d, 3-player)]['children'][(*e, player)]['children'][(*f, 3-player)]['children'])
                cloned = deep._new_tree_node(root['board'])
                deep._copy_cold_subtree(cloned, root)
                self.assertEqual(cloned['move_result'], root['move_result'])
                self.assertEqual(cloned['contribution_unit'], 1000.)
                self.assertIsNot(cloned['children'], root['children'])
                return rows, cold_calls, cold_hits, None
            first = root
            second = root
            for r, col, turn in [(a[0], a[1], player), (c[0], c[1], 3-player),
                                (b[0], b[1], player), (d[0], d[1], 3-player),
                                (e[0], e[1], player), (f[0], f[1], 3-player)]:
                first = first['children'][(r, col, turn)]
            for r, col, turn in [(a[0], a[1], player), (c[0], c[1], 3-player),
                                (e[0], e[1], player), (d[0], d[1], 3-player),
                                (b[0], b[1], player), (f[0], f[1], 3-player)]:
                second = second['children'][(r, col, turn)]
            self.assertEqual(first['board'], second['board'])
            self.assertIsNot(first, second)
            self.assertIsNot(first['children'], second['children'])
            for point in (g, h):
                one = first['children'][(*point, player)]
                two = second['children'][(*point, player)]
                self.assertEqual(one['move_result'], two['move_result'])
                self.assertIsNot(one['move_result'], two['move_result'])
                if one['frontier'] is not None:
                    self.assertEqual(one['frontier'], two['frontier'])
                    self.assertIsNot(one['frontier'], two['frontier'])
                    old = one['frontier']['temperature']
                    two['frontier']['temperature'] -= 1.
                    self.assertEqual(one['frontier']['temperature'], old)
                    two['frontier']['temperature'] += 1.
            warm_rows = None
            if warm:
                for point, turn in ((a, player), (c, 3-player)):
                    board[point[0]][point[1]] = turn
                    engine.on_move(board, *point, turn)
                before_warm = snapshot()
                with patch.object(deep, '_copy_cold_subtree', side_effect=AssertionError('热算不得使用冷算表'), create=True):
                    warm_rows = deep.cached_turn(board, player)[2]
                self.assertEqual(before_warm, snapshot())
                self.assertIsNone(getattr(deep, '_cold_results', None))
                self.assertEqual(deep.cache_stats.get('cold_result_hits', 0), 0)
        return rows, cold_calls, cold_hits, warm_rows

    def test_transposed_full_result_keeps_weighted_scores_both_colors(self):
        for player in (BLACK, WHITE):
            for rotated in (False, True):
                with self.subTest(player=player, rotated=rotated):
                    reference = self.run_tree(False, player, rotated)
                    actual = self.run_tree(True, player, rotated)
                    self.assertEqual(actual[0], reference[0])
                    # 独立手算式的运算顺序不同；上面的缓存开关对照必须逐字段精确相等。
                    self.assertAlmostEqual(actual[0][0]['score'], (120.-80.*.8)/1.8, places=12)
                    self.assertGreater(actual[2], 0)
                    self.assertLess(actual[1], reference[1])

    def test_full_result_preserves_hard_state_and_tree_evidence(self):
        for player in (BLACK, WHITE):
            with self.subTest(player=player):
                reference = self.run_tree(False, player, terminal=True)
                actual = self.run_tree(True, player, terminal=True)
                self.assertEqual(actual[0], reference[0])
                self.assertEqual(actual[0][0]['state'], DeepSearch.STATE_WIN)
                self.assertEqual(actual[0][0]['score'], 1000.)
                self.assertEqual(actual[1], reference[1])
                self.assertEqual(actual[2], 0)  # 证明先完成，转置分支没有访问。

    def test_reused_tree_has_independent_frontiers_and_warm_path(self):
        reference = self.run_tree(False, warm=True)
        actual = self.run_tree(True, warm=True)
        self.assertEqual(actual[0], reference[0])
        self.assertEqual(actual[3], reference[3])
        self.assertGreater(actual[2], 0)

    def test_default_scoring_is_identical(self):
        reference = self.run_tree(False, complete=False)
        actual = self.run_tree(True, complete=False)
        self.assertEqual(actual[0], reference[0])
        self.assertGreater(actual[2], 0)

    def test_production_leaf_scoring_is_identical(self):
        for player in (BLACK, WHITE):
            with self.subTest(player=player):
                reference = self.run_tree(False, player, real_leaf=True)
                actual = self.run_tree(True, player, real_leaf=True)
                self.assertEqual(actual[0], reference[0])
                self.assertGreater(actual[2], 0)
                self.assertLess(actual[1], reference[1])

    def test_clone_keeps_direct_grandchild_soft_evidence(self):
        deep = Engine().deep_search
        for risk in (False, True):
            with self.subTest(risk=risk):
                source = deep._new_tree_node(bytes(225))
                source['mover'] = BLACK
                source['edges'] = ((0, 0, WHITE, 100., 5., .25), (0, 1, WHITE, 100., 5., .75))
                for c, score, state in ((0, 11., deep.STATE_LOSE), (1, -7., deep.STATE_NONE)):
                    child = deep._new_tree_node(bytes(225))
                    child['mover'] = WHITE
                    deep._store_cached_move(child, score, state, 0, 0, 1, BLACK)
                    source['children'][(0, c, WHITE)] = child
                child = source['children'][(0, 0, WHITE)]
                child['edges'] = ((1, 0, BLACK, 100., 2., 1.), (1, 1, BLACK, 80., 2., .8))
                for c, state in ((0, deep.STATE_WIN), (1, deep.STATE_LOSE if risk else deep.STATE_NONE)):
                    grandchild = deep._new_tree_node(bytes(225))
                    grandchild['mover'] = BLACK
                    deep._store_cached_move(grandchild, 3., state, 0, 0, 1, BLACK)
                    child['children'][(1, c, BLACK)] = grandchild
                expected = dict(deep._aggregate_cached_move(source, BLACK))
                target = deep._new_tree_node(bytes(225))
                deep._copy_cold_subtree(target, source)
                self.assertEqual(deep._aggregate_cached_move(target, BLACK), expected)
                self.assertEqual(expected['state'], deep.STATE_EXISTS_LOSE if risk else deep.STATE_EXISTS_WIN)
                self.assertEqual(expected['score'], .25*1000.+.75*(-7.))

    def test_exception_clears_per_analysis_table(self):
        deep = Engine().deep_search
        with patch.object(deep, '_rank_candidates_impl', side_effect=RuntimeError('fixture')):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                deep.rank_candidates(empty_board(), {(7, 7): 100.}, BLACK)
        self.assertIsNone(deep._cold_results)

    def test_key_rejects_different_search_context_and_proof_mode(self):
        deep = Engine().deep_search
        board = empty_board()
        deep._cold_results = {}
        key = deep._cold_result_key(board, BLACK, WHITE, BLACK, 2., 1., 6)
        self.assertIsNotNone(key)
        for args in [(WHITE, BLACK, BLACK, 2., 1., 6),
                     (BLACK, WHITE, WHITE, 2., 1., 6),
                     (BLACK, WHITE, BLACK, 3., 1., 6),
                     (BLACK, WHITE, BLACK, 2., .8, 6),
                     (BLACK, WHITE, BLACK, 2., 1., 8)]:
            self.assertNotEqual(key, deep._cold_result_key(board, *args))
        deep.search.zero_history[BLACK].append(1)
        self.assertNotEqual(key, deep._cold_result_key(board, BLACK, WHITE, BLACK, 2., 1., 6))
        deep._root_pass = {'mode': 'proof', 't0': 20.}
        self.assertIsNone(deep._cold_result_key(board, BLACK, WHITE, BLACK, 2., 1., 6))
        deep._root_pass = None
        deep._cold_results = None
        self.assertIsNone(deep._cold_result_key(board, BLACK, WHITE, BLACK, 2., 1., 6))

    def test_custom_scorer_and_existing_tree_do_not_enable_cold_table(self):
        deep = Engine().deep_search
        board = empty_board()
        with patch.object(deep, '_leaf_score_impl', return_value=0.):
            deep._begin_search_cache(board, False)
            self.assertIsNone(getattr(deep, '_cold_results', None))
        deep._begin_search_cache(board, False)
        self.assertIsNotNone(deep._cold_results)
        deep._tree_root['main_player'] = BLACK
        deep._begin_search_cache(board, False)
        self.assertIsNone(deep._cold_results)


if __name__ == '__main__':
    unittest.main()
