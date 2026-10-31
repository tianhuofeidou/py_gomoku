# -*- coding: utf-8 -*-
"""每层加权均值的尺度不变性、分数界与真实候选排序。"""
import copy
import unittest

from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK, WHITE, empty_board
from gomoku.tests.test_four_with_outer_gap import TPTC_PREFIX, point
from gomoku.tests.test_leaf_turn_boundary import capture_leaves


class WeightedNormalizationTest(unittest.TestCase):
    def test_duplicate_identical_branch_does_not_double_penalty(self):
        for value in (-700.0, 0.0, 850.0):
            self.assertAlmostEqual(DeepSearch._weighted_average([(1, value)]), value)
            self.assertAlmostEqual(DeepSearch._weighted_average([(1, value)] * 7), value)

    def test_common_weight_scale_does_not_change_result(self):
        values = [(1, -700), (.6, -400), (.2, 850)]
        expected = DeepSearch._weighted_average(values)
        for scale in (.02, .1, 8, 100):
            self.assertAlmostEqual(DeepSearch._weighted_average(
                [(w * scale, v) for w, v in values]), expected)

    def test_result_stays_within_child_range_and_zero_weight_is_ignored(self):
        for values in ([(1, -920), (.8, -500)], [(1, 800), (.6, 900)],
                       [(1, -920), (.5, 850)], [(.01, 450)]):
            result = DeepSearch._weighted_average(values)
            self.assertGreaterEqual(result, min(v for _w, v in values))
            self.assertLessEqual(result, max(v for _w, v in values))
            self.assertAlmostEqual(DeepSearch._weighted_average(values + [(0, 99999)]), result)
        self.assertEqual(DeepSearch._weighted_average([]), 0.0)
        self.assertEqual(DeepSearch._weighted_average([(0, 123)]), 0.0)

    def test_tptc_move48_ranks_three_defenses_in_both_colors_and_score_paths(self):
        for defender in (WHITE, BLACK):
            for complete in (False, True):
                with self.subTest(defender=defender, complete=complete):
                    engine = Engine()
                    engine._memory_bonus = lambda: {}
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
                    result = engine.analyze_turn(board, defender, complete_scores=complete)
                    self.assertEqual(tuple(result['part4_result']['move']), point('I6'))
                    ranked = result['part4_result']['ranked']
                    # I6 为修复后的近端防守点；三点均走零耗温战术，普通补步已禁止。
                    self.assertEqual([(row['r'], row['c']) for row in ranked],
                                     [point('I6'), point('D1'), point('H5')])
                    self.assertEqual([row['state_name'] for row in ranked], ['无'] * 3)
                    for higher, lower_row in zip(ranked, ranked[1:]):
                        self.assertGreater(higher['score'], lower_row['score'])
                    lower = min(row['score'] for row in leaves)
                    upper = max(row['score'] for row in leaves)
                    for row in ranked:
                        if row['score'] is None:
                            self.assertFalse(complete)
                            self.assertEqual(row['state_name'], '必败')
                            self.assertFalse(row['score_complete'])
                        else:
                            self.assertTrue(row['score_complete'])
                            self.assertGreaterEqual(row['score'], lower)
                            self.assertLessEqual(row['score'], upper)
                    self.assertEqual((board, engine.moves, engine.search.sb, engine.search.sw,
                        engine.search._direction_levels, engine.search.gear,
                        engine.search.zero_history, engine.search.zero_active), before)
                    self.assertIsNone(engine.deep_search._root_pass)


if __name__ == '__main__':
    unittest.main()
