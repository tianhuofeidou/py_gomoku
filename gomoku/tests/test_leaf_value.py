# -*- coding: utf-8 -*-
"""叶子评估 v3 单元测试：
当前局面全盘和、必杀优先级、平滑累进税、路径质量（_step_quality）、
加权平均（_weighted_average）、剪枝（_prune_branches）、胜负叶恒值。"""
import math
import unittest
from unittest import mock

from gomoku.core.utils import empty_board, place, BLACK, WHITE
from gomoku.core.engine import Engine


def engine_with(stones):
    e = Engine()
    e.deep_search.use_nn = False
    b = empty_board()
    for r, c, p in stones:
        place(b, r, c, p)
        e.on_move(b, r, c, p)
    return e, b


class StepQualityTest(unittest.TestCase):

    def test_step_quality_ratio(self):
        ds = Engine().deep_search
        self.assertEqual(ds._step_quality(100.0, [100, 80, 60]), 1.0)
        self.assertEqual(ds._step_quality(80.0, [100, 80, 60]), 0.8)
        self.assertEqual(ds._step_quality(60.0, [100, 80, 60]), 0.6)
        # 防御：极端值不超 1
        self.assertEqual(ds._step_quality(120.0, [100, 80]), 1.0)
        # 空候选 → 1.0
        self.assertEqual(ds._step_quality(60.0, []), 1.0)

    def test_weighted_average(self):
        ds = Engine().deep_search
        self.assertAlmostEqual(ds._weighted_average([(1.0, 800), (0.6, 900)]),
                               800 + 0.6 * 900)
        self.assertEqual(ds._weighted_average([]), 0.0)

    def test_prune_branches(self):
        ds = Engine().deep_search
        branches = [(0.999, 'a'), (0.001, 'b')]
        kept = ds._prune_branches(branches)
        self.assertEqual(kept, [(0.999, 'a')])
        # 全保留：占比都 >= 阈值
        self.assertEqual(len(ds._prune_branches([(0.9, 'a'), (0.6, 'b'), (0.4, 'c')])), 3)
        # 全被剪（占比都 < 阈值，仅异常防御）→ 保底最强一支
        kept = ds._prune_branches([(0.999, 'a'), (0.0005, 'b'), (0.0004, 'c')])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][0], 0.999)


class CurrentLeafScoreTest(unittest.TestCase):
    """第六章目标口径：读当前棋盘两张状态表，不读历史 ctx。"""

    @staticmethod
    def _score(e, b, player=WHITE, ctx_branch=None):
        return e.deep_search._leaf_score(
            b, [(7, 7, player, 60.0)], 'stale', player, ctx_branch or [])

    def test_root_five_wins_920(self):
        e, b = engine_with([])
        e.search.sw[7][7] = 0
        self.assertEqual(self._score(e, b), 920.0)

    def test_root_strong_kill_850(self):
        for v in (1, 2, 3):
            e, b = engine_with([])
            e.search.sw[7][7] = v
            self.assertEqual(self._score(e, b), 850.0)

    def test_root_weak_kill_650(self):
        e, b = engine_with([])
        e.search.sw[7][7] = 4
        self.assertEqual(self._score(e, b), 650.0)

    def test_opponent_five_strong_weak(self):
        cases = ((0, -920.0), (1, -700.0), (2, -700.0), (3, -700.0), (4, -500.0))
        for v, expected in cases:
            e, b = engine_with([])
            e.search.sb[7][7] = v
            self.assertEqual(self._score(e, b), expected)

    def test_priority_order(self):
        # 对方成五优先于我方强必杀
        e, b = engine_with([])
        e.search.sw[7][7] = 1
        e.search.sb[7][8] = 0
        self.assertEqual(self._score(e, b), -920.0)
        # 我方成五优先于对方成五
        e, b = engine_with([])
        e.search.sw[7][7] = 0
        e.search.sb[7][8] = 0
        self.assertEqual(self._score(e, b), 920.0)
        # 我方强必杀优先于对方弱必杀
        e, b = engine_with([])
        e.search.sw[7][7] = 3
        e.search.sb[7][8] = 4
        self.assertEqual(self._score(e, b), 850.0)
        # 对方强必杀优先于我方弱必杀
        e, b = engine_with([])
        e.search.sw[7][7] = 4
        e.search.sb[7][8] = 2
        self.assertEqual(self._score(e, b), -700.0)

    def test_count_does_not_matter(self):
        e, b = engine_with([])
        for c in (7, 8, 9):
            e.search.sb[7][c] = 1
        self.assertEqual(self._score(e, b), -700.0)
        e, b = engine_with([])
        for c in (7, 8, 9):
            e.search.sw[7][c] = 4
        self.assertEqual(self._score(e, b), 650.0)

    def test_no_kill_sum_and_tax(self):
        e, b = engine_with([])
        e.search.sw[7][7] = 5          # a1_bbb = 400
        e.search.sb[7][8] = 13         # b_bbb = 60
        with mock.patch.object(e.search, '_target_gear', return_value=3):
            score = self._score(e, b)
        expected = 0.05 * 340 + 475 * (1 - math.exp(-340 / 500.0))
        self.assertAlmostEqual(score, expected)

    def test_gear_coefficient(self):
        e, b = engine_with([])
        e.search.sw[7][7] = 5
        e.search.sb[7][8] = 13
        with mock.patch.object(e.search, '_target_gear', return_value=1):
            score_attack = self._score(e, b)   # 黑方 ×1.2 → D=328
        with mock.patch.object(e.search, '_target_gear', return_value=4):
            score_defend = self._score(e, b)   # 黑方 ×0.9 → D=346
        self.assertLess(score_attack, score_defend)
        self.assertAlmostEqual(score_attack, 0.05 * 328 + 475 * (1 - math.exp(-328 / 500.0)))
        self.assertAlmostEqual(score_defend, 0.05 * 346 + 475 * (1 - math.exp(-346 / 500.0)))

    def test_negative_diff(self):
        e, b = engine_with([])
        e.search.sw[7][7] = 13         # 我方 b_bbb = 120
        e.search.sb[7][8] = 5          # 对方 a1_bbb = 250
        e.search.sb[8][8] = 5          # 对方 a1_bbb = 250
        with mock.patch.object(e.search, '_target_gear', return_value=3):
            score = self._score(e, b)
        expected = -(0.05 * 380 + 475 * (1 - math.exp(-380 / 500.0)))
        self.assertAlmostEqual(score, expected)

    def test_occupied_cells_ignored(self):
        e, b = engine_with([])
        b[7][7] = WHITE          # 只改棋盘，不调用 on_move，隔离测试扫描跳过占位格
        e.search.sw[7][7] = 5
        with mock.patch.object(e.search, '_target_gear', return_value=3):
            self.assertEqual(self._score(e, b), 0.0)

    def test_no_clamp_extreme(self):
        e, b = engine_with([])
        for r in range(15):
            for c in range(15):
                e.search.sw[r][c] = 5
        with mock.patch.object(e.search, '_target_gear', return_value=3):
            score = self._score(e, b)
        self.assertGreater(score, 1000.0)
        self.assertAlmostEqual(score, 0.05 * 90000 + 475 * (1 - math.exp(-180)))

    def test_ctx_branch_does_not_affect_score(self):
        e, b = engine_with([])
        e.search.sw[7][7] = 5
        e.search.sb[7][8] = 13
        old_style_ctx = [{'camp': 1, 'cand_sub_ids': [0, 0], 'gear': 3},
                         {'camp': 0, 'cand_sub_ids': [1, 2, 3], 'gear': 3}]
        with mock.patch.object(e.search, '_target_gear', return_value=3):
            s1 = self._score(e, b, ctx_branch=old_style_ctx)
            s2 = self._score(e, b, ctx_branch=[])
        self.assertEqual(s1, s2)

    def test_black_root_symmetry(self):
        e, b = engine_with([])
        e.search.sb[7][7] = 0
        self.assertEqual(self._score(e, b, player=BLACK), 920.0)
        e, b = engine_with([])
        e.search.sw[7][7] = 0
        self.assertEqual(self._score(e, b, player=BLACK), -920.0)

    def test_empty_board_zero(self):
        e, b = engine_with([])
        with mock.patch.object(e.search, '_target_gear', return_value=3):
            self.assertEqual(self._score(e, b), 0.0)


class LeafScoreTest(unittest.TestCase):

    def test_win_lose_constant(self):
        e = Engine()
        e.deep_search.use_nn = False
        b = empty_board()
        branch = [(7, 7, WHITE, 100.0)]
        self.assertEqual(e.deep_search._leaf_score(b, branch, 'win', WHITE, []), 1000.0)
        self.assertEqual(e.deep_search._leaf_score(b, branch, 'lose', WHITE, []), -1000.0)

    def test_stale_stats_path(self):
        e = Engine()
        e.deep_search.use_nn = False
        b = empty_board()
        branch = [(7, 7, WHITE, 60.0)]
        e.deep_search.reset_leaf_eval_stats()
        e.deep_search._leaf_score(b, branch, 'stale', WHITE, [])
        st = e.deep_search.leaf_eval_status()
        self.assertEqual(st['mode'], 'heuristic')
        self.assertEqual(st['total'], 1)


class PickKeyTest(unittest.TestCase):

    def test_forced_layers(self):
        ds = Engine().deep_search
        win = ds._pick_key(1, 0.0, 100)
        mid = ds._pick_key(0, -0.6, 99999)
        los = ds._pick_key(-1, 0.0, 100)
        # 必胜档 > 无判定档 > 必败档（层级第一优先）
        self.assertGreater(win, mid)
        self.assertGreater(mid, los)

    def test_ratio_orders_within_neutral(self):
        ds = Engine().deep_search
        high = ds._pick_key(0, 0.5, 1000)
        low = ds._pick_key(0, -0.6, 99999)     # 比例低者即使 score 再高也靠后
        self.assertGreater(high, low)

    def test_non_positive_ratio_falls_back_to_candidate_score(self):
        ds = Engine().deep_search
        high_score = ds._pick_key(0, -0.9, 500)
        low_score = ds._pick_key(0, 0.0, 100)
        self.assertGreater(high_score, low_score)

    def test_forced_loss_is_below_every_non_losing_candidate(self):
        ds = Engine().deep_search
        losing = ds._pick_key(-1, 1.0, 999999)
        ordinary = ds._pick_key(0, -1.0, -999999)
        self.assertGreater(ordinary, losing)


class FatalMergeTest(unittest.TestCase):

    def test_undecided_child_prevents_forced_loss(self):
        ds = Engine().deep_search
        self.assertEqual(ds._merge_fatal_states([-1, 0]), 0)
        self.assertEqual(ds._merge_fatal_states([-1, 0, 0]), 0)

    def test_forced_state_requires_node_level_condition(self):
        ds = Engine().deep_search
        self.assertEqual(ds._merge_fatal_states([1, 0]), 1)
        self.assertEqual(ds._merge_fatal_states([-1, -1]), -1)
        self.assertEqual(ds._merge_fatal_states([0, 0]), 0)


class FatalRatioTest(unittest.TestCase):

    def test_denominator_is_all_children(self):
        """分母必须是全部子节点数，而不是仅胜败子节点数。"""
        ds = Engine().deep_search
        # 旧口径：1 / (2 + 1) ≈ 0.333；新口径：1 / 10 = 0.1
        self.assertAlmostEqual(ds._fatal_ratio(2, 1, 10), 0.1)
        self.assertAlmostEqual(ds._fatal_ratio(1, 0, 4), 0.25)

    def test_zero_denominator_is_zero(self):
        ds = Engine().deep_search
        self.assertEqual(ds._fatal_ratio(0, 0, 0), 0.0)
        self.assertEqual(ds._fatal_ratio(1, 0, 0), 0.0)

    def test_root_ratio_counts_undecided_children(self):
        """根节点比例用全部子节点做分母；未判定子节点也计入。"""
        e = Engine()
        e.deep_search.T0 = 160.0
        board = empty_board()
        root = (7, 7)

        def fake_candidate_points(_board, player, n=5, plies=0):
            if player == BLACK:
                return [(7, 7, 60.0)]
            return [(7, 8, 60.0)]

        with mock.patch.object(e.deep_search, '_candidate_points',
                               side_effect=fake_candidate_points), \
                mock.patch.object(e.deep_search, '_recursive',
                                  return_value=(0.0, 0, 2, 1, 10, 0)):
            ranked = e.deep_search.rank_candidates(board, [root], BLACK)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]['forced'], 0)
        # 根候选自身 1 个 + 递归子树全部节点 10 个 = 分母 11；
        # 分子为递归子树里的 2 胜 1 负。
        self.assertAlmostEqual(ranked[0]['fatal_ratio'], 1 / 11.0)


class IntegrationTest(unittest.TestCase):

    def test_rank_returns_desc_and_valid(self):
        """普通局面：候选全部打分、降序、空位（回归场景）。"""
        e, b = engine_with([(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (6, 8, WHITE),
                            (5, 6, BLACK), (7, 6, WHITE), (4, 6, BLACK), (8, 8, WHITE),
                            (5, 8, BLACK), (6, 7, WHITE)])
        res = e.analyze_turn(b, WHITE)
        r4 = res['part4_result']
        self.assertEqual(r4['type'], 'searched')
        ranked = r4['ranked']
        self.assertTrue(len(ranked) >= 2)
        keys = [e.deep_search._pick_key_state(x.get('state'), x['score'])
                for x in ranked]
        self.assertEqual(keys, sorted(keys, reverse=True))
        self.assertEqual(r4['move'], (ranked[0]['r'], ranked[0]['c']))
        for x in ranked:
            self.assertEqual(b[x['r']][x['c']], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
