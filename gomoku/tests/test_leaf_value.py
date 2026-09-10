# -*- coding: utf-8 -*-
"""叶子评估 v2 单元测试：
18 棋型结果分（_stale_result_score）、路径质量（_step_quality）、
加权平均（_weighted_average）、剪枝（_prune_branches）、胜负叶恒值。"""
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


class StaleResultScoreTest(unittest.TestCase):

    @staticmethod
    def ctx(camp, subs, gear=3):
        return {'camp': camp, 'cand_sub_ids': list(subs), 'gear': gear,
                'move_w': 60.0, 'cand_ws': [60.0] * len(subs), 'kill': 0}

    def test_my_kill_five_highest(self):
        e = Engine()
        score = e.deep_search._stale_result_score([self.ctx(1, [0])], WHITE)
        self.assertEqual(score, 900.0)

    def test_my_kill_live4_vs_d33_order(self):
        e = Engine()
        s1 = e.deep_search._stale_result_score([self.ctx(1, [1])], WHITE)
        s2 = e.deep_search._stale_result_score([self.ctx(1, [4])], WHITE)
        self.assertEqual(s1, 880.0)
        self.assertEqual(s2, 650.0)
        self.assertGreater(s1, s2)

    def test_opp_two_kill_unsolvable(self):
        e = Engine()
        # 我方一步（无杀）+ 对方一步：对方候选含两个 kill 级点 → 无解
        seq = [self.ctx(1, [5]), self.ctx(0, [0, 0])]
        score = e.deep_search._stale_result_score(seq, WHITE)
        self.assertEqual(score, -920.0)

    def test_opp_single_kill_counts_as_forcing_only(self):
        e = Engine()
        seq = [self.ctx(0, [0])]          # 对方单杀点：可先手占 → 逼应级
        score = e.deep_search._stale_result_score(seq, WHITE)
        self.assertGreaterEqual(score, -500.0)
        self.assertLess(score, 0.0)

    def test_regular_threat_diff(self):
        e = Engine()
        # 我方强活三 a2_bbb(9)=350；对方弱冲四 a1_ccc(8)=120；gear=3 → coef 1.0
        seq = [self.ctx(1, [9]), self.ctx(0, [8])]
        score = e.deep_search._stale_result_score(seq, WHITE)
        self.assertAlmostEqual(score, 350.0 - 120.0)

    def test_no_candidates_is_zero(self):
        e = Engine()
        score = e.deep_search._stale_result_score([], WHITE)
        self.assertEqual(score, 0.0)


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
