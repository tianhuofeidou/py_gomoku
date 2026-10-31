# -*- coding: utf-8 -*-
"""内部普通候选收窄到 3 个（2 攻 1 防）的针对性回归。

覆盖：
1. 黑白对称；
2. 保留防守点（不是把 3攻2防 直接截成 3 个攻点）；
3. 原始权重与顺序；
4. 根层配比兼容（根层仍为 3攻2防 / 动态档位，不受内部 3 点影响）；
5. 各内部深度最多 3 点；
6. 战术候选不受 3 点限制；
7. 根层/内部普通候选缓存键隔离；
8. 模拟深推后棋盘、着法、sb/sw、方向状态、历史与档位完整回滚。
"""
import copy
import unittest
from unittest.mock import patch

from gomoku.core.utils import empty_board, place, BLACK, WHITE, SIZE
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import Search, GEAR_PROFILE, DEFAULT_GEAR
from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine

# 180° 旋转 + 黑白互换后与原盘面相同的对称局面（用于黑白对称测试）。
SYM_STONES = [(5, 5, BLACK), (6, 7, BLACK), (4, 8, BLACK),
              (9, 9, WHITE), (8, 7, WHITE), (10, 6, WHITE)]

# 普通/战术混合的中盘局面（用于深推回滚与实际深度采样）。
M10 = [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (6, 8, WHITE),
       (5, 6, BLACK), (7, 6, WHITE), (4, 6, BLACK), (8, 8, WHITE),
       (5, 8, BLACK), (6, 7, WHITE)]


def clamp(w):
    return min(max(float(w), 1.0), 100.0)


def sym_board():
    b = empty_board()
    for r, c, p in SYM_STONES:
        place(b, r, c, p)
    return b


def make_search(board):
    s = Search(PatternAnalyzer())
    s.rebuild_all(board)
    return s


def rot180(r, c):
    return SIZE - 1 - r, SIZE - 1 - c


def replay(moves):
    e = Engine()
    b = empty_board()
    for (r, c, p) in moves:
        place(b, r, c, p)
        e.on_move(b, r, c, p)
    return e, b


def expected_internal(s, board, player):
    return [(r, c, clamp(w)) for (r, c, w)
            in s._fallback_by_count(board, 2, 1, player=player, with_score=True)]


class InternalCandidateWidthTest(unittest.TestCase):

    def test_black_white_symmetry(self):
        """对称局面下，黑白内部 2攻1防 候选互为 180° 镜像且权重一致。"""
        board = sym_board()
        s = make_search(board)
        d = DeepSearch(s)
        black = d._candidate_points(board, BLACK, 5, plies=1)
        white = d._candidate_points(board, WHITE, 5, plies=1)
        self.assertEqual(len(black), 3)
        self.assertEqual(len(white), 3)
        mapped = [(rot180(r, c)[0], rot180(r, c)[1], w) for (r, c, w) in black]
        self.assertEqual(mapped, white)

    def test_defense_slot_retained(self):
        """内部第 3 个候选必须是防守名额，而不是 3攻2防 截断后的第 3 个攻点。"""
        board = sym_board()
        s = make_search(board)
        d = DeepSearch(s)
        cands = d._candidate_points(board, BLACK, 5, plies=1)
        attack_two = s._fallback_by_count(board, 2, 1, player=BLACK, with_score=True)[:2]
        attack_pts = {p[:2] for p in attack_two}
        defense_all = s._fallback_by_count(board, 0, 1, player=BLACK, with_score=True)
        defense_expected = next(p for p in defense_all if p[:2] not in attack_pts)
        self.assertEqual(len(cands), 3)
        self.assertEqual(cands[2][:2], defense_expected[:2])
        self.assertAlmostEqual(cands[2][2], clamp(defense_expected[2]))
        # 第 3 点不能是攻点，且第 1/2 点确实来自攻点序列。
        self.assertNotIn(cands[2][:2], attack_pts)
        self.assertEqual([p[:2] for p in cands[:2]], [p[:2] for p in attack_two])

    def test_original_weights_and_order(self):
        """内部候选 = 2攻1防 原顺序 + _candidate_score 原权重（clamp 后），不重排、不改权。"""
        board = sym_board()
        s = make_search(board)
        d = DeepSearch(s)
        got = d._candidate_points(board, BLACK, 5, plies=1)
        expected = expected_internal(s, board, BLACK)
        self.assertEqual(got, expected)
        # 权重逐个按来源语义复算：前两点攻点（BLACK 视角）、第 3 点防守点
        # （WHITE 威胁视角）；dual 点用 dual 权重，否则按列表语义取 attack/defend。
        for index, (r, c, w) in enumerate(got):
            if index < 2:
                score_player, mode = BLACK, 'attack'
            else:
                score_player, mode = WHITE, 'defend'
            if s._dual_score(board, r, c) > 0:
                mode = 'dual'
            self.assertAlmostEqual(
                w, clamp(s._candidate_score(board, r, c, score_player, mode)))
        # 攻点按综合分降序；防守点单列在最后，不与攻点重排混合。
        self.assertGreaterEqual(got[0][2], got[1][2])

    def test_root_ratio_unchanged(self):
        """根层仍保持 3攻2防（5 点）与动态档位配比，不被内部 3 点收窄。"""
        board = sym_board()
        s = make_search(board)
        d = DeepSearch(s)
        root = d._candidate_points(board, BLACK, 5, plies=0)
        expected = [(r, c, clamp(w)) for (r, c, w)
                    in s._fallback(board, player=BLACK, with_score=True)]
        self.assertEqual(root, expected)
        self.assertEqual(len(root), 5)
        self.assertGreater(len(root), 3)
        # 动态档位配比（1~4 档）仍是原 (攻, 防) 名额。
        for gear in range(1, 5):
            a_n, d_n = GEAR_PROFILE[gear]
            got = s._fallback_by_count(board, a_n, d_n, player=BLACK, with_score=True)
            ref = s._fallback(board, gear=gear, player=BLACK, with_score=True)
            self.assertEqual(got, ref)
        # 引擎根层普通候选仍按动态档位生成，数量不少于 5 点（允许谱点/记忆补点）。
        e, b2 = replay(SYM_STONES)
        res = e.analyze_turn(b2, BLACK)
        self.assertGreaterEqual(len(res['part3_candidates']['points']), 5)

    def test_each_internal_depth_at_most_three(self):
        """plies>0 的普通候选在任意内部深度都固定 2攻1防、最多 3 点。"""
        board = sym_board()
        s = make_search(board)
        d = DeepSearch(s)
        expected = expected_internal(s, board, BLACK)
        for plies in (1, 2, 3, 5, 8, 13, 21):
            with self.subTest(plies=plies):
                got = d._candidate_points(board, BLACK, 5, plies=plies)
                self.assertEqual(got, expected)
                self.assertLessEqual(len(got), 3)

    def test_actual_recursion_uses_two_attack_one_defense(self):
        """真实深推递归中，内部普通候选调用全部是 (2攻,1防) 且结果 ≤3，并到达多个深度。"""
        # 对称稀疏盘面在新规则下自然只展开两手；使用既有六手局面覆盖深层，
        # 保持默认 T0=160，不靠补普通步或提高温度制造深度。
        stones = [(7, 7, BLACK), (7, 6, WHITE), (6, 6, BLACK),
                  (6, 7, WHITE), (5, 8, BLACK), (8, 5, WHITE)]
        e, board = replay(stones)
        self.assertEqual(e.deep_search.T0, 160.0)
        fallback_calls = []
        orig_fb = Search._fallback_by_count

        def record_fb(self, b, attack_n, defend_n, player=WHITE, with_score=False):
            out = orig_fb(self, b, attack_n, defend_n, player, with_score)
            fallback_calls.append((attack_n, defend_n, len(out)))
            return out

        plies_seen = []
        orig_cp = DeepSearch._candidate_points

        def record_cp(self, b, player, n=5, plies=0):
            out = orig_cp(self, b, player, n, plies)
            plies_seen.append(plies)
            return out

        points = e.search._fallback(board, player=BLACK, with_score=True)
        with patch.object(Search, '_fallback_by_count', record_fb), \
                patch.object(DeepSearch, '_candidate_points', record_cp):
            rows = e.deep_search.rank_candidates(
                board, {(r, c): w for (r, c, w) in points}, BLACK)
        self.assertTrue(rows)
        internal_calls = [c for c in fallback_calls if c[:2] == (2, 1)]
        self.assertTrue(internal_calls, '真实递归未出现内部普通候选')
        for attack_n, defend_n, n in fallback_calls:
            self.assertIn((attack_n, defend_n), ((3, 2), (2, 1)))
            if (attack_n, defend_n) == (2, 1):
                self.assertLessEqual(n, 3)
        self.assertTrue(any(p > 0 for p in plies_seen))
        self.assertGreaterEqual(max(plies_seen), 2, '应到达多个内部深度')

    def test_tactical_candidates_not_limited_to_three(self):
        """战术候选沿用原上限（最多 5 个），不受内部普通候选 3 点限制。"""
        d = DeepSearch(make_search(empty_board()))
        tactical = ('my-vcf', {(0, 0): 100, (0, 1): 100, (0, 2): 100,
                               (0, 3): 100, (0, 4): 100})
        with patch.object(d, '_tactical_result', return_value=tactical):
            internal = d._candidate_points(empty_board(), BLACK, 5, plies=1)
            root = d._candidate_points(empty_board(), BLACK, 5, plies=0)
        self.assertEqual(len(internal), 5)
        self.assertEqual(len(root), 5)

    def test_root_and_internal_candidate_cache_isolated(self):
        """同一盘面的根层 5 点与内部 3 点分别缓存，键含落子方与根层/内部标记。"""
        e, board = replay(SYM_STONES)
        d = e.deep_search
        try:
            d._begin_search_cache(board, False)
            internal = d._candidate_points(board, BLACK, 5, plies=1)
            root = d._candidate_points(board, BLACK, 5, plies=0)
            self.assertEqual(len(internal), 3)
            self.assertEqual(len(root), 5)
            keys = set(d._tree_root['candidates'].keys())
            self.assertEqual(keys, {(BLACK, True), (BLACK, False)})
            base = d.cache_stats['candidate_hits']
            self.assertEqual(d._candidate_points(board, BLACK, 5, plies=1), internal)
            self.assertEqual(d._candidate_points(board, BLACK, 5, plies=0), root)
            self.assertEqual(d.cache_stats['candidate_hits'] - base, 2)
        finally:
            d._reuse_active = False
            d._cache_positions = {}
            d._cache_paths = {}

    def test_simulation_rollback(self):
        """深推模拟后棋盘、真实着法、sb/sw、方向状态、历史与档位完整回滚。"""
        e, board = replay(M10)
        search = e.search

        def snapshot():
            return copy.deepcopy((board, e.moves, search.sb, search.sw,
                                  search._direction_levels, search.zero_history,
                                  search.zero_active, search.gear))

        before = snapshot()
        points = search._fallback(board, player=WHITE, with_score=True)
        rows = e.deep_search.rank_candidates(
            board, {(r, c): w for (r, c, w) in points}, WHITE)
        self.assertTrue(rows)
        self.assertEqual(snapshot(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
