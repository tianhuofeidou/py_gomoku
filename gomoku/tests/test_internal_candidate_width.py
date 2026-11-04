# -*- coding: utf-8 -*-
"""普通候选固定 4 点（2 攻 2 防）的发布回归。

覆盖：
1. 黑白对称；
2. 保留两个防守点，不把 4 点截成纯攻点；
3. 原始权重与顺序；
4. 根层与内部都使用同一套 2攻2防；
5. 各内部深度最多 4 点；
6. 战术候选不受 4 点限制；
7. 根层/内部普通候选缓存键隔离；
8. 模拟深推后棋盘、着法、sb/sw、方向状态、历史与档位完整回滚。
"""
import copy
import unittest
from unittest.mock import patch

from gomoku.core.utils import empty_board, place, BLACK, WHITE, SIZE
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import Search
from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine

SYM_STONES = [(5, 5, BLACK), (6, 7, BLACK), (4, 8, BLACK),
              (9, 9, WHITE), (8, 7, WHITE), (10, 6, WHITE)]
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


def expected_candidates(s, board, player):
    return [(r, c, clamp(w)) for (r, c, w)
            in s._fallback_by_count(board, 2, 2, player=player, with_score=True)]


class InternalCandidateWidthTest(unittest.TestCase):

    def test_black_white_symmetry(self):
        board = sym_board()
        s = make_search(board)
        d = DeepSearch(s)
        black = d._candidate_points(board, BLACK, 5, plies=1)
        white = d._candidate_points(board, WHITE, 5, plies=1)
        self.assertEqual(len(black), 4)
        self.assertEqual(len(white), 4)
        mapped = [(rot180(r, c)[0], rot180(r, c)[1], w) for (r, c, w) in black]
        self.assertEqual(mapped, white)

    def test_two_defense_slots_retained(self):
        board = sym_board()
        s = make_search(board)
        d = DeepSearch(s)
        cands = d._candidate_points(board, BLACK, 5, plies=1)
        expected = expected_candidates(s, board, BLACK)
        attack_two, defense_two = expected[:2], expected[2:]
        attack_pts = {p[:2] for p in attack_two}
        self.assertEqual(len(cands), 4)
        self.assertEqual([p[:2] for p in cands[:2]], [p[:2] for p in attack_two])
        self.assertEqual([p[:2] for p in cands[2:]], [p[:2] for p in defense_two])
        self.assertTrue(all(p[:2] not in attack_pts for p in cands[2:]))

    def test_original_weights_and_order(self):
        board = sym_board()
        s = make_search(board)
        d = DeepSearch(s)
        got = d._candidate_points(board, BLACK, 5, plies=1)
        expected = expected_candidates(s, board, BLACK)
        self.assertEqual(got, expected)
        for index, (r, c, w) in enumerate(got):
            score_player, mode = (BLACK, 'attack') if index < 2 else (WHITE, 'defend')
            if s._dual_score(board, r, c) > 0:
                mode = 'dual'
            self.assertAlmostEqual(w, clamp(s._candidate_score(board, r, c, score_player, mode)))
        self.assertGreaterEqual(got[0][2], got[1][2])

    def test_root_uses_same_fixed_four(self):
        board = sym_board()
        s = make_search(board)
        d = DeepSearch(s)
        root = d._candidate_points(board, BLACK, 5, plies=0)
        expected = expected_candidates(s, board, BLACK)
        self.assertEqual(root, expected)
        self.assertEqual(len(root), 4)
        e, b2 = replay(SYM_STONES)
        res = e.analyze_turn(b2, BLACK)
        self.assertEqual(len(res['part3_candidates']['points']), 4)

    def test_each_internal_depth_at_most_four(self):
        board = sym_board()
        s = make_search(board)
        d = DeepSearch(s)
        expected = expected_candidates(s, board, BLACK)
        for plies in (1, 2, 3, 5, 8, 13, 21):
            with self.subTest(plies=plies):
                got = d._candidate_points(board, BLACK, 5, plies=plies)
                self.assertEqual(got, expected)
                self.assertLessEqual(len(got), 4)

    def test_actual_recursion_uses_two_attack_two_defense(self):
        stones = [(7, 7, BLACK), (7, 6, WHITE), (6, 6, BLACK),
                  (6, 7, WHITE), (5, 8, BLACK), (8, 5, WHITE)]
        e, board = replay(stones)
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
        internal_calls = [c for c in fallback_calls if c[:2] == (2, 2)]
        self.assertTrue(internal_calls, '真实递归未出现2攻2防普通候选')
        for attack_n, defend_n, n in fallback_calls:
            self.assertEqual((attack_n, defend_n), (2, 2))
            self.assertLessEqual(n, 4)
        self.assertTrue(any(p > 0 for p in plies_seen))
        self.assertGreaterEqual(max(plies_seen), 2, '应到达多个内部深度')

    def test_tactical_candidates_not_limited_to_four(self):
        d = DeepSearch(make_search(empty_board()))
        tactical = ('my-vcf', {(0, 0): 100, (0, 1): 100, (0, 2): 100,
                               (0, 3): 100, (0, 4): 100, (0, 5): 100,
                               (0, 6): 100, (0, 7): 100})
        with patch.object(d, '_tactical_result', return_value=tactical):
            internal = d._candidate_points(empty_board(), BLACK, 5, plies=1)
            root = d._candidate_points(empty_board(), BLACK, 5, plies=0)
        self.assertEqual(len(internal), 8)
        self.assertEqual(len(root), 8)

    def test_root_and_internal_candidate_cache_shared(self):
        """根层与内部使用同一固定4点候选，同一盘面按行棋方共享候选缓存。"""
        e, board = replay(SYM_STONES)
        d = e.deep_search
        try:
            d._begin_search_cache(board, False)
            internal = d._candidate_points(board, BLACK, 5, plies=1)
            root = d._candidate_points(board, BLACK, 5, plies=0)
            self.assertEqual(len(internal), 4)
            self.assertEqual(len(root), 4)
            keys = set(d._tree_root['candidates'].keys())
            self.assertEqual(keys, {BLACK})
            base = d.cache_stats['candidate_hits']
            self.assertEqual(d._candidate_points(board, BLACK, 5, plies=1), internal)
            self.assertEqual(d._candidate_points(board, BLACK, 5, plies=0), root)
            self.assertEqual(d.cache_stats['candidate_hits'] - base, 2)
        finally:
            d._reuse_active = False
            d._cache_positions = {}
            d._cache_paths = {}

    def test_simulation_rollback(self):
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
