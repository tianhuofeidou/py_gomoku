# -*- coding: utf-8 -*-
"""普通候选 D（v1.15.0）内部宽度、根层一致、权重与回滚定向检查。

v1.15.0 起普通入口由 `ordinary_selector.select_candidates` 统一生成：
五档配额只决定攻防角色倾向，最终合榜最多 4 点；权重取强侧
max（精筛 TYPE_SCORE 部分含 a1/a2 倍率，潜力按新威胁单独加分），
不再使用固定 2 攻 2 防、旧 Δ 或召回补位。

冻结盘面 SYM_STONES / M10 来自本目录既有普通候选回归，均为真实小棋谱。
"""
import copy
import unittest
from unittest.mock import patch

from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine
from gomoku.core.ordinary_selector import _potential_strength
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import (
    GEAR_THREAT_POINTS, ORDINARY_A1_MULT, ORDINARY_A2_MULT,
    ORDINARY_GEAR_PROFILE, ORDINARY_POTENTIAL_WEIGHT, SCORE_MAX,
    TYPE_SCORE, Search,
)
from gomoku.core.utils import empty_board, place, BLACK, WHITE

SYM_STONES = [(5, 5, BLACK), (6, 7, BLACK), (4, 8, BLACK),
              (9, 9, WHITE), (8, 7, WHITE), (10, 6, WHITE)]
M10 = [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (6, 8, WHITE),
       (5, 6, BLACK), (7, 6, WHITE), (4, 6, BLACK), (8, 8, WHITE),
       (5, 8, BLACK), (6, 7, WHITE)]


def clamp(weight):
    return min(max(float(weight), 1.0), 100.0)


def sym_board():
    board = empty_board()
    for r, c, p in SYM_STONES:
        place(board, r, c, p)
    return board


def make_search(board):
    search = Search(PatternAnalyzer())
    search.rebuild_all(board)
    return search


def replay(moves):
    engine = Engine()
    board = empty_board()
    for r, c, p in moves:
        place(board, r, c, p)
        engine.on_move(board, r, c, p)
    return engine, board


def ordinary_points(search, board, player):
    return search._ordinary_candidates(board, player, with_score=True)


def refined_score(search, board, r, c, side, player):
    """按 D 公式重算单侧精筛分，用于解释候选权重来源。"""
    subtype = (search.sb if side == BLACK else search.sw)[r][c]
    mode = 'dual' if search._dual_score(board, r, c) > 0 else (
        'attack' if side == player else 'defend')
    base = search._candidate_score(board, r, c, side, mode)
    multiplier = (ORDINARY_A1_MULT if subtype in (5, 6, 7, 8)
                  else ORDINARY_A2_MULT if subtype in (9, 10, 11, 12)
                  else 1.0)
    base += ((multiplier - 1.0) * TYPE_SCORE.get(subtype, 0.0)
             / SCORE_MAX * 100.0)
    strength = _potential_strength(search, board, r, c, side,
                                   GEAR_THREAT_POINTS)
    bonus = ORDINARY_POTENTIAL_WEIGHT * strength / 100.0
    return min(100.0, base + bonus / SCORE_MAX * 100.0)


class InternalCandidateWidthTest(unittest.TestCase):

    def test_black_white_symmetry_without_forcing_coordinates(self):
        board = sym_board()
        search = make_search(board)
        black = ordinary_points(search, board, BLACK)
        white = ordinary_points(search, board, WHITE)
        self.assertLessEqual(len(black), 4)
        self.assertLessEqual(len(white), 4)
        self.assertEqual(sorted(weight for _r, _c, weight in black),
                         sorted(weight for _r, _c, weight in white))

    def test_five_gear_quota_is_not_fixed_two_two(self):
        for gear, quota in ORDINARY_GEAR_PROFILE.items():
            self.assertEqual(sum(quota), 4)
        board = empty_board()
        place(board, 7, 7, BLACK)
        search = make_search(board)
        points = ordinary_points(search, board, WHITE)
        # 只有黑表有可用候选；D 不会为凑满 (2,2) 制造白侧点。
        self.assertEqual(ORDINARY_GEAR_PROFILE[search._ordinary_gear(WHITE)],
                         (2, 2))
        self.assertLess(len(points), 4)
        self.assertGreater(len(points), 0)

    def test_strong_side_refined_weights_and_order(self):
        """SYM_STONES 黑方 D 输出；期望来自 v1.15.0 实测与 269 例同构验收。"""
        board = sym_board()
        search = make_search(board)
        got = ordinary_points(search, board, BLACK)
        expected = [(5, 7, 45.19230769230769),
                    (6, 6, 45.14423076923077),
                    (5, 8, 43.07692307692308),
                    (3, 7, 40.96153846153846)]
        self.assertEqual([(r, c) for r, c, _w in got],
                         [(r, c) for r, c, _w in expected])
        for (r, c, weight), (_er, _ec, expected_weight) in zip(got, expected):
            self.assertAlmostEqual(weight, expected_weight, places=12)
            strong = max(refined_score(search, board, r, c, BLACK, BLACK),
                         refined_score(search, board, r, c, WHITE, BLACK))
            self.assertAlmostEqual(weight, clamp(strong), places=12)
        weights = [weight for _r, _c, weight in got]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_attack_defense_overlap_uses_dual_raw_score_not_weak_blend(self):
        engine, board = replay(M10)
        search = engine.search
        point = (6, 4)
        attack = refined_score(search, board, *point, BLACK, WHITE)
        defense = refined_score(search, board, *point, WHITE, WHITE)
        self.assertIsNotNone(attack)
        self.assertIsNotNone(defense)
        points = ordinary_points(search, board, WHITE)
        weight = next(weight for r, c, weight in points if (r, c) == point)
        self.assertAlmostEqual(weight, clamp(max(attack, defense)), places=12)

    def test_ordinary_selection_restores_board_and_search_state(self):
        engine, board = replay(M10)
        search = engine.search

        def snapshot():
            return copy.deepcopy((board, engine.moves, search.sb, search.sw,
                                  search._direction_levels, search.zero_history,
                                  search.zero_active, search.gear,
                                  search.gear_progress, search._gear_active))

        before = snapshot()
        for player in (BLACK, WHITE):
            ordinary_points(search, board, player)
            _potential_strength(search, board, 6, 4, player,
                                GEAR_THREAT_POINTS)
        self.assertEqual(snapshot(), before)

    def test_root_uses_same_selector_width(self):
        board = sym_board()
        search = make_search(board)
        deep = DeepSearch(search)
        expected = ordinary_points(search, board, BLACK)
        root = deep._candidate_points(board, BLACK, 5, plies=0)
        self.assertEqual(root, expected)
        self.assertEqual(len(root), 4)
        engine, board2 = replay(SYM_STONES)
        result = engine.analyze_turn(board2, BLACK)
        self.assertEqual(result['part3_candidates']['type'], 'fallback')
        self.assertEqual(len(result['part3_candidates']['points']), 4)

    def test_each_internal_depth_at_most_four(self):
        board = sym_board()
        search = make_search(board)
        deep = DeepSearch(search)
        expected = ordinary_points(search, board, BLACK)
        for plies in (1, 2, 3, 5, 8, 13, 21):
            with self.subTest(plies=plies):
                got = deep._candidate_points(board, BLACK, 5, plies=plies)
                self.assertEqual(got, expected)
                self.assertLessEqual(len(got), 4)

    def test_actual_recursion_uses_current_five_gear_quota(self):
        stones = [(7, 7, BLACK), (7, 6, WHITE), (6, 6, BLACK),
                  (6, 7, WHITE), (5, 8, BLACK), (8, 5, WHITE)]
        engine, board = replay(stones)
        fallback_calls = []
        orig_fb = Search._fallback_by_count

        def record_fb(self, b, attack_n, defend_n, player=WHITE,
                      with_score=False):
            out = orig_fb(self, b, attack_n, defend_n, player, with_score)
            fallback_calls.append(
                (attack_n, defend_n, len(out),
                 ORDINARY_GEAR_PROFILE[self._ordinary_gear(player)]))
            return out

        plies_seen = []
        orig_cp = DeepSearch._candidate_points

        def record_cp(self, b, player, n=5, plies=0):
            out = orig_cp(self, b, player, n, plies)
            plies_seen.append(plies)
            return out

        points = engine.search._fallback(board, player=BLACK, with_score=True)
        with patch.object(Search, '_fallback_by_count', record_fb), \
                patch.object(DeepSearch, '_candidate_points', record_cp):
            rows = engine.deep_search.rank_candidates(
                board, {(r, c): weight for (r, c, weight) in points}, BLACK)
        self.assertTrue(rows)
        internal_calls = [call for call in fallback_calls
                          if call[:2] == (2, 2)]
        self.assertTrue(internal_calls, '真实递归未出现普通候选初选调用')
        for attack_n, defend_n, count, expected in fallback_calls:
            self.assertEqual((attack_n, defend_n), expected)
            self.assertEqual(attack_n + defend_n, 4)
            self.assertLessEqual(count, 4)
        self.assertTrue(any(depth > 0 for depth in plies_seen))
        self.assertGreaterEqual(max(plies_seen), 2, '应到达多个内部深度')

    def test_tactical_candidates_not_limited_to_four(self):
        deep = DeepSearch(make_search(empty_board()))
        tactical = ('my-vcf', {(0, index): 100 for index in range(8)})
        with patch.object(deep, '_tactical_result', return_value=tactical):
            internal = deep._candidate_points(empty_board(), BLACK, 5, plies=1)
            root = deep._candidate_points(empty_board(), BLACK, 5, plies=0)
        self.assertEqual(len(internal), 8)
        self.assertEqual(len(root), 8)

    def test_root_and_internal_candidate_cache_shared(self):
        """同盘面、行棋方与档位上下文相同，根层/内部共享候选缓存。"""
        engine, board = replay(SYM_STONES)
        deep = engine.deep_search
        try:
            deep._begin_search_cache(board, False)
            internal = deep._candidate_points(board, BLACK, 5, plies=1)
            root = deep._candidate_points(board, BLACK, 5, plies=0)
            self.assertEqual(len(internal), 4)
            self.assertEqual(len(root), 4)
            keys = set(deep._tree_root['candidates'].keys())
            self.assertEqual(keys, {(BLACK, engine.search._gear_context())})
            base = deep.cache_stats['candidate_hits']
            self.assertEqual(deep._candidate_points(board, BLACK, 5, plies=1),
                             internal)
            self.assertEqual(deep._candidate_points(board, BLACK, 5, plies=0),
                             root)
            self.assertEqual(deep.cache_stats['candidate_hits'] - base, 2)
        finally:
            deep._reuse_active = False
            deep._cache_positions = {}
            deep._cache_paths = {}

    def test_simulation_rollback(self):
        engine, board = replay(M10)
        search = engine.search

        def snapshot():
            return copy.deepcopy((board, engine.moves, search.sb, search.sw,
                                  search._direction_levels, search.zero_history,
                                  search.zero_active, search.gear))

        before = snapshot()
        points = ordinary_points(search, board, WHITE)
        rows = engine.deep_search.rank_candidates(
            board, {(r, c): weight for (r, c, weight) in points}, WHITE)
        self.assertTrue(rows)
        self.assertEqual(snapshot(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
