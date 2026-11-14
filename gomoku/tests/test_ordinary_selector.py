# -*- coding: utf-8 -*-
"""普通候选 D（v1.15.0）选择器定向回归。

冻结盘面说明：
- SYM_STONES / M10 来自本目录既有普通候选测试，均为真实小棋谱；
- 五连盘面只用于验证战术路径不进入普通入口；
- 期望值来自 v1.15.0 正式实现与本轮 269 例精确同构验收
  （storages/experiments/20261114-v1150-原D正式接入-01），
  正式测试本身不读取实验目录。

覆盖：
1. `_choose` 只取强侧 max，不做弱侧 0.25 混合；
2. a1/a2 倍率只精筛 TYPE_SCORE 部分，潜力加分不乘倍率；
3. 纯棋型耗温仍读原 TYPE_SCORE；
4. 首阶段原生候选补入池；
5. 五档角色、配额和签名视角；
6. 潜力只计新威胁、单轴重算，且不污染真实 Search；
7. 战术路径不调用普通入口；
8. 空/稀疏/已占点最多 4 点且合法；
9. 缓存参数身份包含 ORDINARY_A1_MULT / ORDINARY_POTENTIAL_WEIGHT。
"""
import copy
import unittest
from unittest.mock import patch

from gomoku.core import ordinary_selector
from gomoku.core import search as search_mod
from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine
from gomoku.core.ordinary_selector import _axis, _choose, _potential_strength
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import (
    GEAR_THREAT_POINTS, ORDINARY_A1_MULT, ORDINARY_A2_MULT,
    ORDINARY_GEAR_PROFILE, ORDINARY_POTENTIAL_WEIGHT, POTEN, SCORE_MAX,
    SUBCLASS_PARENT, THREAT, TYPE_SCORE, Search,
)
from gomoku.core.utils import (
    BLACK, DIRECTIONS, EMPTY, SIZE, WHITE, empty_board, place,
)

SYM_STONES = [(5, 5, BLACK), (6, 7, BLACK), (4, 8, BLACK),
              (9, 9, WHITE), (8, 7, WHITE), (10, 6, WHITE)]
M10 = [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (6, 8, WHITE),
       (5, 6, BLACK), (7, 6, WHITE), (4, 6, BLACK), (8, 8, WHITE),
       (5, 8, BLACK), (6, 7, WHITE)]
FIVE = [(7, 3, BLACK), (7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)]


def board_of(stones):
    board = empty_board()
    for r, c, p in stones:
        place(board, r, c, p)
    return board


def make_search(board):
    search = Search(PatternAnalyzer())
    search.rebuild_all(board)
    return search


def replay(stones):
    engine = Engine()
    board = empty_board()
    for r, c, p in stones:
        place(board, r, c, p)
        engine.on_move(board, r, c, p)
    return engine, board


def snapshot(board, search):
    return copy.deepcopy((
        board, search.sb, search.sw, search._direction_levels,
        search.gear, search.gear_progress, search._gear_active,
        search.zero_history, search.zero_active,
    ))


def row(r, c, a=None, d=None, sig_a=None, sig_d=None):
    return dict(r=r, c=c, a=a, d=d,
                ascore=a, dscore=d, sig_a=sig_a, sig_d=sig_d)


class ChooseTest(unittest.TestCase):
    """纯 `_choose`：只看已算好的行，验证合榜语义。"""

    def test_strong_side_max_without_weak_side_blend(self):
        # 60 + 0.25 * 90 = 82.5；D 应取强侧 90。
        rows = [row(4, 5, a=60.0, d=90.0, sig_a=('a',), sig_d=('d',))]
        self.assertEqual(_choose(rows, (2, 2)), [(4, 5, 90.0)])
        self.assertNotEqual(_choose(rows, (2, 2))[0][2], 82.5)

    def test_score_cap_and_floor(self):
        high = [row(0, 0, a=150.0, sig_a=('h',))]
        low = [row(1, 1, a=0.0, sig_a=('l',))]
        self.assertEqual(_choose(high, (2, 2)), [(0, 0, 100.0)])
        self.assertEqual(_choose(low, (2, 2)), [(1, 1, 1.0)])

    def test_quota_roles_and_tie_prefer_attack_on_equal_priority(self):
        rows = [
            row(1, 1, a=90.0, d=80.0, sig_a=('a1',), sig_d=('d1',)),
            row(2, 2, a=80.0, d=90.0, sig_a=('a2',), sig_d=('d2',)),
            row(3, 3, a=70.0, d=70.0, sig_a=('a3',), sig_d=('d3',)),
        ]
        # 首轮 A/B 同强侧 90，a>=d 平局取 a，再按 (r,c) 取 A；
        # 第二轮 A 已用，B/C 都因攻额已满转 d，B 的强侧 90 先取；
        # merged 精筛仍会继续到宽度上限，第三点 C 保留（配额只是角色倾向）。
        self.assertEqual(_choose(rows, (1, 1)),
                         [(1, 1, 90.0), (2, 2, 90.0), (3, 3, 70.0)])

    def test_signature_dedup_is_role_scoped(self):
        rows = [
            row(1, 1, a=90.0, sig_a=('shape',)),
            row(2, 2, a=85.0, sig_a=('shape',)),
            row(3, 3, a=80.0, sig_a=('other',)),
        ]
        self.assertEqual(_choose(rows, (2, 0)),
                         [(1, 1, 90.0), (3, 3, 80.0)])

    def test_same_signature_allowed_in_other_role(self):
        rows = [
            row(1, 1, a=90.0, sig_a=('shape',)),
            row(2, 2, d=85.0, sig_d=('shape',)),
        ]
        self.assertEqual(_choose(rows, (1, 1)),
                         [(1, 1, 90.0), (2, 2, 85.0)])


class RefineFormulaTest(unittest.TestCase):
    """精筛公式：倍率只加在 TYPE_SCORE 上，潜力单独加在强侧分。"""

    def test_multipliers_only_refine_type_score_component(self):
        board = board_of(M10)
        search = make_search(board)
        captured = []

        def fake_fallback(self, b, attack_n, defend_n, player=WHITE,
                          with_score=False):
            # 故意只给防守侧（黑表）三个已知子类点。
            return [(5, 4, 1.0), (6, 4, 1.0), (4, 4, 1.0)]

        def fake_choose(rows, quota, limit=4):
            captured.extend(rows)
            return []

        with patch.object(Search, '_candidate_score', return_value=50.0), \
                patch.object(Search, '_fallback_by_count', fake_fallback), \
                patch.object(ordinary_selector, '_potential_strength',
                             return_value=30), \
                patch.object(ordinary_selector, '_choose',
                             side_effect=fake_choose):
            ordinary_selector.select_candidates(search, board, WHITE)

        by_point = {(item['r'], item['c']): item for item in captured}
        bonus = ORDINARY_POTENTIAL_WEIGHT * 30 / 100.0 / SCORE_MAX * 100.0
        # 黑表子类：8=A1(5..8), 12=A2(9..12), 16=B(13..16)。
        for point, subtype, mult in (
                ((5, 4), 8, ORDINARY_A1_MULT),
                ((6, 4), 12, ORDINARY_A2_MULT),
                ((4, 4), 16, 1.0)):
            with self.subTest(point=point, subtype=subtype):
                item = by_point[point]
                base = 50.0 + (mult - 1.0) * TYPE_SCORE[subtype] / SCORE_MAX * 100.0
                # 原始综合分/邻近/攻防一体都没有被倍率放大。
                self.assertAlmostEqual(item['d'], base)
                # 潜力加分不乘 a1/a2 倍率。
                self.assertAlmostEqual(item['dscore'],
                                       min(100.0, base + bonus))

    def test_pattern_temperature_keeps_original_type_score(self):
        board = board_of(M10)
        search = make_search(board)
        deep = DeepSearch(search)
        before = deep._pattern_temperature_weight(BLACK, 5, 4)
        self.assertAlmostEqual(before, TYPE_SCORE[8] / SCORE_MAX * 100.0)
        with patch.object(search_mod, 'ORDINARY_A1_MULT', 9.0), \
                patch.object(search_mod, 'ORDINARY_A2_MULT', 9.0), \
                patch.object(search_mod, 'ORDINARY_POTENTIAL_WEIGHT', 999.0):
            self.assertEqual(deep._pattern_temperature_weight(BLACK, 5, 4),
                             before)


class PoolAndRoleTest(unittest.TestCase):
    """首阶段原生候选补池与攻防角色/签名视角。"""

    def _top8(self, search, board, side, score):
        table = search.sb if side == BLACK else search.sw
        points = [(r, c) for r in range(SIZE) for c in range(SIZE)
                  if board[r][c] == EMPTY
                  and SUBCLASS_PARENT[table[r][c]] in (THREAT, POTEN)]
        points.sort(key=score, reverse=True)
        return set(points[:8])

    def test_native_first_stage_candidates_enter_pool(self):
        board = board_of(SYM_STONES)
        search = make_search(board)
        sentinel = (9, 10)

        def score(point):
            # 让高行列点排在后面，确保 sentinel 不在任何一侧前 8。
            return 100.0 - (point[0] * SIZE + point[1]) * 0.01

        self.assertNotIn(sentinel, self._top8(search, board, BLACK, score))
        self.assertNotIn(sentinel, self._top8(search, board, WHITE, score))

        def capture(native):
            rows = []

            def fake_choose(captured_rows, quota, limit=4):
                rows.extend(captured_rows)
                return []

            def fake_fallback(self, b, attack_n, defend_n, player=WHITE,
                              with_score=False):
                return [(*sentinel, 1.0)] if native else []

            with patch.object(Search, '_candidate_score',
                              side_effect=lambda b, r, c, p, m: score((r, c))), \
                    patch.object(Search, '_fallback_by_count', fake_fallback), \
                    patch.object(ordinary_selector, '_potential_strength',
                                 return_value=0), \
                    patch.object(ordinary_selector, '_choose',
                                 side_effect=fake_choose):
                ordinary_selector.select_candidates(search, board, BLACK)
            return {(item['r'], item['c']) for item in rows}

        self.assertIn(sentinel, capture(True))
        self.assertNotIn(sentinel, capture(False))

    def test_roles_use_five_gear_quota_and_side_signatures(self):
        board = board_of(M10)
        search = make_search(board)
        player = WHITE
        rows, sig_calls = [], []

        def fake_fallback(self, b, attack_n, defend_n, player=WHITE,
                          with_score=False):
            return []

        def fake_sig(b, syms, r, c, side):
            sig_calls.append((r, c, side))
            return ('sig', side, r, c)

        def fake_choose(captured_rows, quota, limit=4):
            rows.extend(captured_rows)
            return []

        with patch.object(Search, '_fallback_by_count', fake_fallback), \
                patch.object(Search, '_dedup_sig', side_effect=fake_sig), \
                patch.object(ordinary_selector, '_potential_strength',
                             return_value=0), \
                patch.object(ordinary_selector, '_choose',
                             side_effect=fake_choose):
            ordinary_selector.select_candidates(search, board, player)

        self.assertEqual(ORDINARY_GEAR_PROFILE[search._ordinary_gear(player)],
                         (2, 2))
        for item in rows:
            point = (item['r'], item['c'])
            if item['a'] is not None:
                self.assertIn((*point, player), sig_calls)
            if item['d'] is not None:
                self.assertIn((*point, 3 - player), sig_calls)
        self.assertTrue(sig_calls)


class PotentialTest(unittest.TestCase):
    """潜力只计新威胁、单轴重算，局部副本不污染真实 Search。"""

    def test_potential_single_axis_only_new_threats_and_rollback(self):
        board = board_of(M10)
        search = make_search(board)
        candidate = (3, 7)
        before = snapshot(board, search)
        rescans = []
        original = search._direction_level

        def spy(b, r, c, player, dr, dc):
            rescans.append((r, c, dr, dc))
            return original(b, r, c, player, dr, dc)

        with patch.object(search, '_direction_level', side_effect=spy):
            strength = _potential_strength(
                search, board, candidate[0], candidate[1], BLACK,
                GEAR_THREAT_POINTS)

        self.assertEqual(strength, 40)
        # (5,7) 在落子前已是威胁(1)，必须跳过，不能重复计新威胁。
        self.assertEqual(search.sb[5][7], 1)
        self.assertIn(search.sb[5][7], GEAR_THREAT_POINTS)
        self.assertNotIn((5, 7), {(r, c) for r, c, _dr, _dc in rescans})
        # 每次续点只重扫候选到续点所在的那一条轴。
        for r, c, dr, dc in rescans:
            self.assertEqual((dr, dc), DIRECTIONS[_axis(candidate[0],
                                                        candidate[1], r, c)])
        self.assertEqual(len(rescans), len({(r, c) for r, c, _dr, _dc in rescans}))

        # 独立全轴重算应与缓存复用结果一致。
        local = [line[:] for line in board]
        local[candidate[0]][candidate[1]] = BLACK
        best = 0
        for r, c in sorted(search.scope._segment_scope(
                local, candidate[0], candidate[1], BLACK)):
            if local[r][c] != EMPTY or search.sb[r][c] in GEAR_THREAT_POINTS:
                continue
            local[r][c] = BLACK
            try:
                levels = search._point_levels(local, r, c, BLACK)
                after = search._classify_levels(levels)
            finally:
                local[r][c] = EMPTY
            if after in GEAR_THREAT_POINTS:
                best = max(best, GEAR_THREAT_POINTS[after])
        self.assertEqual(best, strength)

        self.assertEqual(snapshot(board, search), before)
        self.assertEqual(board[candidate[0]][candidate[1]], EMPTY)


class TacticalPathTest(unittest.TestCase):
    """战术路径不应进入普通候选入口。"""

    def test_engine_immediate_five_skips_ordinary_selector(self):
        engine, board = replay(FIVE)
        with patch.object(Search, '_ordinary_candidates',
                          side_effect=AssertionError('战术路径不得调用普通入口')):
            result = engine.analyze_turn(board, BLACK)
        self.assertEqual(result['part3_candidates']['reason'], 'my-five')
        self.assertEqual(result['part4_result']['type'], 'direct')
        self.assertIn(result['part4_result']['move'], {(7, 2), (7, 7)})

    def test_deep_search_tactical_candidates_not_truncated_or_ordinary(self):
        search = make_search(empty_board())
        deep = DeepSearch(search)
        tactical = ('my-vcf', {(0, index): 100 for index in range(8)})
        with patch.object(deep, '_tactical_result', return_value=tactical), \
                patch.object(Search, '_ordinary_candidates',
                             side_effect=AssertionError('战术路径不得调用普通入口')):
            internal = deep._candidate_points(empty_board(), BLACK, 5, plies=1)
            root = deep._candidate_points(empty_board(), BLACK, 5, plies=0)
        self.assertEqual(len(internal), 8)
        self.assertEqual(len(root), 8)


class WidthAndLegalityTest(unittest.TestCase):
    """空/稀疏/已占点边界：最多 4 点且均为合法空点。"""

    def test_empty_board_has_no_ordinary_candidates(self):
        board = empty_board()
        search = make_search(board)
        self.assertEqual(search._ordinary_candidates(board, BLACK, True), [])

    def test_sparse_board_returns_legal_points_within_width(self):
        board = board_of([(7, 7, BLACK)])
        search = make_search(board)
        points = search._ordinary_candidates(board, WHITE, True)
        self.assertGreater(len(points), 0)
        self.assertLessEqual(len(points), 4)
        self._assert_legal(board, points)

    def test_occupied_points_never_returned_and_width_capped(self):
        board = board_of(SYM_STONES)
        search = make_search(board)
        for player in (BLACK, WHITE):
            with self.subTest(player=player):
                points = search._ordinary_candidates(board, player, True)
                self.assertEqual(len(points), 4)
                self.assertLessEqual(len(points), 4)
                self._assert_legal(board, points)
                self.assertEqual(len({(r, c) for r, c, _w in points}), 4)

    def _assert_legal(self, board, points):
        occupied = {(r, c) for r in range(SIZE) for c in range(SIZE)
                    if board[r][c] != EMPTY}
        for r, c, _weight in points:
            self.assertTrue(0 <= r < SIZE and 0 <= c < SIZE)
            self.assertEqual(board[r][c], EMPTY)
            self.assertNotIn((r, c), occupied)


class CacheIdentityTest(unittest.TestCase):
    """普通候选参数变化必须改变缓存身份，不能继承旧树结论。"""

    def test_signature_includes_ordinary_parameters(self):
        board = board_of(SYM_STONES)
        deep = DeepSearch(make_search(board))
        deep._begin_search_cache(board, False)
        signature = deep._reuse_signature()
        names = {name for name, _value in signature[1]}
        for name in ('ORDINARY_A1_MULT', 'ORDINARY_A2_MULT',
                     'ORDINARY_POTENTIAL_WEIGHT', 'ORDINARY_POOL_SIDE'):
            self.assertIn(name, names)

        root = deep._tree_root
        with patch.object(search_mod, 'ORDINARY_A1_MULT', 2.0):
            self.assertNotEqual(deep._reuse_signature(), signature)
            deep._begin_search_cache(board, False)
            self.assertIsNot(deep._tree_root, root)
            self.assertNotEqual(deep._tree_signature, signature)

        with patch.object(search_mod, 'ORDINARY_POTENTIAL_WEIGHT', 60.0):
            self.assertNotEqual(deep._reuse_signature(), signature)


if __name__ == '__main__':
    unittest.main(verbosity=2)
