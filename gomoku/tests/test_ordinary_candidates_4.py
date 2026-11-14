# -*- coding: utf-8 -*-
"""普通候选 D（v1.15.0）宽度、对称、回滚与原生初选定向检查。

v1.15.0 起普通入口不再是固定“2 攻 2 防”或旧 Δ+召回：
- `_ordinary_candidates` 调用 `ordinary_selector.select_candidates`；
- 五档档位只提供攻防配额，最终合榜最多 4 点；
- 首阶段初选复用 `_fallback_by_count` 的原生评分、签名去重和占点排除；
- 精筛权重取强侧 max，不再叠加弱侧 0.25，也不再把潜力算作胜势。

冻结盘面 SYM_STONES / M10 来自本目录既有普通候选回归，均为真实小棋谱。
"""
import copy
import unittest
from unittest.mock import patch

from gomoku.core import ordinary_selector
from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine
from gomoku.core.ordinary_selector import _potential_strength
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import GEAR_THREAT_POINTS, Search
from gomoku.core.utils import empty_board, place, BLACK, WHITE, SIZE, EMPTY

SYM_STONES = [(5, 5, BLACK), (6, 7, BLACK), (4, 8, BLACK),
              (9, 9, WHITE), (8, 7, WHITE), (10, 6, WHITE)]
M10 = [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (6, 8, WHITE),
       (5, 6, BLACK), (7, 6, WHITE), (4, 6, BLACK), (8, 8, WHITE),
       (5, 8, BLACK), (6, 7, WHITE)]


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


def rot180(r, c):
    return SIZE - 1 - r, SIZE - 1 - c


def legal_points(board, points):
    for r, c, _weight in points:
        if not (0 <= r < SIZE and 0 <= c < SIZE) or board[r][c] != EMPTY:
            return False
    return True


class OrdinaryCandidatesFourTest(unittest.TestCase):

    def test_ordinary_entry_delegates_to_selector(self):
        board = sym_board()
        search = make_search(board)
        sentinel = [(1, 1, 50.0)]
        with patch.object(ordinary_selector, 'select_candidates',
                          return_value=sentinel) as selector:
            self.assertEqual(search._ordinary_candidates(board, BLACK, True),
                             sentinel)
            selector.assert_called_once_with(search, board, BLACK)
            selector.reset_mock()
            self.assertEqual(search._ordinary_candidates(board, BLACK, False),
                             [(1, 1)])
            selector.assert_called_once_with(search, board, BLACK)

    def test_root_and_internal_share_selector_width(self):
        board = sym_board()
        search = make_search(board)
        deep = DeepSearch(search)
        expected = search._ordinary_candidates(board, BLACK, with_score=True)
        root = deep._candidate_points(board, BLACK, 5, plies=0)
        internal = deep._candidate_points(board, BLACK, 5, plies=1)
        self.assertEqual(root, expected)
        self.assertEqual(internal, expected)
        # D 的硬上限是 4；SYM_STONES 是既有冻结稠密盘面，本轮实测恰为 4。
        self.assertEqual(len(root), 4)
        self.assertLessEqual(len(root), 4)
        self.assertLessEqual(len(internal), 4)
        self.assertTrue(legal_points(board, root))
        self.assertEqual(len({(r, c) for r, c, _w in root}), len(root))

    def test_black_white_symmetry_without_forcing_coordinates(self):
        """同构盘面只保证评分镜像；同分按行列的稳定排序不保证 D4 坐标一一对应。"""
        board = sym_board()
        search = make_search(board)
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] != EMPTY:
                    continue
                rr, cc = rot180(r, c)
                for mode in ('attack', 'defend'):
                    self.assertAlmostEqual(
                        search._candidate_score(board, r, c, BLACK, mode),
                        search._candidate_score(board, rr, cc, WHITE, mode))
                self.assertEqual(
                    _potential_strength(search, board, r, c, BLACK,
                                        GEAR_THREAT_POINTS),
                    _potential_strength(search, board, rr, cc, WHITE,
                                        GEAR_THREAT_POINTS))
        black = search._ordinary_candidates(board, BLACK, with_score=True)
        white = search._ordinary_candidates(board, WHITE, with_score=True)
        self.assertLessEqual(len(black), 4)
        self.assertLessEqual(len(white), 4)
        self.assertEqual(sorted(weight for _r, _c, weight in black),
                         sorted(weight for _r, _c, weight in white))
        self.assertTrue(legal_points(board, black))
        self.assertTrue(legal_points(board, white))

    def test_tactical_candidates_not_limited_to_four(self):
        deep = DeepSearch(make_search(empty_board()))
        tactical = ('my-vcf', {(0, index): 100 for index in range(8)})
        with patch.object(deep, '_tactical_result', return_value=tactical):
            internal = deep._candidate_points(empty_board(), BLACK, 5, plies=1)
            root = deep._candidate_points(empty_board(), BLACK, 5, plies=0)
        self.assertEqual(len(internal), 8)
        self.assertEqual(len(root), 8)

    def test_simulation_rollback(self):
        engine, board = replay(M10)
        search = engine.search

        def snapshot():
            return copy.deepcopy((board, engine.moves, search.sb, search.sw,
                                  search._direction_levels, search.zero_history,
                                  search.zero_active, search.gear,
                                  search.gear_progress, search._gear_active))

        before = snapshot()
        points = search._ordinary_candidates(board, WHITE, with_score=True)
        rows = engine.deep_search.rank_candidates(
            board, {(r, c): weight for (r, c, weight) in points}, WHITE)
        self.assertTrue(rows)
        self.assertEqual(snapshot(), before)

    def test_native_first_stage_dedups_and_skips_occupied(self):
        """首阶段只按配额取原生去重代表；不再用旧召回层补满 4 点。"""
        board = sym_board()
        search = make_search(board)
        occupied = {(r, c) for r, c, _p in SYM_STONES}
        # 攻防两侧各给一个高分合法点；两侧都只保留一个签名。
        black_hot = {(3, 7)}
        white_hot = {(8, 8)}
        hot = black_hot | white_hot

        def fake_score(b, r, c, player, mode):
            side_hot = black_hot if player == BLACK else white_hot
            return 100.0 if (r, c) in side_hot else 0.0

        def fake_sig(b, syms, r, c, player):
            return (player, 'same')

        with patch.object(Search, '_candidate_score', side_effect=fake_score), \
                patch.object(Search, '_dual_score', return_value=0.0), \
                patch.object(Search, '_dedup_sig', side_effect=fake_sig):
            points = search._fallback_by_count(board, 2, 2, BLACK, True)
            # 两侧各只有一个唯一签名：各保留 1 个代表，共 2 点；
            # 旧召回层会继续补到 4 点，原生初选不会。
            self.assertEqual(points, [(3, 7, 100.0), (8, 8, 100.0)])

            no_dup = search._fallback_by_count(board, 2, 2, BLACK, False)
            self.assertEqual(no_dup, [(3, 7), (8, 8)])
            self.assertEqual(len(no_dup), 2)

        for r, c in no_dup:
            self.assertNotIn((r, c), occupied)
            self.assertEqual(board[r][c], EMPTY)
            self.assertIn((r, c), hot)


if __name__ == '__main__':
    unittest.main(verbosity=2)
