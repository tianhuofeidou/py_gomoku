# -*- coding: utf-8 -*-
"""固定普通候选 2攻2防/最多4点的定向检查。"""
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


def replay(moves):
    e = Engine()
    b = empty_board()
    for r, c, p in moves:
        place(b, r, c, p)
        e.on_move(b, r, c, p)
    return e, b


def rot180(r, c):
    return SIZE - 1 - r, SIZE - 1 - c


class OrdinaryFourTest(unittest.TestCase):

    def test_ordinary_calls_two_attack_two_defense(self):
        board = sym_board()
        s = make_search(board)
        seen = []
        sentinel = [(1, 1, 50.0)]

        def fake(board, attack_n, defend_n, player=WHITE, with_score=False):
            seen.append((attack_n, defend_n, player, with_score))
            return sentinel

        s._fallback_by_count = fake
        self.assertEqual(s._ordinary_candidates(board, BLACK, True), sentinel)
        self.assertEqual(seen, [(2, 2, BLACK, True)])

    def test_root_and_internal_use_four_and_keep_defense_slots(self):
        board = sym_board()
        s = make_search(board)
        d = DeepSearch(s)
        self.assertEqual(d.CAND_N_ORDINARY, 4)
        expected = [(r, c, clamp(w)) for (r, c, w)
                    in s._fallback_by_count(board, 2, 2, player=BLACK, with_score=True)]
        root = d._candidate_points(board, BLACK, 5, plies=0)
        internal = d._candidate_points(board, BLACK, 5, plies=1)
        self.assertEqual(root, expected)
        self.assertEqual(internal, expected)
        self.assertEqual(len(root), 4)
        self.assertEqual(len(internal), 4)
        attack = {p[:2] for p in expected[:2]}
        self.assertNotIn(expected[2][:2], attack)
        self.assertNotIn(expected[3][:2], attack)

    def test_black_white_symmetry(self):
        board = sym_board()
        s = make_search(board)
        d = DeepSearch(s)
        black = d._candidate_points(board, BLACK, 5, plies=1)
        white = d._candidate_points(board, WHITE, 5, plies=1)
        self.assertEqual(len(black), 4)
        self.assertEqual(len(white), 4)
        self.assertEqual([(rot180(r, c)[0], rot180(r, c)[1], w) for (r, c, w) in black],
                         white)

    def test_tactical_candidates_not_limited_to_four(self):
        d = DeepSearch(make_search(empty_board()))
        tactical = ('my-vcf', {(0, i): 100 for i in range(8)})
        with patch.object(d, '_tactical_result', return_value=tactical):
            internal = d._candidate_points(empty_board(), BLACK, 5, plies=1)
            root = d._candidate_points(empty_board(), BLACK, 5, plies=0)
        self.assertEqual(len(internal), 8)
        self.assertEqual(len(root), 8)

    def test_simulation_rollback(self):
        e, board = replay(M10)
        search = e.search

        def snapshot():
            return copy.deepcopy((board, e.moves, search.sb, search.sw,
                                  search._direction_levels, search.zero_history,
                                  search.zero_active, search.gear))

        before = snapshot()
        points = search._fallback_by_count(board, 2, 2, player=WHITE, with_score=True)
        rows = e.deep_search.rank_candidates(
            board, {(r, c): w for (r, c, w) in points}, WHITE)
        self.assertTrue(rows)
        self.assertEqual(snapshot(), before)


if __name__ == '__main__':
    unittest.main()
