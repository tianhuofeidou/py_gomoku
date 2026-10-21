# -*- coding: utf-8 -*-
"""连续成五不能被外围跨缝棋子降级；同时覆盖真实搜索中的胜败回传。"""
import copy
import unittest

from gomoku.core.deep_search import DeepSearch
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import Search
from gomoku.core.utils import BLACK, WHITE, DIRECTIONS, empty_board
from gomoku.game import check_win


def point(label):
    return int(label[1:]) - 1, ord(label[0]) - ord('A')


def screenshot_board(player):
    """截图中轮到白方的局面；player 可交换双方颜色。"""
    board = empty_board()
    for color, labels in (
        (3 - player, 'E4 G4 J4 E5 F5 G5 H5 E6 I6 H7 G8 H8 I8 E9 G9 F10 G11 H11'),
        (player, 'H3 D5 I5 F6 G6 H6 F7 G7 I7 J7 F8 F9 H9 G10 H10 E11 H12'),
    ):
        for label in labels.split():
            r, c = point(label)
            board[r][c] = color
    return board


class FiveWithOuterGapTest(unittest.TestCase):
    def test_contiguous_five_survives_outer_gap(self):
        """五/长连各落点、四轴及镜像、黑白两方：外围跳子不改变成五。"""
        search = Search(PatternAnalyzer())
        for player in (BLACK, WHITE):
            for dr, dc in DIRECTIONS:
                for sign in (-1, 1):
                    for length in (5, 6):
                        for pivot in range(length):
                            for sides in ((-1,), (1,), (-1, 1)):
                                with self.subTest(player=player, direction=(dr, dc),
                                                  sign=sign, length=length,
                                                  pivot=pivot, sides=sides):
                                    board = empty_board()
                                    start, end = -pivot, length - pivot - 1
                                    offsets = list(range(start, end + 1))
                                    for side in sides:
                                        offsets.append(start - 2 if side < 0 else end + 2)
                                    for offset in offsets:
                                        board[7 + sign * offset * dr][7 + sign * offset * dc] = player
                                    self.assertTrue(check_win(board, 7, 7, player))
                                    records = search.pattern.analyze_direction(
                                        board, 7, 7, player, dr, dc)
                                    self.assertEqual([x['parent'] for x in records], ['FIVE'])
                                    self.assertEqual(search._direction_level(
                                        board, 7, 7, player, dr, dc), 0)

    def test_split_stones_are_not_a_five(self):
        """总子数超过五，但连续段只有四时，不能因为忽略缝而误报成五。"""
        pattern = PatternAnalyzer()
        for player in (BLACK, WHITE):
            for dr, dc in DIRECTIONS:
                board = empty_board()
                for offset in (-3, -2, -1, 0, 2, 3):
                    board[7 + offset * dr][7 + offset * dc] = player
                self.assertFalse(check_win(board, 7, 7, player))
                records = pattern.analyze_direction(board, 7, 7, player, dr, dc)
                self.assertNotIn('FIVE', [x['parent'] for x in records])

    def test_simulated_win_point_and_journal_restore(self):
        """白 F3、黑 E3 后，白堵 E2 仍必须保留黑 E7 成五点。"""
        for player in (BLACK, WHITE):
            board = screenshot_board(player)
            search = Search(PatternAnalyzer())
            search.rebuild_all(board)
            before_board = copy.deepcopy(board)
            before = copy.deepcopy((search.sb, search.sw, search._direction_levels,
                                    search.gear, search.zero_history, search.zero_active))
            journals = []
            try:
                for label, color in (('F3', player), ('E3', 3 - player), ('E2', player)):
                    r, c = point(label)
                    journal = {}
                    board[r][c] = color
                    journals.append((r, c, journal))
                    search.on_move(board, r, c, color, journal=journal, record_history=False)
                state = search.sb if player == WHITE else search.sw
                self.assertEqual(state[6][4], 0)
                reason, candidates = search.tactical_candidates(board, 3 - player)
                self.assertEqual(reason, 'my-five')
                self.assertIn(point('E7'), candidates)
            finally:
                for r, c, journal in reversed(journals):
                    search.restore(journal)
                    board[r][c] = 0
            self.assertEqual(board, before_board)
            self.assertEqual((search.sb, search.sw, search._direction_levels,
                              search.gear, search.zero_history, search.zero_active), before)

    def test_f3_is_hard_loss_in_actual_search(self):
        """真实候选与递归回传必须把 F3 标成必败，并完整恢复搜索状态。"""
        for player in (BLACK, WHITE):
            with self.subTest(player=player):
                board = screenshot_board(player)
                search = Search(PatternAnalyzer())
                search.rebuild_all(board)
                deep = DeepSearch(search)
                deep.T0 = 160.0
                before = copy.deepcopy((board, search.sb, search.sw, search._direction_levels,
                                        search.gear, search.zero_history, search.zero_active))
                reason, points = search.tactical_candidates(board, player)
                self.assertEqual(reason, 'opp-vcf')
                ranked = deep.rank_candidates(board, points, player)
                f3 = next(x for x in ranked if (x['r'], x['c']) == point('F3'))
                self.assertEqual(f3['state'], DeepSearch.STATE_LOSE)
                self.assertEqual(f3['state_name'], '必败')
                self.assertEqual(f3['forced'], -1)
                self.assertEqual((board, search.sb, search.sw, search._direction_levels,
                                  search.gear, search.zero_history, search.zero_active), before)
                self.assertIsNone(deep._root_pass)


if __name__ == '__main__':
    unittest.main()
