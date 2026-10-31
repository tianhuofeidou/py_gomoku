# -*- coding: utf-8 -*-
"""连续四的真实两端不能被外围跳子覆盖；含 TPTC 白48漏防的完整前缀。"""
import copy
import unittest

from gomoku.core.engine import Engine
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import Search
from gomoku.core.utils import BLACK, WHITE, DIRECTIONS, empty_board
from gomoku.game import check_win


def point(label):
    return int(label[1:]) - 1, ord(label[0]) - ord('A')


TPTC_PREFIX = (
    'H8 I8 H11 H9 J7 J9 K10 G9 I9 G7 G8 H7 G6 E7 F7 E8 F8 F6 '
    'G5 F9 E9 C6 D7 E5 E6 F5 D5 C4 D6 D8 D4 D3 E2 C7 C5 C8 '
    'C9 B8 A8 F2 B6 E3 F3 E4 G4 C2 B1'
).split()


class FourWithOuterGapTest(unittest.TestCase):
    def test_four_two_empty_ends_survives_outer_stones_at_every_pivot(self):
        """已连续四且紧邻两端空，四轴、反向、双方与各基子均须识别活四。"""
        pattern = PatternAnalyzer()
        search = Search(pattern)
        for player in (BLACK, WHITE):
            for dr, dc in DIRECTIONS:
                for sign in (-1, 1):
                    for pivot in range(4):
                        for outer in ((-1,), (1,), (-1, 1)):
                            with self.subTest(player=player, direction=(dr, dc), sign=sign,
                                              pivot=pivot, outer=outer):
                                board = empty_board()
                                start, end = -pivot, 3 - pivot
                                offsets = list(range(start, end + 1))
                                offsets += [start - 2 if s < 0 else end + 2 for s in outer]
                                for k in offsets:
                                    board[7 + sign * k * dr][7 + sign * k * dc] = player
                                self.assertFalse(check_win(board, 7, 7, player))
                                # 独立规则事实：两端落下分别成五，不能用待测扫描器生成期望。
                                for k in (start - 1, end + 1):
                                    r, c = 7 + sign * k * dr, 7 + sign * k * dc
                                    self.assertEqual(board[r][c], 0)
                                    board[r][c] = player
                                    self.assertTrue(check_win(board, r, c, player))
                                    board[r][c] = 0
                                records = pattern.analyze_direction(board, 7, 7, player, dr, dc)
                                self.assertEqual([r['parent'] for r in records], ['LIVE4'])
                                self.assertEqual(search._direction_level(board, 7, 7, player, dr, dc), 1)

    def test_blocked_end_and_real_jump_four_are_not_promoted(self):
        pattern = PatternAnalyzer()
        for player in (BLACK, WHITE):
            for dr, dc in DIRECTIONS:
                for sign in (-1, 1):
                    for offsets, blocked in (((-3, -2, -1, 0, 2), -4),
                                              ((0, 1, 2, 4), None),
                                              ((0, 1, 3, 4), None)):
                        with self.subTest(player=player, direction=(dr, dc), sign=sign,
                                          offsets=offsets):
                            board = empty_board()
                            for k in offsets:
                                board[7 + sign * k * dr][7 + sign * k * dc] = player
                            if blocked is not None:
                                board[7 + sign * blocked * dr][7 + sign * blocked * dc] = 3 - player
                            records = pattern.analyze_direction(board, 7, 7, player, dr, dc)
                            self.assertEqual([r['parent'] for r in records], ['SLEEP4'])

    def test_single_empty_cell_at_board_edge_is_enough(self):
        pattern = PatternAnalyzer()
        for player in (BLACK, WHITE):
            for columns, pivot, expected in (
                    ((1, 2, 3, 4, 6), 4, 'LIVE4'),
                    ((8, 10, 11, 12, 13), 10, 'LIVE4'),
                    ((0, 1, 2, 3, 5), 3, 'SLEEP4'),
                    ((9, 11, 12, 13, 14), 11, 'SLEEP4')):
                with self.subTest(player=player, columns=columns):
                    board = empty_board()
                    for c in columns:
                        board[7][c] = player
                    self.assertEqual([r['parent'] for r in pattern.analyze_direction(
                        board, 7, pivot, player, 0, 1)], [expected])

    def test_tptc_move48_defends_after_own_forcing_four_and_restores_search(self):
        """白46冲四、黑47堵四后仍须识别黑H5活四；交换颜色也覆盖。"""
        for defender in (WHITE, BLACK):
            with self.subTest(defender=defender):
                engine = Engine()
                engine._memory_bonus = lambda: {}
                engine.deep_search.T0 = 160
                engine.deep_search.use_nn = False
                board = empty_board()
                for i, label in enumerate(TPTC_PREFIX):
                    r, c = point(label)
                    color = 3 - defender if i % 2 == 0 else defender
                    board[r][c] = color
                    engine.on_move(board, r, c, color)
                    if i + 1 in (45, 46, 47):
                        state = engine.search.sb if defender == WHITE else engine.search.sw
                        h5 = point('H5')
                        self.assertEqual(state[h5[0]][h5[1]], 1)
                        self.assertEqual(engine.search._eval(board, *h5, 3 - defender), 1)
                    if i + 1 == 46:
                        reason, points = engine.search.tactical_candidates(board, defender)
                        self.assertEqual(reason, 'my-five')
                        self.assertIn(point('B1'), points)
                attack_state = engine.search.sb if defender == WHITE else engine.search.sw
                self.assertEqual(attack_state[point('H5')[0]][point('H5')[1]], 1)
                reason, points = engine.search.tactical_candidates(board, defender)
                self.assertEqual(reason, 'opp-vcf')
                self.assertIn(point('H5'), points)
                before = copy.deepcopy((board, engine.moves, engine.search.sb, engine.search.sw,
                    engine.search._direction_levels, engine.search.gear,
                    engine.search.zero_history, engine.search.zero_active))
                result = engine.analyze_turn(board, defender)
                self.assertEqual(result['part3_candidates']['reason'], 'opp-vcf')
                # 独立规则核验：假设对手落 H5，两端下一手都实际成五。
                # 占中心或任意一端都可破坏活四；I6 是 d=1 近端，不能漏掉。
                h5 = point('H5')
                board[h5[0]][h5[1]] = 3 - defender
                winning_ends = set()
                try:
                    for rr in range(15):
                        for cc in range(15):
                            if board[rr][cc] != 0:
                                continue
                            board[rr][cc] = 3 - defender
                            try:
                                if check_win(board, rr, cc, 3 - defender):
                                    winning_ends.add((rr, cc))
                            finally:
                                board[rr][cc] = 0
                finally:
                    board[h5[0]][h5[1]] = 0
                self.assertEqual(winning_ends, {point('D1'), point('I6')})
                self.assertEqual(points, {h5: 100.0, **dict.fromkeys(winning_ends, 80.0)})
                selected = tuple(result['part4_result']['move'])
                self.assertIn(selected, winning_ends | {h5})
                self.assertEqual(selected, (result['part4_result']['ranked'][0]['r'],
                                            result['part4_result']['ranked'][0]['c']))
                self.assertNotEqual(tuple(result['part4_result']['move']), point('K7'))
                self.assertEqual((board, engine.moves, engine.search.sb, engine.search.sw,
                    engine.search._direction_levels, engine.search.gear,
                    engine.search.zero_history, engine.search.zero_active), before)
                self.assertIsNone(engine.deep_search._root_pass)


if __name__ == '__main__':
    unittest.main()
