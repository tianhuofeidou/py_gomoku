"""离线研究统计的颜色、对称与共同坐标系回归。"""
import contextlib
import io
import unittest

from scripts import verify_opening_research as research


class OpeningResearchTest(unittest.TestCase):
    def test_symmetric_replies_share_one_orbit(self):
        black, white = [(7, 7), (9, 7)], [(8, 7)]
        left, _ = research.canon_move(black, white, (8, 6))
        right, _ = research.canon_move(black, white, (8, 8))
        self.assertEqual(left, right)

    def test_board_and_reply_are_invariant_under_all_symmetries(self):
        for black, white, move in (
            ([(7, 7), (9, 7)], [(8, 7)], (8, 6)),
            ([(7, 7), (5, 9)], [(6, 7)], (6, 6)),
        ):
            expected, _ = research.canon_move(black, white, move)
            for symmetry in range(8):
                actual, _ = research.canon_move(
                    [research.tf(*p, symmetry) for p in black],
                    [research.tf(*p, symmetry) for p in white],
                    research.tf(*move, symmetry),
                )
                self.assertEqual(actual, expected)

    def test_distinct_replies_on_asymmetric_board_stay_distinct(self):
        black, white = [(7, 7), (5, 9)], [(6, 7)]
        self.assertNotEqual(research.canon_move(black, white, (8, 6))[0],
                            research.canon_move(black, white, (8, 8))[0])

    def test_concentration_uses_same_symmetry_for_anchor_and_reply(self):
        # 偏离天元的对称棋形：候选决定残余对称时，黑1也必须跟着变换。
        games = [('opening="14"', 'h7 h8 h9 g9')] * 100
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            research.sec_313({'14': 'd1'}, games)
        self.assertIn('(-2, -1)', output.getvalue())

    def test_white_win_rate_counts_white_wins_and_keeps_draws_in_denominator(self):
        games = [(f'rule="7" bresult="{result}"', '')
                 for result in ('1', '1', '0', '0.5') for _ in range(100)]
        # 未知结果不得混进分母。
        games += [('rule="7" bresult="2"', '')] * 100
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            research.sec_317({}, {'7': ('Gomoku-Pro', 2)}, games)
        self.assertIn('25.0%', output.getvalue())
        self.assertRegex(output.getvalue(), r'\b400\b')


if __name__ == '__main__':
    unittest.main()
