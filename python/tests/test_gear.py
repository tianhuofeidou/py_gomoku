# -*- coding: utf-8 -*-
"""动态攻防配比单元测试。

运行: python test_gear.py
覆盖:
  1. 压力统计
  2. 四档映射与硬约束
  3. 滞回（最多一次调一档）
  4. 历史 0 级事件去重
  5. _fallback 候选数量与攻防配比
"""
import unittest

from python.core.utils import empty_board, SIZE, EMPTY, BLACK, WHITE
from python.core.pattern import PatternAnalyzer
from python.core.search import (
    Search, A1_SUBCLASSES, A2_SUBCLASSES, B_SUBCLASSES,
    KILL, GEAR_PROFILE, DEFAULT_GEAR,
)


def make_search():
    return Search(PatternAnalyzer())


def set_cell_state(search, state, r, c, subclass):
    """直接写状态表（仅用于单元测试，不保证与棋盘一致）。"""
    state[r][c] = subclass


class GearUnitTest(unittest.TestCase):

    def setUp(self):
        self.s = make_search()
        self.board = empty_board()

    def test_empty_pressure_is_zero(self):
        stats = self.s._pressure_stats(self.board)
        self.assertEqual(stats, (0, 0, 0, 0, 0, 0))

    def test_pressure_stats_counts_subclasses(self):
        set_cell_state(self.s, self.s.sw, 0, 0, A1_SUBCLASSES[0])
        set_cell_state(self.s, self.s.sw, 0, 1, A2_SUBCLASSES[0])
        set_cell_state(self.s, self.s.sw, 0, 2, B_SUBCLASSES[0])
        set_cell_state(self.s, self.s.sb, 1, 0, A1_SUBCLASSES[1])
        set_cell_state(self.s, self.s.sb, 1, 1, A2_SUBCLASSES[1])
        set_cell_state(self.s, self.s.sb, 1, 2, B_SUBCLASSES[1])
        stats = self.s._pressure_stats(self.board)
        self.assertEqual(stats, (1, 1, 1, 1, 1, 1))

    def test_opp_double_a1_hard_constraint_to_gear1(self):
        # 对方双冲四、我方无冲四 → 强制 1攻4防
        self.s.sb[3][0] = A1_SUBCLASSES[0]
        self.s.sb[3][1] = A1_SUBCLASSES[1]
        self.s.gear[WHITE] = DEFAULT_GEAR
        self.assertEqual(self.s._choose_gear(self.board), 1)

    def test_my_double_a1_hard_constraint_to_gear4(self):
        # 我方双冲四、对方无冲四 → 强制 4攻1防
        self.s.sw[3][0] = A1_SUBCLASSES[0]
        self.s.sw[3][1] = A1_SUBCLASSES[1]
        self.s.gear[WHITE] = DEFAULT_GEAR
        self.assertEqual(self.s._choose_gear(self.board), 4)

    def test_opp_overwhelm_soft_constraint_to_gear1(self):
        # 对方 3 个 1 级威胁、我方 <=1 → 强制 1攻4防
        self.s.sb[3][0] = A1_SUBCLASSES[0]
        self.s.sb[3][1] = A2_SUBCLASSES[0]
        self.s.sb[3][2] = A2_SUBCLASSES[1]
        self.s.sw[3][3] = A2_SUBCLASSES[2]
        self.s.gear[WHITE] = DEFAULT_GEAR
        self.assertEqual(self.s._choose_gear(self.board), 1)

    def test_my_overwhelm_soft_constraint_to_gear4(self):
        # 我方 3 个 1 级威胁、对方 <=1 → 强制 4攻1防
        self.s.sw[3][0] = A1_SUBCLASSES[0]
        self.s.sw[3][1] = A2_SUBCLASSES[0]
        self.s.sw[3][2] = A2_SUBCLASSES[1]
        self.s.sb[3][3] = A2_SUBCLASSES[2]
        self.s.gear[WHITE] = DEFAULT_GEAR
        self.assertEqual(self.s._choose_gear(self.board), 4)

    def test_normal_mapping_to_gear4(self):
        # 我方明显压制：多个 a1，对方几乎无威胁 → 目标 4 档
        self.s.sw[2][0] = A1_SUBCLASSES[0]
        self.s.sw[2][1] = A1_SUBCLASSES[1]
        self.s.sw[2][2] = A2_SUBCLASSES[0]
        self.s.sw[2][3] = A2_SUBCLASSES[1]
        self.s.sw[2][4] = B_SUBCLASSES[0]
        self.s.gear[WHITE] = DEFAULT_GEAR
        self.assertEqual(self.s._choose_gear(self.board), 4)

    def test_normal_mapping_to_gear1(self):
        # 对方明显压制，我方几乎没有威胁 → 目标 1 档
        self.s.sb[2][0] = A1_SUBCLASSES[0]
        self.s.sb[2][1] = A1_SUBCLASSES[1]
        self.s.sb[2][2] = A2_SUBCLASSES[0]
        self.s.sb[2][3] = A2_SUBCLASSES[1]
        self.s.sb[2][4] = B_SUBCLASSES[0]
        self.s.gear[WHITE] = DEFAULT_GEAR
        self.assertEqual(self.s._choose_gear(self.board), 1)

    def test_hysteresis_moves_one_gear_at_a_time(self):
        # 常规劣势（未触发硬约束）：从 3 档开始只能先到 2 档，下次再到 1 档
        self.s.sb[2][0] = A1_SUBCLASSES[0]
        self.s.sb[2][1] = A2_SUBCLASSES[0]
        self.s.gear[WHITE] = DEFAULT_GEAR
        self.assertEqual(self.s._choose_gear(self.board), 2)
        self.assertEqual(self.s._choose_gear(self.board), 1)

    def test_zero_event_deduplicates_while_active(self):
        # 出现 0 级杀势记 1 次；持续存在不重复计数；解除后再现再记
        self.s.sw[5][5] = 0   # 白表某个空位为 five（KILL）
        self.s._record_zero_event(self.board, WHITE)
        self.assertEqual(sum(self.s.zero_history[WHITE]), 1)
        # 同一杀势仍在 → 不重复记
        self.s._record_zero_event(self.board, WHITE)
        self.assertEqual(sum(self.s.zero_history[WHITE]), 1)
        # 杀势消失
        self.s.sw[5][5] = 17
        self.s._record_zero_event(self.board, WHITE)
        self.assertEqual(sum(self.s.zero_history[WHITE]), 1)
        # 再次出现 → 再记一次
        self.s.sw[5][5] = 0
        self.s._record_zero_event(self.board, WHITE)
        self.assertEqual(sum(self.s.zero_history[WHITE]), 2)

    def test_fallback_gear_profiles(self):
        # 给黑白表都塞满候选点，验证各档位返回的攻防数量
        for i in range(10):
            self.s.sw[0][i] = B_SUBCLASSES[0]
            self.s.sb[1][i] = B_SUBCLASSES[0]
        for gear, (a, d) in GEAR_PROFILE.items():
            pts = self.s._fallback(self.board, gear=gear)
            self.assertEqual(len(pts), a + d)
            # 前 a 个是白表候选，后 d 个是黑表候选（不重复）
            self.assertEqual(len(set(pts)), a + d)
            self.assertTrue(all(p[0] == 0 for p in pts[:a]))
            self.assertTrue(all(p[0] == 1 for p in pts[a:]))

    def test_fallback_default_is_3_2(self):
        for i in range(10):
            self.s.sw[0][i] = B_SUBCLASSES[0]
            self.s.sb[1][i] = B_SUBCLASSES[0]
        pts = self.s._fallback(self.board)
        self.assertEqual(len(pts), 5)
        self.assertEqual(len([p for p in pts if p[0] == 0]), 3)
        self.assertEqual(len([p for p in pts if p[0] == 1]), 2)

    def test_fallback_black_symmetric(self):
        # 黑方视角：前攻来自黑表(sb)，后防来自白表(sw)，与白方视角对称
        for i in range(10):
            self.s.sw[0][i] = B_SUBCLASSES[0]
            self.s.sb[1][i] = B_SUBCLASSES[0]
        pts = self.s._fallback(self.board, player=BLACK)
        self.assertEqual(len(pts), 5)
        self.assertTrue(all(p[0] == 1 for p in pts[:3]))   # 黑攻
        self.assertTrue(all(p[0] == 0 for p in pts[3:]))   # 白防

    def test_gear_disabled_returns_default(self):
        from python.core import search as S
        old = S.GEAR_ENABLED
        S.GEAR_ENABLED = False
        try:
            self.s.sw[3][0] = A1_SUBCLASSES[0]
            self.s.sw[3][1] = A1_SUBCLASSES[1]
            self.s.gear[WHITE] = DEFAULT_GEAR
            self.assertEqual(self.s._choose_gear(self.board), DEFAULT_GEAR)
        finally:
            S.GEAR_ENABLED = old


if __name__ == '__main__':
    unittest.main(verbosity=2)
