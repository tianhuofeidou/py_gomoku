# -*- coding: utf-8 -*-
"""深推状态表同步测试：
1. on_move(journal) 能记录修改并在 restore 后完全恢复。
2. deep_search 模拟推演后，sb/sw 与调用前完全一致（回滚正确）。
3. 深推模拟时不会污染历史 0 级/档位状态。
"""
import unittest

from python.core.utils import empty_board, place, BLACK, WHITE
from python.core.pattern import PatternAnalyzer
from python.core.search import Search, DEFAULT_GEAR
from python.core.deep_search import DeepSearch
from python.core.engine import Engine


def snapshot(s):
    return [row[:] for row in s.sb], [row[:] for row in s.sw]


class DeepSyncTest(unittest.TestCase):

    def test_on_move_journal_restore(self):
        s = Search(PatternAnalyzer())
        b = empty_board()
        place(b, 7, 7, BLACK)
        s.on_move(b, 7, 7, BLACK)
        sb0, sw0 = snapshot(s)

        # 模拟白落子：journal 记录，之后 restore
        place(b, 6, 6, WHITE)
        j = {}
        s.on_move(b, 6, 6, WHITE, journal=j, record_history=False)
        self.assertTrue(len(j) > 0)
        s.restore(j)
        place(b, 6, 6, 0)   # 撤销棋盘（测试用）
        sb1, sw1 = snapshot(s)
        self.assertEqual(sb0, sb1)
        self.assertEqual(sw0, sw1)

    def test_deep_search_restores_state(self):
        e = Engine()
        b = empty_board()
        # 一个常见开局，让状态表有内容
        for r, c, p in [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (6, 8, WHITE)]:
            place(b, r, c, p)
            e.on_move(b, r, c, p)
        sb0, sw0 = snapshot(e.search)
        gear0 = e.search.gear
        hist0 = {k: list(v) for k, v in e.search.zero_history.items()}

        # 随便给几个候选点，跑一次深推
        cands = [(5, 6), (6, 5), (7, 6), (6, 7), (5, 7)]
        e.deep_search.deep_search(b, cands, WHITE)

        sb1, sw1 = snapshot(e.search)
        self.assertEqual(sb0, sb1)
        self.assertEqual(sw0, sw1)
        self.assertEqual(e.search.gear, gear0)
        self.assertEqual({k: list(v) for k, v in e.search.zero_history.items()}, hist0)

    def test_deep_search_sees_simulated_threat(self):
        """核心：深推模拟黑落子后，状态表应能看到新形成的 0 级杀棋。"""
        e = Engine()
        b = empty_board()
        # 黑已有 F6-F7-F8 竖三；模拟黑在 F9 落子会成四连（若 F5 空且非墙则为活四/冲四）
        for r, c, p in [(5, 5, BLACK), (6, 5, BLACK), (7, 5, BLACK),
                        (7, 7, WHITE), (6, 6, WHITE)]:
            place(b, r, c, p)
            e.on_move(b, r, c, p)
        # 模拟黑落 F9（8,5）
        b[8][5] = BLACK
        j = {}
        e.search.on_move(b, 8, 5, BLACK, journal=j, record_history=False)
        # 黑表 F5(4,5) 或 F10(9,5) 应出现 0 级（five/live4）
        kills = [p for p in [(4, 5), (9, 5)] if e.search.sb[p[0]][p[1]] in (0, 1, 2, 3, 4)]
        self.assertTrue(kills, '深推模拟后黑表应出现必杀点')
        # 恢复
        e.search.restore(j)
        b[8][5] = 0


if __name__ == '__main__':
    unittest.main(verbosity=2)
