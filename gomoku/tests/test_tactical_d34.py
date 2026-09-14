# -*- coding: utf-8 -*-
"""四三（d34）「反四」特例的回归。

背景（桌面版实战败局，第 35 手）：白棋状态表里 D11 是 d34（四三＝冲四逼应 +
活三转活四），引擎把它当必杀档独占直取，0.0s 就拍了；但黑棋在必应点 E10 堵四
的同时自己成了四（E7-E10 活四）——反四，先手当场易主，白棋的三步杀跑不过，
最后被 E11 成五。四三逼得动的只有一手，那一手若被对手"顺手用掉"，就不成立。

特殊处理（状态表不动，只降本次决策身份）：
`Search._d34_is_defensible` 模拟本手落子 → 求成五点（对手必应点）→ 对手在该点
能成四即反四 → 该点从必杀档降级，让位给下一档（对方 VCF 的反推堵点）。
活四/双眠四的成五点 ≥2，对手一手堵不完 → 不降级，仍按必杀直取。

本文件在旧实现上失败（旧实现落 D11）、在新实现上通过（落 E6）。
"""
import os
import tempfile
import unittest
from unittest.mock import patch

from gomoku import memory
from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK, EMPTY, WHITE, empty_board, place

# 桌面版那盘的完整棋谱（人类执黑先行）；前 35 手到白棋败着的决策点
GAME = ('H8 G7 F6 H6 F8 F7 E7 G6 G5 D8 H4 I3 J6 G9 I5 G3 K7 L8 K5 G8 '
        'G10 J5 I7 L4 J7 H7 I6 I8 J9 H5 I4 D9 E8 F9 E9 D11 E10 E6 E11').split()

D11 = (10, 3)          # 白棋那个可被反四防住的四三
E10 = (9, 4)           # D11 的成五点（黑棋必应点，同时也是黑棋自己的活四格）
E6 = (5, 4)            # 第四档反推出来的堵点，改后应落这里


def rc(token):
    return (int(token[1:]) - 1, ord(token[0]) - 65)


def cd(r, c):
    return chr(65 + c) + str(r + 1)


def rot90_swap(r, c, player):
    """90° 旋转 + 黑白互换：局面等价变换，用于黑白对称回归。"""
    return c, 14 - r, 3 - player


def build(upto=35, transform=None):
    """重放前 upto 手，返回 (engine, board)；transform 用于对称局。"""
    engine = Engine()
    engine.deep_search.use_nn = False
    board = empty_board()
    for index, token in enumerate(GAME[:upto]):
        r, c = rc(token)
        player = BLACK if index % 2 == 0 else WHITE
        if transform is not None:
            r, c, player = transform(r, c, player)
        place(board, r, c, player)
        engine.on_move(board, r, c, player)
    return engine, board


def put(engine, board, stones):
    """按 (r, c, player) 摆子并同步引擎状态表（等同实战连续落子）。"""
    for r, c, player in stones:
        place(board, r, c, player)
        engine.on_move(board, r, c, player)


class D34DefensibleTest(unittest.TestCase):
    """实战败局：可防的四三必须让位，不能独占直取。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for context in (
                patch.dict(os.environ, {'GOMOKU_HOME': tmp.name,
                                        'DSH_GOMOKU_GA': '0'}),
                patch.object(memory, '_global_memory', None)):
            context.start()
            self.addCleanup(context.stop)

    def test_game_35_abandons_defensible_d34(self):
        """旧实现落 D11（必败），新实现让位到第四档并落 E6。"""
        engine, board = build(35)
        reason, points = engine.search.tactical_candidates(board, WHITE)
        self.assertEqual(reason, 'opp-vcf',
                         '可防的四三让位后，本档应由对方 VCF 的反推堵点接手')
        self.assertNotIn(D11, points)
        move = engine.compute_move(board, WHITE)
        self.assertNotEqual(move, D11, '可被反四防住的四三不得直取')
        self.assertEqual(move, E6)

    def test_defensible_d34_is_detected(self):
        """D11 的成五点是 E10，而 E10 是黑棋的活四格（反四）→ 判可防。"""
        engine, board = build(35)
        self.assertTrue(engine.search._d34_is_defensible(board, D11[0], D11[1], WHITE))
        self.assertEqual(engine.search._five_points(
            self._board_with(board, D11, WHITE), D11[0], D11[1], WHITE), [E10])

    def test_live4_and_d44_are_not_downgraded(self):
        """成五点 ≥2（活四/双眠四）＝对手一手堵不完 → 不判可防。"""
        engine = Engine()
        engine.deep_search.use_nn = False
        board = empty_board()
        # 白 (7,4)(7,5)(7,6) + 黑 (7,7)：白落 (7,3) 成眠四，成五点 (7,2)
        # 白 (4,3)(5,3)(6,3) + 黑 (3,3)：同一手在 3 列再成眠四 → 双眠四 d44
        put(engine, board, [(7, 4, WHITE), (7, 5, WHITE), (7, 6, WHITE), (7, 7, BLACK),
                            (4, 3, WHITE), (5, 3, WHITE), (6, 3, WHITE), (3, 3, BLACK)])
        self.assertEqual(engine.search._eval(board, 7, 3, WHITE), 2, '应为双眠四 d44')
        pts = engine.search._five_points(
            self._board_with(board, (7, 3), WHITE), 7, 3, WHITE)
        self.assertEqual(sorted(pts), sorted([(7, 2), (8, 3)]))
        self.assertFalse(engine.search._d34_is_defensible(board, 7, 3, WHITE))

    def test_defensible_d34_keeps_kill_when_no_counter_four(self):
        """成五点唯一、但对手在该点成不了四 → 不降级，仍是必杀。"""
        engine = Engine()
        engine.deep_search.use_nn = False
        board = empty_board()
        # 白 (7,4)(7,5)(7,6) + 黑 (7,7)：落 (7,3) 成眠四（成五点 (7,2)）
        # 白 (5,3)(6,3)：同一手在 3 列成活三 → 合起来是四三 d34
        put(engine, board, [(7, 4, WHITE), (7, 5, WHITE), (7, 6, WHITE), (7, 7, BLACK),
                            (5, 3, WHITE), (6, 3, WHITE)])
        self.assertEqual(engine.search._eval(board, 7, 3, WHITE), 3, '应为四三 d34')
        self.assertFalse(engine.search._d34_is_defensible(board, 7, 3, WHITE))

    def test_black_white_symmetry(self):
        """90° 旋转 + 黑白互换后，执黑一方同样放弃对应的四三。"""
        engine, board = build(35, transform=rot90_swap)
        move = engine.compute_move(board, BLACK)
        self.assertNotEqual(move, rot90_swap(*D11, WHITE)[:2])
        self.assertEqual(move, rot90_swap(*E6, WHITE)[:2])

    def test_probe_restores_board_and_tables(self):
        """特例检测只临时落子：返回前棋盘与黑白状态表必须逐格复原。"""
        engine, board = build(35)
        before_board = [row[:] for row in board]
        before_sb = [row[:] for row in engine.search.sb]
        before_sw = [row[:] for row in engine.search.sw]
        engine.search.tactical_candidates(board, WHITE)
        self.assertEqual(board, before_board)
        self.assertEqual(engine.search.sb, before_sb)
        self.assertEqual(engine.search.sw, before_sw)

    @staticmethod
    def _board_with(board, point, player):
        """复制棋盘并补一子（供 _five_points 直接测试）。"""
        copy = [row[:] for row in board]
        copy[point[0]][point[1]] = player
        return copy


if __name__ == '__main__':
    unittest.main()
