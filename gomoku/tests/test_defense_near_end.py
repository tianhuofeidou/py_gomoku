# -*- coding: utf-8 -*-
"""防守近端（d=1）回归：端点不能再因“紧贴成形点”被跳过。

背景：`Search._defense_candidates` 旧实现在端点提取时一律跳过 `d <= 1`，
导致紧贴威胁点的近端堵点漏防。固定 16 手局面（黑先）：

    H8 G8 G7 H7 I6 F9 F6 I9 E5 D4 H6 G6 G9 J9 F5 I10

此时黑方防守白 I8 的四三，旧实现返回 {I8:100, K10:80, I11:80}，漏掉 I7:80；
I7 能破坏白 I8 的四三形状。新实现按端类型、边界与空位正常筛选近端。

本文件在旧实现失败（缺 I7）、在新实现通过；不依赖本机 games.json。
"""
import copy
import os
import tempfile
import unittest
from unittest.mock import patch

from gomoku import memory
from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK, EMPTY, SIZE, WHITE, empty_board, place

GAME = 'H8 G8 G7 H7 I6 F9 F6 I9 E5 D4 H6 G6 G9 J9 F5 I10'.split()

# 黑方防守白 I8 四三的期望候选（中心 100 + 三个端点 80）
TARGET = {
    (7, 8): 100.0,    # I8：威胁中心，占掉落点即破坏杀型
    (6, 8): 80.0,     # I7：紧贴威胁点的近端（旧实现漏点）
    (10, 8): 80.0,    # I11：远端
    (9, 10): 80.0,    # K10：另一条线的端点
}

TRANSFORMS = {
    'identity': lambda r, c, p: (r, c, p),
    'rot90': lambda r, c, p: (c, 14 - r, p),
    'rot180': lambda r, c, p: (14 - r, 14 - c, p),
    'rot270': lambda r, c, p: (14 - c, r, p),
    'flip_h': lambda r, c, p: (r, 14 - c, p),
    'flip_v': lambda r, c, p: (14 - r, c, p),
    'diag': lambda r, c, p: (c, r, p),
    'anti_diag': lambda r, c, p: (14 - c, 14 - r, p),
}

GAP_MOVES = [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (5, 7, WHITE),
             (7, 5, BLACK), (4, 5, WHITE), (7, 6, BLACK), (7, 4, WHITE),
             (6, 5, BLACK), (5, 4, WHITE), (6, 3, BLACK), (4, 4, WHITE),
             (6, 4, BLACK), (4, 6, WHITE), (4, 7, BLACK), (3, 5, WHITE),
             (6, 8, BLACK)]


def rc(token):
    return (int(token[1:]) - 1, ord(token[0]) - 65)


def build(transform=None):
    """重放 16 手；transform 用于对称局。返回 (engine, board)。"""
    engine = Engine()
    board = empty_board()
    for index, token in enumerate(GAME):
        r, c = rc(token)
        player = BLACK if index % 2 == 0 else WHITE
        if transform is not None:
            r, c, player = transform(r, c, player)
        place(board, r, c, player)
        engine.on_move(board, r, c, player)
    return engine, board


def board_engine(stones):
    """按 (r,c,player) 摆子并同步引擎状态表。"""
    engine = Engine()
    board = empty_board()
    for r, c, player in stones:
        place(board, r, c, player)
        engine.on_move(board, r, c, player)
    return engine, board


class DefenseNearEndTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for context in (
                patch.dict(os.environ, {'GOMOKU_HOME': tmp.name,
                                        'DSH_GOMOKU_GA': '0'}),
                patch.object(memory, '_global_memory', None)):
            context.start()
            self.addCleanup(context.stop)

    def assert_points(self, points, expected):
        for point, weight in expected.items():
            self.assertIn(point, points, '缺少防守点 %s' % (point,))
            self.assertAlmostEqual(points[point], weight)

    def test_target_position_includes_near_end_i7(self):
        """固定局面：黑方防守白 I8 四三，必须包含近端 I7。"""
        engine, board = build()
        snapshot = (copy.deepcopy(board), copy.deepcopy(engine.search.sb),
                    copy.deepcopy(engine.search.sw), list(engine.moves))
        reason, points = engine.search.tactical_candidates(board, BLACK)
        self.assertEqual(reason, 'opp-vcf')
        self.assert_points(points, TARGET)
        # 旧实现恰好漏掉紧贴威胁点的近端，这里显式钉住。
        self.assertIn((6, 8), points, '近端 I7 被跳过（旧实现缺陷）')
        for (r, c) in points:
            self.assertTrue(0 <= r < SIZE and 0 <= c < SIZE)
            self.assertEqual(board[r][c], EMPTY, '候选点必须是空位')
        # 候选生成期间临时落子必须完全回滚。
        self.assertEqual(board, snapshot[0])
        self.assertEqual(engine.search.sb, snapshot[1])
        self.assertEqual(engine.search.sw, snapshot[2])
        self.assertEqual(engine.moves, snapshot[3])

    def test_color_swap_is_symmetric(self):
        """黑白互换 + 旋转后，白方防守黑方威胁应得到同样权重的镜像候选。"""
        transform = lambda r, c, p: (c, 14 - r, 3 - p)     # noqa: E731
        engine, board = build(transform)
        reason, points = engine.search.tactical_candidates(board, WHITE)
        self.assertEqual(reason, 'opp-vcf')
        expected = {}
        for (r, c), weight in TARGET.items():
            er, ec, _ = transform(r, c, BLACK)
            expected[(er, ec)] = weight
        self.assert_points(points, expected)

    def test_all_eight_symmetries_keep_near_end(self):
        """8 种旋转/镜像下，近端防守点与权重都应保持一致。"""
        for name, transform in TRANSFORMS.items():
            with self.subTest(symmetry=name):
                engine, board = build(transform)
                reason, points = engine.search.tactical_candidates(board, BLACK)
                self.assertEqual(reason, 'opp-vcf')
                expected = {}
                for (r, c), weight in TARGET.items():
                    er, ec, _ = transform(r, c, BLACK)
                    expected[(er, ec)] = weight
                self.assert_points(points, expected)

    def test_live4_near_end_included(self):
        """活四：紧贴威胁点的近端也必须进入防守候选。"""
        engine, board = board_engine([(7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK)])
        points = engine.search._defense_candidates(board, 7, 7, player=BLACK)
        self.assertEqual(points.get((7, 7)), 100.0)    # 中心
        self.assertEqual(points.get((7, 8)), 80.0)     # d=1 近端
        self.assertEqual(points.get((7, 3)), 80.0)     # 远端
        self.assertEqual(board[7][7], EMPTY)           # 临时子已清除

    def test_sleep4_near_end_included(self):
        """冲四（单端被堵）：唯一可用的近端不能被跳过。"""
        engine, board = board_engine([(7, 3, WHITE), (7, 4, BLACK),
                                      (7, 5, BLACK), (7, 6, BLACK)])
        points = engine.search._defense_candidates(board, 7, 7, player=BLACK)
        self.assertEqual(points.get((7, 7)), 100.0)
        self.assertEqual(points.get((7, 8)), 80.0)     # d=1 近端
        self.assertNotIn((7, 3), points)               # 已被白子占住
        self.assertEqual(board[7][3], WHITE)
        self.assertEqual(board[7][7], EMPTY)

    def test_live3_near_end_included(self):
        """活三：两端都应是防守候选，近端（d=1）不得缺失。"""
        engine, board = board_engine([(7, 5, BLACK), (7, 6, BLACK)])
        points = engine.search._defense_candidates(board, 7, 7, player=BLACK)
        self.assertEqual(points.get((7, 7)), 100.0)
        self.assertEqual(points.get((7, 4)), 80.0)     # 远端
        self.assertEqual(points.get((7, 8)), 80.0)     # d=1 近端
        self.assertEqual(board[7][7], EMPTY)

    def test_occupied_end_is_not_a_candidate(self):
        """已占端点不能作为防守候选。"""
        engine, board = board_engine([(7, 5, BLACK), (7, 6, BLACK), (7, 8, WHITE)])
        points = engine.search._defense_candidates(board, 7, 7, player=BLACK)
        self.assertIn((7, 7), points)
        self.assertNotIn((7, 8), points)
        self.assertEqual(board[7][8], WHITE)

    def test_board_edge_points_stay_inside(self):
        """边界：只返回盘内空点，越界端点被过滤。"""
        engine, board = board_engine([(14, 10, BLACK), (14, 11, BLACK)])
        points = engine.search._defense_candidates(board, 14, 12, player=BLACK)
        self.assertEqual(points.get((14, 12)), 100.0)
        self.assertEqual(points.get((14, 13)), 80.0)   # 近端
        for (r, c) in points:
            self.assertTrue(0 <= r < SIZE and 0 <= c < SIZE)
            self.assertEqual(board[r][c], EMPTY)
        # 右端越界：黑 (14,13),(14,14) 落 (14,12) 时不得产生 c=15 的候选。
        engine2, board2 = board_engine([(14, 13, BLACK), (14, 14, BLACK)])
        points2 = engine2.search._defense_candidates(board2, 14, 12, player=BLACK)
        self.assertTrue(all(0 <= r < SIZE and 0 <= c < SIZE for r, c in points2))

    def test_gap_candidates_still_present(self):
        """缝位（xx_xx 中间格）路径保持原行为：中心 100、缝 80。"""
        engine, board = board_engine(GAP_MOVES)
        reason, points = engine.search.tactical_candidates(board, WHITE)
        self.assertEqual(reason, 'opp-vcf')
        self.assertEqual(points.get((9, 5)), 100.0)    # F10 威胁中心
        self.assertEqual(points.get((8, 5)), 80.0)     # F9 缝位
        self.assertEqual(points.get((8, 6)), 80.0)     # G9 缝位

    def test_rollback_for_both_colors(self):
        """黑白两侧生成候选后，棋盘/状态表/着法记录必须完整恢复。"""
        engine, board = build()
        snapshot = (copy.deepcopy(board), copy.deepcopy(engine.search.sb),
                    copy.deepcopy(engine.search.sw), list(engine.moves))
        for player in (BLACK, WHITE):
            engine.search.tactical_candidates(board, player)
        engine.search._defense_candidates(board, 7, 8, player=WHITE)
        self.assertEqual(board, snapshot[0])
        self.assertEqual(engine.search.sb, snapshot[1])
        self.assertEqual(engine.search.sw, snapshot[2])
        self.assertEqual(engine.moves, snapshot[3])
        first = engine.search.tactical_candidates(board, BLACK)[1]
        second = engine.search.tactical_candidates(board, BLACK)[1]
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main(verbosity=2)
