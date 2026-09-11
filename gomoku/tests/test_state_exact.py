# -*- coding: utf-8 -*-
"""状态表增量更新的精确性回归。

`Search.on_move` 的增量刷新必须与全盘重算 `Search.rebuild_all` 逐格一致：
刷新范围是落子所在棋段的段内空缝 + 该棋段两端外侧的尾部空位（全取），
取满尾部后即为完备范围，不需要额外的兜底区域。

这两个性质在旧实现上都不成立，因此本文件是"旧实现失败、新实现通过"的回归：

1. 旧范围只取尾部前 2 格，会漏掉"距离 3、紧邻两空"的两类格子——
   `● _ _ □`（□ 的双空跳搭档就是本手）与 `□ _ _ x ●`（□ 的搭档是 x，
   本手贴着搭档外侧，改的是搭档的外侧端口）。实测 24 局累计漂移 524 格。
2. 旧实现按落子方颜色分叉（黑落子刷段1+段2、白落子刷段3），叠加漂移后
   颜色互换的两张表不一致（实测 40 格 / 15 个局面）。
"""
import random
import unittest

from gomoku.core.engine import Engine
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import Search
from gomoku.core.utils import BLACK, EMPTY, SIZE, WHITE, empty_board, place

NAME = 'ABCDEFGHIJKLMNO'


def nm(point):
    return NAME[point[1]] + str(point[0] + 1)


# 两个固定棋谱：覆盖黑先、白先以及成片棋子相连的形态
FIXED_GAMES = [
    [(7, 7), (6, 6), (5, 7), (6, 8), (5, 5), (7, 6), (4, 6), (8, 8), (5, 8), (6, 7)],
    [(7, 7), (7, 8), (7, 6), (6, 7), (8, 7), (6, 6), (8, 8), (5, 7), (9, 7), (4, 7)],
]


def play(cells, swap_colors=False):
    """按黑白交替落子，返回 (board, engine)。swap_colors=True 时黑白互换。"""
    board = empty_board()
    engine = Engine()
    for index, (r, c) in enumerate(cells):
        first, second = (WHITE, BLACK) if swap_colors else (BLACK, WHITE)
        player = first if index % 2 == 0 else second
        place(board, r, c, player)
        engine.on_move(board, r, c, player)
    return board, engine


def full_tables(board):
    """全盘重算得到的黑白状态表，作为增量更新的基准。"""
    search = Search(PatternAnalyzer())
    search.rebuild_all(board)
    return search.sb, search.sw


def random_games(seed, count, moves):
    rng = random.Random(seed)
    cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    return [rng.sample(cells, moves) for _ in range(count)]


class IncrementalExactnessTest(unittest.TestCase):

    def assert_exact(self, engine, board, label):
        sb, sw = full_tables(board)
        drift = [(nm((r, c)), BLACK if engine.search.sb[r][c] != sb[r][c] else WHITE)
                 for r in range(SIZE) for c in range(SIZE)
                 if engine.search.sb[r][c] != sb[r][c] or engine.search.sw[r][c] != sw[r][c]]
        self.assertEqual(engine.search.sb, sb, '%s：黑表与全盘重算不一致 %s' % (label, drift[:5]))
        self.assertEqual(engine.search.sw, sw, '%s：白表与全盘重算不一致 %s' % (label, drift[:5]))

    def test_fixed_games_match_full_rebuild(self):
        for index, cells in enumerate(FIXED_GAMES):
            board, engine = play(cells)
            self.assert_exact(engine, board, '固定棋谱#%d' % index)

    def test_random_games_match_full_rebuild(self):
        for index, cells in enumerate(random_games(20240607, 12, 20)):
            board, engine = play(cells)
            self.assert_exact(engine, board, '随机棋谱#%d' % index)

    def test_single_move_from_exact_tables(self):
        """从完全正确的表出发只走一手——直接检验"本手会不会漏刷"。

        落子点用随机空位：若固定取最左上的空位，落子点会被挤到角上，
        角上的双空跳有一端在盘外（判为 SLEEP2/DEAD，档位不变），测不出漏刷。
        """
        rng = random.Random(13579)
        for index, cells in enumerate(random_games(2468, 8, 20)):
            board, engine = play(cells)
            engine.search.rebuild_all(board)
            empties = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == EMPTY]
            for point in rng.sample(empties, 3):
                r, c = point
                board[r][c] = BLACK
                engine.search.on_move(board, r, c, BLACK, record_history=False)
                sb, sw = full_tables(board)
                self.assertEqual(engine.search.sb, sb, '棋谱#%d 落子 %s 后黑表不一致' % (index, nm(point)))
                self.assertEqual(engine.search.sw, sw, '棋谱#%d 落子 %s 后白表不一致' % (index, nm(point)))
                board[r][c] = EMPTY
                engine.search.rebuild_all(board)   # 复原为精确状态，供下一个落子点使用

    def test_color_swap_tables_are_mirrored(self):
        """颜色互换后两张表必须互为镜像：更新路径不再按落子方分叉。"""
        for index, cells in enumerate(random_games(20240607, 12, 20)):
            _, engine = play(cells)
            _, swapped = play(cells, swap_colors=True)
            self.assertEqual(engine.search.sb, swapped.search.sw, '棋谱#%d 黑表/白表不镜像' % index)
            self.assertEqual(engine.search.sw, swapped.search.sb, '棋谱#%d 白表/黑表不镜像' % index)


if __name__ == '__main__':
    unittest.main(verbosity=2)
