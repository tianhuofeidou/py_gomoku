# -*- coding: utf-8 -*-
"""热路径档位查询（Search._direction_level / Search._eval）的等价性回归。

`Search._eval` 现在只取每方向档位（`_direction_level`），不再构造棋型记录、
不算子类名与 gap_sides。它必须与改前的口径**严格等价**：

    analyze_point 的记录 → 每方向取 DIR_LEVEL 最小值 → _classify_levels

任何不等价都会静默改变状态表编号，进而改变候选分、攻防档位与叶子评估。
本文件同时钉住三条最容易改错的分支：

  1. 预筛必须查【两侧】3 格 —— 只查单侧会在随机局面上错约 5%；
  2. 预筛必须查到第 3 格 —— 双空前瞻会读到距离 3；
  3. 端 5 缝合不可省 —— 它把 SINGLE(档位 5) 换成 LIVE2(档位 4)，
     而档位 4 在 _classify_levels 里要计入 B 的数量。

注意：`_direction_level` 与 `_scan_two` 一样，假定**落子已经在棋盘上**
（调用方负责临时落子），参考实现同理。
"""
import random
import types
import unittest

from gomoku.core.engine import Engine
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import DIR_LEVEL, Search
from gomoku.core.utils import BLACK, DIRECTIONS, EMPTY, SIZE, WHITE, empty_board, place

# 抽样规模：完整线型空间是 3^14 = 4,782,969，CI 里用抽样 + 边界穷举覆盖。
LINE_SAMPLES = 2000
EVAL_BOARDS = 3


def reference_eval(self, board, r, c, player):
    """改前 `_eval` 的逐字复刻，作为整点比对基准。"""
    board[r][c] = player
    try:
        recs = self.pattern.analyze_point(board, r, c, player)
        levels = []
        for (dr, dc) in DIRECTIONS:
            dir_level = 5
            for rec in recs:
                if rec['dir'] != (dr, dc):
                    continue
                lv = DIR_LEVEL.get(rec['parent'], 5)
                if lv < dir_level:
                    dir_level = lv
            levels.append(dir_level)
        return self._classify_levels(levels)
    finally:
        board[r][c] = EMPTY


def reference_direction_level(self, board, r, c, player, dr, dc):
    """改前 `_eval` 对单个方向的贡献：记录取 DIR_LEVEL 最小值。"""
    best = 5
    for rec in self.pattern.analyze_direction(board, r, c, player, dr, dc):
        lv = DIR_LEVEL.get(rec['parent'], 5)
        if lv < best:
            best = lv
    return best


def random_boards(seed, count, lo=4, hi=32):
    rng = random.Random(seed)
    cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    boards = []
    for _ in range(count):
        board = empty_board()
        for (r, c) in rng.sample(cells, rng.randint(lo, hi)):
            board[r][c] = rng.choice((BLACK, WHITE))
        boards.append(board)
    return boards


def diff(a, b, path=''):
    """递归找出第一处差异（用于端到端全字段比对）。"""
    if type(a) is not type(b):
        return '%s 类型不同 %r vs %r' % (path, type(a), type(b))
    if isinstance(a, dict):
        if set(a) != set(b):
            return '%s 键不同 %r vs %r' % (path, sorted(a), sorted(b))
        for key in a:
            found = diff(a[key], b[key], '%s[%r]' % (path, key))
            if found:
                return found
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return '%s 长度不同 %d vs %d' % (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            found = diff(x, y, '%s[%d]' % (path, i))
            if found:
                return found
        return None
    if a != b:
        return '%s 值不同 %r vs %r' % (path, a, b)
    return None


class DirectionLevelTest(unittest.TestCase):
    """单方向档位：抽样线型 + 边界穷举。"""

    def test_line_patterns_sampled(self):
        """沿四个方向各铺随机线型，逐条比对档位。"""
        rng = random.Random(20240607)
        search = Search(PatternAnalyzer())
        checked = 0
        for _ in range(LINE_SAMPLES):
            dr, dc = rng.choice(DIRECTIONS)
            player = rng.choice((BLACK, WHITE))
            board = empty_board()
            for k in range(-7, 8):
                rr, cc = 7 + k * dr, 7 + k * dc
                if not (0 <= rr < SIZE and 0 <= cc < SIZE) or (rr, cc) == (7, 7):
                    continue
                board[rr][cc] = rng.choice((EMPTY, EMPTY, EMPTY, player, 3 - player))
            board[7][7] = player
            try:
                got = search._direction_level(board, 7, 7, player, dr, dc)
                expected = reference_direction_level(search, board, 7, 7, player, dr, dc)
            finally:
                board[7][7] = EMPTY
            self.assertEqual(got, expected,
                             '方向 %s 玩家 %d 线型不一致' % ((dr, dc), player))
            checked += 1
        self.assertEqual(checked, LINE_SAMPLES)

    def test_boundary_lines_exhaustive(self):
        """基子贴墙时穷举：col=0/1/2，另一侧 6 格全枚举（3^6 每种位置）。"""
        search = Search(PatternAnalyzer())
        values = (EMPTY, WHITE, BLACK)
        checked = 0
        for base_col in (0, 1, 2):
            free = list(range(base_col + 1, base_col + 7))
            digits = [0] * len(free)
            for _ in range(3 ** len(free)):
                board = empty_board()
                for k, cc in enumerate(free):
                    board[7][cc] = values[digits[k]]
                board[7][base_col] = WHITE
                try:
                    got = search._direction_level(board, 7, base_col, WHITE, 0, 1)
                    expected = reference_direction_level(search, board, 7, base_col, WHITE, 0, 1)
                finally:
                    board[7][base_col] = EMPTY
                self.assertEqual(got, expected, '边界基子 col=%d' % base_col)
                checked += 1
                i = 0
                while i < len(free) and digits[i] == 2:
                    digits[i] = 0
                    i += 1
                if i < len(free):
                    digits[i] += 1
                else:
                    break
        self.assertEqual(checked, 3 * (3 ** 6))

    def test_prefilter_checks_both_sides(self):
        """预筛必须查两侧：己方子只在基子负侧时不能返回 5。"""
        search = Search(PatternAnalyzer())
        board = empty_board()
        place(board, 7, 4, WHITE)          # 只在 (0,1) 方向的负侧
        board[7][7] = WHITE
        try:
            self.assertNotEqual(search._direction_level(board, 7, 7, WHITE, 0, 1), 5)
        finally:
            board[7][7] = EMPTY

    def test_prefilter_checks_third_cell(self):
        """预筛必须查到第 3 格：双空前瞻会读距离 3，那里有己方子即 LIVE2。"""
        search = Search(PatternAnalyzer())
        board = empty_board()
        place(board, 7, 10, WHITE)         # 距离 3
        board[7][7] = WHITE
        try:
            self.assertEqual(search._direction_level(board, 7, 7, WHITE, 0, 1), 4)
            self.assertEqual(search._direction_level_scan(board, 7, 7, WHITE, 0, 1), 4)
        finally:
            board[7][7] = EMPTY

    def test_double_gap_edge5_is_not_skipped(self):
        """端 5 缝合不可省：SINGLE(5) → LIVE2(4)，整点编号 16 而不是 17。

        最小反例：棋盘只有白子 (7,10)，(7,7) 落白时 (0,1) 方向为 LIVE2。
        """
        board = empty_board()
        place(board, 7, 10, WHITE)
        search = Search(PatternAnalyzer())
        self.assertEqual(search._direction_level(board, 7, 7, WHITE, 0, 1), 4)
        self.assertEqual(search._eval(board, 7, 7, WHITE), 16)


class EvalEquivalenceTest(unittest.TestCase):
    """整点求值：新路径与改前实现逐点一致。"""

    def test_eval_matches_reference(self):
        checked = 0
        for board in random_boards(2024, EVAL_BOARDS):
            search = Search(PatternAnalyzer())
            reference = Search(PatternAnalyzer())
            reference._eval = types.MethodType(reference_eval, reference)
            for r in range(SIZE):
                for c in range(SIZE):
                    if board[r][c] != EMPTY:
                        continue
                    for player in (BLACK, WHITE):
                        self.assertEqual(
                            search._eval(board, r, c, player),
                            reference._eval(board, r, c, player),
                            '整点 (%d,%d) 玩家 %d 编号不一致' % (r, c, player))
                        checked += 1
        self.assertGreater(checked, 500)

    def test_state_tables_identical_after_moves(self):
        """增量更新路径整盘一致：on_move 后两张状态表逐格相同。"""
        moves = [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (6, 8, WHITE),
                 (5, 6, BLACK), (7, 6, WHITE), (4, 6, BLACK), (8, 8, WHITE),
                 (5, 8, BLACK), (6, 7, WHITE)]
        fast = Engine()
        slow = Engine()
        slow.search._eval = types.MethodType(reference_eval, slow.search)
        board_fast, board_slow = empty_board(), empty_board()
        for (r, c, p) in moves:
            place(board_fast, r, c, p)
            place(board_slow, r, c, p)
            fast.on_move(board_fast, r, c, p)
            slow.on_move(board_slow, r, c, p)
        self.assertEqual(fast.search.sb, slow.search.sb)
        self.assertEqual(fast.search.sw, slow.search.sw)


class RankingEquivalenceTest(unittest.TestCase):
    """端到端：analyze_turn 的全部输出逐字段一致（含浮点分与五态）。"""

    POSITIONS = [
        [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (6, 8, WHITE), (5, 6, BLACK),
         (7, 6, WHITE), (4, 6, BLACK), (8, 8, WHITE), (5, 8, BLACK), (6, 7, WHITE)],
        [(7, 7, BLACK), (7, 8, WHITE), (7, 6, BLACK), (6, 7, WHITE), (8, 7, BLACK),
         (6, 6, WHITE), (8, 8, BLACK), (5, 7, WHITE)],
        [(7, 7, BLACK), (6, 6, WHITE), (8, 8, BLACK), (6, 8, WHITE), (8, 6, BLACK),
         (5, 5, WHITE), (9, 9, BLACK)],
    ]

    def _analyze(self, moves, player, use_reference):
        engine = Engine()
        if use_reference:
            engine.search._eval = types.MethodType(reference_eval, engine.search)
        board = empty_board()
        for (r, c, p) in moves:
            place(board, r, c, p)
            engine.on_move(board, r, c, p)
        engine.deep_search.T0 = 40.0        # 压低温度，保证 CI 里跑得快
        engine.deep_search.reset_leaf_eval_stats()
        result = engine.analyze_turn(board, player)
        return result, engine.deep_search.leaf_eval_status(), engine.decision_net_used

    def test_ranked_identical_with_reference_eval(self):
        for index, moves in enumerate(self.POSITIONS):
            for player in (BLACK, WHITE):
                fast = self._analyze(moves, player, False)
                slow = self._analyze(moves, player, True)
                found = diff(fast[0], slow[0], 'result') or diff(fast[1], slow[1], 'leaf')
                self.assertIsNone(found, '局面#%d 玩家 %d → %s' % (index, player, found))
                self.assertEqual(fast[2], slow[2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
