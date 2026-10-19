# -*- coding: utf-8 -*-
"""Gomocup / Pikvork 协议适配层回归。

锁定三件事：
1. 握手与命令契约：START 尺寸校验、INFO 不回复、ABOUT 格式、未知命令回 UNKNOWN、
   END 退出、单手命令（BEGIN/TURN）回合法坐标；
2. 坐标映射：协议 (X, Y) → 内核 (r, c) = (Y, X)；空盘首手必须是天元 (7,7) → "7,7"；
3. 端到端：两个协议进程通过管道跑完一局（管理器侧用 gomoku.tools.match），
   每一步都合法、能判出结果。
"""
import io
import sys
import unittest

from gomoku.core.utils import BLACK, WHITE, SIZE, empty_board, place
from gomoku.protocol import Brain, run
from gomoku.tools.match import (_elo_delta, _wilson, gen_openings,
                                load_official_openings, play_one)


class HandshakeTest(unittest.TestCase):

    def setUp(self):
        self.brain = Brain()

    def test_start_sizes(self):
        self.assertEqual(self.brain.handle('START %d' % SIZE), 'OK')
        self.assertTrue(self.brain.handle('START 20').startswith('ERROR'),
                        '20×20 必须明确回 ERROR，由管理器另试尺寸')

    def test_info_and_about_and_unknown(self):
        self.brain.handle('START %d' % SIZE)
        self.assertIsNone(self.brain.handle('INFO timeout_turn 30000'))
        self.assertEqual(self.brain.timeout_turn, 30000)
        self.assertIsNone(self.brain.handle('INFO max_memory 83886080'))
        self.assertEqual(self.brain.max_memory, 83886080)
        self.assertTrue(self.brain.handle('ABOUT').startswith('name="'))
        self.assertTrue(self.brain.handle('RECTSTART 30,20').startswith('UNKNOWN'))
        self.assertEqual(self.brain.handle('END'), '__END__')

    def test_begin_opens_at_center(self):
        self.brain.handle('START %d' % SIZE)
        self.assertEqual(self.brain.handle('BEGIN'), '7,7',
                         '空盘首手是天元；协议 (X,Y)=(列,行) → "7,7"')
        self.assertEqual(self.brain.my_color, BLACK)
        self.assertEqual(len(self.brain.moves), 1)

    def test_turn_maps_axis_and_answers_legal(self):
        self.brain.handle('START %d' % SIZE)
        self.brain.handle('BEGIN')
        # 对手下在协议坐标 (0,0) → 内核 (r,c)=(0,0)
        reply = self.brain.handle('TURN 0,0')
        self.assertEqual(self.brain.board[0][0], WHITE)
        self.assertEqual(self.brain.my_color, BLACK)
        r, c = [int(v) for v in reply.split(',')]
        self.assertTrue(0 <= r < SIZE and 0 <= c < SIZE)
        self.assertEqual(self.brain.board[c][r], BLACK)

    def test_board_command_then_move(self):
        self.brain.handle('START %d' % SIZE)
        # 自动开局：管理器直接给盘面（1=己方，2=对方），随后要一手
        stream = io.StringIO('7,7,1\n7,8,2\nDONE\n')
        reply = self.brain.read_board(stream)
        self.assertEqual(len(self.brain.moves), 3, '2 颗盘面 + 我方回复 1 手')
        self.assertEqual(self.brain.board[7][7], BLACK)
        self.assertEqual(self.brain.board[8][7], WHITE)
        self.assertEqual(self.brain.my_color, BLACK, '双方子数相等且轮到己方 → 我方执黑')
        self.assertIsNotNone(reply)
        r, c = [int(v) for v in reply.split(',')]
        self.assertEqual(self.brain.board[c][r], BLACK)
        self.assertEqual(self.brain.moves[-1]['r'], c, '协议 (X,Y) → 内核 (r,c)=(Y,X)')
        self.assertEqual(self.brain.moves[-1]['c'], r)

    def test_board_opponent_has_one_more_stone_means_white(self):
        """对方多一子且轮到己方 → 我方执白；不依赖第一条记录。"""
        self.brain.handle('START %d' % SIZE)
        stream = io.StringIO('7,7,2\n7,8,1\n6,6,2\nDONE\n')
        reply = self.brain.read_board(stream)
        self.assertEqual(self.brain.my_color, WHITE)
        self.assertEqual(self.brain.board[7][7], BLACK)
        self.assertEqual(self.brain.board[8][7], WHITE)
        self.assertEqual(self.brain.board[6][6], BLACK)
        self.assertIsNotNone(reply)
        self.assertFalse(reply.startswith('ERROR'), reply)

    def test_run_loop_terminates_on_end(self):
        stream = io.StringIO('START %d\nINFO timeout_turn 1000\nABOUT\nEND\n' % SIZE)
        out = io.StringIO()
        old = sys.stdout
        sys.stdout = out
        try:
            self.assertEqual(run(stream_in=stream), 0)
        finally:
            sys.stdout = old
        text = out.getvalue()
        self.assertIn('OK', text)
        self.assertIn('name="', text)


class EndToEndTest(unittest.TestCase):

    def test_two_protocol_processes_play_a_game(self):
        """两个协议进程通过管道跑完一局：每手合法、能判出结果。"""
        cmd = '%s -u -m gomoku.protocol --t0 40' % sys.executable
        winner, moves, log = play_one(cmd, cmd, opening=[], max_moves=8,
                                      timeout_turn=60000)
        self.assertIn(winner, ('B', 'W', 'draw'), 'msg=%s' % log)
        self.assertEqual(len(moves), 8)
        board = empty_board()
        for i, m in enumerate(moves):
            player = BLACK if i % 2 == 0 else WHITE
            self.assertEqual(m['player'], player, '轮次必须黑白交替')
            self.assertEqual(board[m['r']][m['c']], 0, '不得落在已占点')
            place(board, m['r'], m['c'], player)
        self.assertEqual(moves[0]['r'], 7)
        self.assertEqual(moves[0]['c'], 7)


class ArenaTest(unittest.TestCase):
    """对战台纯函数：开局生成与统计口径。"""

    SYMS = (
        lambda r, c: (r, c), lambda r, c: (c, 14 - r), lambda r, c: (14 - r, 14 - c),
        lambda r, c: (14 - c, r), lambda r, c: (r, 14 - c), lambda r, c: (14 - r, c),
        lambda r, c: (c, r), lambda r, c: (14 - c, 14 - r),
    )

    def canon(self, moves):
        return min(tuple(sorted(f(r, c) for (r, c) in moves)) for f in self.SYMS)

    def test_gen_openings_deterministic_and_legal(self):
        a = gen_openings(6, seed=7, stones=3)
        b = gen_openings(6, seed=7, stones=3)
        self.assertEqual(a, b, '同一种子必须生成同一批开局（可复现）')
        self.assertEqual(len(a), 6)
        for moves in a:
            self.assertEqual(len(moves), 3)
            self.assertEqual(len(set(moves)), 3, '开局内不得重复落点')
            self.assertEqual(moves[0], (7, 7), '黑首手固定天元')
            for (r, c) in moves:
                self.assertTrue(0 <= r < SIZE and 0 <= c < SIZE)

    def test_gen_openings_symmetry_dedup(self):
        openings = gen_openings(12, seed=11, stones=3)
        canons = [self.canon(m) for m in openings]
        self.assertEqual(len(canons), len(set(canons)), '不得存在对称等价的开局')

    def test_different_seeds_differ(self):
        self.assertNotEqual(gen_openings(6, seed=1), gen_openings(6, seed=2))

    def test_official_openings_are_usable(self):
        """官方开局集：条目数与合法性。

        注意：原文件里绝大多数条目以天元开局，但**存在被平移过的条目**
        （例如偏移 `4,4,...` → 首手 (11,11)），因此这里只断言"合法 + 绝大多数居中"，
        不强制每条都天元。
        """
        openings = load_official_openings()
        self.assertGreaterEqual(len(openings), 40)
        centered = 0
        for moves in openings:
            self.assertIn(len(moves), (3, 5), '官方开局只有 3 手与 5 手两种')
            self.assertEqual(len(set(moves)), len(moves), '开局内不得重复落点')
            for (r, c) in moves:
                self.assertTrue(0 <= r < SIZE and 0 <= c < SIZE, '必须落在 15×15 内')
            centered += 1 if moves[0] == (7, 7) else 0
        self.assertGreaterEqual(centered, len(openings) - 3,
                                '除少数平移/远角变体外，官方开局均以天元开局')

    def test_official_openings_limit(self):
        self.assertEqual(len(load_official_openings(limit=5)), 5)

    def test_official_openings_provenance_recorded(self):
        """数据文件必须记录来源，便于核对与许可说明。"""
        import json
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data', 'official_openings.json')
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.assertIn('Piskvork', data.get('source', ''))
        self.assertTrue(data.get('url'))
        self.assertIn('unusable_on_15', data, '超出 15 路棋盘的条目应单列，不能悄悄丢')

    def test_stats_helpers(self):
        self.assertAlmostEqual(_elo_delta(0.5), 0.0, places=6)
        self.assertGreater(_elo_delta(0.75), 150.0)      # 75% 胜率 ≈ +190 Elo
        lo, hi = _wilson(5, 10)
        self.assertTrue(0.0 <= lo < 0.5 < hi <= 1.0)
        self.assertLess(_wilson(0, 10)[1], 0.35)         # 0 胜 10 负：上界很低
        self.assertGreater(_wilson(10, 10)[0], 0.65)


    def test_game_with_official_opening_stays_legal(self):
        """带自动开局的对局必须全程合法（管理器要用 BOARD 把**全部**开局子告诉引擎）。

        旧实现只把最后一颗开局子用 TURN 发出，引擎看不见前面的开局子，会落在已有子上——
        实测 80 局官方开局基线里有 34/40 局以"落子非法"结束（数据无效）。本用例锁死该回归。
        """
        cmd = '%s -u -m gomoku.protocol --t0 40' % sys.executable
        opening = load_official_openings(limit=1)[0]          # 官方第一条（3 手）
        self.assertEqual(len(opening), 3)
        winner, moves, log = play_one(cmd, cmd, opening=list(opening),
                                      max_moves=len(opening) + 8, timeout_turn=60000)
        self.assertEqual(log, [], '带开局的对局不得出现非法落子或超时')
        self.assertIn(winner, ('B', 'W', 'draw'))
        for i, (r, c) in enumerate(opening):
            self.assertEqual((moves[i]['r'], moves[i]['c']), (r, c), '开局子必须原样保留')
            self.assertEqual(moves[i]['player'], BLACK if i % 2 == 0 else WHITE)
        board = empty_board()
        for m in moves:
            self.assertEqual(board[m['r']][m['c']], 0, '不得落在已占点')
            place(board, m['r'], m['c'], m['player'])


if __name__ == '__main__':
    unittest.main()
