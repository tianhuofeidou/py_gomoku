# -*- coding: utf-8 -*-
"""自我对弈记账 + 训练采样器测试：
1. SelfMemory.record_game：胜/负方视角两条记录、killType、winLine 正确；文件落盘可回读。
2. loss_type_of：活四 / 冲四 / 跳四 / 其他。
3. SamplePool：人类胜率 → 重复上限；batch 人机占比 ≤20%；退役逻辑。
4. data.play_one_game(record_self=True) 端到端记账。
"""
import json
import os
import tempfile
import unittest

from gomoku.core.utils import empty_board, place, BLACK, WHITE
from gomoku.nn.self_memory import SelfMemory, loss_type_of, win_line_of, default_self_memory_file
from gomoku.nn.sample_pool import SamplePool, human_cap_of
from gomoku.nn import data as nn_data


def b_with(moves):
    b = empty_board()
    for (r, c, p) in moves:
        place(b, r, c, p)
    return b


class LossTypeTest(unittest.TestCase):

    def test_live4(self):
        b = empty_board()
        moves = [(7, 3, BLACK), (7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK)]
        for (r, c, p) in moves:
            place(b, r, c, p)
        self.assertEqual(loss_type_of(b, 7, 7, BLACK), '活四')

    def test_sleep4(self):
        b = empty_board()
        moves = [(7, 3, BLACK), (7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK), (6, 8, WHITE)]
        for (r, c, p) in moves:
            place(b, r, c, p)
        # 右端 (7,8) 空、左端 (7,2) 空 → 还是活四；(7,2) 堵住 → 冲四
        place(b, 7, 2, WHITE)
        self.assertEqual(loss_type_of(b, 7, 7, BLACK), '冲四')

    def test_jump4(self):
        b = empty_board()
        # 跳四：xx_xx 中间是空位（7,6），两端封死 → 跳四
        moves = [(7, 3, BLACK), (7, 4, BLACK), (7, 5, BLACK), (7, 7, BLACK), (7, 8, BLACK),
                 (7, 2, WHITE), (7, 9, WHITE)]
        for (r, c, p) in moves:
            place(b, r, c, p)
        self.assertEqual(loss_type_of(b, 7, 8, BLACK), '跳四')

    def test_win_line(self):
        b = b_with([(7, 3, BLACK), (7, 4, BLACK), (7, 5, BLACK), (7, 6, BLACK), (7, 7, BLACK)])
        wl = win_line_of(b, 7, 7, BLACK)
        self.assertEqual(wl, [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)])


class SelfMemoryTest(unittest.TestCase):

    def _tmp_path(self):
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        os.unlink(path)
        return path

    def test_record_game_two_views(self):
        path = self._tmp_path()
        try:
            m = SelfMemory(path)
            # 一局：黑五连获胜（黑 H8-F6... 用简单横线）
            moves = [(7, 3, BLACK), (6, 6, WHITE), (7, 4, BLACK), (6, 7, WHITE),
                     (7, 5, BLACK), (6, 5, WHITE), (7, 6, BLACK), (6, 4, WHITE), (7, 7, BLACK)]
            self.assertTrue(m.record_game(moves, BLACK))
            self.assertEqual(len(m.good_lines), 1)   # 胜方视角
            self.assertEqual(len(m.bad_lines), 1)    # 负方视角
            g = m.good_lines[0]
            self.assertEqual(g['killType'], '活四')
            self.assertEqual(g['winLine'], [{'r': 7, 'c': 3}, {'r': 7, 'c': 4}, {'r': 7, 'c': 5},
                                            {'r': 7, 'c': 6}, {'r': 7, 'c': 7}])
            # good：opp=负方(白)，ai=胜方(黑)
            self.assertEqual([(x['r'], x['c']) for x in g['opp']], [(6, 6), (6, 7), (6, 5), (6, 4)])
            self.assertEqual([(x['r'], x['c']) for x in g['ai']], [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)])
            # bad：opp=胜方(黑)，ai=负方(白)
            bd = m.bad_lines[0]
            self.assertEqual([(x['r'], x['c']) for x in bd['ai']], [(6, 6), (6, 7), (6, 5), (6, 4)])
            # 落盘可回读
            m2 = SelfMemory(path)
            self.assertEqual(len(m2.good_lines), 1)
            self.assertEqual(len(m2.bad_lines), 1)
            self.assertEqual(m2.good_lines[0]['killType'], '活四')
        finally:
            os.unlink(path)

    def test_draw_not_recorded(self):
        path = self._tmp_path()
        try:
            m = SelfMemory(path)
            self.assertFalse(m.record_game([(7, 7, BLACK)], 0))   # 和棋不记
            self.assertEqual(len(m.good_lines), 0)
            self.assertEqual(len(m.bad_lines), 0)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_default_path_under_dsh_home(self):
        p = default_self_memory_file()
        self.assertTrue(p.endswith(os.path.join('storages', 'gomoku', 'self-memory.json')))


class SamplePoolTest(unittest.TestCase):

    def test_human_cap(self):
        self.assertEqual(human_cap_of(0.8), 5)
        self.assertEqual(human_cap_of(0.5), 3)
        self.assertEqual(human_cap_of(0.3), 1)

    def test_batch_share_and_retire(self):
        pool = SamplePool(human_winrate=0.8, rng=__import__('random').Random(1))
        for i in range(10):
            pool.add(('human', i), 'human')
        for i in range(100):
            pool.add(('self', i), 'self')
        total_human = 0
        for _ in range(20):
            batch = pool.sample_batch(50)
            h = sum(1 for s in batch if s[0] == 'human')
            total_human += h
            self.assertLessEqual(h, 10)   # 50×20% = 10 封顶
        # R=0.8 → 上限 5：10 个人机样本 × 5 次 = 最多 50 次采样后全部退役
        self.assertEqual(pool.retired_human_count(), 10)
        batch = pool.sample_batch(50)
        self.assertEqual(sum(1 for s in batch if s[0] == 'human'), 0)  # 全退役 → 纯自我

    def test_low_winrate_one_pass(self):
        pool = SamplePool(human_winrate=0.3)
        for i in range(4):
            pool.add(('human', i), 'human')
        for i in range(50):
            pool.add(('self', i), 'self')
        for _ in range(10):
            pool.sample_batch(20)
        self.assertEqual(pool.retired_human_count(), 4)   # 上限 1 → 一遍退役

    def test_self_pool_backfill(self):
        pool = SamplePool(human_winrate=0.5)
        pool.add(('self', 0), 'self')
        pool.add(('self', 1), 'self')
        batch = pool.sample_batch(10)
        self.assertEqual(len(batch), 10)   # 自我池不足 → 重复补齐


class DataRecordSelfTest(unittest.TestCase):

    def test_play_one_game_records(self):
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        os.unlink(path)   # 只借用目录位置，SelfMemory 会写 storages/gomoku/self-memory.json
        try:
            os.environ['DSH_HOME'] = os.path.dirname(path)
            try:
                nn_data.play_one_game(max_moves=8, t0=80.0, record_self=True)
                sm_file = os.path.join(os.path.dirname(path), 'storages', 'gomoku', 'self-memory.json')
                self.assertTrue(os.path.exists(sm_file))
                with open(sm_file, encoding='utf-8') as f:
                    data = json.load(f)
                # 8 手内未必分胜负（和棋/未完不记），但若记录则结构必须完整
                if data.get('selfGoodLines'):
                    self.assertEqual(len(data['selfGoodLines']), len(data['selfBadLines']))
            finally:
                del os.environ['DSH_HOME']
        finally:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
