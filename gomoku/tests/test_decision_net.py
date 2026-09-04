# -*- coding: utf-8 -*-
"""决策网络输入/结构/接入测试：
1. encode_decision：8 层棋盘 + 候选清单 + mask 形状与数值正确。
2. MemoryLookup：对称匹配命中/不命中。
3. DecisionNet：前向输出形状、padding 行不参与选择。
4. engine 接入：use_decision_net=False 回退第一名；传入随机网络 → 候选内选点。
"""
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from gomoku.core.utils import empty_board, place, BLACK, WHITE
from gomoku.core.engine import Engine
from gomoku.nn.decision_input import encode_decision, board_layers, candidate_rows, MAX_CANDS, FEAT_BOARD_CH, FEAT_CAND
from gomoku.nn.memory_lookup import MemoryLookup
from gomoku.nn.decision_net import DecisionNet


def replay(e, b, moves):
    for (r, c, p) in moves:
        place(b, r, c, p)
        e.on_move(b, r, c, p)


MIDGAME = [
    (7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (6, 8, WHITE),
    (5, 6, BLACK), (7, 6, WHITE), (4, 6, BLACK), (8, 8, WHITE),
    (5, 8, BLACK), (6, 7, WHITE),
]


class EncodeTest(unittest.TestCase):

    def test_shapes(self):
        e = Engine()
        b = empty_board()
        replay(e, b, MIDGAME)
        ranked = [{'r': 5, 'c': 6, 'score': 1050.0}, {'r': 6, 'c': 9, 'score': 48.0},
                  {'r': 4, 'c': 5, 'score': -320.5}]
        wins, losses = [2, 0, 1], [0, 3, 1]
        xb, xc, mask = encode_decision(b, e.search, ranked, wins, losses, [1, 5, 0], [0, 2, 1])
        self.assertEqual(xb.shape, (1, FEAT_BOARD_CH, 15, 15))
        self.assertEqual(xc.shape, (1, MAX_CANDS, FEAT_CAND))
        self.assertEqual(mask.shape, (1, MAX_CANDS))
        self.assertEqual(mask[0][:3].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(mask[0][3:].sum(), 0.0)
        # 数值：候选分数层 = tanh 归一；排名层 = 1/k
        bl = board_layers(b, e.search, ranked)
        self.assertAlmostEqual(bl[6][5][6], 0.5 + 0.5 * np.tanh(1050.0 / 500.0), places=5)
        self.assertAlmostEqual(bl[7][6][9], 0.5, places=5)   # 第 2 名 1/2
        self.assertAlmostEqual(bl[2][7][7], 1.0, places=5)   # 已占格 sb=17(none) → 父类3 → /3 = 1.0
        # 已占格状态表是 17(none) → 父类 3 → /3 = 1.0；黑子层该格为 1
        self.assertEqual(bl[0][7][7], 1.0)
        self.assertEqual(bl[1][6][6], 1.0)
        # 候选清单：坐标/分数/排名/人机胜负/自我胜负归一
        rows = candidate_rows(ranked, wins, losses, [1, 5, 0], [0, 2, 1])
        self.assertAlmostEqual(rows[0][0], 5 / 14.0)
        self.assertAlmostEqual(rows[0][3], 1.0)
        self.assertAlmostEqual(rows[1][4], 0.0)
        self.assertAlmostEqual(rows[1][5], 3 / 9.0)
        self.assertAlmostEqual(rows[2][4], 1 / 9.0)
        self.assertAlmostEqual(rows[0][6], 1 / 9.0)          # self_wins[0]=1
        self.assertAlmostEqual(rows[1][7], 2 / 9.0)          # self_losses[1]=2

    def test_overflow_truncation(self):
        """候选超过 MAX_CANDS → 截断尾部，mask 只标前 MAX_CANDS 个"""
        e = Engine()
        b = empty_board()
        replay(e, b, MIDGAME)
        ranked = [{'r': i // 15, 'c': i % 15, 'score': float(i)} for i in range(MAX_CANDS + 5)]
        _, _, mask = encode_decision(b, e.search, ranked, [0] * len(ranked), [0] * len(ranked))
        self.assertEqual(mask[0].sum(), float(MAX_CANDS))


class MemoryTest(unittest.TestCase):

    def test_hits_and_misses(self):
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump({
                'badLines': [{'opp': [{'r': 7, 'c': 7}], 'ai': [{'r': 6, 'c': 6}], 'killType': '冲四'}],
                'goodLines': [{'opp': [{'r': 7, 'c': 7}], 'ai': [{'r': 7, 'c': 8}], 'killType': None}],
            }, f)
            path = f.name
        try:
            mem = MemoryLookup(path, os.path.join(tempfile.gettempdir(), 'no-such-self-mem.json'))
            # 黑 H8 起手，白（player=2）候选 (6,6)=G7 → 命中 badLines；候选 (7,8)=H9 → 命中 goodLines
            moves = [(7, 7, BLACK)]
            cands = [{'r': 6, 'c': 6, 'score': 1.0}, {'r': 7, 'c': 8, 'score': 1.0}, {'r': 5, 'c': 5, 'score': 1.0}]
            wins, losses, sw, sl = mem.batch_stats(moves, WHITE, cands)
            self.assertEqual((wins[0], losses[0]), (0, 1))   # G7 败过
            self.assertEqual((wins[1], losses[1]), (1, 0))   # H9 赢过
            self.assertEqual((wins[2], losses[2]), (0, 0))   # F6 无记录
            self.assertEqual(sw, [0, 0, 0])                  # 临时文件无自我对弈数据
            self.assertEqual(sl, [0, 0, 0])
            # 对称性：镜像变换后仍命中（黑 H8 镜像 = A8？(7,7) 对称不变；换个开局验证变换）
            moves2 = [(7, 7, BLACK), (6, 6, WHITE)]
            cands2 = [{'r': 8, 'c': 8, 'score': 1.0}]
            w2, l2, _, _ = mem.batch_stats(moves2, BLACK, cands2)
            # 黑候选 (8,8)：历史黑 ai 前缀只有 [(6,6) 白？] 不对——player=BLACK 时 ai_moves=黑着法
            # 黑着法 = [(7,7)]，候选 (8,8) 后 [(7,7),(8,8)]；goodLines ai=[(7,8)] 长度不够 → 0
            self.assertEqual((w2[0], l2[0]), (0, 0))
        finally:
            os.unlink(path)

    def test_missing_file_empty(self):
        mem = MemoryLookup(os.path.join(tempfile.gettempdir(), 'no-such-gomoku-mem.json'))
        wins, losses, sw, sl = mem.batch_stats([], WHITE, [{'r': 0, 'c': 0}])
        self.assertEqual(wins, [0])
        self.assertEqual(losses, [0])
        self.assertEqual(sw, [0])
        self.assertEqual(sl, [0])


class DecisionNetTest(unittest.TestCase):

    def test_forward_and_padding(self):
        torch = __import__('torch')
        net = DecisionNet()
        net.eval()
        e = Engine()
        b = empty_board()
        replay(e, b, MIDGAME)
        ranked = [{'r': 5, 'c': 6, 'score': 90.0}, {'r': 6, 'c': 9, 'score': 85.0}]
        xb, xc, mask = encode_decision(b, e.search, ranked, [0, 0], [0, 0])
        logits = net(torch.from_numpy(xb), torch.from_numpy(xc), torch.from_numpy(mask))
        self.assertEqual(logits.shape, (1, MAX_CANDS))
        idx = net.pick(torch.from_numpy(xb), torch.from_numpy(xc), torch.from_numpy(mask))[0]
        self.assertIn(idx, (0, 1))       # 只可能是有效候选
        # padding 行分数必须极低
        self.assertLess(logits[0, 2].item(), -1e8)

    def test_engine_fallback_without_net(self):
        """use_decision_net=False → 纯算法第一名，行为与之前一致"""
        e = Engine()
        b = empty_board()
        replay(e, b, MIDGAME)
        r4 = e.analyze_turn(b, WHITE)['part4_result']
        self.assertEqual(r4['move'], (r4['ranked'][0]['r'], r4['ranked'][0]['c']))

    def test_engine_with_random_net_picks_from_candidates(self):
        """传入随机网络 → 最终 move 必须是候选之一（不越界、不崩）"""
        torch = __import__('torch')
        from gomoku.nn.decision_net import DecisionNet
        net = DecisionNet()
        net.eval()
        e = Engine(decision_net=net, use_decision_net=True)
        b = empty_board()
        replay(e, b, MIDGAME)
        r4 = e.analyze_turn(b, WHITE)['part4_result']
        cand_set = {(x['r'], x['c']) for x in r4['ranked']}
        self.assertIn(r4['move'], cand_set)

    def test_engine_net_load_failure_fallback(self):
        """模型加载失败（torch.load 抛错）→ 静默回退第一名"""
        with mock.patch('torch.load', side_effect=RuntimeError('mock load fail')):
            e = Engine(use_decision_net=True)
            b = empty_board()
            replay(e, b, MIDGAME)
            r4 = e.analyze_turn(b, WHITE)['part4_result']
            self.assertEqual(r4['move'], (r4['ranked'][0]['r'], r4['ranked'][0]['c']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
