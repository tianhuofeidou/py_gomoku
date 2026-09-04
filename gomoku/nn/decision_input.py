# -*- coding: utf-8 -*-
"""decision_input.py —— 决策网络输入编码（棋盘分层 + 全候选清单）。

输入三块：
  ① 棋盘分层 [8, 15, 15]：
     0 黑子位置（1/0）  1 白子位置（1/0）
     2 黑棋型父类（0-3 → /3）  3 白棋型父类
     4 黑棋型子类（0-17 → /17）  5 白棋型子类
     6 候选分数层（该格=候选 → 0.5+0.5*tanh(score/500)，否则 0）
     7 候选排名层（该格=第 k 名 → 1/k，否则 0）
  ② 全候选清单 [max_cands, 8]：
     坐标 r/14、c/14、深推分（tanh 归一）、排名 1/k、
     人机历史赢/输、自我对弈历史赢/输（各自 clip 0-9 → /9）
  ③ mask [max_cands]：有效候选 1 / padding 0

max_cands=16：全候选通常 ≤16（防御反推去重后更少），超出截断尾部。
"""
import math

import numpy as np

from gomoku.core.utils import SIZE, BLACK, WHITE
from gomoku.core.search import SUBCLASS_PARENT

MAX_CANDS = 16
FEAT_BOARD_CH = 8
FEAT_CAND = 8


def _norm_score(score):
    """深推分支分 → (0,1)：0.5+0.5*tanh(score/500)。
    0 分 → 0.5；超大值（唯一候选 1e9）→ ~1.0；负分 → <0.5。"""
    return 0.5 + 0.5 * math.tanh(float(score) / 500.0)


def _norm_count(n):
    """历史胜负次数 → (0,1)：clip(0,9)/9"""
    return min(max(int(n), 0), 9) / 9.0


def board_layers(board, search, ranked):
    """棋盘 8 层编码 → [8, 15, 15] float32 ndarray。
    board: 15×15 0/1/2；search: 提供 sb/sw 状态表（18 子类编号）；
    ranked: [{'r','c','score'},...] 降序候选列表。"""
    layers = np.zeros((FEAT_BOARD_CH, SIZE, SIZE), dtype=np.float32)
    sb, sw = search.sb, search.sw
    for r in range(SIZE):
        for c in range(SIZE):
            v = board[r][c]
            if v == BLACK:
                layers[0][r][c] = 1.0
            elif v == WHITE:
                layers[1][r][c] = 1.0
            layers[2][r][c] = SUBCLASS_PARENT[sb[r][c]] / 3.0
            layers[3][r][c] = SUBCLASS_PARENT[sw[r][c]] / 3.0
            layers[4][r][c] = sb[r][c] / 17.0
            layers[5][r][c] = sw[r][c] / 17.0
    for k, cand in enumerate(ranked):
        r, c = cand['r'], cand['c']
        layers[6][r][c] = _norm_score(cand['score'])
        layers[7][r][c] = 1.0 / (k + 1)
    return layers


def candidate_rows(ranked, wins, losses, self_wins=None, self_losses=None, max_cands=MAX_CANDS):
    """全候选清单 → [max_cands, 8] float32 ndarray（不足补 0）。
    wins/losses: 人机历史赢/输；self_wins/self_losses: 自我对弈历史赢/输
    （缺省视为全 0，与 ranked 等长或按索引取）。"""
    self_wins = self_wins or [0] * len(ranked)
    self_losses = self_losses or [0] * len(ranked)
    rows = np.zeros((max_cands, FEAT_CAND), dtype=np.float32)
    for k, cand in enumerate(ranked[:max_cands]):
        rows[k][0] = cand['r'] / 14.0
        rows[k][1] = cand['c'] / 14.0
        rows[k][2] = _norm_score(cand['score'])
        rows[k][3] = 1.0 / (k + 1)
        rows[k][4] = _norm_count(wins[k] if k < len(wins) else 0)
        rows[k][5] = _norm_count(losses[k] if k < len(losses) else 0)
        rows[k][6] = _norm_count(self_wins[k] if k < len(self_wins) else 0)
        rows[k][7] = _norm_count(self_losses[k] if k < len(self_losses) else 0)
    return rows


def encode_decision(board, search, ranked, wins, losses, self_wins=None, self_losses=None, max_cands=MAX_CANDS):
    """完整输入编码 → (xb, xc, mask) numpy 数组（batch=1）。
    xb: [1,8,15,15]；xc: [1,max_cands,8]；mask: [1,max_cands]。"""
    n = min(len(ranked), max_cands)
    xb = board_layers(board, search, ranked)[np.newaxis, ...]
    xc = candidate_rows(ranked, wins, losses, self_wins, self_losses, max_cands=max_cands)[np.newaxis, ...]
    mask = np.zeros((1, max_cands), dtype=np.float32)
    mask[0, :n] = 1.0
    return xb, xc, mask
