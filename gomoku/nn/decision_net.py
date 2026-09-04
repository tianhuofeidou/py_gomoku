# -*- coding: utf-8 -*-
"""decision_net.py —— 决策网络（纯算法系统的「决策大脑」）。

双通道输入：
  ① 棋盘分层 [B, 8, 15, 15]（黑白子 / 棋型父类×2 / 棋型子类×2 / 候选分数 / 候选排名）
     → 小 CNN（2×Conv3x3 + 全局均值池）→ 棋盘特征 [B, C]
  ② 全候选清单 [B, N, 6]（坐标 / 深推分 / 排名 / 历史赢 / 历史输）
     → MLP → 每候选特征 [B, N, H]

每候选 = concat(棋盘特征, 候选特征) → 打分头 → 该候选「该不该下」的分。
输出 [B, N]：选分最高者为最终落子；padding 行（mask=0）压到极小不参与选择。

设计意图：纯算法（深推）出候选和棋理分，网络凭棋盘棋感 + 实战记忆修正排序——
「90 分的候选历史上总走输，改选 85 分那个」。训练数据与训练方式暂定（监督模仿
深推分 / 实战胜负），本文件只负责网络结构与前向。
"""
import torch
import torch.nn as nn

from gomoku.nn.decision_input import FEAT_BOARD_CH, FEAT_CAND, MAX_CANDS


class DecisionNet(nn.Module):
    def __init__(self, board_channels=FEAT_BOARD_CH, cand_dim=FEAT_CAND,
                 max_cands=MAX_CANDS, cnn_ch=32, mlp_hidden=64, dropout=0.2):
        super().__init__()
        self.max_cands = max_cands
        # 棋盘编码：2 层卷积 + 全局均值池 → [B, cnn_ch*2]
        self.board_enc = nn.Sequential(
            nn.Conv2d(board_channels, cnn_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_ch, cnn_ch * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.board_pool = nn.AdaptiveAvgPool2d(1)
        # 候选清单编码：每候选 6 维 → [B, N, mlp_hidden]
        self.cand_enc = nn.Sequential(
            nn.Linear(cand_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 打分头：棋盘特征 + 候选特征 → 该候选分数
        self.head = nn.Sequential(
            nn.Linear(cnn_ch * 2 + mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, xb, xc, mask):
        """xb [B,8,15,15]；xc [B,N,6]；mask [B,N]（1=有效候选）。
        返回 [B,N] logits：有效候选为打分，padding 行≈-1e9。"""
        bf = self.board_pool(self.board_enc(xb)).flatten(1)          # [B, C]
        cf = self.cand_enc(xc)                                        # [B, N, H]
        bf = bf.unsqueeze(1).expand(-1, cf.size(1), -1)               # [B, N, C]
        logits = self.head(torch.cat([bf, cf], dim=-1)).squeeze(-1)   # [B, N]
        # padding 行压到极小：mask=0 → -1e9
        logits = logits + (mask - 1.0) * 1e9
        return logits

    def pick(self, xb, xc, mask):
        """返回 (候选索引, logits)：有效候选中分最高者的索引。"""
        with torch.no_grad():
            logits = self.forward(xb, xc, mask)
            idx = int(logits[0].argmax().item())
        return idx, logits
