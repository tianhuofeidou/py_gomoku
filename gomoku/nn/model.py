# -*- coding: utf-8 -*-
"""分支价值网络：输入 [B, T, 16] 分支矩阵，输出该路线分数（标量）。

结构：
  输入 [B,T,16]
    → GRU(hidden=64) 处理序列
    → 取最后有效步输出
    → Linear(64→1)
"""
import torch
import torch.nn as nn

from gomoku.nn.features import FEAT_DIM


class BranchValueNet(nn.Module):
    def __init__(self, input_dim=FEAT_DIM, hidden=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, lengths=None):
        """x: [B, T, 16]；lengths: [B] 实际步数（用于取最后有效步）。

        返回 [B, 1] 分数。
        """
        out, _ = self.gru(x)  # out: [B, T, hidden]
        if lengths is not None:
            batch = x.size(0)
            idx = (lengths - 1).clamp(min=0).long()
            last = out[torch.arange(batch), idx]  # [B, hidden]
        else:
            last = out[:, -1, :]
        return self.head(last)
