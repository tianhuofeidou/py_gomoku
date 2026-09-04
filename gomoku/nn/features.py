# -*- coding: utf-8 -*-
"""分支特征编码：把深推分支的每一步上下文编码成 [T, 16] 矩阵。

每一步特征（16 维）：
  0      阵营（自家=1，对家=0）
  1      该步落子候选点权重
  2-9    当前局面全候选点权重（最多 8 个，不足补 0）
  10     是否触发必杀（1/0）
  11-15  攻防比 one-hot（0=必杀，1=1攻4防，2=2攻3防，3=3攻2防，4=4攻1防）

矩阵形状： [max_steps, 16]
"""
import numpy as np

FEAT_DIM = 16
DEFAULT_MAX_STEPS = 20
DEFAULT_MAX_CANDS = 8


def gear_onehot(gear):
    """攻防比 0~4 → 5 维 one-hot。"""
    g = int(round(gear))
    g = max(0, min(4, g))
    vec = [0.0] * 5
    vec[g] = 1.0
    return vec


def encode_step(ctx, max_cands=DEFAULT_MAX_CANDS):
    """一步上下文 → 16 维向量。

    ctx 字段：
      camp    : int, 自家=1 对家=0
      move_w  : float, 该步落子候选点权重
      cand_ws : list[float], 当前局面全候选点权重
      kill    : int, 是否触发必杀 1/0
      gear    : int, 攻防比 0~4（0=必杀）
    """
    row = [0.0] * FEAT_DIM
    row[0] = 1.0 if ctx.get('camp') else 0.0
    row[1] = float(ctx.get('move_w', 0.0))

    cands = list(ctx.get('cand_ws') or [])[:max_cands]
    for i, w in enumerate(cands):
        row[2 + i] = float(w)
    # 2~9 不足部分保持 0

    row[10] = 1.0 if ctx.get('kill') else 0.0

    gear_vec = gear_onehot(ctx.get('gear', 0))
    row[11:16] = gear_vec
    return row


def branch_matrix(ctxs, max_steps=DEFAULT_MAX_STEPS, max_cands=DEFAULT_MAX_CANDS):
    """步上下文列表 → [max_steps, 16] 矩阵 + 实际步数。

    返回 (matrix, length)。matrix 为 float32 ndarray，不足补 0。
    """
    steps = list(ctxs)[:max_steps]
    matrix = np.zeros((max_steps, FEAT_DIM), dtype=np.float32)
    for i, ctx in enumerate(steps):
        matrix[i] = encode_step(ctx, max_cands=max_cands)
    return matrix, len(steps)


def decode_matrix(matrix):
    """调试用：把矩阵行还原成可读字典列表。"""
    out = []
    for row in matrix:
        out.append({
            'camp': int(round(row[0])),
            'move_w': float(row[1]),
            'cand_ws': [float(x) for x in row[2:10]],
            'kill': int(round(row[10])),
            'gear': int(np.argmax(row[11:16])),
        })
    return out
