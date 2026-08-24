# ============================================================
# evaluate.py —— 估值层（神经网络，学习核心）
# 职责：
#   小网络（粗筛）：225×10 父类矩阵 → 225 分 → top-K 候选点
#   大网络（精评）：候选点全量特征 → 三个洞🕳️🕳️🕳️ [进攻分, 防守分, 大局分]
#
# 实现选择：
#   - 起步：NumPy 手写 MLP（零依赖，可读可调）
#   - 后续：可换 PyTorch（GPU）
# 训练：TD 学习 + 自对弈 + 人类对局加权（train.py）
# ============================================================

import json
import os

try:
    import numpy as np
except ImportError:
    np = None

# 特征维度配置（待定方案的统一入口；子类数按 pattern.py 实际 152 计）
# - 720：全量 152 子类 × 4 方向 × 2 攻防 = 1216 → 压缩至 720（待定）
# - 240：4 方向 × 30 共享槽位 × 2 攻防（压缩）
# - 180：方向合并 152 子类 × 2 攻防 = 304 → 压缩至 180（待定）
FEATURE_SCHEME = '720'  # TODO: '720' | '240' | '180'


class SmallNet:
    """小网络：粗筛
    输入: 225 × 10 父类矩阵（扁平 2250 或 225×10）
    输出: 225 个分数 → top-K 候选点
    """

    def __init__(self, input_dim=2250, hidden=64, output=225):
        self.input_dim = input_dim
        self.hidden = hidden
        self.output = output
        # TODO: 权重初始化（棋理初值 or 随机）
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

    def forward(self, board_features):
        """board_features: 225×10 矩阵 → 225 分"""
        # TODO: 前向传播
        raise NotImplementedError

    def top_k(self, board_features, k=30):
        """返回 top-K 候选点 [(r, c), ...]"""
        # TODO: 取分最高的 k 个位置
        raise NotImplementedError

    def train_step(self, batch_x, batch_y, lr=0.01):
        """一步梯度更新（TD/监督）"""
        # TODO: 反向传播
        raise NotImplementedError


class EvalNet:
    """大网络：精评（三个洞🕳️🕳️🕳️）
    输入: 候选点全量特征（720/240/180 + 位置 + 全局）
    输出: [进攻分, 防守分, 大局分]
    """

    def __init__(self, input_dim=None, hidden=64, output=3):
        self.input_dim = input_dim or _feature_dim(FEATURE_SCHEME)
        self.hidden = hidden
        self.output = output
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

    def forward(self, point_features):
        """点特征 → [进攻分, 防守分, 大局分]"""
        # TODO: 前向传播（三个输出头）
        raise NotImplementedError

    def train_step(self, batch_x, batch_targets, lr=0.01):
        """TD 训练一步：targets 是三个洞的目标值"""
        # TODO: 反向传播
        raise NotImplementedError

    def save(self, path):
        """保存权重 JSON"""
        # TODO: 序列化权重
        raise NotImplementedError

    def load(self, path):
        """加载权重"""
        # TODO: 反序列化
        raise NotImplementedError


def _feature_dim(scheme):
    """特征维度换算"""
    if scheme == '720':
        return 720
    if scheme == '240':
        return 240
    if scheme == '180':
        return 180
    raise ValueError('unknown scheme: ' + scheme)


class Evaluate:
    """估值总入口：组合小网络 + 大网络 + 特征编码"""

    def __init__(self, pattern, small_net=None, eval_net=None):
        self.pattern = pattern
        self.small_net = small_net or SmallNet()
        self.eval_net = eval_net or EvalNet()

    def coarse_select(self, board, player, k=30):
        """无威胁粗筛：全盘父类矩阵 → 小网络 → top-K 候选点"""
        # TODO: 全盘 225 点父类特征（pattern 父类级）→ small_net.top_k
        raise NotImplementedError

    def refine(self, board, player, candidates):
        """精评：对候选点逐个算全量特征 → 大网络 → [攻, 防, 大局]"""
        # TODO: 每点 720 特征（含对方视角）→ eval_net.forward
        raise NotImplementedError

    def global_score(self, board, player):
        """无候选点时的整盘大局分（兜底）"""
        # TODO: 全局聚合特征 → eval_net
        raise NotImplementedError
