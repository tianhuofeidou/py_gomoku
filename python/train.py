# ============================================================
# train.py —— 自对弈 + TD 学习
# 职责：生成训练数据并更新 evaluate 的权重
#
# 数据源：
#   - 自对弈（两个 AI 对下）
#   - 人类对局（engine.py 记录，质量高 → 样本加权）
#
# 训练方法：TD(0)
#   每步局面特征 → 网络评分 → 终局结果回传
#   w ← w + α × (目标 − 评分) × 特征
#   目标：越靠终局越接近真实结果（折扣 γ）
# ============================================================

from utils import SIZE, EMPTY, BLACK, WHITE

GAMMA = 0.9          # TD 折扣因子（待调）
HUMAN_WEIGHT = 5.0   # 人类对局样本加权
LEARNING_RATE = 0.01


class Trainer:
    """训练器：自对弈 + 对局回放 → 更新网络权重"""

    def __init__(self, evaluate, search, deep_search, pattern):
        self.evaluate = evaluate
        self.search = search
        self.deep_search = deep_search
        self.pattern = pattern

    # ---------- 对局生成 ----------

    def play_game(self, ai_black, ai_white, max_moves=225):
        """两个 AI 对下一局，返回 (moves, winner)"""
        # TODO: 交替调用 search/deep_search 落子，记录每步特征
        raise NotImplementedError

    def collect_selfplay(self, games=100):
        """批量自对弈 → 训练样本 [(特征, 终局结果), ...]"""
        # TODO: 循环 play_game，收集 (局面特征, 结果)
        raise NotImplementedError

    def collect_human_games(self, history):
        """人类对局 → 训练样本（加权 HUMAN_WEIGHT）"""
        # TODO: 从 engine 的历史对局提取
        raise NotImplementedError

    # ---------- 训练 ----------

    def td_update(self, trajectory, final_result):
        """TD(0) 更新：把终局结果沿轨迹回传
        trajectory: [(特征, 网络当时评分), ...] 从终局往前
        """
        # TODO: w ← w + α×(目标−评分)×特征，目标随折扣衰减
        raise NotImplementedError

    def train_loop(self, epochs, games_per_epoch):
        """主训练循环：自对弈 + 人类数据混合 → 更新权重"""
        # TODO: 主循环
        raise NotImplementedError

    def save_checkpoint(self, path):
        """保存训练进度"""
        # TODO: evaluate.save
        raise NotImplementedError
