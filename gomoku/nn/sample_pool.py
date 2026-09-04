# -*- coding: utf-8 -*-
"""sample_pool.py —— 训练样本池（人机/自我对弈按人类胜率 R 混合，防过拟合）。

定稿机制：
  - 自我对弈样本是主体（海量，每批 80%+）；
  - 人机对弈样本少而珍贵：每批占比 ≤ HUMAN_MAX_SHARE（20%），且每个样本有
    「重复学习上限」——学满上限即退役，不再进 batch（防止少数据被无限重学而过拟合）；
  - 人类胜率 R 只调节人机样本的重复上限（不调节比例）：
      R > 0.6（人强，人类套路值钱）→ 上限 5 次
      0.4 ≤ R ≤ 0.6              → 上限 3 次
      R < 0.4（机强）            → 上限 1 次（一遍过，主力压自我对弈）
"""
import random

HUMAN_CAP_HIGH = 5   # R > 0.6
HUMAN_CAP_MID = 3    # 0.4 ~ 0.6
HUMAN_CAP_LOW = 1    # R < 0.4
HUMAN_MAX_SHARE = 0.2   # 人机样本每批占比上限


def human_cap_of(winrate):
    """人类胜率 → 人机样本重复学习上限"""
    if winrate > 0.6:
        return HUMAN_CAP_HIGH
    if winrate < 0.4:
        return HUMAN_CAP_LOW
    return HUMAN_CAP_MID


class SamplePool:
    """样本池：add(sample, src) 加入（src='human' 人机 / 'self' 自我对弈）；
    sample_batch(n) 按上限与比例混合采样，返回样本列表。"""

    def __init__(self, human_winrate=0.5, rng=None):
        self.human_winrate = human_winrate
        self.rng = rng or random.Random()
        self.human = []          # [sample, learn_count]
        self.self_samples = []

    def add(self, sample, src):
        if src == 'human':
            self.human.append([sample, 0])
        else:
            self.self_samples.append(sample)

    def add_many(self, samples, src):
        for s in samples:
            self.add(s, src)

    def sample_batch(self, n):
        """混合采样 n 个样本：人机 ≤20%（未达上限者）；不足由自我对弈补齐。
        返回样本列表（长度 ≤ n；两池皆空时为空）。"""
        cap = human_cap_of(self.human_winrate)
        active = [h for h in self.human if h[1] < cap]
        max_human = max(1, int(n * HUMAN_MAX_SHARE))
        n_human = min(len(active), max_human)
        human_pick = self.rng.sample(active, n_human) if active else []
        for h in human_pick:
            h[1] += 1                      # 命中一次 → 学习次数 +1
        n_self = n - n_human
        self_pick = []
        if self.self_samples:
            self_pick = self.rng.sample(self.self_samples, min(n_self, len(self.self_samples)))
            # 自我对弈池不足时允许重复补齐（自我样本无限生成，重复无害）
            while len(self_pick) < n_self:
                self_pick.append(self.rng.choice(self.self_samples))
        return [s for (s, _) in human_pick] + self_pick

    def retired_human_count(self):
        """已退役（学满上限）的人机样本数"""
        return sum(1 for h in self.human if h[1] >= human_cap_of(self.human_winrate))

    def stats(self):
        return {'human': len(self.human), 'self': len(self.self_samples),
                'human_retired': self.retired_human_count(),
                'winrate': self.human_winrate}
