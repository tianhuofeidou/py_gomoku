# -*- coding: utf-8 -*-
"""memory_lookup.py —— 决策网络的历史记忆查询（读全局记忆文件）。

数据源：DSH_HOME/storages/gomoku/global-memory.json（与 Node 侧 memory.js 同一文件）：
  badLines  = [{opp:[{r,c}], ai:[{r,c}], killType, winLine}, ...]  败局棋谱
  goodLines = [{opp:[...], ai:[...]}, ...]                          胜局棋谱

查询语义（与 Node symDualPrefixMatch 一致）：
  当前局面着法（对手前缀 + 我方前缀）与候选点组成「我方前缀+候选」，
  在 8 种对称变换下匹配历史棋谱前缀；命中 badLines → 该候选历史败绩 +1，
  命中 goodLines → 历史胜绩 +1。网络据此学习「这步在实战中灵不灵」。
"""
import json
import os
from os.path import join, dirname, expanduser

from gomoku import paths
from gomoku.nn.self_memory import default_self_memory_file

# 15×15 棋盘 8 种对称变换（4 旋转 × 2 镜像），与 Node memory.js SYMMS 一致
SYMMS = [
    lambda r, c: (r, c),
    lambda r, c: (c, 14 - r),
    lambda r, c: (14 - r, 14 - c),
    lambda r, c: (14 - c, r),
    lambda r, c: (r, 14 - c),
    lambda r, c: (14 - r, c),
    lambda r, c: (c, r),
    lambda r, c: (14 - c, 14 - r),
]


def default_memory_file():
    return paths.global_memory_file()


class MemoryLookup:
    """全局记忆只读查询（人机 + 自我对弈两套）。
    文件缺失/损坏时静默空载（返回全 0，不阻塞决策）。"""

    def __init__(self, path=None, self_path=None):
        self.bad_lines = []
        self.good_lines = []
        self.self_bad_lines = []
        self.self_good_lines = []
        path = path or default_memory_file()
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                if isinstance(data.get('badLines'), list):
                    self.bad_lines = data['badLines']
                if isinstance(data.get('goodLines'), list):
                    self.good_lines = data['goodLines']
        except Exception:
            pass
        self_path = self_path or default_self_memory_file()
        try:
            with open(self_path, encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                if isinstance(data.get('selfBadLines'), list):
                    self.self_bad_lines = data['selfBadLines']
                if isinstance(data.get('selfGoodLines'), list):
                    self.self_good_lines = data['selfGoodLines']
        except Exception:
            pass

    # ---------- 内部：对称前缀匹配 ----------

    @staticmethod
    def _to_pairs(arr):
        """历史棋谱条目 opp/ai（[{r,c},...]）→ [(r,c),...]"""
        return [(int(x['r']), int(x['c'])) for x in arr] if arr else []

    def _hits(self, cur_opp, cur_ai, lines):
        """历史 lines 中与「当前对手前缀 + 我方前缀（含候选）」对称匹配的条数。"""
        cnt = 0
        for ln in lines:
            ho = self._to_pairs(ln.get('opp'))
            ha = self._to_pairs(ln.get('ai'))
            if len(ho) < len(cur_opp) or len(ha) < len(cur_ai):
                continue
            for f in SYMMS:
                to = [f(r, c) for (r, c) in cur_opp]
                ta = [f(r, c) for (r, c) in cur_ai]
                if to == ho[:len(to)] and ta == ha[:len(ta)]:
                    cnt += 1
                    break
        return cnt

    # ---------- 对外接口 ----------

    def batch_stats(self, moves, player, candidates):
        """对每个候选点返回两套历史战绩：
        人机（badLines/goodLines）与自我对弈（selfBadLines/selfGoodLines）各自
        的 (赢, 输) 次数。moves: [(r,c,player)] 当前局；player: 当前落子方；
        candidates: [{'r','c','score'}, ...]（ranked）。
        返回 (wins, losses, self_wins, self_losses) 四个等长列表。"""
        opp = 3 - player
        opp_moves = [(m[0], m[1]) for m in moves if m[2] == opp]
        ai_moves = [(m[0], m[1]) for m in moves if m[2] == player]
        wins, losses = [], []
        self_wins, self_losses = [], []
        for cand in candidates:
            cur_ai = ai_moves + [(cand['r'], cand['c'])]
            wins.append(self._hits(opp_moves, cur_ai, self.good_lines))
            losses.append(self._hits(opp_moves, cur_ai, self.bad_lines))
            self_wins.append(self._hits(opp_moves, cur_ai, self.self_good_lines))
            self_losses.append(self._hits(opp_moves, cur_ai, self.self_bad_lines))
        return wins, losses, self_wins, self_losses
