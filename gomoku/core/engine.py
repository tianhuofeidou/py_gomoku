# ============================================================
# engine.py —— 对外接口层（决策编排 + JSON 协议）
# 职责：组装全部底层（pattern / search / deep_search），
#       对外提供决策接口：
#         - on_move():          落子后更新状态表（委托 Search）
#         - compute_move():     轮到白（AI）时选点（原 ai.decide 逻辑）
#         - analyze_turn():     4 部分分析（对方威胁/己方威胁/候选点/深推结果）
#         - describe():         黑白威胁分布概览
#
# 协议（stdin/stdout 或子进程参数，TODO 未实现）：
#   请求:  {"type": "move", "board": [[...]], "player": 1, "last_move": [r,c]}
#   响应:  {"row": 7, "col": 8, "score": 0.83, "reason": "..."}
#
# 用法:
#   python engine.py < request.json          # 单次（TODO）
#   python engine.py --serve                 # 常驻 stdin/stdout 循环（TODO）
# ============================================================

import os
import sys
import json

from .utils import SIZE, EMPTY, BLACK, WHITE
from .pattern import PatternAnalyzer
from .search import Search, SUBCLASS_NAMES, SUBCLASS_PARENT
from .deep_search import DeepSearch


class Engine:
    """引擎：组装全部底层，负责决策编排（AI 执白，对手执黑）。
    可选「决策网络」（决策大脑）：use_decision_net=True 且模型可用时，
    analyze_turn 出全候选后由网络打分选点（结合棋盘棋感 + 实战记忆）；
    网络未启用/未加载/异常 → 自动回退纯算法第一名，两套系统互不干扰。"""

    def __init__(self, decision_net=None, use_decision_net=False):
        self.pattern = PatternAnalyzer()
        self.search = Search(self.pattern)        # 状态表 + 候选 + 评分兜底
        self.deep_search = DeepSearch(self.search)  # 温度银行深推
        # 决策网络（可选）：None 或加载失败 → 纯算法回退
        self.decision_net = decision_net
        self.use_decision_net = use_decision_net
        self._decision_model_tried = decision_net is not None
        self._memory = None                       # MemoryLookup 懒加载缓存
        self.moves = []                           # 真实对局着法 [(r,c,player)]（记忆匹配用）

    # ---------- 状态维护 ----------

    def on_move(self, board, r, c, player):
        """任意方落子后：委托 Search 做 3 段增量状态更新（黑白状态表），
        并记录真实着法序列（记忆匹配用；深推模拟走 search.on_move，不经过这里）。"""
        self.search.on_move(board, r, c, player)
        self.moves.append((r, c, player))

    # ---------- 核心动作：选点 ----------

    def compute_move(self, board, player=WHITE):
        """AI（执白）决策入口，检测优先级【命中即止】（命中第一个就返回，不再看后面）：
        0. 开局（盘面 0/1 子）→ 返回固定候选列表给 agent（不深度推演）
        1. 白五连 w_five → 落子即胜 → 直接深度推演该点（必然 WIN）
        2. 黑五连 b_five → 必须堵   → 直接深度推演堵点
        3. 白VCF  w_vcf  → 活四/双眠四/眠四+活三，落子即无解杀 → 直接深度推演
        4. 黑VCF  b_vcf  → 对方有连杀（含活四/44/43）：对每个黑必杀点反推堵点
                           （_defense_candidates：黑落点中心100 + 各线端80，
                             多杀型并集取最高分）→ 深度推演这些防守点
        5. 白双三 w_d33  → 己方双三必杀 → 直接深度推演
        6. 黑双三 b_d33  → 对方双三，同上反推堵点 → 深度推演
        兜底（全无）     → 3攻2防 5 点 → 深度推演
        返回：深度推演选出的最佳 (r, c)；开局则返回候选列表。
        """
        opening = self._opening_candidates(board)
        if opening is not None:
            return opening     # 开局：候选列表直接给 agent
        c = self.search.candidates(board)
        if c['w_five']:
            return self.deep_search.deep_search(board, c['w_five'], WHITE)
        if c['b_five']:
            return self.deep_search.deep_search(board, c['b_five'], WHITE)
        if c['w_vcf']:
            return self.deep_search.deep_search(board, c['w_vcf'], WHITE)
        if c['b_vcf']:
            # 黑VCF：对每个黑必杀点反推防守候选（必杀点100/反推80）
            pts = {}
            for (r, col) in c['b_vcf']:
                for (rr, cc), score in self.search._defense_candidates(board, r, col).items():
                    if (rr, cc) not in pts or score > pts[(rr, cc)]:
                        pts[(rr, cc)] = score
            return self.deep_search.deep_search(board, pts, WHITE)
        if c['w_d33']:
            return self.deep_search.deep_search(board, c['w_d33'], WHITE)
        if c['b_d33']:
            # 黑双三：必杀点反推堵法
            pts = {}
            for (r, col) in c['b_d33']:
                for (rr, cc), score in self.search._defense_candidates(board, r, col).items():
                    if (rr, cc) not in pts or score > pts[(rr, cc)]:
                        pts[(rr, cc)] = score
            return self.deep_search.deep_search(board, pts, WHITE)
        # 无必杀兜底：AI 执白第一层候选按局势动态切换 4/3/2/1 攻防配比
        return self.deep_search.deep_search(
            board,
            self.search._fallback(board, gear=self.search._choose_gear(board)),
            WHITE,
        )

    def _opening_candidates(self, board):
        """开局候选（固定逻辑，不进深度推演，返回候选【列表】给 agent 自行选）：
        - 盘面 0 子 → [(7,7)]：天元（坐标 0 起，即 H8）
        - 盘面 1 子 → 天元 (7,7) 的一格邻域 8 点中仍为空者（列表）
                      （AI 执白：无论黑第一手落哪，第二手都围绕天元应）
        - 盘面 ≥2 子 → None（进入正常检测/深度推演流程）
        记忆权重预留：后续对第二手 8 点的选择偏好（如贴黑子方向）由此处接入。"""
        cnt = 0
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] != EMPTY:
                    cnt += 1
        if cnt == 0:
            return [(7, 7)]
        if cnt == 1:
            pts = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = 7 + dr, 7 + dc
                    if board[rr][cc] == EMPTY:
                        pts.append((rr, cc))
            return pts
        return None

    # ---------- 分析接口（4 部分输出） ----------

    def _threat_stats(self, board, state):
        """按父类统计某视角状态表的威胁分布（只统计空位，已占格不参与）：
        - state：要统计的状态表（黑表 sb 或白表 sw）
        - 返回 {父类: [(r, c, 子类名), ...]}，父类 4 种：
            0=必杀(KILL) / 1=威胁(THREAT) / 2=潜力(POTEN) / 3=无用(USELESS)
        - 用途：analyze_turn 第 1/2 部分输出，向 agent 展示双方场上威胁全貌"""
        stats = {0: [], 1: [], 2: [], 3: []}
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] != EMPTY:
                    continue
                parent = SUBCLASS_PARENT[state[r][c]]
                stats[parent].append((r, c, SUBCLASS_NAMES[state[r][c]]))
        return stats

    def analyze_turn(self, board, player):
        """轮到 player 时的完整分析，输出 4 部分（agent 决策界面）：
        1. part1_opp_threat：对方视角威胁分布（0必杀/1威胁/2潜力/3无用，各含点位+子类名）
        2. part2_my_threat ：己方视角威胁分布（同上）
        3. part3_candidates：初始候选点，按检测优先级【命中即止】取第一个命中的类型：
             my-five→direct(落子即胜)   / opp-five→direct(直接堵)
             my-vcf(活四/44/43)→direct  / opp-vcf→defense(反推堵点,带权重dict)
             my-d33→direct              / opp-d33→defense(反推堵点,带权重dict)
             全无→fallback(3攻2防列表)
           points 形态：direct/fallback 为 [(r,c),...] 列表；
                        defense 为 {(r,c): 权重} 字典（权重供温度分配/排序用）
        4. part4_result：{'type': 'direct'|'searched', 'move': (r,c), 'ranked': [...]}
             direct（唯一候选）→ 不深推，ranked 给超大分 1e9；
             searched（候选 ≥2）→ 全部候选逐个深推打分，ranked 全返回。
        """
        opp = 3 - player
        mine = self.search.candidates(board)
        opp_state = self.search.sw if player == BLACK else self.search.sb
        my_state = self.search.sb if player == BLACK else self.search.sw
        result = {
            'turn': 'black' if player == BLACK else 'white',
            'part1_opp_threat': self._threat_stats(board, opp_state),
            'part2_my_threat': self._threat_stats(board, my_state),
        }
        # 3. 候选点（初始，类型标记）——只看 0 必杀级
        if player == BLACK:
            my_five, opp_five = mine['b_five'], mine['w_five']
            my_vcf, opp_vcf = mine['b_vcf'], mine['w_vcf']
            my_d33, opp_d33 = mine['b_d33'], mine['w_d33']
        else:
            my_five, opp_five = mine['w_five'], mine['b_five']
            my_vcf, opp_vcf = mine['w_vcf'], mine['b_vcf']
            my_d33, opp_d33 = mine['w_d33'], mine['b_d33']
        cand = None
        if my_five:
            cand = {'type': 'direct', 'reason': 'my-five', 'points': [(r, c) for (r, c) in my_five]}
        elif opp_five:
            cand = {'type': 'direct', 'reason': 'opp-five', 'points': [(r, c) for (r, c) in opp_five]}
        elif my_vcf:
            cand = {'type': 'direct', 'reason': 'my-vcf', 'points': [(r, c) for (r, c) in my_vcf]}
        elif opp_vcf:
            # 对方 VCF（活四/44/43）：对每个对方必杀点反推堵点（中心100 + 端80）
            pts = {}
            for (r, col) in opp_vcf:
                for (rr, cc), s in self.search._defense_candidates(board, r, col, is_kill=True, player=opp).items():
                    if (rr, cc) not in pts or s > pts[(rr, cc)]:
                        pts[(rr, cc)] = s
            cand = {'type': 'defense', 'reason': 'opp-vcf', 'points': pts}
        elif my_d33:
            cand = {'type': 'direct', 'reason': 'my-d33', 'points': [(r, c) for (r, c) in my_d33]}
        elif opp_d33:
            # 对方双三：必杀点加搜索（对每个 d33 点反推堵法）
            pts = {}
            for (r, col) in opp_d33:
                for (rr, cc), s in self.search._defense_candidates(board, r, col, is_kill=True, player=opp).items():
                    if (rr, cc) not in pts or s > pts[(rr, cc)]:
                        pts[(rr, cc)] = s
            cand = {'type': 'defense', 'reason': 'opp-d33', 'points': pts}
        else:
            # 机机对战中黑白双方都可用动态配比；
            # 深推内部仍用固定 3攻2防，保证搜索树稳定。
            gear = self.search._choose_gear(board, player)
            # fallback 候选带 _candidate_score 真实分（dict 形态），
            # rank_candidates 按分分配温度：强候选（冲四/活三成型点）推深。
            # 叠加记忆经验修正（move_memory_bonus）：历史赢谱落点加分、
            # 败谱落点减分——记忆参与候选打分（8 对称归一化，见 memory.py）。
            mem_bonus = self._memory_bonus()
            pts = {}
            for (rr, cc), s in [(p[:2], p[2]) for p in
                                self.search._fallback(board, gear=gear, player=player, with_score=True)]:
                if (rr, cc) not in pts or s > pts[(rr, cc)]:
                    pts[(rr, cc)] = min(max(float(s) + mem_bonus.get((rr, cc), 0.0), 1.0), 100.0)
            cand = {'type': 'fallback', 'reason': 'none', 'points': pts}
        result['part3_candidates'] = cand
        # 4. 深度推演结果
        #   绝对必杀（my-five/opp-five，落子即胜/必堵）→ 直取第一，不深推；
        #   VCF/双三/defense/fallback → 深推逐个打分（ranked 全返回）。
        #   注意两处重构修复：
        #   a) defense 候选必须保留 dict 权重（100/80 驱动防守点推深）——9947a96
        #      曾转 list 等权 60 → dt(60)=67>T0 → 防守深推浅化、乱选（黑弱根因）；
        #   b) VCF/双三候选（列表无权重）补 100 权重——等权 60 同样推不动，
        #      浅推会漏杀/犹豫；深推验证后再落子。
        pts = cand['points']
        if cand['type'] == 'direct' and cand.get('reason') in ('my-five', 'opp-five'):
            first = pts[0] if isinstance(pts, list) else next(iter(pts))
            result['part4_result'] = {
                'type': 'direct',
                'move': first,
                'ranked': [{'r': first[0], 'c': first[1], 'score': 1e9}],
            }
        else:
            if cand['type'] == 'direct' and isinstance(pts, list):
                # my-vcf / my-d33：列表候选补高权重，深推验证真杀
                pts = {p: 100.0 for p in pts}
            if isinstance(pts, dict) and len(pts) == 1:
                (r0, c0) = next(iter(pts))
                result['part4_result'] = {
                    'type': 'direct',
                    'move': (r0, c0),
                    'ranked': [{'r': r0, 'c': c0, 'score': 1e9}],
                }
            else:
                ranked = self.deep_search.rank_candidates(board, pts, player)
                best = ranked[0] if ranked else None
                result['part4_result'] = {
                    'type': 'searched',
                    'move': (best['r'], best['c']) if best else None,
                    'ranked': ranked,
                }
        # 决策网络选点（可选）：覆盖 move，ranked 全候选保持原样（供大模型/网络输入）
        if result['part4_result'].get('ranked'):
            mv = self._decision_pick(board, player, result['part4_result']['ranked'])
            result['part4_result']['move'] = mv
        return result

    # ---------- 决策网络（决策大脑，可选） ----------

    def _decision_pick(self, board, player, ranked):
        """决策网络选点：唯一候选直取；多候选时网络打分选最高。
        网络未启用/未加载/异常 → 回退纯算法第一名（两套系统并行，互不干扰）。"""
        if not ranked:
            return None
        if len(ranked) == 1:
            return (ranked[0]['r'], ranked[0]['c'])
        if not self.use_decision_net:
            return (ranked[0]['r'], ranked[0]['c'])
        try:
            net = self._ensure_decision_net()
            if net is None:
                return (ranked[0]['r'], ranked[0]['c'])
            mem = self._ensure_memory()
            if mem is not None:
                wins, losses, self_wins, self_losses = mem.batch_stats(self.moves, player, ranked)
            else:
                wins = losses = self_wins = self_losses = [0] * len(ranked)
            import torch
            from gomoku.nn.decision_input import encode_decision
            xb, xc, mask = encode_decision(board, self.search, ranked, wins, losses, self_wins, self_losses)
            dev = next(net.parameters()).device          # 跟随网络所在设备（cuda/cpu）
            xb = torch.from_numpy(xb).to(dev)
            xc = torch.from_numpy(xc).to(dev)
            mask = torch.from_numpy(mask).to(dev)
            idx = net.pick(xb, xc, mask)[0]
            pt = ranked[idx]
            return (pt['r'], pt['c'])
        except Exception:
            return (ranked[0]['r'], ranked[0]['c'])

    def _ensure_decision_net(self):
        """懒加载决策网络权重 gomoku/nn/decision.pt；不存在/失败 → None（回退纯算法）。"""
        if self._decision_model_tried:
            return self.decision_net
        self._decision_model_tried = True
        try:
            import torch
            from gomoku.nn.decision_net import DecisionNet
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), 'gomoku', 'nn', 'decision.pt')
            net = DecisionNet()
            net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            net.eval()
            self.decision_net = net
        except Exception:
            self.decision_net = None
        return self.decision_net

    def _ensure_memory(self):
        """懒加载全局记忆查询；文件缺失/失败 → None（候选历史战绩全 0）。"""
        if self._memory is None:
            try:
                from gomoku.nn.memory_lookup import MemoryLookup
                self._memory = MemoryLookup()
                if not self._memory.bad_lines and not self._memory.good_lines:
                    self._memory = False
            except Exception:
                self._memory = False
        return self._memory if self._memory else None

    def _memory_bonus(self):
        """历史经验修正表（memory.move_memory_bonus）：赢谱落点加分/败谱落点减分。
        memory.global_memory 为进程内缓存（对局结束写入后自动最新），
        每步重建成本 <1ms（遍历棋谱 ×8 对称变换），无需自缓存。"""
        try:
            from gomoku import memory as _mem
            return _mem.move_memory_bonus()
        except Exception:
            return {}

    # ---------- 状态描述 ----------

    def describe(self, board):
        """当前棋盘的黑白威胁分布概览（委托 Search）"""
        return self.search.describe(board)

    # ---------- JSON 协议 ----------

    def handle_request(self, req):
        """处理单条请求，返回响应 dict"""
        # TODO: 解析 req → compute_move → 响应
        raise NotImplementedError

    def serve_stdin(self):
        """常驻模式：循环读 stdin 一行 JSON，写 stdout 一行 JSON"""
        # TODO: 循环
        raise NotImplementedError


def main():
    """CLI 入口：
    - 无参数: 读 stdin 单条 JSON → 输出响应
    - --serve: 常驻循环
    - --weights <path>: 加载权重
    """
    # TODO: 参数解析 + 分发
    raise NotImplementedError


if __name__ == '__main__':
    main()
