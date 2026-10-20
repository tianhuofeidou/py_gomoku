# ============================================================
# deep_search.py —— 深度推演层（温度银行）
# 职责：对候选点做多步递归推演，返回分支分最高的落子点
#
# 核心机制：
#   - 温度银行：每个根候选从 T0 开始，沿每条路径按候选权重扣温
#       DT = DT_M*((100-w)/100)^DT_POW + DT_N（权重高→耗温少）
#       每轮双方模拟落子后检查温度，耗尽则评估当前叶子
#   - 一层 = 本方+对方一个完整回合；候选与分数保留路径权重回溯，
#     胜败状态另按双方最优选择汇总，不将分数汇总误当成胜败证明
#   - 叶子评估：实际成五终局 ±1000；僵持叶子读当前棋盘两张状态表，
#     按成五/强必杀/弱必杀优先级判定，其余全盘加和走平滑累进税
#   - cache：按"已落子序列"缓存分支分，序列前缀复用（同序列只推一次）
#
# TODO（骨架保留，本次结构整理未实现）：
#   - 分支定界：UB/LB 剪枝（先深推一枝拿标杆，砍追不上的枝）
#   - 保底 3 枝：无条件开枝（防漏）
#
# 依赖：search（候选/应手来源）、utils（棋盘常量）
# ============================================================

import math
import random
import os

from .utils import SIZE, EMPTY, BLACK, WHITE


class DeepSearch:
    """深推器：温度银行驱动的递归推演。
    持有 Search 实例（提供 candidates/_fallback 取候选点），
    供 engine 的 compute_move / analyze_turn 调用。"""

    # 温度参数（结构：幂函数降温，参数交给 GA 寻优）
    # 本机实测（24 个 KataGo 中盘局面单步耗时）：80→avg 0.30s(最慢 1.2s)、
    # 160→avg 3.08s(最慢 17.2s)、240→avg 15.2s(最慢 97.1s)。
    # 更早用 3 个局面测过 320：复杂局面 90s+ 仍未完成 → 档位定稿 80/160/240。
    T0 = 160.0            # 路径初始温度（默认标准档，GUI 可切 80/160/240；不是全局节点额度）
    DT_M = 400.0           # 单层耗温系数 DT = DT_M*((100-w)/100)^DT_POW + DT_N
    DT_POW = 2.0           # 幂次：控制曲线形状（GA 可调）
    DT_N = 3.0             # 基础耗温
    # 深度收窄（越深越窄）：每层展开的候选数随深度递减——深度价值集中在
    # 最强势的几条线上（强制链），收窄宽度可指数级减少叶子数，基本不损失棋力：
    #   回合数(plies//2) >= NARROW_ROUNDS     → 候选 5 → 3
    #   回合数(plies//2) >= HARD_NARROW_ROUNDS → 候选 3 → 1
    NARROW_ROUNDS = 4
    HARD_NARROW_ROUNDS = 8
    CAND_N_FULL = 5
    CAND_N_MID = 3
    CAND_N_DEEP = 1
    WIN_SCORE = 1000.0     # 实际成五终局的基准分（区别于僵持叶子的成五点判定）
    # ------------------------------------------------------------
    # 叶子评估基础分：仅 5~16 号点型参与全盘加和。
    #   0-4 必杀点由必杀优先级单独判定，这里保留 0 占位；
    #   5-8 a1 冲四系，9-12 a2 活三系，13-16 b 潜力系，17 none=0。
    # 根决策方与对手使用两套分值，体现先手优势。
    MY_SUB_VALUE = (
        0.0, 0.0, 0.0, 0.0, 0.0,
        400.0, 350.0, 300.0, 250.0,
        350.0, 300.0, 220.0, 150.0,
        120.0, 90.0, 60.0, 30.0,
        0.0,
    )
    OPP_SUB_VALUE = (
        0.0, 0.0, 0.0, 0.0, 0.0,
        250.0, 200.0, 160.0, 120.0,
        200.0, 160.0, 120.0, 80.0,
        60.0, 45.0, 30.0, 15.0,
        0.0,
    )
    # 必杀优先级分值（当前决策方视角）：
    KILL_VALUE_WIN = 920.0           # 成五（0）
    KILL_VALUE_STRONG = 850.0        # 我方强必杀（1/2/3）
    KILL_VALUE_STRONG_OPP = -700.0   # 对方强必杀（已含 150 先手差）
    KILL_VALUE_WEAK = 650.0          # 我方弱必杀（4）
    KILL_VALUE_WEAK_OPP = -500.0     # 对方弱必杀（已含 150 先手差）
    KILL_VALUE_WIN_OPP = -920.0      # 对方成五（0）
    # 无必杀全盘和的平滑累进税参数：
    LEAF_TAX_R = 0.05                # 长期边际保留比例
    LEAF_TAX_S = 500.0               # 过渡尺度
    # 无必杀时全盘差 D 的叶子分：
    #   sign(D) * (r*|D| + (1-r)*S*(1 - exp(-|D|/S)))
    PRUNE_W = 0.02                 # 剪枝阈值：分支绝对路径权重低于此值不往下推（至少保留最强一支）
    GEAR_OPP_COEF = {1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9}  # 攻防档位系数：乘在对手全盘和上
    THREAT_BONUS_WHITE = 100.0   # （遗留常量，不再用于叶子分）
    THREAT_BONUS_BLACK = 50.0    # （遗留常量，不再用于叶子分）

    def __init__(self, search):
        self.search = search        # Search 实例：候选点来源
        self.use_nn = False         # 暂时停用所有神经网络：默认纯算法叶子评估（显式传参可开启）
        self.collect_samples = False  # 是否收集深推叶子分支样本
        self.samples = []             # 收集到的 (ctx_list, label)
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nn', 'value.pt')  # 叶子 NN 模型路径（可切换）
        self._nn_model = None         # 已加载的 NN 模型缓存
        self._nn_load_tried = False   # 加载失败后不在每个叶子重复尝试
        self._nn_unavailable_reason = None
        self._root_pass = None  # 仅本次固定上下文共享，返回后释放。
        self.reset_leaf_eval_stats()

    def reset_leaf_eval_stats(self):
        """重置本次决策的叶子评估统计，供界面展示实际使用路径。"""
        if self._root_pass is not None:
            return  # 筛选和补分累计本次实际叶子计算，不重复清零。
        self.leaf_eval_stats = {
            'nn': 0,
            'heuristic': 0,
            'total': 0,
            'reason': None,
        }

    def leaf_eval_status(self):
        """返回本次决策的叶子评估状态快照。"""
        out = dict(self.leaf_eval_stats)
        if out['nn'] and out['heuristic']:
            out['mode'] = 'mixed'
        elif out['nn']:
            out['mode'] = 'neural'
        elif out['heuristic']:
            out['mode'] = 'heuristic'
        else:
            out['mode'] = 'not_triggered'
        return out

    # ---------- 基础工具 ----------

    @staticmethod
    def _check_win(board, r, c, player):
        """胜负检测：落子 (r,c) 后是否成五（四方向数连续同色 ≥5）"""
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cnt = 1
            for s in (1, -1):
                rr, cc = r + dr * s, c + dc * s
                while 0 <= rr < SIZE and 0 <= cc < SIZE and board[rr][cc] == player:
                    cnt += 1
                    rr += dr * s
                    cc += dc * s
            if cnt >= 5:
                return True
        return False

    @staticmethod
    def _dt(w):
        """单层耗温：DT = DT_M*((100-w)/100)^DT_POW + DT_N（w 为候选权重 50~100）。
        权重高（强手）→ DT 小 → 温度剩多 → 推得深；权重低 → DT 大 → 浅推。
        幂函数让 100 与 80 差异小、80 与 60 差异明显，形状由 DT_POW 控制。"""
        x = max(100.0 - float(w), 0.0) / 100.0
        return DeepSearch.DT_M * (x ** DeepSearch.DT_POW) + DeepSearch.DT_N

    @staticmethod
    def _step_quality(w, cand_ws):
        """单步路径质量 = 该步候选分 / 当步最高候选分（走最优候选 → 1.0）。"""
        top = max(cand_ws) if cand_ws else 0.0
        if top <= 0:
            return 1.0
        return min(1.0, float(w) / top)

    # ---------- 终局种子与兼容统计（不参与分数计算） ----------
    # 实际成五是确定的终局事实；能否保证走到该终局由上层最优选择判定，
    # 不再要求此前每一手都为强制威胁。fatal 只保留为实验上下文特征。

    def _has_forcing_threat(self, player):
        """实验上下文的潜在威胁特征；空点棋型不能作为强制胜败证据。"""
        state = self.search.sb if player == BLACK else self.search.sw
        for row in state:
            for v in row:
                if v <= 12:
                    return True
        return False

    @staticmethod
    def _leaf_forced_state(ctx_branch, winner, main_player):
        """仅在已确认实际成五后调用，返回胜方视角的硬必胜。"""
        return 1

    @staticmethod
    def _fatal_ratio(n_win, n_lose, n_total):
        """旧版胜败比（兼容保留）：分母为全部子节点数（含未判定子节点），零时取 0。

        默认决策已切到 _pick_key_state；此字段仅随结果返回作兼容。"""
        if n_total <= 0:
            return 0.0
        return (n_win - n_lose) / float(n_total)

    @staticmethod
    def _pick_key(forced, ratio, score):
        """旧版排序键（兼容保留）：根必胜 > 正胜败比 > 候选分 > 根必败。

        默认决策已切到 _pick_key_state；此函数只留给旧调用方和旧测试。"""
        if forced == 1:
            return (3, 0.0, score)
        if forced == -1:
            return (0, 0.0, score)
        if ratio > 0:
            return (2, ratio, score)
        return (1, 0.0, score)

    @classmethod
    def _merge_fatal_states(cls, child_states):
        """同一行棋方选择候选：有硬胜就选胜，全部硬败才认败。

        输入均为该选择方视角；软状态必须按未确定处理，不能删掉分支。
        这是内部选择集合，不是父子落子节点间的状态翻转。
        """
        if not child_states:
            return 0
        hard_states = [cls._state_forced(s) for s in child_states]
        if any(s == 1 for s in hard_states):
            return 1
        if all(s == -1 for s in hard_states):
            return -1
        return 0

    # ---------- 5 态离散判定 ----------
    # 编码：+1 必胜 / +2 存在必胜 / 0 无 / -2 存在必败 / -1 必败。
    # 每个落子节点描述该步落子方。子节点由对方选择，孙节点又属本方。
    # 硬结论优先；其余只读取直接子/孙节点的硬状态，软状态不再上传。
    STATE_WIN = 1
    STATE_EXISTS_WIN = 2
    STATE_NONE = 0
    STATE_EXISTS_LOSE = -2
    STATE_LOSE = -1

    STATE_NAMES = {
        STATE_WIN: '必胜',
        STATE_EXISTS_WIN: '存在必胜',
        STATE_NONE: '无',
        STATE_EXISTS_LOSE: '存在必败',
        STATE_LOSE: '必败',
    }

    @classmethod
    def _merge_child_states(cls, child_states, grandchild_states=()):
        """由直接子/孙落子节点汇总当前落子方的五态。

        child_states 是对方各应手自身的状态；grandchild_states 是本方
        下一手各落点自身的状态，不是更深层任意败点，也不是软标签翻转。
        对方有硬胜应手即本方必败；对方全硬败才本方必胜。
        未定时：本方孙节点有硬败 → 自身失误风险；否则对方子节点有
        硬败 → 对手失误机会。两者同时出现时风险优先；软状态按 0 参与。
        """
        if not child_states:
            return cls.STATE_NONE
        if any(s == cls.STATE_WIN for s in child_states):
            return cls.STATE_LOSE
        if all(s == cls.STATE_LOSE for s in child_states):
            return cls.STATE_WIN
        if any(s == cls.STATE_LOSE for s in grandchild_states):
            return cls.STATE_EXISTS_LOSE
        if any(s == cls.STATE_LOSE for s in child_states):
            return cls.STATE_EXISTS_WIN
        return cls.STATE_NONE

    @staticmethod
    def _flip_state(state):
        """仅翻转硬状态；软证据局限于原落子节点，换层按未确定处理。"""
        return -state if state in (1, -1) else 0

    @classmethod
    def _state_forced(cls, state):
        """导出硬结论字段：只能取 -1 / 0 / +1。"""
        return state if state in (cls.STATE_WIN, cls.STATE_LOSE) else 0

    @classmethod
    def _state_child_state(cls, state):
        """导出软证据字段：none / exists_win / exists_lose。"""
        if state == cls.STATE_EXISTS_WIN:
            return 'exists_win'
        if state == cls.STATE_EXISTS_LOSE:
            return 'exists_lose'
        return 'none'

    @classmethod
    def _state_name(cls, state):
        return cls.STATE_NAMES.get(state, '无')

    @classmethod
    def _candidate_result(cls, r, c, score, state=0, fatal_ratio=0.0):
        """候选对外字段只由该实际落子节点的一个五态结论导出。"""
        return {'r': r, 'c': c, 'score': score, 'score_complete': True, 'state': state,
                'state_name': cls._state_name(state),
                'child_state': cls._state_child_state(state),
                'forced': cls._state_forced(state), 'fatal_ratio': fatal_ratio}

    # 决策排序档位：必胜 > 存在必胜 > 无 > 存在必败 > 必败。
    STATE_ORDER = {
        STATE_WIN: 4,
        STATE_EXISTS_WIN: 3,
        STATE_NONE: 2,
        STATE_EXISTS_LOSE: 1,
        STATE_LOSE: 0,
    }

    @classmethod
    def _pick_key_state(cls, state, score):
        """决策排序键：先比状态档位，同档比候选分。"""
        return (cls.STATE_ORDER.get(state, cls.STATE_ORDER[cls.STATE_NONE]),
                float('-inf') if score is None else score)

    @staticmethod
    def _weighted_average(items):
        """按路径权重累加（不归一化）：items = [(w, value), ...]。
        权重 = 路径质量连乘（全程最优 → 1.0），走差分支贡献按 W 缩水；
        深层 W 连乘趋近 0 → 贡献自然衰减，无需除以 ΣW。"""
        return sum(w * v for w, v in items)

    def _prune_branches(self, branches):
        """分支剪枝：branches = [(abs_weight, ...)]，绝对路径权重低于 PRUNE_W 剔除；
        至少保留权重最大的一支（防止整层被剪空）。深层权重 = 连乘衰减，
        因此到一定深度弱分支自然被剪，树宽收敛。"""
        if not branches:
            return []
        total = sum(b[0] for b in branches)
        if total <= 0:
            return branches
        kept = [b for b in branches if b[0] >= self.PRUNE_W]
        if not kept:
            kept = [max(branches, key=lambda b: b[0])]
        return kept

    def _sub_at(self, player, r, c):
        """当前状态表中 (r,c) 的 18 子类编号（该方若在此落子会形成什么）。"""
        state = self.search.sb if player == BLACK else self.search.sw
        return state[r][c]

    def _scan_leaf_tables(self, board, my_state, opp_state):
        """一次扫描双方状态表，返回 (我方档, 我方和, 对方档, 对方和)。

        必杀档：0=无、1=弱必杀(4)、2=强必杀(1/2/3)、3=成五(0)。
        只统计空点；5~16 号按双方基础分累加。
        我方成五直接返回；双方都已有必杀时只比档位，总分丢弃。
        """
        my_tier = opp_tier = 0
        my_total = opp_total = 0.0
        my_values = self.MY_SUB_VALUE
        opp_values = self.OPP_SUB_VALUE
        for r in range(SIZE):
            brow = board[r]
            mrow = my_state[r]
            orow = opp_state[r]
            for c in range(SIZE):
                if brow[c] != EMPTY:
                    continue
                v = mrow[c]
                if v == 0:
                    my_tier = 3
                elif v <= 3:
                    if my_tier < 2:
                        my_tier = 2
                elif v == 4:
                    if my_tier < 1:
                        my_tier = 1
                elif v <= 16:
                    my_total += my_values[v]
                v = orow[c]
                if v == 0:
                    opp_tier = 3
                elif v <= 3:
                    if opp_tier < 2:
                        opp_tier = 2
                elif v == 4:
                    if opp_tier < 1:
                        opp_tier = 1
                elif v <= 16:
                    opp_total += opp_values[v]
                if my_tier == 3:
                    # 我方成五优先级最高，不需要继续确认对手
                    return my_tier, 0.0, 0, 0.0
                if my_tier > 0 and opp_tier > 0:
                    # 双方都有必杀：只比档位，总分不再需要
                    return my_tier, 0.0, opp_tier, 0.0
        return my_tier, my_total, opp_tier, opp_total

    def _stale_result_score(self, board, player):
        """僵持叶子评估：读当前棋盘两张状态表，按目标设计判定。

        优先级（当前决策方视角，命中即停，只看有无、不数个数）：
          我方成五(0) → +920；对方成五(0) → -920；
          我方强必杀(1/2/3) → +850；对方强必杀(1/2/3) → -700；
          我方弱必杀(4) → +650；对方弱必杀(4) → -500；
          双方都无 0~4 → 全盘 5~16 基础分求和：
            D = 我方和 − 对方和 × 攻防系数
            叶子分 = sign(D) × [r|D| + (1−r)S(1−e^(−|D|/S))]
        不读历史 ctx 快照，也不使用候选名次权重。
        """
        my_state = self.search.sw if player == WHITE else self.search.sb
        opp_state = self.search.sb if player == WHITE else self.search.sw
        my_tier, my_total, opp_tier, opp_total = self._scan_leaf_tables(
            board, my_state, opp_state)
        if my_tier == 3:
            return self.KILL_VALUE_WIN
        if opp_tier == 3:
            return self.KILL_VALUE_WIN_OPP
        if my_tier == 2:
            return self.KILL_VALUE_STRONG
        if opp_tier == 2:
            return self.KILL_VALUE_STRONG_OPP
        if my_tier == 1:
            return self.KILL_VALUE_WEAK
        if opp_tier == 1:
            return self.KILL_VALUE_WEAK_OPP
        gear = self.search._target_gear(board, player)
        opp_total *= self.GEAR_OPP_COEF.get(gear, 1.0)
        diff = my_total - opp_total
        x = abs(diff)
        value = self.LEAF_TAX_R * x + (1.0 - self.LEAF_TAX_R) * self.LEAF_TAX_S * (
            1.0 - math.exp(-x / self.LEAF_TAX_S))
        return value if diff >= 0 else -value

    def _rollout(self, board, player, max_moves=10):
        """随机下到终局，返回 player 视角的胜负（1/-1/0）。
        用于给 stale 叶子生成训练标签。"""
        b = [row[:] for row in board]
        turn = player
        for _ in range(max_moves):
            empty = [(r, c) for r in range(SIZE) for c in range(SIZE) if b[r][c] == EMPTY]
            if not empty:
                return 0.0
            r, c = random.choice(empty)
            b[r][c] = turn
            if self._check_win(b, r, c, turn):
                return 1.0 if turn == player else -1.0
            turn = 3 - turn
        return 0.0

    def _leaf_score(self, board, branch, status, player, ctx_branch=None):
        context = self._root_pass
        if context is None or not context['memo_leaves']:
            return self._leaf_score_impl(board, branch, status, player, ctx_branch)
        key = (tuple(branch), status, player)
        if key not in context['leaves']:
            context['leaves'][key] = self._leaf_score_impl(
                board, branch, status, player, ctx_branch)
        return context['leaves'][key]

    def _leaf_score_impl(self, board, branch, status, player, ctx_branch=None):
        """叶子评估：以 player 视角计分。
        - 实际成五终局：player 成五 → +WIN_SCORE；对方成五 → -WIN_SCORE。
        - stale（温度耗尽僵持）：默认纯算法读当前棋盘两张状态表；
          神经网络路径属于实验能力，启用时仍使用旧的 ctx 分支特征。
        纯算法口径详见 _stale_result_score。"""
        if self.collect_samples and ctx_branch:
            # 收集原始样本：分支上下文 + 视角方 + 叶子状态；
            # 标签由对局结束后真实胜负回传（强化学习式）
            self.samples.append((list(ctx_branch), player, status))

        if status == 'win':
            return self.WIN_SCORE
        if status == 'lose':
            return -self.WIN_SCORE
        self.leaf_eval_stats['total'] += 1
        nn_score = self._nn_score(ctx_branch)
        if nn_score is not None:
            self.leaf_eval_stats['nn'] += 1
            # NN 输出约 [-1,1]，缩放到与胜负分(WIN_SCORE≈1000)同量级
            return nn_score * self.WIN_SCORE
        self.leaf_eval_stats['heuristic'] += 1
        if self.leaf_eval_stats['reason'] is None:
            self.leaf_eval_stats['reason'] = self._nn_unavailable_reason or (
                '已关闭' if not self.use_nn else '神经网络不可用')
        return self._stale_result_score(board, player)

    def _nn_score(self, ctx_branch):
        """用神经网络评估分支路线分数；模型不可用/未训练/被禁用时返回 None。"""
        if not self.use_nn:
            self._nn_unavailable_reason = '已关闭'
            return None
        if not ctx_branch:
            self._nn_unavailable_reason = '无分支特征'
            return None
        if self._nn_model is None:
            if self._nn_load_tried:
                return None
            self._nn_load_tried = True
            try:
                import torch
                from gomoku.nn.model import BranchValueNet
                model = BranchValueNet()
                model.load_state_dict(torch.load(
                    self.model_path, map_location='cpu', weights_only=True))
                model.eval()
                self._nn_model = model
                self._nn_unavailable_reason = None
            except ImportError:
                self._nn_unavailable_reason = '缺少 PyTorch/NumPy'
                return None
            except FileNotFoundError:
                self._nn_unavailable_reason = '模型文件不存在'
                return None
            except Exception:
                self._nn_unavailable_reason = '模型加载失败'
                return None
        try:
            import torch
            from gomoku.nn.features import branch_matrix
            matrix, length = branch_matrix(ctx_branch)
            x = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)   # [1,T,16]
            lengths = torch.tensor([length], dtype=torch.long)
            with torch.no_grad():
                score = self._nn_model(x, lengths).item()
            return score
        except Exception:
            self._nn_unavailable_reason = '模型推理失败'
            self._nn_model = None
            return None

    # ---------- 候选点展开（供推演层） ----------

    def _candidate_points(self, board, player, n=5, plies=0):
        """当前 player 的候选点（带权重），供推演层展开：
        与顶层共用强制攻防优先级；普通候选为 3攻2防，使用真实综合分。
        战术及普通候选均最多取 n 个（默认 5）。
        plies：当前分支已落子数（0=根层），用于「越深越窄」：
          回合数 = plies//2；>= NARROW_ROUNDS → 最多 CAND_N_MID 个；
          >= HARD_NARROW_ROUNDS → 最多 CAND_N_DEEP 个。
        返回 [(r, c, w), ...]。权重同时用于温度分配与叶子过程分。"""
        rounds = plies // 2
        if rounds >= self.HARD_NARROW_ROUNDS:
            n = min(n, self.CAND_N_DEEP)
        elif rounds >= self.NARROW_ROUNDS:
            n = min(n, self.CAND_N_MID)
        reason, tactical = self.search.tactical_candidates(board, player)
        if reason:
            # 保留现有候选顺序及权重，限制递归分支数量。
            return [(r, c, w) for (r, c), w in list(tactical.items())[:n]]
        # 兜底：3攻2防，视角与当前 player 对称（白攻+黑防 / 黑攻+白防）
        fb = self.search._fallback(board, player=player, with_score=True)
        # 权重 = _candidate_score 真实综合分（0~100），温度银行按此分配深度：
        # 强候选（落子成冲四/活三等先手点，分高）推得深，弱候选自动浅推。
        # clamp 到 [1,100]：超 100 会让 dt=(100-w)^2 回升，温度非单调。
        w_lst = [(r, c, min(max(float(w), 1.0), 100.0)) for (r, c, w) in fb[:n]]
        return w_lst

    def _make_ctx(self, board, player, main_player, w, cand_ws, cand_pts=None, temp=None, depth=None):
        """构造一步上下文（神经网络分支矩阵特征/叶子结果分的完整输入，8 字段）。
        cand_pts：与 cand_ws 对位的候选点 [(r,c,w)]（用于按状态表取 18 子类）。
        temp/depth：该步的剩余温度与已推层数。"""
        cand_ws = [float(x) for x in (cand_ws or [])]
        kill = any(cw >= 100 for cw in cand_ws)
        gear = 0 if kill else self.search._target_gear(board, player)
        return {
            'camp': 1 if player == main_player else 0,
            'move_w': float(w),
            'cand_ws': cand_ws,
            # 与 cand_ws 对位的候选 18 子类（状态表直查 O(1)，不重扫棋型）
            'cand_sub_ids': [self._sub_at(player, r, c) for (r, c, _w) in (cand_pts or [])],
            'temp': float(temp) if temp is not None else None,
            'depth': int(depth) if depth is not None else None,
            'kill': 1 if kill else 0,
            'gear': gear,
        }

    # ---------- 递归推演 ----------

    def _mark_state_cut(self):
        """早停只确定硬态，不能缓存未完成的累计分冒充完整分。"""
        context = self._root_pass
        context['partial_roots'].add(context['root'])
        if context['stack']:
            context['stack'][-1] = False

    def _recursive(self, board, player, opp, main_player, temp, branch, ctx_branch,
                   cache, trace=False, indent=0, path_w=1.0):
        context = self._root_pass
        args = (board, player, opp, main_player, temp, branch, ctx_branch, cache, trace, indent)
        if context is None or context['mode'] == 'full':
            return self._recursive_impl(*args, path_w=path_w)
        key = tuple((r, c) for r, c, _p, _w in branch)
        context['stack'].append(key not in context['incomplete'])
        try:
            value = self._recursive_impl(*args, path_w=path_w)
        except BaseException:
            context['stack'].pop()
            raise
        complete = context['stack'].pop()
        if complete:
            context['complete'][key] = value
        else:
            context['incomplete'].add(key)
            if context['stack']:
                context['stack'][-1] = False
        return value

    def _recursive_impl(self, board, player, opp, main_player, temp, branch, ctx_branch, cache, trace=False, indent=0, path_w=1.0):
        """递归推演：player 落一子（候选）→ opp 应一手 → 递归下一层。
        一层 = 双方完整回合；回合后温度耗尽则进行叶子评估。
        模拟落子后同步更新状态表（journal 哈希表回滚），保证候选点/必杀检测
        基于当前分支的真实棋盘，而不是进入深推前的过期状态表。
        ctx_branch 与 branch 同步记录每步上下文，供神经网络分支矩阵使用。
        cache 按落子序列复用，包含硬结论和直接候选落子节点的状态。
        trace=True 时按树形缩进输出推演内部（调试用）。
        内部返回 (分支分, 硬状态, n_win, n_lose, n_total, 硬状态, 直接候选状态)。
        返回值描述 player 从本层候选中最优选择的结果，两个硬状态字段同源；
        末项保留每个实际落子节点的五态，供上一轮读取本方孙节点的硬败风险。
        不通过软状态翻转传播证据；内部七项返回结构不变。
        """
        key = tuple((r, c) for (r, c, _p, _w) in branch)
        if key in cache:
            return cache[key]
        cands = self._candidate_points(board, player, 5, plies=len(branch))
        if not cands:
            cache[key] = (0.0, 0, 0, 0, 1, 0, ())
            return cache[key]
        cand_ws = [x[2] for x in cands]
        # 我方候选分支：绝对路径权重 = path_w × 本步质量（深层连乘衰减 → 自动剪弱枝）
        raw = []
        for (r, c, w) in cands:
            q = self._step_quality(w, cand_ws)
            raw.append((path_w * q, r, c, w))
        kept = self._prune_branches(raw)
        weighted = []
        cand_states = []                     # 各实际落子节点的硬状态（player 视角）
        new_cand_states = []                 # 各实际落子节点五态；仅直接候选硬败可作孙节点证据
        cand_wins = []                       # 子子树 main 视角必胜判定节点数
        cand_losses = []
        cand_totals = []                     # 子分支全部节点数（含未判定）
        for (q, r, c, w) in kept:
            if board[r][c] != EMPTY:
                continue
            ctx = self._make_ctx(board, player, main_player, w, cand_ws,
                                 cand_pts=cands, temp=temp, depth=len(branch))
            board[r][c] = player
            branch.append((r, c, player, w))
            ctx_branch.append(ctx)
            j_w = {}
            self.search.on_move(board, r, c, player, journal=j_w, record_history=False)
            if trace:
                print('  ' * indent + '+- %s落%s%d w=%g t=%g' % (
                    'W' if player == WHITE else 'B', 'ABCDEFGHIJKLMNO'[c], r + 1, w, temp))
            if self._check_win(board, r, c, player):
                ctx['fatal'] = 1          # 成五 = 必杀步（新字段）
                score = self._leaf_score(board, branch, 'win', player, ctx_branch)
                branch_state = self._leaf_forced_state(ctx_branch, player, main_player)   # player 视角
                new_branch_state = 1 if branch_state == 1 else 0
                cw = 1 if branch_state == 1 else 0
                cl = 0
                c_total = 1              # 该候选子节点本身
                if trace:
                    print('  ' * indent + '   -> WIN %g' % score)
            else:
                # 仅保留实验上下文特征，不作为胜败种子的门槛。
                ctx['fatal'] = 1 if self._has_forcing_threat(player) else 0
                # 黑方候选逻辑与白方完全一致：有 0 级先取 0 级，否则 fallback top5
                opp_pts = self._candidate_points(board, opp, 5, plies=len(branch))
                if not opp_pts:
                    score = self._leaf_score(board, branch, 'stale', player, ctx_branch)
                    branch_state = 0
                    new_branch_state = 0
                    cw = cl = 0
                    c_total = 1          # 该候选子节点本身（无应手）
                else:
                    opp_cand_ws = [x[2] for x in opp_pts]
                    # 对方应手分支：绝对路径权重 = path_w × q × oq（连乘衰减）
                    opp_raw = []
                    for (or_, oc, ow) in opp_pts:
                        oq = self._step_quality(ow, opp_cand_ws)
                        opp_raw.append((path_w * q * oq, or_, oc, ow))
                    opp_kept = self._prune_branches(opp_raw)
                    opp_weighted = []
                    opp_states = []          # 直接对方应手的硬状态（opp 视角）
                    grandchild_states = []   # 本方直接下一手的状态，不收集更深层败点
                    cw = 0
                    cl = 0
                    c_total = 0
                    for (oq, or_, oc, ow) in opp_kept:
                        if board[or_][oc] != EMPTY:
                            continue
                        temp2 = temp - self._dt(w) - self._dt(ow)
                        ctx_b = self._make_ctx(board, opp, main_player, ow, opp_cand_ws,
                                               cand_pts=opp_pts, temp=temp2, depth=len(branch))
                        board[or_][oc] = opp
                        branch.append((or_, oc, opp, ow))
                        ctx_branch.append(ctx_b)
                        j_b = {}
                        self.search.on_move(board, or_, oc, opp, journal=j_b, record_history=False)
                        opp_won = self._check_win(board, or_, oc, opp)
                        ctx_b['fatal'] = 1 if (opp_won or self._has_forcing_threat(opp)) else 0
                        if opp_won:
                            child = self._leaf_score(board, branch, 'lose', player, ctx_branch)
                            st = self._leaf_forced_state(ctx_branch, opp, main_player)  # opp 视角
                            opp_states.append(st)
                            cw += 0
                            cl += 1 if st == 1 else 0
                            c_total += 1      # 对方应手子节点
                            if trace:
                                print('  ' * indent + '   -> B应%s%d LOSE %g' % ('ABCDEFGHIJKLMNO'[oc], or_ + 1, child))
                        elif temp2 <= 0:
                            child = self._leaf_score(board, branch, 'stale', player, ctx_branch)
                            opp_states.append(0)
                            c_total += 1      # 对方应手子节点（未判定）
                            if trace:
                                print('  ' * indent + '   -> 温度尽(t=%g<dt=%g) leaf=%g' % (temp2 + self._dt(w), self._dt(w), child))
                        else:
                            child, st, cw2, cl2, ct2, new_st, next_states = self._recursive(board, player, opp, main_player, temp2, branch, ctx_branch, cache, trace, indent + 1, path_w=path_w * q * oq)
                            opp_states.append(-st)   # 递归返回 player 视角 → 投影到 opp 视角
                            grandchild_states.extend(next_states)
                            cw += cw2
                            cl += cl2
                            c_total += ct2        # 递归子节点及其子树
                        self.search.restore(j_b)
                        branch.pop()
                        ctx_branch.pop()
                        board[or_][oc] = EMPTY
                        opp_weighted.append((path_w * q * oq, child))
                        if self._root_pass is not None and self._root_pass['mode'] == 'proof' and opp_states[-1] == 1:
                            # 对方已有硬胜应手，本方该落子必败。
                            self._mark_state_cut()
                            break
                    score = self._weighted_average(opp_weighted)
                    # 对方最优应手决定硬胜败；局部机会/风险只取直接子/孙节点。
                    opp_layer = self._merge_fatal_states(opp_states)
                    # opp 应手层聚合判定计入"判定节点数"（统一 main 视角：opp 必胜=main 败）
                    if opp_layer == 1:
                        cl += 1
                    elif opp_layer == -1:
                        cw += 1
                    c_total += 1          # 该候选子节点本身（应手层聚合节点，含未判定）
                    new_branch_state = self._merge_child_states(opp_states, grandchild_states)
                    branch_state = self._state_forced(new_branch_state)
            self.search.restore(j_w)
            branch.pop()
            ctx_branch.pop()
            board[r][c] = EMPTY
            if trace:
                print('  ' * indent + '    branch=%g' % score)
            weighted.append((path_w * q, score))
            cand_states.append(branch_state)
            new_cand_states.append(new_branch_state)
            cand_wins.append(cw)
            cand_losses.append(cl)
            cand_totals.append(c_total)
            if self._root_pass is not None and self._root_pass['mode'] == 'proof' and branch_state == 1:
                # 同方选择层有硬胜可结束，单个可避开的硬败不能提前结束。
                self._mark_state_cut()
                break
        val = self._weighted_average(weighted)
        node_state = self._merge_fatal_states(cand_states)
        # 此处是 player 主动选落点，不能再次当作对方应手层合并。
        new_node_state = node_state
        n_win = sum(cand_wins) + (1 if node_state == 1 else 0)
        n_lose = sum(cand_losses) + (1 if node_state == -1 else 0)
        n_total = sum(cand_totals) + 1          # 当前节点本身
        cache[key] = (val, node_state, n_win, n_lose, n_total, new_node_state, tuple(new_cand_states))
        return cache[key]

    def deep_search(self, board, points, player, trace=False):
        """温度银行入口：逐个推演候选，返回状态优先、同档分数最高的 (r, c)。
        points 可为 {(r,c): score}（带权重）或 [(r,c), ...]（等权 60）。
        每候选：模拟落子 → 对方应手 → 温度检查 → 递归 → 分支分和五态分别回溯。
        全候选由 rank_candidates 排序，本方法取首位返回（向后兼容）。
        trace=True 时输出推演树（调试用）。"""
        ranked = self.rank_candidates(board, points, player, trace)
        if not ranked:
            return (SIZE // 2, SIZE // 2)
        return (ranked[0]['r'], ranked[0]['c'])

    def rank_candidates(self, board, points, player, trace=False, *, complete_scores=False):
        """硬态筛选后补齐非硬败候选评分；全硬败时全部补齐兜底。

        被淘汰且早停的硬败候选 score=None、score_complete=False；其余候选的
        分、根五态和兼容统计保持完整搜索含义。网络/采样及 complete_scores
        沿原完整路径。所有派生结果仅在本次固定棋盘、温度、历史中复用。
        """
        if complete_scores or self.use_nn or self.collect_samples:
            return self._rank_candidates_pass(board, points, player, trace)
        self.reset_leaf_eval_stats()
        context = {'mode': 'full', 'proof': {}, 'complete': {}, 'leaves': {},
                   'incomplete': set(), 'partial_roots': set(), 'stack': [],
                   'root': None, 'memo_leaves': False, 't0': self.T0}
        self._root_pass = context
        try:
            candidates = list(points) if isinstance(points, (dict, list)) else []
            rows = []
            has_nonlosing = False
            for point in candidates:
                point = tuple(point)
                # 尚无非硬败替代时必须完整评分。全败局面因此沿原完整路径，
                # 不先剪一遍再重算；已完成的早期硬败分也无需丢弃。
                context['mode'] = 'proof' if has_nonlosing else 'full'
                context['memo_leaves'] = has_nonlosing
                needed = {point: points[point]} if isinstance(points, dict) else [point]
                if trace and has_nonlosing:
                    print('=== 状态筛选：中间累计分尚未补齐 ===')
                current = self._rank_candidates_pass(board, needed, player, trace)
                if not current:
                    continue
                row = current[0]
                if point in context['partial_roots']:
                    if row['state'] == self.STATE_LOSE:
                        # 已有非硬败替代，此硬败不再用于选点；不展示部分累计分。
                        row['score'] = None
                        row['score_complete'] = False
                    else:
                        # 软态均补齐直接子/孙证据，且同档比较必须使用完整原始分。
                        context['mode'] = 'full'
                        if trace:
                            print('=== 补齐参与比较的根候选评分 ===')
                        row = self._rank_candidates_pass(board, needed, player, trace)[0]
                rows.append(row)
                has_nonlosing = has_nonlosing or row['state'] != self.STATE_LOSE
                # 不同根前缀没有可复用的完整落子序列；派生缓存仅跨本根两阶段。
                for name in ('proof', 'complete', 'leaves', 'incomplete', 'partial_roots'):
                    context[name].clear()
            rows.sort(key=lambda row: self._pick_key_state(row['state'], row['score']), reverse=True)
            return rows
        finally:
            self._root_pass = None

    def _rank_candidates_pass(self, board, points, player, trace=False):
        """对全部候选逐个深推，按五态优先、同档分支分降序返回完整列表
        [{'r': r, 'c': c, 'score': val, ...}, ...]；中间累计分由外层判定是否需要补齐。
        已占格跳过（不打分）；空候选返回 []。
        每条结果额外带新 5 态字段 state/state_name/child_state；
        排序规则：状态档位优先，同档按候选分降序。
        旧 forced/fatal_ratio 仅作兼容字段保留，不再参与排序。
        供 analyze_turn 输出全部候选及分数（决策网络输入层的数据源）。"""
        self.reset_leaf_eval_stats()
        if isinstance(points, dict):
            cands = list(points.keys())
        elif isinstance(points, list):
            cands = points
        else:
            cands = []
        if not cands:
            return []
        context = self._root_pass
        t0 = self.T0 if context is None else context['t0']
        if trace:
            print('=== rank_candidates turn=%s T0=%g cands=%d ===' % ('W' if player == WHITE else 'B', t0, len(cands)))
        cache = (context['proof'] if context['mode'] == 'proof' else context['complete']) if context is not None else {}
        opp = 3 - player
        main_player = player
        results = []
        for (r, c) in cands:
            if context is not None:
                context['root'] = (r, c)
            if board[r][c] != EMPTY:
                continue
            w = points.get((r, c), 60.0) if isinstance(points, dict) else 60.0
            root_cands = self._candidate_points(board, player, 5)
            cand_ws = [x[2] for x in root_cands]
            ctx = self._make_ctx(board, player, main_player, w, cand_ws,
                                 cand_pts=root_cands, temp=t0, depth=0)
            board[r][c] = player
            branch = [(r, c, player, w)]
            ctx_branch = [ctx]
            j_w = {}
            self.search.on_move(board, r, c, player, journal=j_w, record_history=False)
            if trace:
                print('== cand: %s%d w=%g' % ('ABCDEFGHIJKLMNO'[c], r + 1, w))
            if self._check_win(board, r, c, player):
                ctx['fatal'] = 1          # 成五 = 必杀步（新字段）
                val = self._leaf_score(board, branch, 'win', player, ctx_branch)
                root_state = self._leaf_forced_state(ctx_branch, player, main_player)   # player 视角
                new_root_state = 1 if root_state == 1 else 0
                cw, cl = (1, 0) if root_state == 1 else (0, 0)
                c_total = 1              # 该候选根子节点本身
                if trace:
                    print('   -> direct WIN %g' % val)
            else:
                # 仅保留实验上下文特征，不作为胜败种子的门槛。
                ctx['fatal'] = 1 if self._has_forcing_threat(player) else 0
                # 黑方候选逻辑与白方完全一致：有 0 级先取 0 级，否则 fallback top5
                opp_pts = self._candidate_points(board, opp, 5, plies=len(branch))
                if not opp_pts:
                    val = self._leaf_score(board, branch, 'stale', player, ctx_branch)
                    root_state = 0
                    new_root_state = 0
                    cw = cl = 0
                    c_total = 1          # 该候选根子节点本身（无应手）
                else:
                    opp_cand_ws = [x[2] for x in opp_pts]
                    # 对方应手分支：按路径质量权重剪枝 + 加权求和（替代 min）
                    opp_raw = []
                    for (or_, oc, ow) in opp_pts:
                        oq = self._step_quality(ow, opp_cand_ws)
                        opp_raw.append((oq, or_, oc, ow))
                    opp_kept = self._prune_branches(opp_raw)
                    opp_weighted = []
                    opp_states = []          # 根层直接对方应手的硬状态（opp 视角）
                    grandchild_states = []   # 根候选方下一手各落点的状态
                    cw = 0
                    cl = 0
                    c_total = 0
                    for (oq, or_, oc, ow) in opp_kept:
                        if board[or_][oc] != EMPTY:
                            continue
                        temp2 = t0 - self._dt(w) - self._dt(ow)
                        ctx_b = self._make_ctx(board, opp, main_player, ow, opp_cand_ws,
                                               cand_pts=opp_pts, temp=temp2, depth=len(branch))
                        board[or_][oc] = opp
                        branch.append((or_, oc, opp, ow))
                        ctx_branch.append(ctx_b)
                        j_b = {}
                        self.search.on_move(board, or_, oc, opp, journal=j_b, record_history=False)
                        opp_won = self._check_win(board, or_, oc, opp)
                        ctx_b['fatal'] = 1 if (opp_won or self._has_forcing_threat(opp)) else 0
                        if opp_won:
                            child = self._leaf_score(board, branch, 'lose', player, ctx_branch)
                            st = self._leaf_forced_state(ctx_branch, opp, main_player)
                            opp_states.append(st)
                            cl += 1 if st == 1 else 0      # opp 必胜 → main 必败节点
                            c_total += 1                  # 对方应手子节点
                            if trace:
                                print('   -> B resp %s%d LOSE %g' % ('ABCDEFGHIJKLMNO'[oc], or_ + 1, child))
                        elif temp2 <= 0:
                            child = self._leaf_score(board, branch, 'stale', player, ctx_branch)
                            opp_states.append(0)
                            c_total += 1                  # 对方应手子节点（未判定）
                        else:
                            child, st, cw2, cl2, ct2, new_st, next_states = self._recursive(board, player, opp, main_player, temp2, branch, ctx_branch, cache, trace, 1)
                            opp_states.append(-st)   # 递归返回 player 视角 → 投影到 opp 视角
                            grandchild_states.extend(next_states)
                            cw += cw2
                            cl += cl2
                            c_total += ct2              # 递归子节点及其子树
                        self.search.restore(j_b)
                        branch.pop()
                        ctx_branch.pop()
                        board[or_][oc] = EMPTY
                        opp_weighted.append((oq, child))
                        if context is not None and context['mode'] == 'proof' and opp_states[-1] == 1:
                            self._mark_state_cut()
                            break
                    val = self._weighted_average(opp_weighted)
                    opp_layer = self._merge_fatal_states(opp_states)
                    if opp_layer == 1:
                        cl += 1              # opp 层聚合：opp 必胜 → main 必败节点
                    elif opp_layer == -1:
                        cw += 1              # opp 层聚合：opp 必败 → main 必胜节点
                    c_total += 1              # 该候选根子节点本身（应手层聚合节点，含未判定）
                    new_root_state = self._merge_child_states(opp_states, grandchild_states)
                    root_state = self._state_forced(new_root_state)
            self.search.restore(j_w)
            branch.pop()
            ctx_branch.pop()
            board[r][c] = EMPTY
            # 必胜/必败只取候选根节点的聚合结论；根节点本身不重复计入比例。
            # 未定时保留兼容统计，但比例不参与排序。
            # 分母为全部子节点数（含未判定），不是仅胜败子节点数。
            if root_state == 0:
                ratio = self._fatal_ratio(cw, cl, c_total)
            else:
                ratio = 0.0
            if trace:
                print('   cand%s%d val=%g' % ('ABCDEFGHIJKLMNO'[c], r + 1, val))
            # forced/fatal_ratio 为兼容字段，决策排序不再使用。
            results.append(self._candidate_result(r, c, val, new_root_state, ratio))
        results.sort(key=lambda x: self._pick_key_state(x.get('state'), x.get('score', 0.0)),
                     reverse=True)
        if trace:
            top = results[0] if results else None
            if top:
                print('=== pick: %s%d val=%g ===' % ('ABCDEFGHIJKLMNO'[top['c']], top['r'] + 1, top['score']))
            else:
                print('=== no candidates ===')
        return results
