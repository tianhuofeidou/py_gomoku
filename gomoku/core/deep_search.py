# ============================================================
# deep_search.py —— 深度推演层（温度银行）
# 职责：对候选点做多步递归推演，返回分支分最高的落子点
#
# 核心机制：
#   - 温度银行：每手决策一个温度总预算 T0，每层推演按候选权重扣温
#       DT = DT_M*((100-w)/100)^DT_POW + DT_N（权重高→耗温少）
#       每轮双方模拟落子后检查温度，耗尽则评估当前叶子
#   - 一层 = 白+黑一个完整回合；黑应手展开 BLACK_CAND_N 个候选，
#     取对白方最不利的分支（因为候选评分是半成品，单点最凶会漏杀）
#   - 叶子评估：成五 ±1000×过程权重；僵持 → 过程分（快杀/干净杀得分高）
#   - cache：按"已落子序列"缓存分支分，序列前缀复用（同序列只推一次）
#
# TODO（骨架保留，本次结构整理未实现）：
#   - 分支定界：UB/LB 剪枝（先深推一枝拿标杆，砍追不上的枝）
#   - 保底 3 枝：无条件开枝（防漏）
#
# 依赖：search（候选/应手来源）、utils（棋盘常量）
# ============================================================

import random
import os

from .utils import SIZE, EMPTY, BLACK, WHITE


class DeepSearch:
    """深推器：温度银行驱动的递归推演。
    持有 Search 实例（提供 candidates/_fallback 取候选点），
    供 engine 的 compute_move / analyze_turn 调用。"""

    # 温度参数（结构：幂函数降温，参数交给 GA 寻优）
    # 本机实测（3 个中盘局面单步耗时）：80→avg 0.11s(最慢 0.13s)、160→avg 0.47s(最慢 1.3s)、
    # 240→avg 10s(复杂局面 18~29s)。320 在复杂局面 90s+ 仍未完成 → 档位定稿 80/160/240。
    T0 = 160.0            # 初始温度（每手决策总预算，默认=标准档，GUI 可切 80/160/240）
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
    WIN_SCORE = 1000.0     # 成五基准分
    # ------------------------------------------------------------
    # 18 棋型结果分（无胜负 stale 叶值）——纯终局判定，不看路径。
    # 索引 = 18 子类编号 0-17（search.SUBCLASS_NAMES 顺序）：
    #   0-4 必杀：five/live4/d44/d34/d33
    #   5-8 a1 眠四系，9-12 a2 活三系，13-16 b 潜力系，17 none
    # 排名递减权重：候选榜第 0 个是第 1 名 → 满权重；第 2 名 0.5；……
    RANK_W = (1.0, 0.5, 0.3, 0.2, 0.1)
    # 我方候选点（我下一手用它）：子类威胁等级分。
    # 第一档（0-4）——叶子处轮到我（先手），我任意一个 kill 级点即必胜/必胜结构：
    #   five +900（直接成五）> live4 +880（活四不可解）> d44 +750 > d34 +700 > d33 +650
    MY_SUB_VALUE = (
        900.0, 880.0, 750.0, 700.0, 650.0,
        400.0, 350.0, 300.0, 250.0,
        350.0, 300.0, 220.0, 150.0,
        120.0, 90.0, 60.0, 30.0,
        0.0,
    )
    # 对方候选点（对方下一手用它）：我是先手可抢先占/堵 → 单杀点按"逼应"级计，
    # 而不是-1000；仅当对方 kill 级点 ≥2（双杀/活四成型）才无解。
    OPP_SUB_VALUE = (
        400.0, 380.0, 450.0, 430.0, 400.0,
        250.0, 200.0, 160.0, 120.0,
        200.0, 160.0, 120.0, 80.0,
        60.0, 45.0, 30.0, 15.0,
        0.0,
    )
    OPP_KILL_UNSOLVABLE = -920.0   # 对方 kill 级点 ≥2 → 无解（防不住）
    STALE_CLAMP = 500.0            # 常规威胁对比的 clamp（第一档大分不纳入）
    PRUNE_W = 0.02                 # 剪枝阈值：分支绝对路径权重低于此值不往下推（至少保留最强一支）
    GEAR_OPP_COEF = {1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9}  # 攻防档位：偏防放大对方威胁
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
        self.reset_leaf_eval_stats()

    def reset_leaf_eval_stats(self):
        """重置本次决策的叶子评估统计，供界面展示实际使用路径。"""
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

    # ---------- 必杀链识别（VCF/VCT，旁路：不参与打分） ----------
    # 状态：+1 = 该节点决策方必胜；-1 = 该节点决策方必败；0 = 不确定。
    # 判定种子 = 胜负叶子 + 胜方全程"必应威胁"步：
    #   每步落子后把"该步是否必杀（成五或形成必应威胁）"写入 ctx['fatal']（新字段 1/0）；
    #   叶子直接读该字段判断"胜方全程必杀"（胜方每步 fatal 均为 1）。

    def _has_forcing_threat(self, player):
        """落子后：player 是否存在'对方必应'的威胁点（子类 0-4 必杀/5-8 冲四/9-12 活三）。"""
        state = self.search.sb if player == BLACK else self.search.sw
        for row in state:
            for v in row:
                if v <= 12:
                    return True
        return False

    @staticmethod
    def _leaf_forced_state(ctx_branch, winner, main_player):
        """叶子强制链判定（用新字段 fatal）：胜方路径上每一步都是必杀步 → +1；否则 0。
        （普通胜负叶子——非全程必杀——不作为传播种子。）"""
        steps = [c.get('fatal', 0) for c in ctx_branch
                 if (c.get('camp') == 1) == (winner == main_player)]
        if steps and all(steps):
            return 1
        return 0

    @staticmethod
    def _pick_key(forced, ratio, score):
        """决策排序键：根必胜 > 正胜败比 > 候选分 > 根必败。"""
        if forced == 1:
            return (3, 0.0, score)
        if forced == -1:
            return (0, 0.0, score)
        if ratio > 0:
            return (2, ratio, score)
        return (1, 0.0, score)

    @staticmethod
    def _merge_fatal_states(child_states):
        """合并同层子分支状态（已投影为当前决策方视角）：
        全部必败 → 当前节点必败；任一必胜 → 当前节点必胜；否则不确定。"""
        if not child_states:
            return 0
        if any(s == 1 for s in child_states):
            return 1
        if all(s == -1 for s in child_states):
            return -1
        return 0

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

    def _stale_result_score(self, ctx_seq, player):
        """18 棋型结果分（纯终局判定，输入 = NN 路径上下文，不重扫盘面）：
        从 ctx_seq（每步含 camp / cand_sub_ids）取【最近的】我方一步与对方一步
        的候选子类，作为叶子局面的双方威胁结构：
          - 我方最近 camp=1 步的 cand_sub_ids → 我方候选子类（排名序）；
          - 对方最近 camp=0 步的 cand_sub_ids → 对方候选子类（排名序）。
        - 我方任意 kill 级点（0-4）→ 先手必胜/必胜结构 → +650~+900；
        - 对方 kill 级点 ≥2 → 双杀/活四成型，防不住 → -920；
        - 对方 kill 级点 ==1 → 我先手可占掉，按"逼应"级计入常规对比；
        - 无 kill → 常规威胁对比（SUB_VALUE × RANK_W 排名加权 × gear），clamp ±500。"""
        my_subs = []
        opp_subs = []
        for ctx in reversed(ctx_seq):
            subs = list(ctx.get('cand_sub_ids') or [])
            if ctx.get('camp') == 1:
                if not my_subs:
                    my_subs = subs
            else:
                if not opp_subs:
                    opp_subs = subs
        my_kill = [s for s in my_subs if s <= 4]
        if my_kill:
            return max(self.MY_SUB_VALUE[s] for s in my_kill)
        opp_kill = [s for s in opp_subs if s <= 4]
        if len(opp_kill) >= 2:
            return self.OPP_KILL_UNSOLVABLE
        my_v = sum(self.MY_SUB_VALUE[s] * self.RANK_W[k] for k, s in enumerate(my_subs[:5]))
        opp_v = sum(self.OPP_SUB_VALUE[s] * self.RANK_W[k] for k, s in enumerate(opp_subs[:5]))
        gear = ctx_seq[-1].get('gear', 3) if ctx_seq else 3
        opp_v *= self.GEAR_OPP_COEF.get(gear, 1.0)
        v = my_v - opp_v
        return max(-self.STALE_CLAMP, min(self.STALE_CLAMP, v))

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
        """叶子评估：以 player 视角计分。
        - player 成五（win）→ +WIN_SCORE + WA×我方威胁 - WB×对方威胁
        - 对方成五（lose）→ -WIN_SCORE - WB×对方威胁 + WA×我方威胁
        - stale（温度尽僵持）→ 优先神经网络评估；不可用则 _stale_score
        我方威胁权重高于对方；长链/连续威胁不再被稀释，黑白对称。"""
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
        return self._stale_result_score(ctx_branch, player)

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

    def _recursive(self, board, player, opp, main_player, temp, branch, ctx_branch, cache, trace=False, indent=0, path_w=1.0):
        """递归推演：player 落一子（候选）→ opp 应一手 → 递归下一层。
        一层 = 双方完整回合；回合后温度耗尽则进行叶子评估。
        模拟落子后同步更新状态表（journal 哈希表回滚），保证候选点/必杀检测
        基于当前分支的真实棋盘，而不是进入深推前的过期状态表。
        ctx_branch 与 branch 同步记录每步上下文，供神经网络分支矩阵使用。
        cache: {落子序列: (评估分, 必杀链状态)} 前缀复用。
        trace=True 时按树形缩进输出推演内部（调试用）。
        返回 (分支评估分, forced_state)：forced_state 为该节点决策方(player)视角的
        必杀链必然性：+1 必胜 / -1 必败 / 0 不确定（旁路识别，不参与打分）。"""
        key = tuple((r, c) for (r, c, _p, _w) in branch)
        if key in cache:
            return cache[key]
        cands = self._candidate_points(board, player, 5, plies=len(branch))
        if not cands:
            cache[key] = (0.0, 0, 0, 0)
            return (0.0, 0, 0, 0)
        cand_ws = [x[2] for x in cands]
        # 我方候选分支：绝对路径权重 = path_w × 本步质量（深层连乘衰减 → 自动剪弱枝）
        raw = []
        for (r, c, w) in cands:
            q = self._step_quality(w, cand_ws)
            raw.append((path_w * q, r, c, w))
        kept = self._prune_branches(raw)
        weighted = []
        cand_states = []                     # 子分支状态（player 视角，必杀链传播）
        cand_wins = []                       # 子子树 main 视角必胜判定节点数
        cand_losses = []
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
                cw = 1 if branch_state == 1 else 0
                cl = 0
                if trace:
                    print('  ' * indent + '   -> WIN %g' % score)
            else:
                # 落子后未成五：该步是否"必应威胁步"写入 fatal 字段
                ctx['fatal'] = 1 if self._has_forcing_threat(player) else 0
                # 黑方候选逻辑与白方完全一致：有 0 级先取 0 级，否则 fallback top5
                opp_pts = self._candidate_points(board, opp, 5, plies=len(branch))
                if not opp_pts:
                    score = self._leaf_score(board, branch, 'stale', player, ctx_branch)
                    branch_state = 0
                    cw = cl = 0
                else:
                    opp_cand_ws = [x[2] for x in opp_pts]
                    # 对方应手分支：绝对路径权重 = path_w × q × oq（连乘衰减）
                    opp_raw = []
                    for (or_, oc, ow) in opp_pts:
                        oq = self._step_quality(ow, opp_cand_ws)
                        opp_raw.append((path_w * q * oq, or_, oc, ow))
                    opp_kept = self._prune_branches(opp_raw)
                    opp_weighted = []
                    opp_states = []          # 该候选下 opp 应手层（opp 视角状态）
                    cw = 0
                    cl = 0
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
                        ctx_b['fatal'] = 1 if (opp_won or self._has_forcing_threat(opp)) else 0   # 新字段
                        if opp_won:
                            child = self._leaf_score(board, branch, 'lose', player, ctx_branch)
                            st = self._leaf_forced_state(ctx_branch, opp, main_player)  # opp 视角
                            opp_states.append(st)
                            cw += 0
                            cl += 1 if st == 1 else 0
                            if trace:
                                print('  ' * indent + '   -> B应%s%d LOSE %g' % ('ABCDEFGHIJKLMNO'[oc], or_ + 1, child))
                        elif temp2 <= 0:
                            child = self._leaf_score(board, branch, 'stale', player, ctx_branch)
                            opp_states.append(0)
                            if trace:
                                print('  ' * indent + '   -> 温度尽(t=%g<dt=%g) leaf=%g' % (temp2 + self._dt(w), self._dt(w), child))
                        else:
                            child, st, cw2, cl2 = self._recursive(board, player, opp, main_player, temp2, branch, ctx_branch, cache, trace, indent + 1, path_w=path_w * q * oq)
                            opp_states.append(-st)   # 递归返回 player 视角 → 投影到 opp 视角
                            cw += cw2
                            cl += cl2
                        self.search.restore(j_b)
                        branch.pop()
                        ctx_branch.pop()
                        board[or_][oc] = EMPTY
                        opp_weighted.append((path_w * q * oq, child))
                    score = self._weighted_average(opp_weighted)
                    # 对方必杀链证据：opp 层全败 → opp 必败（=player 必胜）；任一 opp 必胜 → player 必败
                    opp_layer = self._merge_fatal_states(opp_states)
                    # opp 应手层聚合判定计入"判定节点数"（统一 main 视角：opp 必胜=main 败）
                    if opp_layer == 1:
                        cl += 1
                    elif opp_layer == -1:
                        cw += 1
                    branch_state = -opp_layer
            self.search.restore(j_w)
            branch.pop()
            ctx_branch.pop()
            board[r][c] = EMPTY
            if trace:
                print('  ' * indent + '    branch=%g' % score)
            weighted.append((path_w * q, score))
            cand_states.append(branch_state)
            cand_wins.append(cw)
            cand_losses.append(cl)
        val = self._weighted_average(weighted)
        node_state = self._merge_fatal_states(cand_states)
        n_win = sum(cand_wins) + (1 if node_state == 1 else 0)
        n_lose = sum(cand_losses) + (1 if node_state == -1 else 0)
        cache[key] = (val, node_state, n_win, n_lose)
        return val, node_state, n_win, n_lose

    def deep_search(self, board, points, player, trace=False):
        """温度银行深度推演入口：对候选点逐一开始推演，返回分支分最高的 (r, c)。
        points 可为 {(r,c): score}（带权重）或 [(r,c), ...]（等权 60）。
        每候选：模拟落子 → 黑最优应手 → 温度检查 → 递归 → 分支分回溯取最高。
        全候选的分数由 rank_candidates 计算，本方法取最高者返回（向后兼容）。
        trace=True 时输出推演树（调试用）。"""
        ranked = self.rank_candidates(board, points, player, trace)
        if not ranked:
            return (SIZE // 2, SIZE // 2)
        return (ranked[0]['r'], ranked[0]['c'])

    def rank_candidates(self, board, points, player, trace=False):
        """对全部候选点逐个深推打分，按分支分降序返回完整列表
        [{'r': r, 'c': c, 'score': val}, ...]——全返回、不截断、分数直接暴露原始值。
        已占格跳过（不打分）；空候选返回 []。
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
        if trace:
            print('=== rank_candidates turn=%s T0=%g cands=%d ===' % ('W' if player == WHITE else 'B', self.T0, len(cands)))
        cache = {}
        opp = 3 - player
        main_player = player
        results = []
        for (r, c) in cands:
            if board[r][c] != EMPTY:
                continue
            w = points.get((r, c), 60.0) if isinstance(points, dict) else 60.0
            root_cands = self._candidate_points(board, player, 5)
            cand_ws = [x[2] for x in root_cands]
            ctx = self._make_ctx(board, player, main_player, w, cand_ws,
                                 cand_pts=root_cands, temp=self.T0, depth=0)
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
                cw, cl = (1, 0) if root_state == 1 else (0, 0)
                if trace:
                    print('   -> direct WIN %g' % val)
            else:
                # 落子后未成五：该步是否"必应威胁步"写入 fatal 字段
                ctx['fatal'] = 1 if self._has_forcing_threat(player) else 0
                # 黑方候选逻辑与白方完全一致：有 0 级先取 0 级，否则 fallback top5
                opp_pts = self._candidate_points(board, opp, 5, plies=len(branch))
                if not opp_pts:
                    val = self._leaf_score(board, branch, 'stale', player, ctx_branch)
                    root_state = 0
                    cw = cl = 0
                else:
                    opp_cand_ws = [x[2] for x in opp_pts]
                    # 对方应手分支：按路径质量权重剪枝 + 加权求和（替代 min）
                    opp_raw = []
                    for (or_, oc, ow) in opp_pts:
                        oq = self._step_quality(ow, opp_cand_ws)
                        opp_raw.append((oq, or_, oc, ow))
                    opp_kept = self._prune_branches(opp_raw)
                    opp_weighted = []
                    opp_states = []          # 根层对方应手（opp 视角状态，必杀链传播）
                    cw = 0
                    cl = 0
                    for (oq, or_, oc, ow) in opp_kept:
                        if board[or_][oc] != EMPTY:
                            continue
                        temp2 = self.T0 - self._dt(w) - self._dt(ow)
                        ctx_b = self._make_ctx(board, opp, main_player, ow, opp_cand_ws,
                                               cand_pts=opp_pts, temp=temp2, depth=len(branch))
                        board[or_][oc] = opp
                        branch.append((or_, oc, opp, ow))
                        ctx_branch.append(ctx_b)
                        j_b = {}
                        self.search.on_move(board, or_, oc, opp, journal=j_b, record_history=False)
                        opp_won = self._check_win(board, or_, oc, opp)
                        ctx_b['fatal'] = 1 if (opp_won or self._has_forcing_threat(opp)) else 0   # 新字段
                        if opp_won:
                            child = self._leaf_score(board, branch, 'lose', player, ctx_branch)
                            st = self._leaf_forced_state(ctx_branch, opp, main_player)
                            opp_states.append(st)
                            cl += 1 if st == 1 else 0      # opp 必胜 → main 必败节点
                            if trace:
                                print('   -> B resp %s%d LOSE %g' % ('ABCDEFGHIJKLMNO'[oc], or_ + 1, child))
                        elif temp2 <= 0:
                            child = self._leaf_score(board, branch, 'stale', player, ctx_branch)
                            opp_states.append(0)
                        else:
                            child, st, cw2, cl2 = self._recursive(board, player, opp, main_player, temp2, branch, ctx_branch, cache, trace, 1)
                            opp_states.append(-st)   # 递归返回 player 视角 → 投影到 opp 视角
                            cw += cw2
                            cl += cl2
                        self.search.restore(j_b)
                        branch.pop()
                        ctx_branch.pop()
                        board[or_][oc] = EMPTY
                        opp_weighted.append((oq, child))
                    val = self._weighted_average(opp_weighted)
                    opp_layer = self._merge_fatal_states(opp_states)
                    if opp_layer == 1:
                        cl += 1              # opp 层聚合：opp 必胜 → main 必败节点
                    elif opp_layer == -1:
                        cw += 1              # opp 层聚合：opp 必败 → main 必胜节点
                    root_state = -opp_layer
            self.search.restore(j_w)
            branch.pop()
            ctx_branch.pop()
            board[r][c] = EMPTY
            # 必胜/必败只取候选根节点的聚合结论；根节点本身不重复计入比例。
            # 只有根节点尚未定论时，才用子节点的胜败统计辅助排序。
            if root_state == 0:
                tot = cw + cl
                ratio = (cw - cl) / float(tot) if tot else 0.0
            else:
                ratio = 0.0
            if trace:
                print('   cand%s%d val=%g' % ('ABCDEFGHIJKLMNO'[c], r + 1, val))
            results.append({'r': r, 'c': c, 'score': val, 'forced': root_state,
                            'fatal_ratio': ratio})
        results.sort(key=lambda x: self._pick_key(x['forced'], x['fatal_ratio'], x['score']),
                     reverse=True)
        if trace:
            top = results[0] if results else None
            if top:
                print('=== pick: %s%d val=%g ===' % ('ABCDEFGHIJKLMNO'[top['c']], top['r'] + 1, top['score']))
            else:
                print('=== no candidates ===')
        return results
