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
    # 温度参数（结构：幂函数降温，参数交给 GA 寻优）
    # 分数驱动后实测：160→~0.7s/手、240→1.4s(最慢4.3s)、320→2.9s(最慢7.6s)、
    # 400→前5局面平均9s+(最慢16s)、480→平均16s+(最慢34s)——320 为可用上限。
    # GUI 档位（play.py TEMP_LEVELS）：轻快160 / 标准240 / 认真320。
    T0 = 320.0            # 初始温度（每手决策总预算，默认=认真档）
    DT_M = 400.0           # 单层耗温系数 DT = DT_M*((100-w)/100)^DT_POW + DT_N
    DT_POW = 2.0           # 幂次：控制曲线形状（GA 可调）
    DT_N = 3.0             # 基础耗温
    WIN_SCORE = 1000.0     # 成五基准分
    THREAT_BONUS_WHITE = 100.0   # 白方威胁步奖励（白威胁权重要高于黑）
    THREAT_BONUS_BLACK = 50.0    # 黑方威胁步惩罚

    def __init__(self, search):
        self.search = search        # Search 实例：候选点来源
        self.use_nn = True          # 是否使用神经网络叶子评估（eval 时可按实例关闭）
        self.collect_samples = False  # 是否收集深推叶子分支样本
        self.samples = []             # 收集到的 (ctx_list, label)
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nn', 'value.pt')  # 叶子 NN 模型路径（可切换）
        self._nn_model = None         # 已加载的 NN 模型缓存

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
    def _threat_counts(branch):
        """分开统计黑白双方的高威胁步数（w≥90：0 级必杀/1 级威胁）。
        返回 (white_threats, black_threats)。"""
        wt = sum(1 for (_r, _c, p, w) in branch if p == WHITE and w >= 90)
        bt = sum(1 for (_r, _c, p, w) in branch if p == BLACK and w >= 90)
        return wt, bt

    @staticmethod
    def _stale_score(branch, player):
        """僵持过程分：以 player 视角计分。
        我方威胁步贡献 THREAT_BONUS_WHITE，对方威胁步贡献 -THREAT_BONUS_BLACK，
        我方潜力步（w<90）贡献 10。机机对战中黑白对称。"""
        total = 0.0
        for (_r, _c, p, w) in branch:
            if p == player:
                if w >= 90:
                    total += DeepSearch.THREAT_BONUS_WHITE
                else:
                    total += 10.0       # 2 潜力
            else:
                if w >= 90:
                    total -= DeepSearch.THREAT_BONUS_BLACK
        return total

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

        wt, bt = self._threat_counts(branch)
        my_t = wt if player == WHITE else bt
        opp_t = bt if player == WHITE else wt
        if status == 'win':
            return self.WIN_SCORE + self.THREAT_BONUS_WHITE * my_t - self.THREAT_BONUS_BLACK * opp_t
        if status == 'lose':
            return -self.WIN_SCORE - self.THREAT_BONUS_BLACK * opp_t + self.THREAT_BONUS_WHITE * my_t
        nn_score = self._nn_score(ctx_branch)
        if nn_score is not None:
            # NN 输出约 [-1,1]，缩放到与胜负分(WIN_SCORE≈1000)同量级
            return nn_score * self.WIN_SCORE
        return self._stale_score(branch, player)

    def _nn_score(self, ctx_branch):
        """用神经网络评估分支路线分数；模型不可用/未训练/被禁用时返回 None。"""
        if not self.use_nn or not ctx_branch:
            return None
        try:
            import torch
            from gomoku.nn.features import branch_matrix
            from gomoku.nn.model import BranchValueNet
            if self._nn_model is None:
                model = BranchValueNet()
                model.load_state_dict(torch.load(
                    self.model_path, map_location='cpu', weights_only=True))
                model.eval()
                self._nn_model = model
            matrix, length = branch_matrix(ctx_branch)
            x = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)   # [1,T,16]
            lengths = torch.tensor([length], dtype=torch.long)
            with torch.no_grad():
                score = self._nn_model(x, lengths).item()
            return score
        except Exception:
            return None

    # ---------- 候选点展开（供推演层） ----------

    def _candidate_points(self, board, player, n=5):
        """当前 player 的候选点（带权重），供推演层展开：
        与顶层共用强制攻防优先级；普通候选为 3攻2防，使用真实综合分。
        返回 [(r, c, w), ...]。权重同时用于温度分配与叶子过程分。"""
        reason, tactical = self.search.tactical_candidates(board, player)
        if reason:
            # 战术防守不得被普通 top-N 名额截断，否则会漏掉唯一救点。
            return [(r, c, w) for (r, c), w in tactical.items()]
        # 兜底：3攻2防，视角与当前 player 对称（白攻+黑防 / 黑攻+白防）
        fb = self.search._fallback(board, player=player, with_score=True)
        # 权重 = _candidate_score 真实综合分（0~100），温度银行按此分配深度：
        # 强候选（落子成冲四/活三等先手点，分高）推得深，弱候选自动浅推。
        # clamp 到 [1,100]：超 100 会让 dt=(100-w)^2 回升，温度非单调。
        w_lst = [(r, c, min(max(float(w), 1.0), 100.0)) for (r, c, w) in fb[:n]]
        return w_lst

    def _make_ctx(self, board, player, main_player, w, cand_ws):
        """构造一步上下文（供神经网络分支矩阵特征使用）。"""
        cand_ws = [float(x) for x in (cand_ws or [])]
        kill = any(cw >= 100 for cw in cand_ws)
        gear = 0 if kill else self.search._target_gear(board, player)
        return {
            'camp': 1 if player == main_player else 0,
            'move_w': float(w),
            'cand_ws': cand_ws,
            'kill': 1 if kill else 0,
            'gear': gear,
        }

    # ---------- 递归推演 ----------

    def _recursive(self, board, player, opp, main_player, temp, branch, ctx_branch, cache, trace=False, indent=0):
        """递归推演：player 落一子（候选）→ opp 应一手 → 递归下一层。
        一层 = 双方完整回合；回合后温度耗尽则进行叶子评估。
        模拟落子后同步更新状态表（journal 哈希表回滚），保证候选点/必杀检测
        基于当前分支的真实棋盘，而不是进入深推前的过期状态表。
        ctx_branch 与 branch 同步记录每步上下文，供神经网络分支矩阵使用。
        cache: {落子序列: 评估分} 前缀复用（同一序列的后续推演直接复用）。
        trace=True 时按树形缩进输出推演内部（调试用）。
        返回该分支评估分（取子分支最高）。"""
        key = tuple((r, c) for (r, c, _p, _w) in branch)
        if key in cache:
            return cache[key]
        best = -1e9
        cands = self._candidate_points(board, player, 5)
        cand_ws = [x[2] for x in cands]
        for (r, c, w) in cands:
            if board[r][c] != EMPTY:
                continue
            ctx = self._make_ctx(board, player, main_player, w, cand_ws)
            board[r][c] = player
            branch.append((r, c, player, w))
            ctx_branch.append(ctx)
            j_w = {}
            self.search.on_move(board, r, c, player, journal=j_w, record_history=False)
            if trace:
                print('  ' * indent + '+- %s落%s%d w=%g t=%g' % (
                    'W' if player == WHITE else 'B', 'ABCDEFGHIJKLMNO'[c], r + 1, w, temp))
            if self._check_win(board, r, c, player):
                score = self._leaf_score(board, branch, 'win', player, ctx_branch)
                if trace:
                    print('  ' * indent + '   -> WIN %g' % score)
            else:
                # 黑方候选逻辑与白方完全一致：有 0 级先取 0 级，否则 fallback top5
                opp_pts = self._candidate_points(board, opp, 5)
                if not opp_pts:
                    score = self._leaf_score(board, branch, 'stale', player, ctx_branch)
                else:
                    opp_cand_ws = [x[2] for x in opp_pts]
                    worst = 1e9
                    for (or_, oc, ow) in opp_pts:
                        if board[or_][oc] != EMPTY:
                            continue
                        temp2 = temp - self._dt(w) - self._dt(ow)
                        ctx_b = self._make_ctx(board, opp, main_player, ow, opp_cand_ws)
                        board[or_][oc] = opp
                        branch.append((or_, oc, opp, ow))
                        ctx_branch.append(ctx_b)
                        j_b = {}
                        self.search.on_move(board, or_, oc, opp, journal=j_b, record_history=False)
                        if self._check_win(board, or_, oc, opp):
                            child = self._leaf_score(board, branch, 'lose', player, ctx_branch)
                            if trace:
                                print('  ' * indent + '   -> B应%s%d LOSE %g' % ('ABCDEFGHIJKLMNO'[oc], or_ + 1, child))
                        elif temp2 <= 0:
                            child = self._leaf_score(board, branch, 'stale', player, ctx_branch)
                            if trace:
                                print('  ' * indent + '   -> 温度尽(t=%g<dt=%g) leaf=%g' % (temp2 + self._dt(w), self._dt(w), child))
                        else:
                            child = self._recursive(board, player, opp, main_player, temp2, branch, ctx_branch, cache, trace, indent + 1)
                        self.search.restore(j_b)
                        branch.pop()
                        ctx_branch.pop()
                        board[or_][oc] = EMPTY
                        if child < worst:
                            worst = child
                    score = worst
            self.search.restore(j_w)
            branch.pop()
            ctx_branch.pop()
            board[r][c] = EMPTY
            if trace:
                print('  ' * indent + '    branch=%g' % score)
            if score > best:
                best = score
        cache[key] = best
        return best

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
            cand_ws = [x[2] for x in self._candidate_points(board, player, 5)]
            ctx = self._make_ctx(board, player, main_player, w, cand_ws)
            board[r][c] = player
            branch = [(r, c, player, w)]
            ctx_branch = [ctx]
            j_w = {}
            self.search.on_move(board, r, c, player, journal=j_w, record_history=False)
            if trace:
                print('== cand: %s%d w=%g' % ('ABCDEFGHIJKLMNO'[c], r + 1, w))
            if self._check_win(board, r, c, player):
                val = self._leaf_score(board, branch, 'win', player, ctx_branch)
                if trace:
                    print('   -> direct WIN %g' % val)
            else:
                # 黑方候选逻辑与白方完全一致：有 0 级先取 0 级，否则 fallback top5
                opp_pts = self._candidate_points(board, opp, 5)
                if not opp_pts:
                    val = self._leaf_score(board, branch, 'stale', player, ctx_branch)
                else:
                    opp_cand_ws = [x[2] for x in opp_pts]
                    worst = 1e9
                    for (or_, oc, ow) in opp_pts:
                        if board[or_][oc] != EMPTY:
                            continue
                        temp2 = self.T0 - self._dt(w) - self._dt(ow)
                        ctx_b = self._make_ctx(board, opp, main_player, ow, opp_cand_ws)
                        board[or_][oc] = opp
                        branch.append((or_, oc, opp, ow))
                        ctx_branch.append(ctx_b)
                        j_b = {}
                        self.search.on_move(board, or_, oc, opp, journal=j_b, record_history=False)
                        if self._check_win(board, or_, oc, opp):
                            child = self._leaf_score(board, branch, 'lose', player, ctx_branch)
                            if trace:
                                print('   -> B resp %s%d LOSE %g' % ('ABCDEFGHIJKLMNO'[oc], or_ + 1, child))
                        elif temp2 <= 0:
                            child = self._leaf_score(board, branch, 'stale', player, ctx_branch)
                        else:
                            child = self._recursive(board, player, opp, main_player, temp2, branch, ctx_branch, cache, trace, 1)
                        self.search.restore(j_b)
                        branch.pop()
                        ctx_branch.pop()
                        board[or_][oc] = EMPTY
                        if child < worst:
                            worst = child
                    val = worst
            self.search.restore(j_w)
            branch.pop()
            ctx_branch.pop()
            board[r][c] = EMPTY
            if trace:
                print('   cand%s%d val=%g' % ('ABCDEFGHIJKLMNO'[c], r + 1, val))
            results.append({'r': r, 'c': c, 'score': val})
        results.sort(key=lambda x: x['score'], reverse=True)
        if trace:
            top = results[0] if results else None
            if top:
                print('=== pick: %s%d val=%g ===' % ('ABCDEFGHIJKLMNO'[top['c']], top['r'] + 1, top['score']))
            else:
                print('=== no candidates ===')
        return results
