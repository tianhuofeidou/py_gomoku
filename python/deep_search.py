# ============================================================
# deep_search.py —— 深度推演层（温度银行）
# 职责：对候选点做多步递归推演，返回分支分最高的落子点
#
# 核心机制：
#   - 温度银行：每手决策一个温度总预算 T0，每层推演按候选权重扣温
#       DT = DT_M / w + DT_N（w=候选权重 50~100：权重高→耗温少→推得深）
#       温度耗尽 → 叶子：不再落子，直接评估当前局面
#   - 一层 = 白+黑一个完整回合；黑应手取单点最凶（_best_response）
#   - 叶子评估：成五 ±1000×过程权重；僵持 → 过程分（快杀/干净杀得分高）
#   - cache：按"已落子序列"缓存分支分，序列前缀复用（同序列只推一次）
#
# TODO（骨架保留，本次结构整理未实现）：
#   - 分支定界：UB/LB 剪枝（先深推一枝拿标杆，砍追不上的枝）
#   - 保底 3 枝：无条件开枝（防漏）
#
# 依赖：search（候选/应手来源）、utils（棋盘常量）
# ============================================================

from utils import SIZE, EMPTY, BLACK, WHITE


class DeepSearch:
    """深推器：温度银行驱动的递归推演。
    持有 Search 实例（提供 candidates/_fallback 取候选点），
    供 engine 的 compute_move / analyze_turn 调用。"""

    # 温度参数（定稿占位，待调优）
    T0 = 80.0              # 初始温度（每手决策总预算）
    DT_M = 800.0           # 单层耗温 DT = DT_M/w + DT_N
    DT_N = 3.0             #   w = 候选权重(50~100)：权重高→耗温少→推得深
    WIN_SCORE = 1000.0     # 成五基准分（有界，乘过程权重）

    def __init__(self, search):
        self.search = search        # Search 实例：候选点来源

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
        """单层耗温：DT = 800/w + 3（w 为候选权重 50~100）。
        权重高（强手）→ DT 小 → 温度剩多 → 推得深；权重低 → DT 大 → 浅推。"""
        return DeepSearch.DT_M / max(w, 1e-6) + DeepSearch.DT_N

    @staticmethod
    def _process_weight(branch):
        """过程权重（层系数连乘）：每层乘 (w白/100) × (w黑/100)²。
        黑白全程 100 → 乘积≈1（衰减极小，深推高权重线）；
        黑弱应/白弱手 → 快速衰减（过程分降低）。
        用于成五(win/lose)时的分数加权。"""
        pos = 1.0
        for (_r, _c, player, w) in branch:
            if player == WHITE:
                pos *= (w / 100.0)
            else:
                pos *= (w / 100.0) ** 2
        return pos

    @staticmethod
    def _stale_score(branch):
        """僵持过程分：累加每层【我方落子】的分值 × 位置权重。
        我方落子分值按权重档：0必杀(w≥100)=50、1威胁(w≥90)=30、2潜力=10。
        位置权重 = 1/(x+1)，x=层号（第1层=1/2、第2层=1/3...靠前权重高）。
        黑层落子不计分（黑是阻力，不贡献我方过程分）。
        效果：快杀/干净杀（必杀威胁多、靠前）的分支得分高。"""
        total = 0.0
        x = 0
        for (_r, _c, player, w) in branch:
            if player == WHITE:
                x += 1
                if w >= 100:
                    val = 50.0          # 0 必杀
                elif w >= 90:
                    val = 30.0          # 1 威胁
                else:
                    val = 10.0          # 2 潜力
                total += val / (x + 1)
        return total

    def _leaf_score(self, board, branch, status):
        """叶子评估：
        - win（我方成五）→ +1000 × 过程权重
        - lose（对方成五）→ -1000 × 过程权重
        - stale（温度尽僵持）→ _stale_score 过程分（不再固定 0，
          让"白压制强"的分支与"白无作为"的分支区分开）"""
        if status == 'win':
            pw = self._process_weight(branch)
            return self.WIN_SCORE * pw
        if status == 'lose':
            pw = self._process_weight(branch)
            return -self.WIN_SCORE * pw
        return self._stale_score(branch)

    # ---------- 候选点展开（供推演层） ----------

    def _candidate_points(self, board, player, n=5):
        """当前 player 的候选点（带权重），供推演层展开：
        优先检测优先级（0 必杀级：五连/VCF/双三，权重 100），否则兜底 3攻2防（权重 60）。
        返回 [(r, c, w), ...]（降序）。权重同时用于温度分配与叶子过程分。"""
        if player == WHITE:
            c = self.search.candidates(board)
            if c['w_five']:
                return [(r, col, 100.0) for (r, col) in c['w_five'][:n]]
            if c['w_vcf']:
                return [(r, col, 100.0) for (r, col) in c['w_vcf'][:n]]
            if c['w_d33']:
                return [(r, col, 100.0) for (r, col) in c['w_d33'][:n]]
        else:
            c = self.search.candidates(board)
            if c['b_five']:
                return [(r, col, 100.0) for (r, col) in c['b_five'][:n]]
            if c['b_vcf']:
                return [(r, col, 100.0) for (r, col) in c['b_vcf'][:n]]
            if c['b_d33']:
                return [(r, col, 100.0) for (r, col) in c['b_d33'][:n]]
        # 兜底：3攻2防（白）/ 黑视角同类
        fb = self.search._fallback(board)
        w_lst = [(r, c, 60.0) for (r, c) in fb[:n]]
        return w_lst

    def _best_response(self, board, player):
        """对手（黑）应手：黑候选分最高 1 个（先取最高权重点）。
        只取 1 个 = 白推演始终面对黑最凶应手，验证杀棋是否真正无解。"""
        pts = self._candidate_points(board, player, 1)
        return pts[0] if pts else None

    # ---------- 递归推演 ----------

    def _recursive(self, board, player, opp, temp, branch, cache, trace=False, indent=0):
        """递归推演：player 落一子（候选）→ opp 应一手 → 递归下一层。
        一层 = 白+黑完整回合；温度不足 → 叶子评估（不落子，只评估当前局面）。
        cache: {落子序列: 评估分} 前缀复用（同一序列的后续推演直接复用）。
        trace=True 时按树形缩进输出推演内部（调试用）。
        返回该分支评估分（取子分支最高）。"""
        key = tuple((r, c) for (r, c, _p, _w) in branch)
        if key in cache:
            return cache[key]
        best = -1e9
        for (r, c, w) in self._candidate_points(board, player, 5):
            if board[r][c] != EMPTY:
                continue
            board[r][c] = player
            branch.append((r, c, player, w))
            if trace:
                print('  ' * indent + '+- %s落%s%d w=%g t=%g' % (
                    'W' if player == WHITE else 'B', 'ABCDEFGHIJKLMNO'[c], r + 1, w, temp))
            if self._check_win(board, r, c, player):
                score = self._leaf_score(board, branch, 'win')
                if trace:
                    print('  ' * indent + '   -> WIN %g' % score)
            else:
                opp_pt = self._best_response(board, opp)
                if opp_pt is None:
                    score = self._leaf_score(board, branch, 'stale')
                else:
                    or_, oc, ow = opp_pt
                    board[or_][oc] = opp
                    branch.append((or_, oc, opp, ow))
                    if self._check_win(board, or_, oc, opp):
                        score = self._leaf_score(board, branch, 'lose')
                        if trace:
                            print('  ' * indent + '   -> B应%s%d LOSE %g' % ('ABCDEFGHIJKLMNO'[oc], or_ + 1, score))
                    else:
                        temp2 = temp - self._dt(w)
                        if temp2 <= 0:
                            score = self._leaf_score(board, branch, 'stale')
                            if trace:
                                print('  ' * indent + '   -> 温度尽(t=%g<dt=%g) leaf=%g' % (temp2 + self._dt(w), self._dt(w), score))
                        else:
                            score = self._recursive(board, player, opp, temp2, branch, cache, trace, indent + 1)
                    branch.pop()
                    board[or_][oc] = EMPTY
            branch.pop()
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
        trace=True 时输出推演树（调试用）。"""
        if isinstance(points, dict):
            cands = list(points.keys())
        elif isinstance(points, list):
            cands = points
        else:
            cands = []
        if not cands:
            return (SIZE // 2, SIZE // 2)
        if trace:
            print('=== deep_search turn=%s T0=%g ===' % ('W' if player == WHITE else 'B', self.T0))
        cache = {}
        opp = 3 - player
        best_pt = None
        best_val = -1e9
        for (r, c) in cands:
            if board[r][c] != EMPTY:
                continue
            w = points.get((r, c), 60.0) if isinstance(points, dict) else 60.0
            board[r][c] = player
            branch = [(r, c, player, w)]
            if trace:
                print('== cand: %s%d w=%g' % ('ABCDEFGHIJKLMNO'[c], r + 1, w))
            if self._check_win(board, r, c, player):
                val = self._leaf_score(board, branch, 'win')
                if trace:
                    print('   -> direct WIN %g' % val)
            else:
                opp_pt = self._best_response(board, opp)
                if opp_pt is None:
                    val = self._leaf_score(board, branch, 'stale')
                else:
                    or_, oc, ow = opp_pt
                    board[or_][oc] = opp
                    branch.append((or_, oc, opp, ow))
                    if self._check_win(board, or_, oc, opp):
                        val = self._leaf_score(board, branch, 'lose')
                        if trace:
                            print('   -> B resp %s%d LOSE %g' % ('ABCDEFGHIJKLMNO'[oc], or_ + 1, val))
                    else:
                        temp2 = self.T0 - self._dt(w)
                        if temp2 <= 0:
                            val = self._leaf_score(board, branch, 'stale')
                        else:
                            val = self._recursive(board, player, opp, temp2, branch, cache, trace, 1)
                    branch.pop()
                    board[or_][oc] = EMPTY
            branch.pop()
            board[r][c] = EMPTY
            if trace:
                print('   cand%s%d val=%g' % ('ABCDEFGHIJKLMNO'[c], r + 1, val))
            if val > best_val:
                best_val = val
                best_pt = (r, c)
        if trace:
            print('=== pick: %s%d val=%g ===' % ('ABCDEFGHIJKLMNO'[best_pt[1]], best_pt[0] + 1, best_val))
        return best_pt if best_pt is not None else cands[0]
