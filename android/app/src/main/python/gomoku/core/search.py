# ============================================================
# search.py —— 搜索层（决策主流程 + 状态表维护）
# 职责：
#   1. 维护黑白状态表（全量 15×15，每格 18 子类编号 0-17）
#   2. 落子后按邻域、棋段及状态区做 3 段局部更新
#   3. 候选点生成：按检测优先级分类（5连/活四/VCF/双三）+ 防守反推
#   4. 无必杀时兜底：3攻2防（棋型分 + 邻近度 + 攻防一体 平滑评分）
#
# 依赖：pattern（棋型分析）、utils（棋盘常量）
# 被 deep_search（取候选/应手）与 engine（决策编排）依赖
#
# 【状态表】
#   父类 4 种：KILL(必杀) / THREAT(威胁) / POTEN(潜力) / USELESS(无用)
#   子类 18 种：必杀5(five/live4/d44/d34/d33) + 威胁8(A1眠四系4+A2活三系4)
#             + 潜力4(b系) + 无用1(none)
#
# 【局部增量更新】：黑落子分两段更新黑白表，白落子按影响区更新双方。
# 更新范围由 SearchScope 的邻域、棋段及状态区决定。
# ============================================================

from .utils import SIZE, EMPTY, BLACK, WHITE, DIRECTIONS, in_board
from .pattern import PatternAnalyzer

# 格子状态：黑白分离的状态表（sb/sw），实现为 15×15 二维数组，
# 每格存 18 子类编号 0-17（见下方 SUBCLASS_NAMES）：
#   0-4   必杀（five/live4/d44/d34/d33）
#   5-12  威胁（a1 眠四系 + a2 活三系）
#   13-16 潜力（b 系）
#   17    无用（none）
# 以下三个常量是旧版「三态」残留（1/2/3），已无任何引用，勿再使用。
STATE_STRONG = 1
STATE_POTENTIAL = 2
STATE_NONE = 3

# ============================================================
# 格子威胁分类：父类 4 种 + 子类 18 种（按 4 方向最高级组合）
#
# 方向档位：
#   A1 = 眠四(冲四)   A2 = 活三
#   B  = 眠三/活二（等价合并）
#   C  = 眠二/无（合并）
#
# 父类（4 种）：
#   KILL    = 必杀（落子后对手无解）
#   THREAT  = 威胁（单 A，对手必应但可防）
#   POTEN   = 潜力（无 A，有发展价值）
#   USELESS = 无用（全弱/无，无价值）
#
# 子类（18 种）：
#   必杀 5：五连 / 活四 / 44(双眠四) / 34·43(眠四+活三) / 33(双三)
#   威胁 8：A1系4（A1+3维B/C无序）+ A2系4
#   潜力 4：(B,B,B,B) (B,B,B,C) (B,B,C,C) (B,C,C,C)
#   无用 1：(C,C,C,C) 全弱/无
# ============================================================

KILL = 0       # 必杀
THREAT = 1     # 威胁
POTEN = 2      # 潜力
USELESS = 3    # 无用

# 子类名（18 个，全局编号 0-17）
SUBCLASS_NAMES = [
    # 必杀 5（0-4）
    'five', 'live4', 'd44', 'd34', 'd33',
    # 威胁 8（5-12）：A1 系 4 + A2 系 4
    'a1_bbb', 'a1_bbc', 'a1_bcc', 'a1_ccc',
    'a2_bbb', 'a2_bbc', 'a2_bcc', 'a2_ccc',
    # 潜力 4（13-16）
    'b_bbb', 'b_bbc', 'b_bcc', 'b_ccc',
    # 无用 1（17）
    'none',
]

# 子类 → 父类
SUBCLASS_PARENT = [
    KILL, KILL, KILL, KILL, KILL,          # five, live4, d44, d34, d33
    THREAT, THREAT, THREAT, THREAT,        # a1 系
    THREAT, THREAT, THREAT, THREAT,        # a2 系
    POTEN, POTEN, POTEN, POTEN,            # b 系
    USELESS,                               # none
]

assert len(SUBCLASS_NAMES) == 18 and len(SUBCLASS_PARENT) == 18

# 父类威胁优先级（数值越大越优先）—— search 分派依据
# 进攻侧 / 防守侧交错，由 owner 定稿后锁定
THREAT_PRIORITY = {
    # 我方进攻
    'FIVE':   100,
    'LIVE4':  90,
    'SLEEP4': 70,
    'LIVE3':  60,
    'SLEEP3': 40,
    'LIVE2':  25,
    'SLEEP2': 10,
    # 对方防守（同父类对方级略低于我方级，实际穿插规则 TODO 确认）
}

# ============================================================
# 方向档位（每方向最高级）：
#   0 = 五连    1 = 活四
#   2 = 眠四(A1)  3 = 活三(A2)
#   4 = 眠三/活二(B)   5 = 眠二/无(C)
# ============================================================
DIR_LEVEL = {
    'FIVE': 0, 'LIVE4': 1,
    'SLEEP4': 2, 'LIVE3': 3,
    'SLEEP3': 4, 'LIVE2': 4,
    'SLEEP2': 5, 'DEAD': 5, 'SINGLE': 5, 'NONE': 5,
}

# ============================================================
# 候选点评分常量（占位值，待调优——集中管理，为遗传算法留接口）
# ============================================================
# 邻近度：距离衰减（1格/2格）与方向价值（同线/异线）
PROX_DIST_W = {1: 1.0, 2: 0.5}      # 距离衰减：紧贴子权重高
PROX_DIR_W = {'line': 2.0, 'other': 0.5}  # 方向：同线可连价值高

# 邻近度综合：自家/对家权重（进攻 vs 防守 vs 攻防一体）
NEIGHBOR_W = {
    'attack': (110, -10),   # 纯攻：自家×110 + 对家×(-10)
    'defend': (30, 70),     # 纯防：自家×30  + 对家×70
    'dual':   (60, 40),     # 攻防一体：自家×60 + 对家×40（自家更高）
}

# 候选点综合分（棋型分基准 + 邻近 + 攻防一体），标准化到满分 100
# 棋型分：TYPE_SCORE 原生值（30~200）
# 邻近：归一化 score/600 × W_NEIGHBOR（保留负值）→ 约 -∞~30
# 攻防一体：黑白双 1 固定 +DUAL_BONUS
# 满分 = 200 + 30 + 30 = 260 → 综合分 / 260 × 100
# 全部占位，待遗传算法调优
W_NEIGHBOR = 30.0     # 邻近权重（等效威胁 30 分）
NEIGHBOR_NORM = 600.0 # 邻近归一化分母
DUAL_BONUS = 30.0     # 攻防一体加分
SCORE_MAX = 260.0     # 综合分满分（200+30+30），用于标准化到 100

# 棋型基础分（占位待调：a2 活三系 9-12、a1 眠四系 5-8、b 潜力系 13-16）
TYPE_SCORE = {
    9: 200, 10: 170, 11: 130, 12: 80,     # a2_bbb/bbc/bcc/ccc
    5: 190, 6: 160, 7: 120, 8: 70,        # a1_bbb/bbc/bcc/ccc
    13: 120, 14: 90, 15: 60, 16: 30,      # b_bbb/bbc/bcc/ccc
}

# ============================================================
# 动态攻防配比（无必杀兜底时使用）
#   档位 1-4：4攻1防 / 3攻2防 / 2攻3防 / 1攻4防
#   GEAR_ENABLED=False 时回退固定 3攻2防（深推内部 / 黑应手保持固定）
#   压力只看 1 级(THREAT) 与 2 级(POTEN)，0 级由上层必杀判定单独处理；
#   历史 0 级事件作为“杀势动量”参与档位切换。
# ============================================================
GEAR_ENABLED = True
GEAR_PROFILE = {
    1: (1, 4),   # 1攻4防
    2: (2, 3),   # 2攻3防
    3: (3, 2),   # 3攻2防
    4: (4, 1),   # 4攻1防
}
DEFAULT_GEAR = 3

# 子类区间（18 子类编号）
A1_SUBCLASSES = (5, 6, 7, 8)     # a1 系：眠四/冲四
A2_SUBCLASSES = (9, 10, 11, 12)  # a2 系：活三
B_SUBCLASSES = (13, 14, 15, 16)  # b 系：眠三/活二

# 压力权重：a1(冲四) > a2(活三) > b(潜力)
GEAR_W_A1 = 4.0
GEAR_W_A2 = 2.0
GEAR_W_B = 0.5
# 历史 0 级杀势动量权重（一次新杀势 ≈ 1.25 个冲四差）
GEAR_W_ZERO = 5.0
# 历史 0 级事件窗口（步数）
GEAR_ZERO_WINDOW = 10
# 档位阈值（score = 当前压力差 + 历史动量）
GEAR_THRESHOLDS = (2.0, -1.0, -3.0)   # score>=t1→4档；t2<=score<t1→3档；t3<=score<t2→2档；score<t3→1档


class SearchScope:
    """3 段搜索范围（黑白对称，AI 执白 / 对手执黑）。

    状态表 state[player]：15×15 二维数组，每格 18 子类编号 0-17（由更新逻辑维护，本类只读取）。
    每段返回候选点集合（空位坐标），供状态更新逻辑消费。

    搜索规则（定稿）：
      定点区 = 最后一步 2 格邻域空位
      棋段区 = 最后一步所在棋段（close/三空为界，单双空 gap 不限量）
               + 对侧端点外延伸 2 格
      状态区 = 状态 1 区（空则降级状态 2 区兜底）
    """

    # ---------- 基础：定点邻域 ----------

    def _neighborhood(self, board, r, c, radius=2):
        """① 最后一步 2 格邻域内的空位"""
        pts = set()
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if in_board(rr, cc) and board[rr][cc] == EMPTY:
                    pts.add((rr, cc))
        return pts

    # ---------- 基础：棋段提取（close / 连续三空 为界，gap 不限量） ----------

    def _scan_side(self, board, r, c, player, dr, dc, sign):
        """从 (r,c) 沿 (dr,dc) 的 sign 方向扫描棋段。
        返回 (段内己方子, 段内 gap 空位, 段尾外侧连续空位, 终止原因)
        终止：越界(wall) / 对方子(close) / 连续三空(triple-gap)
        """
        stones = []
        gaps = []
        tail = []
        empty_run = 0
        step = 1
        while True:
            nr = r + sign * step * dr
            nc = c + sign * step * dc
            if not in_board(nr, nc):
                return stones, gaps, tail, 'wall'
            v = board[nr][nc]
            if v == player:
                gaps.extend(tail)       # 之前的空位成为段内 gap
                tail = []
                empty_run = 0
                stones.append((nr, nc))
            elif v == EMPTY:
                empty_run += 1
                tail.append((nr, nc))
                if empty_run >= 3:
                    return stones, gaps, tail, 'triple-gap'
            else:
                return stones, gaps, tail, 'close'
            step += 1

    # ---------- ② 棋段搜索：段内 gap（不限量）+ 对侧端点外延伸 2 格 ----------

    def _segment_scope(self, board, r, c, player, extend=2):
        """最后一步所在棋段的补充搜索区（含 player 一方的棋段）：
        a. 段内所有 gap（不限量）
        b. 对侧端点外延伸 extend 格（防守反推时取 1，进攻搜索取 2；
           最后一步在段中间时两端外侧都取）
        """
        pts = set()
        for (dr, dc) in DIRECTIONS:
            neg_s, neg_g, neg_t, _ = self._scan_side(board, r, c, player, dr, dc, -1)
            pos_s, pos_g, pos_t, _ = self._scan_side(board, r, c, player, dr, dc, +1)
            stones = neg_s[::-1] + [(r, c)] + pos_s
            if len(stones) <= 1:
                continue                # 无棋段（孤立子）
            for g in neg_g + pos_g:
                pts.add(g)
            if not neg_s:
                for e in pos_t[:extend]:
                    pts.add(e)          # 最后一步是最负端 → 对侧=pos端外侧
            elif not pos_s:
                for e in neg_t[:extend]:
                    pts.add(e)          # 最后一步是最正端 → 对侧=neg端外侧
            else:
                for e in neg_t[:extend]:
                    pts.add(e)          # 段中间 → 负侧端点外侧
                for e in pos_t[:extend]:
                    pts.add(e)          # 段中间 → 正侧端点外侧（各 extend 格，不互相截断）
        return pts

    def _both_segment_scope(self, board, r, c):
        """落子点 4 条线上【双方】棋段的延伸区并集。
        落子会同时改变己方棋段与对方棋段上各点的威胁，
        只刷一方会导致另一方棋段上的点状态过期（如对方棋段被新子堵端）。"""
        pts = self._segment_scope(board, r, c, BLACK)
        pts |= self._segment_scope(board, r, c, WHITE)
        return pts

    # ---------- 状态区（必杀/威胁优先，潜力兜底） ----------

    def _state_region(self, state):
        """搜索状态表：父类为 KILL/THREAT 的区；该区为空时降级 POTEN 区兜底。
        状态表每格存 18 子类编号（0-17），父类由 SUBCLASS_PARENT 得出。
        返回格子集合（含已占格——由调用方过滤空位）。"""
        pts = set()
        for r in range(SIZE):
            for c in range(SIZE):
                if SUBCLASS_PARENT[state[r][c]] in (KILL, THREAT):
                    pts.add((r, c))
        if not pts:
            for r in range(SIZE):
                for c in range(SIZE):
                    if SUBCLASS_PARENT[state[r][c]] == POTEN:
                        pts.add((r, c))
        return pts

    # ---------- 3 段 ----------

    def segment1_black(self, board, last_r, last_c, state_black):
        """段1（黑表）：黑最后一步定点区（双方棋段）∪ 黑状态区"""
        pts = self._neighborhood(board, last_r, last_c, 2)
        pts |= self._both_segment_scope(board, last_r, last_c)
        pts |= self._state_region(state_black)
        return pts

    def segment2_white(self, board, last_r, last_c, state_black, state_white):
        """段2（白表）：段1黑影响区域 ∪ 白状态区"""
        pts = self.segment1_black(board, last_r, last_c, state_black)
        pts |= self._state_region(state_white)
        return pts

    def segment3_update(self, board, last_r, last_c, state_black, state_white):
        """段3（白落子后）：白影响区域（邻域 + 双方棋段）→ 黑白两表都更新"""
        pts = self._neighborhood(board, last_r, last_c, 2)
        pts |= self._both_segment_scope(board, last_r, last_c)
        return pts


class Search:
    """搜索器：状态表维护 + 候选点生成 + 威胁分类 + 评分兜底。
    持有黑白状态表（sb/sw，18 子类编号）与 PatternAnalyzer，
    供 engine（决策编排）与 deep_search（候选/应手来源）使用。"""

    def __init__(self, pattern):
        self.pattern = pattern                # PatternAnalyzer 实例
        self.scope = SearchScope()            # 3 段影响区域计算
        self.sb = [[17] * SIZE for _ in range(SIZE)]   # 黑状态表（none=17）
        self.sw = [[17] * SIZE for _ in range(SIZE)]   # 白状态表
        # 动态攻防配比状态：黑白各自独立，避免机机对战时互相污染
        self.gear = {BLACK: DEFAULT_GEAR, WHITE: DEFAULT_GEAR}
        # 历史 0 级事件用普通 list，窗口大小动态读 GEAR_ZERO_WINDOW（GA 可调）
        self.zero_history = {BLACK: [], WHITE: []}
        self.zero_active = {BLACK: False, WHITE: False}   # 防同一杀势连续多步重复计数

    # ---------- 3 段更新（增量刷新状态表） ----------

    def on_move(self, board, r, c, player, journal=None, record_history=True):
        """任意方落子后的状态更新（增量，不重扫全盘）。
        先清落子点自身，再按 SearchScope 的 3 段局部范围更新黑白表。

        journal：可选哈希表，记录本次更新中“首次被修改格子”的旧值，
                 用于深推模拟后的状态表回滚。
        record_history：是否记录历史 0 级/档位等真实对局状态；
                        深推模拟时应传 False，避免污染真实历史。
        """
        # 落子点已占：双方状态表立即置无用，日志保留旧值供模拟回滚。
        self._write_state(self.sb, r, c, 17, journal)
        self._write_state(self.sw, r, c, 17, journal)
        if player == BLACK:
            reg1 = self.scope.segment1_black(board, r, c, self.sb)
            self._update(board, reg1, BLACK, self.sb, journal)
            reg2 = self.scope.segment2_white(board, r, c, self.sb, self.sw)
            self._update(board, reg2, WHITE, self.sw, journal)
        else:
            reg3 = self.scope.segment3_update(board, r, c, self.sb, self.sw)
            self._update(board, reg3, BLACK, self.sb, journal)
            self._update(board, reg3, WHITE, self.sw, journal)
        # 每个真实棋步更新双方杀势及历史窗口，同一杀势持续存在只计一次。
        # 深推模拟不记录，避免假设棋步污染真实历史。
        if record_history:
            self._record_zero_event(board, BLACK)
            self._record_zero_event(board, WHITE)

    def _write_state(self, state, r, c, new_val, journal=None):
        """写状态表；若传入 journal，首次修改该格时记录旧值。"""
        old = state[r][c]
        if old == new_val:
            return
        if journal is not None:
            key = (id(state), r, c)
            if key not in journal:
                journal[key] = old
        state[r][c] = new_val

    def restore(self, journal):
        """恢复 journal 中记录的所有格子旧值（深推模拟后回滚状态表）。"""
        for (sid, r, c), old in journal.items():
            if sid == id(self.sb):
                self.sb[r][c] = old
            else:
                self.sw[r][c] = old

    def _update(self, board, region, player, state, journal=None):
        """对区域内每个空位模拟落子 → 18 子类编号 → 写回状态表。
        已占格一律置 17（none 无用）。"""
        for (r, c) in region:
            if not (0 <= r < SIZE and 0 <= c < SIZE):
                continue
            if board[r][c] != EMPTY:
                self._write_state(state, r, c, 17, journal)   # 已占格 → 无用
                continue
            self._write_state(state, r, c, self._eval(board, r, c, player), journal)

    # ---------- 单点评估（模拟落子 → 18 子类） ----------

    def rebuild_all(self, board):
        """全盘重建黑白状态表 + 清杀势历史。
        悔棋 / 换新对局后调用：Game 层回退只动 moves，引擎增量状态表
        与棋盘失步（幽灵子残留干扰后续检测/漏杀），需全盘重算。"""
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] == EMPTY:
                    self.sb[r][c] = self._eval(board, r, c, BLACK)
                    self.sw[r][c] = self._eval(board, r, c, WHITE)
                else:
                    self.sb[r][c] = 17
                    self.sw[r][c] = 17
        self.zero_history = {BLACK: [], WHITE: []}
        self.zero_active = {BLACK: False, WHITE: False}
        self.gear = {BLACK: DEFAULT_GEAR, WHITE: DEFAULT_GEAR}

    def _direction_level(self, board, r, c, player, dr, dc):
        """单方向档位（0 五连 … 5 无）：等价于「analyze_direction 的记录取
        DIR_LEVEL 最小值」，但不构造棋型记录、不算子类名与 gap_sides。

        改这里之前必须读完下面两条，否则会静默改变状态表编号（进而改变候选分、
        攻防档位与叶子评估）：

        1. 落子点在该方向【两侧】3 格内都没有己方子 → 该方向必为档位 5，
           直接返回、不做扫描。
           依据：_scan_two 从基子向两侧扫描，遇「连续 3 空 / 对方子 / 边界」即停，
           stones 初值为 1（落子点自身）。某侧 3 格内无己方子时该侧必在 i<=3 内
           结束、stones 保持 1，而 _classify 对 stones==1 只会返回 SINGLE 或
           DEAD，两者 DIR_LEVEL 都是 5；同时该侧端类型不可能是 5，端 5 缝合分支
           也不会触发。
           - 必须查两侧：_scan_two 是双向扫描（只查单侧会在随机局面上错约 5%）。
           - 必须查到第 3 格：双空前瞻会读距离 3，那里有己方子会合成 LIVE2（档位 4）。

        2. 完整路径见 _direction_level_scan，与 analyze_direction 对档位的影响
           逐条对齐（overall 早返回 / normalize / classify / 端 5 缝合）。
           端 5 缝合【不能省】：它把 SINGLE(5) 换成 LIVE2(4)，而档位 4 在
           _classify_levels 里要计入 B 的数量。
        """
        for sgn in (1, -1):
            rr, cc = r + sgn * dr, c + sgn * dc
            for _ in range(3):
                if in_board(rr, cc) and board[rr][cc] == player:
                    return self._direction_level_scan(board, r, c, player, dr, dc)
                rr += sgn * dr
                cc += sgn * dc
        return 5

    def _direction_level_scan(self, board, r, c, player, dr, dc):
        """_direction_level 的完整路径：复刻 analyze_direction 对档位的影响，
        只是不产出记录、子类名与 gap_sides。"""
        res = self.pattern._scan_two(board, r, c, player, dr, dc)
        overall = self.pattern._overall_judge(res)
        if overall is not None:
            return DIR_LEVEL[overall['overall']]
        items = self.pattern._normalize(res)
        best = 5
        for item in items:
            lv = DIR_LEVEL[self.pattern._classify(item)]
            if lv < best:
                best = lv
        if len(items) == 1 and self.pattern._classify(items[0]) == 'SINGLE':
            lt5 = items[0]['l_edge'][1]
            rt5 = items[0]['r_edge'][1]
            if lt5 == 5 or rt5 == 5:
                best = 5        # 与 analyze_direction 一致：此处丢弃 SINGLE 记录
                synth = []
                if lt5 == 5:
                    synth += self.pattern._synth_double_gap(board, r, c, player, dr, dc, 'l')
                if rt5 == 5:
                    synth += self.pattern._synth_double_gap(board, r, c, player, dr, dc, 'r')
                for it in synth:    # 合成可能为空 → 档位保持 5
                    lv = DIR_LEVEL[it['parent']]
                    if lv < best:
                        best = lv
        return best

    def _eval(self, board, r, c, player):
        """模拟 player 在 (r,c) 落子 → 18 子类编号 0-17。
        流程：
          1. 临时落子 → 逐方向取档位（_direction_level，不再构造棋型记录）
          2. _classify_levels 按 4 方向档位组合归类（必杀/威胁/潜力/无用）
        评估后恢复棋盘（finally 清子）。

        这是全引擎最热的函数（一次决策约 6 万次调用）。只取档位、不构造记录、
        不算子类名/gap_sides：_subclass 内部用 list.index() 线性查找，而档位
        根本不读这些字段。等价性由 gomoku/tests/test_pattern_level_fast.py 锁定。"""
        board[r][c] = player
        try:
            direction_level = self._direction_level
            return self._classify_levels(
                [direction_level(board, r, c, player, dr, dc) for (dr, dc) in DIRECTIONS])
        finally:
            board[r][c] = EMPTY

    @staticmethod
    def _classify_levels(levels):
        """4 方向档位列表 → 18 子类编号。
        判定顺序（组合优先于单档）：
          n0>=1 → five(0)      ：任一方向成五
          n1>=1 → live4(1)     ：任一方向活四
          n2>=2 → d44(2)       ：双眠四（两个方向冲四）
          n2>=1且n3>=1 → d34(3)：眠四+活三（三四）
          n3>=2 → d33(4)       ：双三（两个方向活三）
          恰 1 个眠四 → a1 系(5-8)：按其余方向 B(眠三/活二)数量细分
          恰 1 个活三 → a2 系(9-12)：同上
          无 A → b 系(13-16)/none(17)：按 B 数量 4/3/2/1/0
        """
        n0 = levels.count(0)   # 五连
        n1 = levels.count(1)   # 活四
        n2 = levels.count(2)   # 眠四
        n3 = levels.count(3)   # 活三
        if n0 >= 1:
            return 0                    # five
        if n1 >= 1:
            return 1                    # live4
        if n2 >= 2:
            return 2                    # d44
        if n2 >= 1 and n3 >= 1:
            return 3                    # d34
        if n3 >= 2:
            return 4                    # d33
        if n2 == 1:
            # 单眠四：其余 3 维 B/C → a1 系 4 个（索引 5-8）
            bcnt = sum(1 for l in levels if l == 4)
            return 5 + (3 - bcnt)
        if n3 == 1:
            # 单活三：其余 3 维 B/C → a2 系 4 个（索引 9-12）
            bcnt = sum(1 for l in levels if l == 4)
            return 9 + (3 - bcnt)
        # 无 A：4 维 B/C → 潜力 4 / 无用 1（索引 13-17）
        bcnt = sum(1 for l in levels if l == 4)
        return 17 - bcnt      # 4B→13, 3B→14, 2B→15, 1B→16, 0B→17

    # ---------- 候选点分类（检测优先级） ----------

    def candidates(self, board):
        """按检测优先级分类候选点（只看 0 必杀级，命中即止）：
        w_five/b_five : 白/黑五连点   —— direct（落子即胜 / 必须直接堵）
        w_vcf/b_vcf   : 白/黑 VCF     —— live4(活四)+d44(双眠四)+d34(眠四+活三)
                                         白侧直下；黑侧反推堵点后进深推
        w_d33/b_d33   : 白/黑双三点   —— 两个活三叠加，必杀级
        说明：活三（a2 系）是威胁(1)非必杀 → 不进检测优先级，归 fallback(3攻2防) 处理。
        """
        def pts(state, ids):
            return [(r, c) for r in range(SIZE) for c in range(SIZE)
                    if board[r][c] == EMPTY and state[r][c] in ids]
        five = (0,)
        vcf = (1, 2, 3)                     # live4(活四) + d44(双眠四) + d34(眠四+活三)
        return {
            'w_five': pts(self.sw, five),
            'b_five': pts(self.sb, five),
            'w_vcf': pts(self.sw, vcf),
            'b_vcf': pts(self.sb, vcf),
            'w_d33': pts(self.sw, (4,)),
            'b_d33': pts(self.sb, (4,)),
        }

    def tactical_candidates(self, board, player):
        """顶层与深推共用的强制攻防优先级；返回 (原因, 带权候选)。"""
        c = self.candidates(board)
        mine, other = ('b', 'w') if player == BLACK else ('w', 'b')
        for owner, prefix, kind in (
                ('my', mine, 'five'), ('opp', other, 'five'),
                ('my', mine, 'vcf'), ('opp', other, 'vcf'),
                ('my', mine, 'd33'), ('opp', other, 'd33')):
            points = c[prefix + '_' + kind]
            if not points:
                continue
            if owner == 'my' or kind == 'five':
                return owner + '-' + kind, {p: 100.0 for p in points}
            defense = {}
            for r, col in points:
                for point, weight in self._defense_candidates(
                        board, r, col, player=3 - player).items():
                    defense[point] = max(defense.get(point, 0), weight)
            return owner + '-' + kind, defense
        return None, {}

    def _defense_candidates(self, board, r, c, is_kill=True, player=BLACK):
        """防守候选（带权重）——模拟落子反推必杀点 (r,c) 的堵法。
        player 指定“被模拟落子的一方”（即对方威胁方）：
          - 白方防守黑方威胁时 player=BLACK（默认，保持旧行为）
          - 黑方防守白方威胁时 player=WHITE（机机对战对称）
        1. 临时模拟 player 落 (r,c)，analyze_point 得各方向棋型记录
        2. 中心 (r,c) = 100 分（is_kill=True，占掉落点即破坏杀型）或 60 分（活三）
        3. 对必杀级记录（LIVE4/SLEEP4/LIVE3）的每条线，按 l_edge/r_edge
           提取线端空位（类型 1/3/4 = 活端/半活端/断裂端）作为端点候选 = 80 分
           （is_kill）或 60 分；再按 gap_location 提取缝（两个 player 子之间的空位，
           xx_xx 中间格）作为候选，同样 = 80/60 分。数据与 scan_two 一致。
        返回 {(r, c): score}（score 用于深度推演的温度分配/排序）。
        """
        center_w = 100.0 if is_kill else 60.0
        end_w = 80.0 if is_kill else 60.0
        pts = {(r, c): center_w}
        board[r][c] = player
        try:
            recs = self.pattern.analyze_point(board, r, c, player)
            for rec in recs:
                parent = rec['parent']
                if parent not in ('LIVE4', 'SLEEP4', 'LIVE3'):
                    continue
                dr, dc = rec['dir']
                # ① 端点提取：每条线取两端空位，l_edge 用负方向(减)、r_edge 用正方向(加)，
                #    距离 d 与方向 (dr,dc) 结合得端空位坐标。
                #    第一步过滤【有没有子】：d=1 = 端紧贴成形点、中间无 player 子（无子方向，
                #    如活四的外侧空端）→ 不提取；d>1 = 端跨过 player 子链（有子方向）→ 提取。
                #    第二步过滤【端类型】：与 pattern._classify 的 OPEN_END 一致 (1,3,4,5)
                #    1=活(__) 3=半活(_|) 4=断裂端(第二个_x) 5=延伸端(__x)
                for (d, t), sign in ((rec['l_edge'], -1), (rec['r_edge'], +1)):
                    if d <= 1:
                        continue                # 无子方向：中间没有 player 子，端无防守意义
                    if t not in (1, 3, 4, 5):
                        continue
                    pr, pc = r + sign * d * dr, c + sign * d * dc   # 该端空位（紧贴段外）
                    if 0 <= pr < SIZE and 0 <= pc < SIZE and board[pr][pc] == EMPTY:
                        pts[(pr, pc)] = end_w
                # ② gap 提取：缝 = 两个 player 子之间的空位（xx_xx 的中间格）。
                #    gap_location = 缝到基子的距离（0 = 无缝）。左右两侧都试，
                #    验证缝两侧紧邻格都是 player 子才确认是真缝（兼容双缝拆分的记录）。
                #    注意：模拟落子在扫描期间必须保留在棋盘上，否则紧贴威胁点的缝
                #    会把威胁点自身误判成空位而漏掉（F9/G9 类型防守点）。
                gl = rec.get('gap_location') or 0
                if gl:
                    for sign in (-1, 1):
                        pr, pc = r + sign * gl * dr, c + sign * gl * dc
                        if not (0 <= pr < SIZE and 0 <= pc < SIZE) or board[pr][pc] != EMPTY:
                            continue
                        lr, lc = r + sign * (gl - 1) * dr, c + sign * (gl - 1) * dc
                        rr2, rc2 = r + sign * (gl + 1) * dr, c + sign * (gl + 1) * dc
                        if (0 <= lr < SIZE and 0 <= lc < SIZE and board[lr][lc] == player
                                and 0 <= rr2 < SIZE and 0 <= rc2 < SIZE and board[rr2][rc2] == player):
                            pts[(pr, pc)] = end_w
        finally:
            board[r][c] = EMPTY
        return pts

    # ---------- 子类排序键与评分 ----------

    @staticmethod
    def _threat_score(subclass):
        """棋型基础分：查 TYPE_SCORE 表（a2/a1/b 系），未列返回 0。
        用于候选点综合分（无必杀时平滑加权选点）。"""
        return TYPE_SCORE.get(subclass, 0)

    @staticmethod
    def _poten_key(subclass):
        """潜力 4 子类排序：b 越多越强（b_bbb > b_bbc > b_bcc > b_ccc）。
        编号 13-16，b 数 = 16 - subclass（bbb=3 ... ccc=0）"""
        return (16 - subclass,)

    @staticmethod
    def _proximity(board, r, c, player):
        """邻近度（单口径）：周围 2 格内 player 棋子的 距离衰减 × 方向价值 加权和。
        距离：紧贴(1格)×1.0、隔格(2格)×0.5；方向：同线可连×2.0、异线×0.5。
        值越大 = 该点越贴近己方棋群（布局连贯性）。"""
        score = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if not (0 <= rr < SIZE and 0 <= cc < SIZE) or board[rr][cc] != player:
                    continue
                dist = max(abs(dr), abs(dc))          # 切比雪夫距离
                w_dist = PROX_DIST_W.get(dist, 0.5)
                on_line = (dr == 0 or dc == 0 or abs(dr) == abs(dc))
                w_dir = PROX_DIR_W['line'] if on_line else PROX_DIR_W['other']
                score += w_dist * w_dir
        return score

    def _neighbor_score(self, board, r, c, player, mode):
        """邻近度综合 = 自家邻近×w_self + 对家邻近×w_opp。
        mode 决定权重：'attack'(110,-10) 纯攻自家优先/对家为负；
        'defend'(30,70) 纯防对家优先；'dual'(60,40) 攻防一体自家略高。"""
        w_self, w_opp = NEIGHBOR_W[mode]
        self_cnt = self._proximity(board, r, c, player)
        opp_cnt = self._proximity(board, r, c, 3 - player)
        return self_cnt * w_self + opp_cnt * w_opp

    def _dual_score(self, board, r, c):
        """攻防一体加分：黑白双方该点【都是威胁(1)】→ +DUAL_BONUS，否则 0。
        单侧威胁(1)+另一侧潜力(2) 不算攻防一体（12 不算）。"""
        w = SUBCLASS_PARENT[self.sw[r][c]] == THREAT
        b = SUBCLASS_PARENT[self.sb[r][c]] == THREAT
        return DUAL_BONUS if (w and b) else 0.0

    def _candidate_score(self, board, r, c, player, mode):
        """候选点综合分（无必杀时平滑加权，不暴力分档），标准化到满分 100：
        分 = (棋型分 + 邻近归一×30 + 攻防一体) / 260 × 100
        棋型分覆盖威胁(1)与潜力(2)（无必杀时布局点也可入选）。
        三参数平滑融合：威胁为主，邻近/攻防作为加分，避免"只有威胁定生死"的暴力分层。
        注：已回退"查表颜色与邻近视角分离"的修改（该修改导致引擎执黑时防守
        名额不足、黑白失衡、GA 参数寻优失效；原逻辑黑白均衡）。"""
        state = self.sw if player == WHITE else self.sb
        type_score = TYPE_SCORE.get(state[r][c], 0.0)
        neighbor = (self._neighbor_score(board, r, c, player, mode) / NEIGHBOR_NORM) * W_NEIGHBOR
        dual = self._dual_score(board, r, c)
        return (type_score + neighbor + dual) / SCORE_MAX * 100.0

    # ---------- 动态攻防配比 ----------

    def _count_subclasses(self, board, state, ids):
        """统计状态表中属于 ids 子类集合的空位点数。"""
        cnt = 0
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] == EMPTY and state[r][c] in ids:
                    cnt += 1
        return cnt

    def _pressure_stats(self, board, player=WHITE):
        """当前压力统计（按 player 视角）：
        返回 (my_a1, my_a2, my_b, opp_a1, opp_a2, opp_b)，
        只统计空位；0 级必杀不参与（上层已单独处理）。"""
        my_state = self.sw if player == WHITE else self.sb
        opp_state = self.sb if player == WHITE else self.sw
        my_a1 = self._count_subclasses(board, my_state, A1_SUBCLASSES)
        my_a2 = self._count_subclasses(board, my_state, A2_SUBCLASSES)
        my_b = self._count_subclasses(board, my_state, B_SUBCLASSES)
        opp_a1 = self._count_subclasses(board, opp_state, A1_SUBCLASSES)
        opp_a2 = self._count_subclasses(board, opp_state, A2_SUBCLASSES)
        opp_b = self._count_subclasses(board, opp_state, B_SUBCLASSES)
        return my_a1, my_a2, my_b, opp_a1, opp_a2, opp_b

    def _zero_momentum(self, player=WHITE):
        """历史 0 级杀势动量：窗口内‘新出现杀势’事件数之差（按 player 视角）。"""
        return (sum(self.zero_history[player]) - sum(self.zero_history[3 - player])) * GEAR_W_ZERO

    def _record_zero_event(self, board, player):
        """落子后记录：刚落子方是否新形成 0 级杀势。
        同一杀势若连续存在于多步，只在第一次出现时记 1 次。"""
        state = self.sb if player == BLACK else self.sw
        has_kill = any(board[r][c] == EMPTY and SUBCLASS_PARENT[state[r][c]] == KILL
                       for r in range(SIZE) for c in range(SIZE))
        self.zero_history[player].append(int(has_kill and not self.zero_active[player]))
        self.zero_history[player] = self.zero_history[player][-max(1, GEAR_ZERO_WINDOW):]
        self.zero_active[player] = has_kill

    def _target_gear(self, board, player=WHITE):
        """纯计算目标攻防档位（不更新 self.gear，不滞回）。
        用于深推上下文特征记录，避免污染真实档位状态。
        返回 1-4；若 GEAR_ENABLED=False 返回 DEFAULT_GEAR。"""
        if not GEAR_ENABLED:
            return DEFAULT_GEAR

        my_a1, my_a2, my_b, opp_a1, opp_a2, opp_b = self._pressure_stats(board, player)
        if opp_a1 >= 2 and my_a1 == 0:
            return 1
        if my_a1 >= 2 and opp_a1 == 0:
            return 4
        if opp_a1 + opp_a2 >= 3 and my_a1 + my_a2 <= 1:
            return 1
        if my_a1 + my_a2 >= 3 and opp_a1 + opp_a2 <= 1:
            return 4
        my_pressure = my_a1 * GEAR_W_A1 + my_a2 * GEAR_W_A2 + my_b * GEAR_W_B
        opp_pressure = opp_a1 * GEAR_W_A1 + opp_a2 * GEAR_W_A2 + opp_b * GEAR_W_B
        score = (my_pressure - opp_pressure) + self._zero_momentum(player)
        t1, t2, t3 = GEAR_THRESHOLDS
        if score >= t1:
            return 4
        elif score >= t2:
            return 3
        elif score >= t3:
            return 2
        else:
            return 1

    def _choose_gear(self, board, player=WHITE):
        """根据当前局势选择攻防档位（只应在双方都无 0 级时调用）。
        规则：
          1. 硬约束：对方双冲四且我方无冲四 → 1攻4防；我方双冲四且对方无 → 4攻1防。
          2. 常规：score = 当前压力差 + 历史0级动量，按阈值映射四档。
          3. 滞回：每次最多升/降一档，避免临界抖动。
        player 指定视角（WHITE/BLACK），机机对战可让黑白都用动态配比。
        返回档位 1-4。"""
        if not GEAR_ENABLED:
            self.gear[player] = DEFAULT_GEAR
            return self.gear[player]

        my_a1, my_a2, my_b, opp_a1, opp_a2, opp_b = self._pressure_stats(board, player)

        # 硬约束（优先级最高，直接生效不等待滞回）
        if opp_a1 >= 2 and my_a1 == 0:
            self.gear[player] = 1
            return self.gear[player]
        if my_a1 >= 2 and opp_a1 == 0:
            self.gear[player] = 4
            return self.gear[player]
        if opp_a1 + opp_a2 >= 3 and my_a1 + my_a2 <= 1:
            self.gear[player] = 1
            return self.gear[player]
        if my_a1 + my_a2 >= 3 and opp_a1 + opp_a2 <= 1:
            self.gear[player] = 4
            return self.gear[player]

        my_pressure = my_a1 * GEAR_W_A1 + my_a2 * GEAR_W_A2 + my_b * GEAR_W_B
        opp_pressure = opp_a1 * GEAR_W_A1 + opp_a2 * GEAR_W_A2 + opp_b * GEAR_W_B
        score = (my_pressure - opp_pressure) + self._zero_momentum(player)
        t1, t2, t3 = GEAR_THRESHOLDS
        if score >= t1:
            target = 4
        elif score >= t2:
            target = 3
        elif score >= t3:
            target = 2
        else:
            target = 1

        # 常规路径滞回：最多一次调一档（只影响当前 player 的档位）
        if target > self.gear[player]:
            self.gear[player] = min(4, self.gear[player] + 1)
        elif target < self.gear[player]:
            self.gear[player] = max(1, self.gear[player] - 1)
        return self.gear[player]

    # ---------- 候选等效去重（对称点/同型点合并，防止同类型点占满名额） ----------

    def _board_symmetries(self, board):
        """返回保持当前局面不变的 D4 变换列表（恒等+3旋转+4镜像）。
        局面在某变换 f 下不变 → 互为 f 镜像的两个候选点完全等效。"""
        n = SIZE
        def ident(r, c): return (r, c)
        def rot90(r, c): return (c, n - 1 - r)
        def rot180(r, c): return (n - 1 - r, n - 1 - c)
        def rot270(r, c): return (n - 1 - c, r)
        def mir_h(r, c): return (n - 1 - r, c)
        def mir_v(r, c): return (r, n - 1 - c)
        def mir_d1(r, c): return (c, r)
        def mir_d2(r, c): return (n - 1 - c, n - 1 - r)
        syms = []
        for f in (ident, rot90, rot180, rot270, mir_h, mir_v, mir_d1, mir_d2):
            ok = True
            for r in range(n):
                for c in range(n):
                    rr, cc = f(r, c)
                    if board[r][c] != board[rr][cc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                syms.append(f)
        return syms

    def _pattern_sig(self, board, r, c, player):
        """候选点 (r,c) 落子后的方向无关棋型签名（等效点近似检测）。
        签名 = 4 方向棋型记录的排序元组（去掉方向），同签名视为近似等效。"""
        board[r][c] = player
        try:
            recs = self.pattern.analyze_point(board, r, c, player)
        finally:
            board[r][c] = EMPTY
        sig = []
        for rec in recs:
            sig.append((rec['parent'], rec['stones'], rec['gap_location'],
                        rec['l_edge'][0], rec['l_edge'][1],
                        rec['r_edge'][0], rec['r_edge'][1]))
        return tuple(sorted(sig))

    def _dedup_sig(self, board, syms, r, c, player):
        """候选等效签名：
        - 局面有对称性（syms>1）→ 对称轨道（严格等效）；
        - 非对称局面 → 棋型签名（近似等效：落子后棋型相同归一类）。"""
        if len(syms) > 1:
            orbit = tuple(sorted(set(f(r, c) for f in syms)))
            return ('sym', orbit)
        return ('pat', self._pattern_sig(board, r, c, player))

    def _fallback(self, board, gear=None, player=WHITE, with_score=False):
        """兜底候选（无必杀时）：默认 3 攻 + 2 防 = 5 个点，保证不重复。
        传入 gear 时按档位取 (攻, 防) 配比；gear=None 保持固定 3攻2防
        （深推内部与黑应手继续用固定配比，保证搜索树稳定）。

        player 指定“进攻方”视角：
          - WHITE：白攻 TopN + 黑防 TopM（AI 执白的第一层候选）
          - BLACK：黑攻 TopN + 白防 TopM（深推中黑方应手，对称）
        候选池 = 双方状态表中 威胁(1) ∪ 潜力(2) 的点。
        排序键 = 综合分（棋型+邻近+攻防一体），平滑加权。
        with_score=True 时返回 [(r, c, score)]——分数即 _candidate_score
        0~100 综合分，供温度银行按候选质量分配推演深度（原实现只用于
        排序挑选、返回时丢弃，深推展开被硬编码 60，强弱点推得一样浅）。"""
        syms = self._board_symmetries(board)
        def top_pts(state, player, n, mode, exclude=()):
            pts = [(r, c) for r in range(SIZE) for c in range(SIZE)
                   if board[r][c] == EMPTY and (r, c) not in exclude
                   and SUBCLASS_PARENT[state[r][c]] in (THREAT, POTEN)]
            def score(p):
                # 攻防一体点优先用 dual；否则按列表语义取 attack/defend。
                m = 'dual' if self._dual_score(board, p[0], p[1]) > 0 else mode
                return self._candidate_score(board, p[0], p[1], player, m)
            pts.sort(key=score, reverse=True)
            # 等效去重：对称轨道（严格）/棋型签名（近似），每类取最高分代表，
            # 防止同类型点（对称镜像/同棋型近邻）占满名额、漏掉其他类型。
            chosen = []
            seen = set()
            for p in pts:
                sig = self._dedup_sig(board, syms, p[0], p[1], player)
                if sig in seen:
                    continue
                seen.add(sig)
                if with_score:
                    chosen.append((p[0], p[1], score(p)))
                else:
                    chosen.append(p)
                if len(chosen) >= n:
                    break
            return chosen
        if gear is None:
            attack_n, defend_n = GEAR_PROFILE[DEFAULT_GEAR]
        else:
            attack_n, defend_n = GEAR_PROFILE[gear]
        if player == WHITE:
            attack_state, attack_player = self.sw, WHITE
            defend_state, defend_player = self.sb, BLACK
        else:
            attack_state, attack_player = self.sb, BLACK
            defend_state, defend_player = self.sw, WHITE
        # 攻防模式按“列表语义”显式区分，黑白对称：
        #   attack 列表用 attack 权重（自家棋型/连线优先，压对方 -10）；
        #   defense 列表用 defend 权重（对方威胁分 + 靠近己方防守子）。
        # 旧实现按 player==WHITE 硬编码模式，导致黑方进攻候选被当防守评分，
        # 机机/黑方对局不主动造威胁；这里修正为按列表语义传参。
        attack = top_pts(attack_state, attack_player, attack_n, 'attack')
        used = set(p[:2] for p in attack)
        defend = top_pts(defend_state, defend_player, defend_n, 'defend', exclude=used)
        return attack + defend

    # ---------- 状态描述 ----------

    def describe(self, board):
        """当前棋盘的黑白威胁分布概览（调试/对局展示用）：
        分别统计黑表 sb、白表 sw，按父类聚合空位点并附子类名：
          必杀(KILL) / 威胁(THREAT) / 潜力(POTEN)（无用不列出）
        每类最多列出前 10 个（超出加 ' ...'），坐标显示为字母列+数字行（如 H8）。"""
        def collect(state):
            d = {KILL: [], THREAT: [], POTEN: []}
            for r in range(SIZE):
                for c in range(SIZE):
                    if board[r][c] == EMPTY:
                        parent = SUBCLASS_PARENT[state[r][c]]
                        if parent in d:
                            d[parent].append((r, c, SUBCLASS_NAMES[state[r][c]]))
            return d
        def fmt(lst, n=10):
            if not lst:
                return '无'
            s = ', '.join('%s%d(%s)' % ('ABCDEFGHIJKLMNO'[c], r + 1, name) for r, c, name in lst[:n])
            return s + (' ...' if len(lst) > n else '')
        db, dw = collect(self.sb), collect(self.sw)
        lines = []
        lines.append('黑表：必杀=%s | 威胁=%s | 潜力=%s' % (fmt(db[KILL]), fmt(db[THREAT]), fmt(db[POTEN])))
        lines.append('白表：必杀=%s | 威胁=%s | 潜力=%s' % (fmt(dw[KILL]), fmt(dw[THREAT]), fmt(dw[POTEN])))
        return '\n'.join(lines)
