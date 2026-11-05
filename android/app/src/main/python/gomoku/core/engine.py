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

import sys
import json
from random import SystemRandom

from .utils import SIZE, EMPTY, BLACK, WHITE
from .pattern import PatternAnalyzer
from .search import Search, SUBCLASS_NAMES, SUBCLASS_PARENT
from .deep_search import DeepSearch
from .opening_book import (OpeningBook, FRONT_STONES_BEFORE, _opening_move,
                           opening_fallback)


class Engine:
    """纯算法引擎：组织候选、深推和五态排序，不加载或调用神经网络。"""

    def __init__(self):
        self.pattern = PatternAnalyzer()
        self.search = Search(self.pattern)        # 状态表 + 候选 + 评分兜底
        self.opening_book = OpeningBook()
        self._reset_opening_orientation()
        self.opening_source = None  # 本次决策来自可靠谱着、后续谱候选或原算法。
        self.opening_detail = None  # 命中谱着时的数据来源标签（不进入返回结构）。
        self.deep_search = DeepSearch(self.search)  # 温度银行深推
        self.moves = []                           # 真实对局着法 [(r,c,player)]（记忆匹配用）
        self._searched_sides = set()
        self._loaded_position = False

    # ---------- 状态维护 ----------

    def on_move(self, board, r, c, player):
        """任意方落子后：委托 Search 局部更新黑白状态表，
        并记录真实着法序列（记忆匹配用；深推模拟走 search.on_move，不经过这里）。"""
        self.search.on_move(board, r, c, player)
        self.moves.append((r, c, player))
        self.deep_search.advance_search_cache(board, r, c, player)

    def _reset_opening_orientation(self):
        """只在创建引擎或明确新对局时抽取方向，与其他算法的随机序列隔离。"""
        self._opening_symmetry = SystemRandom().randrange(8)

    def reset(self, moves=()):
        """重建真实棋谱及搜索状态，保留温度及本局开局方向（包括悔棋到空盘）。"""
        moves = tuple(moves)
        self._searched_sides.clear()
        self._loaded_position = bool(moves)  # 非空重建/导入标记为中间冷启动，不沿用旧树。
        self.search = Search(self.pattern)
        self.deep_search.search = self.search
        self.deep_search.clear_search_cache()
        self.moves = []
        self.opening_source = None
        board = [[EMPTY] * SIZE for _ in range(SIZE)]
        for move in moves:
            r, c, player = (move['r'], move['c'], move['player']) if isinstance(move, dict) else move
            board[r][c] = player
            self.on_move(board, r, c, player)

    # ---------- 核心动作：选点 ----------

    def compute_move(self, board, player=WHITE):
        """统一经analyze_turn建树；查谱阶段仍保留旧的单点列表返回形态。"""
        opening_form = self._opening_candidates(board, player) is not None
        move = self.analyze_turn(board, player)['part4_result']['move']
        # 这里只兼容返回形态，实际落点始终取本次分析/缓存，不能再绕过深推。
        return [move] if opening_form and move is not None else move

    def _opening_candidates(self, board, player=WHITE, tactical=None):
        """整盘前5手由前5手库接管；计数取盘面，不依赖协议着法顺序。

        compute_move保留开局返回列表的既有接口。成五/堵五交回战术逻辑；
        谱着只固定根落点，仍按预算搜索其后续；出谱交回普通候选。
        第6手起（着手前盘面已有5子）不再接管，也不会回落到旧库接管。
        """
        if (sum(cell != EMPTY for row in board for cell in row) > FRONT_STONES_BEFORE
                or sum(row.count(player) for row in board) >= 3):
            return None
        reason, points = (self.search.tactical_candidates(board, player)
                          if tactical is None else tactical)
        if reason in ('my-five', 'opp-five'):
            return None
        move, self.opening_detail = _opening_move(
            self.opening_book, board, player, points, self._opening_symmetry)
        # 公开取值仍为 'book'；具体数据来源只保存在 opening_detail，不进返回结构。
        self.opening_source = 'book' if move is not None else None
        return [move] if move is not None else None

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

    def _rank_search(self, board, points, player, complete_scores):
        """正式开局160、中途120；显式实验可覆盖二者，续温仍由cached_turn处理。"""
        d = self.deep_search
        middle = self._loaded_position or player in self._searched_sides
        d.last_temperature_kind = 'middle_cold' if middle else 'opening_cold'
        d._cold_temperature = d.MIDDLE_COLD_T0 if middle else d.OPENING_COLD_T0
        self._searched_sides.add(player)
        try:
            return (d.rank_candidates(board, points, player, complete_scores=True)
                    if complete_scores else d.rank_candidates(board, points, player))
        finally:
            d._cold_temperature = None

    def analyze_turn(self, board, player, *, complete_scores=False):
        """轮到 player 时的完整分析，输出 4 部分（agent 决策界面）：
        1. part1_opp_threat：对方视角威胁分布（0必杀/1威胁/2潜力/3无用，各含点位+子类名）
        2. part2_my_threat ：己方视角威胁分布（同上）
        双方合计第1—5手命中前5手库则固定落点并续推（成五/堵五优先）。
        接管以外的旧谱点只补入普通根候选；所有谱点保留原评分与五态语义。
        3. part3_candidates：初始候选点，按检测优先级【命中即止】取第一个命中的类型：
             my-five→direct(落子即胜)   / opp-five→direct(固定堵点后续推)
             my-vcf(活四/44/43)→direct  / opp-vcf→defense(反推堵点,带权重dict)
             my-d33→direct              / opp-d33→defense(反推堵点,带权重dict)
             全无→fallback(五档动态攻防，普通池最多4点)
           points 形态：direct 为 [(r,c),...] 列表；
                        defense/fallback 为 {(r,c): 综合权重} 字典（供排序及路线质量使用）
        4. part4_result：{'type': 'direct'|'searched', 'move': (r,c), 'ranked': [...]}
             direct → 仅实际成五直接结束，硬胜+1000；
             searched → 硬态±1000定分并允许早停，其他候选保留原加权估分。
        """
        self.opening_source = None
        # 本次结果刷新前先清空旧报告；成五直取不能残留上一手汇总。
        self.deep_search.root_contribution_report = None
        # 成五直取不会进入rank_candidates，也要清掉上一手的叶子统计。
        self.deep_search.reset_leaf_eval_stats()
        opp_state = self.search.sw if player == BLACK else self.search.sb
        my_state = self.search.sb if player == BLACK else self.search.sw
        result = {
            'turn': 'black' if player == BLACK else 'white',
            'part1_opp_threat': self._threat_stats(board, opp_state),
            'part2_my_threat': self._threat_stats(board, my_state),
        }
        # 命中已搜索节点时，候选和中间边直接继承；只有叶子允许生成新候选。
        cached = self.deep_search.cached_turn(board, player)
        if cached is not None:
            self._searched_sides.add(player)
            self.deep_search.last_temperature_kind = 'continuation'
            points, reason, ranked = cached
            result['part3_candidates'] = {
                'type': 'defense' if reason in ('opp-vcf', 'opp-d33') else ('direct' if reason else 'fallback'),
                'reason': reason or 'none', 'points': points}
            best = ranked[0] if ranked else None
            result['part4_result'] = {
                'type': 'searched', 'move': (best['r'], best['c']) if best else None,
                'ranked': ranked}
            return result
        reason, tactical = self.search.tactical_candidates(board, player)
        if reason in ('opp-vcf', 'opp-d33') and not tactical:
            ranked = self.deep_search._no_defense_turn(board, player, reason)
            self._searched_sides.add(player)
            self.deep_search.last_temperature_kind = 'hard_terminal'
            best = ranked[0] if ranked else None
            result['part3_candidates'] = {'type': 'defense', 'reason': reason, 'points': {}}
            result['part4_result'] = {'type': 'searched',
                'move': (best['r'], best['c']) if best else None, 'ranked': ranked}
            return result
        opening = self._opening_candidates(board, player, (reason, tactical))
        if opening:
            move = opening[0]
            result['part3_candidates'] = {
                'type': 'direct', 'reason': 'opening-' + self.opening_source, 'points': opening}
            mode = 'dual' if self.search._dual_score(board, *move) > 0 else 'attack'
            weight = min(max(float(self.search._candidate_score(board, *move, player, mode)), 1.0), 100.0)
            # 保留谱点综合分；普通扣温在深推中独立读取落子方纯棋型分。
            ranked = self._rank_search(board, {move: weight}, player, complete_scores)
            result['part4_result'] = {
                'type': 'searched', 'move': move, 'ranked': ranked,
            }
            return result
        if reason:
            defense = reason in ('opp-vcf', 'opp-d33')
            cand = {'type': 'defense' if defense else 'direct',
                    'reason': reason,
                    'points': tactical if defense else list(tactical)}
        else:
            # 根层和深推使用同一五档普通候选入口，普通池最多4点。
            # 模拟层与真实层共用档位政策，缓存节点沿用旧边而不重新取点。
            # fallback 候选带 _candidate_score 真实分（dict 形态），
            # 综合分用于排序和路线质量，普通扣温不包含邻近/经验加分。
            # 叠加记忆经验修正（move_memory_bonus）：历史赢谱落点加分、
            # 败谱落点减分——记忆参与候选打分（8 对称归一化，见 memory.py）。
            mem_bonus = self._memory_bonus()
            pts = {}
            for (rr, cc), s in [(p[:2], p[2]) for p in
                                self.search._ordinary_candidates(board, player=player, with_score=True)]:
                if (rr, cc) not in pts or s > pts[(rr, cc)]:
                    pts[(rr, cc)] = min(max(float(s) + mem_bonus.get((rr, cc), 0.0), 1.0), 100.0)
            # 第四次落子起，谱点只补入根候选；不改递归宽度、温度或五态排序。
            # 使用原综合分作为排序/路线权重，不能把棋谱可信度直接当作温度。
            book_move = self.opening_book.lookup(board, player)
            if book_move is not None:
                self.opening_source = 'candidate'
                if book_move not in pts:
                    rr, cc = book_move
                    mode = 'dual' if self.search._dual_score(board, rr, cc) > 0 else 'attack'
                    score = self.search._candidate_score(board, rr, cc, player, mode)
                    pts[book_move] = min(max(float(score) + mem_bonus.get(book_move, 0.0), 1.0), 100.0)
            if not pts:
                # 谱外仅一子等稀疏局面可能没有潜力/威胁候选；只在空池时补合法点。
                # 复用已有几何工具，不再让它绕过正常多候选搜索或产生开局标签。
                legal = opening_fallback(board, player)
                if legal is not None:
                    score = self.search._candidate_score(board, *legal, player, 'attack')
                    pts[legal] = min(max(float(score), 1.0), 100.0)
            cand = {'type': 'fallback', 'reason': 'none', 'points': pts}
        result['part3_candidates'] = cand
        # 4. 深度推演结果
        #   仅my-five落子即终局可跳过搜索；堵五保持原来第一个堵点并续推。
        #   唯一候选只免选点比较，仍沿该点搜索，保留权重/欠温/子树。
        #   注意两处重构修复：
        #   a) defense 候选必须保留 dict 权重（100/80 驱动防守点推深）——9947a96
        #      曾转 list 等权 60 → dt(60)=67>T0 → 防守深推浅化、乱选（黑弱根因）；
        #   b) VCF/双三候选（列表无权重）补 100 权重——等权 60 同样推不动，
        #      浅推会漏杀/犹豫；深推验证后再落子。
        pts = cand['points']
        if cand['type'] == 'direct' and cand.get('reason') == 'my-five':
            first = pts[0] if isinstance(pts, list) else next(iter(pts))
            result['part4_result'] = {
                'type': 'direct',
                'move': first,
                'ranked': [self.deep_search._candidate_result(
                    first[0], first[1], 1e9,
                    self.deep_search.STATE_WIN)],
            }
        else:
            if cand['type'] == 'direct' and isinstance(pts, list):
                if cand.get('reason') == 'opp-five':
                    # 不改原来堵五直取的落点；多个成五点也不虚构可守。
                    first = pts[0]
                    pts = {first: tactical[first]}
                else:
                    # my-vcf / my-d33原来已用100权重，保持该语义。
                    pts = {p: 100.0 for p in pts}
            ranked = self._rank_search(board, pts, player, complete_scores)
            best = ranked[0] if ranked else None
            result['part4_result'] = {
                'type': 'searched',
                'move': (best['r'], best['c']) if best else None,
                'ranked': ranked,
            }
        return result

    # ---------- 普通对局记忆 ----------

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
