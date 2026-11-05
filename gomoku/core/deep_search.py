# ============================================================
# deep_search.py —— 深度推演层（温度银行）
# 职责：对候选点做多步递归推演，返回分支分最高的落子点
#
# 核心机制：
#   - 温度银行：普通步按落子方的纯棋型分扣温，综合候选分仍用于排序/路线权重
#       w_type = TYPE_SCORE[落子前预测子类]/SCORE_MAX*100
#       DT = DT_M*((100-w_type)/100)^DT_POW + DT_N
#       战术前8手免耗温，之后按2/4/8梯度；温尽由战术预算规则停止，
#       额外8手延伸默认关闭，实际硬胜败优先停止
#   - 一层 = 本方+对方一个完整回合；候选与分数保留路径权重回溯，
#     胜败状态另按双方最优选择汇总，不将分数汇总误当成胜败证明
#   - 叶子评估：实际成五终局 ±1000；僵持叶子读当前棋盘两张状态表，
#     按成五/强必杀/弱必杀优先级判定，其余全盘加和走平滑累进税
#   - cache：按"已落子序列"缓存分支分，序列前缀复用（同序列只推一次）；
#       真实换根后的旧叶子按当前根距离拨温，每4手减半，旧欠温继续抵扣
#
# TODO（骨架保留，本次结构整理未实现）：
#   - 分支定界：UB/LB 剪枝（先深推一枝拿标杆，砍追不上的枝）
#   - 保底 3 枝：无条件开枝（防漏）
#
# 依赖：search（候选/应手来源）、utils（棋盘常量）
# ============================================================

import math

from .utils import SIZE, EMPTY, BLACK, WHITE

ROOT_SHARE = 100.0  # 贡献汇总的根份额；只用于贡献表达，不进入搜索缓存身份
from . import search as search_module


class DeepSearch:
    """深推器：温度银行驱动的递归推演。
    持有 Search 实例（提供 candidates/_fallback 取候选点），
    供 engine 的 compute_move / analyze_turn 调用。"""

    # 温度参数（结构：幂函数降温，参数交给 GA 寻优）
    # 旗舰默认：冷启动160/120，普通耗温系数200，根最低三层；GUI不再分三档。
    T0 = 160.0            # 直接DeepSearch调用预算；跨回合拨温独立设置，不是全局节点额度。
    DT_M = 200.0           # 普通步耗温系数；战术仍走独立免费额度/梯度规则。
    DT_POW = 2.0           # 幂次：控制曲线形状（GA 可调）
    DT_N = 3.0             # 基础耗温
    MIN_SEARCH_PLIES = 3   # 当前根最低三手仅豁免温度停止，不免扣温；实验可设0关闭。
    CAND_N_FULL = 5       # 兼容普通候选请求上界；战术候选不按此值截断。
    TACTICAL_BATCH_SIZE = 3  # 当前发布候选：每节点先展开前3个战术候选，其余保留pending。
    OPENING_COLD_T0 = 160.0
    MIDDLE_COLD_T0 = 120.0
    CAND_N_ORDINARY = 4   # 根层与内部共用五档攻防配额，普通池最多4点。
    # 当前发布候选：前8手战术免费，之后按2/4/8梯度；普通步仍按原公式扣温。
    # 显式None只用于回退到旧的“战术全程免费”实验口径。
    TACTICAL_FREE_PLIES = 8
    # 默认关闭温尽后的8手延伸上限，不截断仍有战术候选的分支。
    # 显式整数只供实验启用延伸限额/威胁差估分；不改变路线权重。
    TACTICAL_EXTENSION_PLIES = None
    TACTICAL_THREAT_VALUE = 200.0
    TACTICAL_THREAT_CLAMP = 4
    TACTICAL_STAGE_PLIES = 8
    TACTICAL_COSTS = (2.0, 4.0, 8.0)
    WIN_SCORE = 1000.0     # 实际成五及已证明硬胜败的定分；其他叶子仍独立估值
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
    # 跨回合拨温：64 * 2^(-当前根深度/4)，与旧欠温相加，负数仍保留。
    # 权重只作原有准入限制，不额外加温；硬结论和实际成五不进入续推。
    CONTINUE_W_MIN = 0.10
    CONTINUE_BASE = 64.0
    CONTINUE_HALF_DEPTH = 4.0
    THREAT_BONUS_WHITE = 100.0   # （遗留常量，不再用于叶子分）
    THREAT_BONUS_BLACK = 50.0    # （遗留常量，不再用于叶子分）

    def __init__(self, search):
        self.search = search        # Search 实例：候选点来源
        self._root_pass = None  # 单次搜索上下文；硬态定分后不再开补分遍。
        self._cold_temperature = None  # Engine仅在本次新建树时指定；不改变续温T0。
        self.last_temperature_kind = None
        self._debug_observer = None  # 桌面调试可选观察者；不参与缓存身份或选点。
        self.clear_search_cache()
        self.reset_leaf_eval_stats()

    def _debug_emit(self, kind, node=None, **data):
        """仅调试进程启用；记录实际事件，不生成候选、不计算替代分数。"""
        if self._debug_observer is not None:
            self._debug_observer.record(kind, node, **data)

    def _debug_candidates(self, board, player, items, reason, cached=False):
        if self._debug_observer is not None:
            self._debug_emit('candidates', self._position_node(board),
                             turn=player, reason=reason, points=items, cached=cached)
        return items

    def _debug_step(self, node, before, after, weight, tactical, count, *, cost_weight=None):
        if self._debug_observer is not None:
            self._debug_emit('temperature', node, temperature_before=before,
                             temperature=after, cost=before-after, weight=weight,
                             cost_weight=weight if cost_weight is None else cost_weight,
                             tactical=tactical, tactical_count=count,
                             extension_state=self._extension_state(node=node))

    def clear_search_cache(self):
        """显式重建/悔棋清空旧树；不影响棋盘、参数或对局记忆。"""
        self._tree_root = None
        self._tree_signature = None
        self._tree_epoch = 0
        self._reuse_active = False
        self._state_cut_serial = 0
        self._cache_positions = {}
        self._cache_paths = {}
        self._cold_results = None  # 仅本次新树冷搜索使用，不跨回合保存。
        self.root_contribution_report = None  # 本次根候选的继承份额/贡献报告；不参与选棋。
        self.cache_stats = dict(tactical_hits=0, tactical_misses=0, candidate_hits=0,
                                leaf_hits=0, feature_hits=0, subtree_hits=0,
                                frontier_resumes=0, cold_result_hits=0,
                                tactical_cutoffs=0, tactical_probe_moves=0,
                                pending_expansions=0)

    @staticmethod
    def _position_key(board):
        # 保持行优先的完整字节键；逐行交给bytes打包，避免Python逐格生成。
        return b''.join(map(bytes, board))

    @staticmethod
    def _new_tree_node(key):
        return dict(board=key, children={}, tactical={}, candidates={}, values={},
                    features={}, results={}, result_epoch=-1, frontier=None,
                    resume_epoch=-1, resume_source=None, resume_values={},
                    resume_blocked=False, resume_blocked_key=None,
                    terminal=None, last_result=None, mover=None, edges=None,
                    main_player=None, move_result=None, continued_epoch=-1,
                    tactical_count=0, extension_state=(0, 0, 0), contribution_unit=None,
                    contribution_complete=False, contribution_leaves=0,
                    pending_edges=(), batch_epoch=-1, batch_moves=(),
                    branch_temperature=None, gear_context=None)

    def advance_search_cache(self, board, r, c, player):
        """只在真实落子后换根；未命中就清掉旧树，模拟落子不调用这里。"""
        key = self._position_key(board)
        previous = self._tree_root
        node = (self._tree_root['children'].get((r, c, player))
                if self._tree_root is not None else None)
        self._tree_root = (node if node is not None and node['board'] == key
                           and node.get('gear_context') in (None, self.search._gear_context())
                           else self._new_tree_node(key))
        self._tree_root['mover'] = player
        if self._debug_observer is not None:
            self._debug_emit('reroot', self._tree_root, previous=previous,
                             move=(r, c, player), hit=self._tree_root is node)
        self._tree_epoch += 1
        self._cache_positions = {}
        self._cache_paths = {}

    def _reuse_signature(self):
        # 参数寻优会临时改类常量/模块权重；身份不同不能继承旧树结论。
        constants = tuple((name, repr(getattr(self, name))) for name in dir(type(self))
                          if name.isupper())
        weights = tuple((name, repr(value)) for name, value in vars(search_module).items()
                        if name.isupper())
        methods = (self._candidate_points, self._leaf_score_impl, self._check_win,
                   self._pattern_temperature_weight,
                   self.search.tactical_candidates, self.search._defense_candidates,
                   self.search._ordinary_candidates, self.search._gear_threats)
        return constants, weights, methods

    def _initial_temperature(self):
        """根建树预算与跨回合拨温分开；直接DeepSearch调用仍使用T0。"""
        return float(self.T0 if self._cold_temperature is None else self._cold_temperature)

    def _begin_search_cache(self, board, trace):
        self._cold_results = None
        self.cache_stats = dict.fromkeys(self.cache_stats, 0)
        # 路径表达式记录器和替换过的评分器不能冒充纯局面评估。
        self._cache_credit = float(self.CONTINUE_BASE)
        self._reuse_active = (type(self) is DeepSearch and not trace
                              and '_leaf_score_impl' not in self.__dict__)
        if not self._reuse_active:
            return
        signature = self._reuse_signature()
        key = self._position_key(board)
        if signature != self._tree_signature:
            self._tree_root = None
            self._tree_signature = signature
        if (self._tree_root is None or self._tree_root['board'] != key
                or self._tree_root.get('gear_context') not in (None, self.search._gear_context())):
            self._tree_root = self._new_tree_node(key)
        self._tree_root['gear_context'] = self.search._gear_context()
        self._cache_positions = {key: self._tree_root}
        self._cache_paths = {(): self._tree_root}
        root = self._tree_root
        if self._debug_observer is not None:
            self._debug_emit('root', root, epoch=self._tree_epoch)
        if (root['main_player'] is None and not root['children']
                and root['edges'] is None and root['frontier'] is None):
            self._cold_results = {}

    def _position_node(self, board):
        if not self._reuse_active:
            return None
        key = self._position_key(board)
        node = self._cache_positions.get(key)
        if node is None:
            node = self._new_tree_node(key)
            self._cache_positions[key] = node
        return node

    def _remember_position(self, board, branch):
        if not self._reuse_active:
            return None
        path = tuple((r, c, player) for r, c, player, _w in branch)
        if not path:
            return self._tree_root
        parent = self._cache_paths[path[:-1]]
        key = self._position_key(board)
        node = parent['children'].get(path[-1])
        if node is None or node['board'] != key:
            node = self._new_tree_node(key)
            parent['children'][path[-1]] = node
        # 同盘面可共享纯局面结果；路径欠温、展开进度与聚合结果不能串到另一条路径。
        position = self._cache_positions.get(key)
        if position is not None and position is not node:
            for name in ('tactical', 'candidates', 'values', 'features'):
                position[name].update(node[name])
                node[name] = position[name]
        node['mover'] = branch[-1][2]
        node['gear_context'] = self.search._gear_context()
        self._cache_positions[key] = node
        self._cache_paths[path] = node
        if self._debug_observer is not None:
            self._debug_emit('enter', node, parent=parent, move=path[-1],
                             weight=branch[-1][3], depth=len(branch))
        return node

    def _history_key(self):
        return (tuple(tuple(self.search.zero_history[p]) for p in (BLACK, WHITE)),
                self.search._gear_context())

    def _tactical_result(self, board, player):
        node = self._position_node(board)
        if node is not None and player in node['tactical']:
            self.cache_stats['tactical_hits'] += 1
            reason, items = node['tactical'][player]
            return reason, dict(items)  # 保留顺序和权重，隔离调用方修改。
        reason, points = self.search.tactical_candidates(board, player)
        if node is not None:
            self.cache_stats['tactical_misses'] += 1
            node['tactical'][player] = (reason, tuple(points.items()))
        return reason, points

    def _resume_temperature(self, node, player, temperature, *, depth, before_move=False):
        if node is None:
            return temperature
        if node['resume_epoch'] != self._tree_epoch:
            node['resume_epoch'] = self._tree_epoch
            # 保留本轮拨温前快照，后续写回欠温不能改变同轮memo的原始依据。
            node['resume_source'] = (dict(node['frontier'])
                                     if node['frontier'] is not None else None)
            node['resume_values'] = {}
            node['resume_blocked'] = False
            node['resume_blocked_key'] = None
        old = node['resume_source']
        # 仅真实换根后处理旧欠温；同盘面反复分析、判态和补分不反复拨温。
        if (old is None or old['epoch'] >= self._tree_epoch or old['player'] != player
                or bool(old.get('before_move', False)) != before_move):
            return temperature
        key = (player, before_move)
        if key not in node['resume_values']:
            # 用当前根距离而非保存时旧深度拨温，不重扣已计算的旧前缀。
            resumed, blocked = self._continuation_temperature(old, depth=depth)
            node['resume_values'][key] = resumed
            node['resume_blocked'] = blocked
            node['resume_blocked_key'] = key
            if not blocked:
                # 即使本轮仍欠温且没有新展开，也保存还款后的余额及当前根深度。
                # 新epoch可继续偿还；同epoch仍读resume_values，不重复领取拨温。
                node['frontier'] = dict(old, temperature=resumed, depth=int(depth),
                                        epoch=self._tree_epoch)
            self.cache_stats['frontier_resumes'] += 1
            if self._debug_observer is not None:
                self._debug_emit('resume', node, old_temperature=old['temperature'],
                                 temperature=resumed, round_t0=float(self.T0),
                                 continuation_base=self._cache_credit, current_depth=depth,
                                 blocked=blocked, before_move=before_move,
                                 path_weight=old['path_weight'], old_score=old['score'],
                                 old_depth=old['depth'], tactical_count=old['tactical_count'])
        return node['resume_values'][key]

    def _resume_blocked(self, node, player, *, before_move=False):
        """返回本次跨回合恢复是否被叶子续推门槛拦截。"""
        return bool(node is not None and node.get('resume_blocked')
                    and node.get('resume_blocked_key') == (player, before_move))

    def _continuation_temperature(self, frontier, *, depth):
        """按当前根深度拨温并抵扣旧欠温，返回(有符号温度, 硬门槛阻断)。

        state=0、+2、-2同公式，评分不再额外加温；+1/-1或终局直接复用。
        不按旧根深度停在第8层，不截到40，也不把尚未还清的欠温抹成0。
        """
        state = int(frontier.get('state', self.STATE_NONE))
        if (state in (self.STATE_WIN, self.STATE_LOSE) or frontier.get('terminal')
                or self._extension_limit_reached(frontier.get('extension_state', (0, 0, 0))[0])):
            return 0.0, True
        weight = min(max(float(frontier.get('path_weight', 0.0)), 0.0), 1.0)
        if weight < self.CONTINUE_W_MIN:
            return 0.0, True
        depth = max(0.0, float(depth))
        grant = float(self.CONTINUE_BASE) * 2.0 ** (-depth / self.CONTINUE_HALF_DEPTH)
        debt = min(0.0, float(frontier.get('temperature', 0.0)))
        return debt + grant, False

    def _remember_frontier(self, node, player, temperature, path_weight, *,
                           before_move=False, depth=0, score=None, state=0,
                           terminal=False):
        if node is not None:
            node['edges'] = None
            node['frontier'] = dict(player=player, temperature=float(temperature),
                                    path_weight=float(path_weight), epoch=self._tree_epoch,
                                    before_move=before_move, depth=int(depth),
                                    score=None if score is None else float(score),
                                    state=int(state), terminal=bool(terminal),
                                    tactical_count=node['tactical_count'],
                                    extension_state=node['extension_state'],
                                    leaf_end_reason='temperature')

    @staticmethod
    def _update_frontier(node, *, score=None, state=None):
        if node is None or node.get('frontier') is None:
            return
        if score is not None:
            node['frontier']['score'] = float(score)
        if state is not None:
            node['frontier']['state'] = int(state)

    def _result_key(self, player, main_player, temperature, path_weight, tactical_count=0,
                    extension_state=(0, 0, 0), *, depth=None):
        return (player, main_player, float(temperature), float(path_weight),
                self._history_key(), tactical_count, tuple(extension_state),
                max(0, self.MIN_SEARCH_PLIES - depth) if depth is not None else 0)

    def _cold_result_key(self, board, player, opp, main_player, temperature,
                         path_weight, depth, tactical_count=0, extension_state=(0, 0, 0)):
        """同次冷算的全分结果键；判态部分分及热树均不能进入此表。

        温度/路径权重影响后续展开与剪枝，历史影响叶子评分，深度影响
        保存的末端元数据。全部精确匹配，不用棋盘键代替有效搜索上下文。
        """
        context = self._root_pass
        if (self._cold_results is None
                or context is not None and context['mode'] != 'full'):
            return None
        t0 = self._initial_temperature() if context is None else context['t0']
        return (self._position_key(board), player, opp, main_player,
                float(temperature), float(path_weight), depth,
                self._history_key(), float(t0), tactical_count, tuple(extension_state))

    @staticmethod
    def _copy_cold_subtree(target, source):
        """复制已完成子树的路径状态；每个父节点仍独立计入原分支权重。

        完整结果在本次冷算内可复用，树结构和欠温状态不能直接共用：
        真实换根后各路径须各自续推。纯局面派生缓存沿原策略共享。
        不重新生成候选、不改变旧边，也不重新计算或折叠任何分数。
        """
        if target is source:
            return
        pending = [(target, source)]
        while pending:
            dst, src = pending.pop()
            dst.update(src)
            for name in ('results', 'resume_values', 'move_result',
                         'frontier', 'resume_source'):
                value = src[name]
                dst[name] = dict(value) if isinstance(value, dict) else value
            dst['children'] = {}
            for move, child in src['children'].items():
                copied = {}
                dst['children'][move] = copied
                pending.append((copied, child))

    def _save_tree_result(self, node, key, result):
        if node is not None:
            main_player = key[1]
            if node['edges']:
                self._aggregate_cached_move(node, main_player)
            elif (node['frontier'] or {}).get('leaf_end_reason') != 'no_common_defense':
                self._store_cached_move(node, result[0], 0, *result[2:5], main_player)
            # 部分分可以留在move_result供本轮暂估，不能进入完整结果缓存。
            value = node['move_result']
            if value is not None and value['complete']:
                node['results'][key] = result
                node['last_result'] = result

    def _activate_tactical_batch(self, node):
        """按显式战术批量解锁旧边；默认225即不分批，同盘面读不扩批。"""
        if node['batch_epoch'] == self._tree_epoch:
            return
        # 上一批可能因判态早停而尚有边未实际展开；这些也要占本轮批量名额。
        pending = tuple(edge for edge in node['edges'] or ()
                        if node['children'].get(edge[:3]) is None
                        or node['children'][edge[:3]]['move_result'] is None)
        if node.get('selection_reason') and pending:
            limit = self.TACTICAL_BATCH_SIZE
            added, remaining = pending[:limit], pending[limit:]
            node['pending_edges'] = remaining
            node['batch_moves'] = tuple(edge[:3] for edge in node['edges']
                                         if edge not in remaining)
            self.cache_stats['pending_expansions'] += len(added)
        elif not pending:
            node['pending_edges'] = ()
            node['batch_moves'] = tuple(edge[:3] for edge in node['edges'] or ())
        node['batch_epoch'] = self._tree_epoch

    def _cache_edges(self, board, node, player, items, temperature, main_player):
        """保存完整保留边，返回本轮可展开子集；原剪枝剔除的点不属待展开点。"""
        if node is None:
            return items  # 无跨回合树的调用不能永久丢点，沿原完整路径计算。
        reason, tactical = self._tactical_result(board, player)
        edges = tuple((r, c, player, w,
                       temperature - self._path_temperature_cost(
                           (w if (r, c) in tactical else self._pattern_temperature_weight(player, r, c)),
                           (r, c) in tactical, node['tactical_count'], temperature),
                       path_weight) for path_weight, r, c, w in items)
        previous = node['edges']
        if previous is None:
            limit = self.TACTICAL_BATCH_SIZE
            if type(limit) is not int or limit < 1:
                raise ValueError('TACTICAL_BATCH_SIZE须为正整数')
            active = edges[:limit] if reason else edges
            node['pending_edges'] = edges[len(active):]
            node['batch_moves'] = tuple(edge[:3] for edge in active)
            node['batch_epoch'] = self._tree_epoch
            node['edges'] = edges
            node['branch_temperature'] = temperature
        elif previous != edges:
            raise RuntimeError('已有节点的候选边/温度上下文变化，不能覆盖待展开进度')
        self._activate_tactical_batch(node)
        node['main_player'] = main_player
        node['selection_reason'] = reason
        node['frontier'] = None
        node['move_result'] = None
        node['contribution_unit'] = None
        node['contribution_complete'] = False
        node['contribution_leaves'] = 0
        if self._debug_observer is not None:
            self._debug_emit('branches', node, turn=player, reason=reason,
                             edges=node['edges'], temperature=temperature, score_player=main_player,
                             pending_moves=[edge[:3] for edge in node['pending_edges']])
        allowed = set(node['batch_moves'])
        return [item for item in items if (item[1], item[2], player) in allowed]

    @staticmethod
    def _pending_state(node, states):
        # 未展开点是未知应手，不能删掉后把“已查全败”当成全体必败。
        return states + [0] if node is not None and node['pending_edges'] else states

    def _hard_score(self, state, mover, score_player, fallback):
        """硬态是节点落子方视角，分数必须转换到本次主评分方视角。"""
        if not self._state_forced(state):
            return fallback
        if mover not in (BLACK, WHITE) or score_player not in (BLACK, WHITE):
            raise ValueError('硬态评分缺少明确落子方/主评分方')
        return self.WIN_SCORE * state * (1 if mover == score_player else -1)

    def _store_cached_move(self, node, score, state, wins, losses, total,
                           main_player, complete=True):
        if node is not None:
            node['main_player'] = main_player
            edges = node['edges'] or ()
            if edges:
                children = [node['children'].get(edge[:3]) for edge in edges]
                if any(child is None or child['move_result'] is None
                       or not child['move_result']['complete'] for child in children):
                    complete = False
                    child_states = [0 if child is None or child['move_result'] is None
                                    else child['move_result']['state'] for child in children]
                    grandchildren = []
                    for child in children:
                        if child is None:
                            continue
                        for edge in child['edges'] or ():
                            grandchild = child['children'].get(edge[:3])
                            if grandchild is not None and grandchild['move_result'] is not None:
                                grandchildren.append(grandchild['move_result']['state'])
                    state = self._merge_child_states(child_states, grandchildren)
            if self._state_forced(state):
                score = self._hard_score(state, node['mover'], main_player, score)
                complete = True  # 定分已完成，不等于全部后代已展开。
            self._update_frontier(node, score=score, state=state)
            node['move_result'] = dict(score=score, state=state, wins=wins,
                                       losses=losses, total=total,
                                       complete=complete)
            node['continued_epoch'] = self._tree_epoch
            self._refresh_node_contribution(node)
            if self._debug_observer is not None:
                self._debug_emit('result', node, score=score, state=state,
                                 score_player=main_player, complete=complete,
                                 wins=wins, losses=losses, total=total)

    def _aggregate_cached_move(self, node, main_player):
        """全候选含未知项判态；硬态直接定分，其他完整分要求全部分支可用。"""
        edges = node['edges']
        if not edges:
            return None
        children = [node['children'].get(edge[:3]) for edge in edges]
        values = [None if child is None else child['move_result'] for child in children]
        ready = [(edge, value) for edge, value in zip(edges, values)
                 if value is not None and value['score'] is not None]
        if not ready:
            return None
        child_states = [0 if value is None else value['state'] for value in values]
        grandchild_states = []
        for child in children:
            if child is None:
                continue
            for edge in child['edges'] or ():
                grandchild = child['children'].get(edge[:3])
                if grandchild is not None and grandchild['move_result'] is not None:
                    grandchild_states.append(grandchild['move_result']['state'])
        score = self._weighted_average([(edge[5], value['score']) for edge, value in ready])
        state = self._merge_child_states(child_states, grandchild_states)
        wins = sum(value['wins'] for _edge, value in ready)
        losses = sum(value['losses'] for _edge, value in ready)
        selected = self._merge_fatal_states(child_states)
        if selected:
            winner = edges[0][2] if selected == self.STATE_WIN else node['mover']
            wins += int(winner == main_player)
            losses += int(winner != main_player)
        complete = not node['pending_edges'] and all(value is not None and value['complete']
                                                    for value in values)
        self._store_cached_move(node, score, state, wins, losses,
                                sum(value['total'] for _edge, value in ready) + 1,
                                main_player, complete=complete)
        return node['move_result']

    def _contribution_leaf_score(self, node):
        """严格读取叶子已有评分；move_result 明确未完成时不得回退 frontier 旧分。"""
        if node is None:
            return None
        move_result = node.get('move_result')
        if move_result is not None:
            if move_result.get('complete') and move_result.get('score') is not None:
                return float(move_result['score'])
            return None
        frontier = node.get('frontier')
        if frontier is not None and frontier.get('score') is not None:
            return float(frontier['score'])
        return None

    def _refresh_node_contribution(self, node):
        """维护与绝对份额无关的子树单位汇总；只读直接子节点汇总。"""
        if node is None:
            return
        value = node.get('move_result')
        if value is not None and value.get('complete') and self._state_forced(value.get('state', 0)):
            node['contribution_unit'] = float(value['score'])
            node['contribution_complete'] = True
            node['contribution_leaves'] = 1  # 证明节点自身是评分叶子，后代只保留作证据。
            return
        edges = node.get('edges') or ()
        if not edges:
            score = self._contribution_leaf_score(node)
            if score is None:
                node['contribution_unit'] = None
                node['contribution_complete'] = False
                node['contribution_leaves'] = 1
            else:
                node['contribution_unit'] = score
                node['contribution_complete'] = True
                node['contribution_leaves'] = 1
            return
        move_result = node.get('move_result')
        if move_result is not None and not move_result.get('complete'):
            node['contribution_unit'] = None
            node['contribution_complete'] = False
            node['contribution_leaves'] = 0
            return
        total_weight = sum(float(edge[3]) for edge in edges)
        if total_weight <= 0:
            node['contribution_unit'] = None
            node['contribution_complete'] = False
            node['contribution_leaves'] = 0
            return
        unit = 0.0
        complete = True
        leaves = 0
        children = node.get('children') or {}
        for edge in edges:
            r, c, player, weight = edge[0], edge[1], edge[2], edge[3]
            child = children.get((r, c, player))
            if child is None:
                complete = False
                continue
            leaves += int(child.get('contribution_leaves', 0))
            if not child.get('contribution_complete') \
                    or child.get('contribution_unit') is None:
                complete = False
                continue
            unit += float(weight) / total_weight * float(child['contribution_unit'])
        node['contribution_unit'] = unit if complete else None
        node['contribution_complete'] = bool(complete)
        node['contribution_leaves'] = leaves

    def _subtree_contribution(self, node, share):
        """读取节点已保存的单位汇总；不遍历后代。

        返回 (share/100 * 单位汇总, 是否完整, 叶子数)。单位汇总是该子树
        独立取得 100 份时的贡献值，与当前根绝对份额无关。
        """
        if node is None:
            return None, False, 0
        leaves = int(node.get('contribution_leaves', 0))
        if not node.get('contribution_complete') or node.get('contribution_unit') is None:
            return None, False, leaves
        return share / ROOT_SHARE * float(node['contribution_unit']), True, leaves

    def _refresh_root_contribution_report(self, ranked):
        """从当前根节点直接读取根候选汇总；覆盖上一次，不跨根累加。"""
        root = getattr(self, '_tree_root', None)
        if root is None:
            self.root_contribution_report = {
                'root_share': ROOT_SHARE, 'roots': [],
                'total': None, 'complete': False, 'leaf_count': 0,
                'reason': 'no_tree'}
            return
        hard = root.get('move_result')
        if hard is not None and hard.get('complete') and self._state_forced(hard.get('state', 0)):
            self.root_contribution_report = dict(root_share=ROOT_SHARE, roots=[],
                total=float(hard['score']), complete=True, leaf_count=1,
                reason='hard_state', root_leaf=True)
            return
        edges = root.get('edges') or ()
        if not edges:
            # 无共同防点等根叶子：没有候选边，但已有完成评分时按根 100 份计入。
            if root.get('contribution_complete') and root.get('contribution_unit') is not None:
                self.root_contribution_report = {
                    'root_share': ROOT_SHARE, 'roots': [],
                    'total': float(root['contribution_unit']),
                    'complete': True,
                    'leaf_count': int(root.get('contribution_leaves', 1)),
                    'reason': 'root_leaf', 'root_leaf': True,
                }
            else:
                self.root_contribution_report = {
                    'root_share': ROOT_SHARE, 'roots': [],
                    'total': None, 'complete': False, 'leaf_count': 0,
                    'reason': 'no_root_edges'}
            return
        rows = {(row.get('r'), row.get('c')): row for row in (ranked or [])}
        total_weight = sum(float(edge[3]) for edge in edges)
        if total_weight <= 0:
            self.root_contribution_report = {
                'root_share': ROOT_SHARE, 'roots': [],
                'total': None, 'complete': False, 'leaf_count': 0,
                'reason': 'no_root_edges'}
            return
        roots = []
        total = 0.0
        complete = True
        leaf_count = 0
        children = root.get('children') or {}
        for edge in edges:
            r, c, player, weight = edge[0], edge[1], edge[2], edge[3]
            share = ROOT_SHARE * float(weight) / total_weight
            child = children.get((r, c, player))
            value, child_complete, child_leaves = self._subtree_contribution(child, share)
            leaf_count += child_leaves
            row = rows.get((r, c))
            if not child_complete or value is None:
                complete = False
            else:
                total += value
            roots.append({
                'r': r, 'c': c, 'share': share,
                'contribution': value, 'complete': bool(child_complete),
                'score': None if row is None else row.get('score'),
                'state': None if row is None else row.get('state'),
                'state_name': None if row is None else row.get('state_name'),
                'leaf_count': child_leaves,
            })
        self.root_contribution_report = {
            'root_share': ROOT_SHARE,
            'roots': roots,
            'total': total if complete else None,
            'complete': bool(complete),
            'leaf_count': leaf_count,
            'reason': None,
        }

    def cached_turn(self, board, player):
        """Engine在生成候选前调用；返回None才允许建立一棵新树。"""
        node = self._tree_root
        if (type(self) is not DeepSearch
                or '_leaf_score_impl' in self.__dict__ or node is None
                or node['board'] != self._position_key(board)
                or node['main_player'] != player
                or node.get('gear_context') not in (None, self.search._gear_context())
                or (node['edges'] is None and node['frontier'] is None)
                or self._tree_signature != self._reuse_signature()):
            return None
        self._begin_search_cache(board, False)
        if self._debug_observer is not None:
            self._debug_emit('cache_hit', node, reason='cached_turn')
        try:
            self.reset_leaf_eval_stats()
            self._continue_cached_move(board, node, player, [], [], self.T0, 1.0)
            edges = node['edges'] or ()
            points = {(r, c): w for r, c, p, w, _t, _q in edges if p == player}
            rows = []
            for r, c, p, _w, _t, _q in edges:
                if p != player or board[r][c] != EMPTY:
                    continue
                child = node['children'].get((r, c, p))
                if child is None or child['move_result'] is None:
                    if (self._state_forced(node['move_result']['state'])
                            or any(edge[:3] == (r, c, p) for edge in node['pending_edges'])):
                        continue
                    raise RuntimeError('缓存候选缺少落子节点结果')
                value = child['move_result']
                row = self._candidate_result(r, c, value['score'], value['state'],
                    self._fatal_ratio(value['wins'], value['losses'], value['total'])
                    if not self._state_forced(value['state']) else 0.0)
                row['score_complete'] = value['complete']
                rows.append(row)
            if not rows:
                # 命中未展开且不允许续温的根叶子，仍须给出合法落子。
                # 此处是在叶子上首次取候选，不重建已有中间节点。
                value = node['move_result']
                if (node['frontier'] or {}).get('leaf_end_reason') == 'no_common_defense':
                    point = self.search._defense_fallback(board, player, node['selection_reason'])
                    if point is not None:
                        rows = [self._candidate_result(*point, value['score'], self.STATE_LOSE)]
                else:
                    cands = self._candidate_points(board, player, self.CAND_N_FULL)
                    points = {(r, c): w for r, c, w in cands}
                    rows = [self._candidate_result(r, c, value['score']) for r, c, _w in cands]
            rows.sort(key=lambda row: self._pick_key_state(row['state'], row['score']), reverse=True)
            self.cache_stats['subtree_hits'] += 1
            self._refresh_root_contribution_report(rows)
            return points, node.get('selection_reason'), rows
        finally:
            self._reuse_active = False
            self._cache_positions = {}
            self._cache_paths = {}
            self._cold_results = None

    def _continue_cached_move(self, board, node, main_player, branch, contexts,
                              temperature, path_weight):
        """冻结完成硬态；旧边复用，叶子续温，内部节点可补推待展开边。"""
        value = node['move_result']
        if self._debug_observer is not None:
            self._debug_emit('cache_visit', node, temperature=temperature, score_player=main_player)
        if value is not None:
            if (node['continued_epoch'] == self._tree_epoch
                    or value['complete'] and self._state_forced(value['state'])):
                self.cache_stats['subtree_hits'] += 1
                if self._debug_observer is not None:
                    self._debug_emit('cache_hit', node, reason='hard_state_or_already_continued',
                                     score=value['score'], state=value['state'])
                return value
        if node['edges']:
            self._walk_cached_edges(board, node, main_player, branch, contexts)
            return node['move_result']
        frontier = node['frontier']
        if frontier is not None:
            if frontier['epoch'] >= self._tree_epoch:
                return value
            temperature = self._resume_temperature(node, frontier['player'],
                frontier['temperature'], depth=len(branch), before_move=frontier['before_move'])
            if self._resume_blocked(node, frontier['player'], before_move=frontier['before_move']):
                node['continued_epoch'] = self._tree_epoch
                return value
            path_weight = frontier['path_weight']
        elif value is not None and value['complete']:
            node['continued_epoch'] = self._tree_epoch
            return value
        mover = node['mover']
        if branch and self._check_win(board, branch[-1][0], branch[-1][1], mover):
            status = 'win' if mover == main_player else 'lose'
            score = self._leaf_score(board, branch, status, main_player, contexts)
            self._store_cached_move(node, score, self.STATE_WIN,
                                    int(mover == main_player), int(mover != main_player),
                                    1, main_player)
        elif mover != main_player:
            self._recursive(board, main_player, 3-main_player, main_player,
                            temperature, branch, contexts, {}, path_w=path_weight)
        else:
            self._expand_cached_leaf(board, node, main_player, temperature,
                                     path_weight, branch, contexts)
        if node['move_result'] is None:
            raise RuntimeError('叶子续推未保存完整落子结果')
        return node['move_result']

    def _walk_cached_edges(self, board, node, main_player, branch, contexts):
        self._activate_tactical_batch(node)
        edges = node['edges']
        pending = set(edge[:3] for edge in node['pending_edges'])
        for r, c, player, weight, temperature, path_weight in edges:
            if (r, c, player) in pending:
                continue
            child = node['children'].get((r, c, player))
            value = child['move_result'] if child is not None else None
            if value is not None and value['complete'] and (
                    self._state_forced(value['state'])
                    or child['continued_epoch'] == self._tree_epoch):
                self.cache_stats['subtree_hits'] += 1
                if self._debug_observer is not None:
                    self._debug_emit('cache_hit', child, reason='frozen_cached_edge',
                                     score=value['score'], state=value['state'])
                if value['state'] == self.STATE_WIN:
                    self._mark_state_cut(node, 'cached_winning_reply')
                    break
                continue
            if board[r][c] != EMPTY:
                raise RuntimeError('缓存分支包含已占点')
            journal = {}
            count = (child['tactical_count'] if child is not None else
                     self._advance_tactical_count(node['tactical_count'],
                         (r, c) in self._tactical_points(board, player)))
            context = dict(camp=int(player == main_player), move_w=weight,
                           temp=temperature, depth=len(branch), fatal=0, tactical_count=count)
            if child is not None and value is not None:
                context['extension_state'] = self._extension_state(node=child)
            else:
                before_temperature = node.get('branch_temperature')
                self._prepare_extension_step(context, board, player,
                    temperature if before_temperature is None else before_temperature,
                    (r, c) in self._tactical_points(board, player),
                    self._extension_state(contexts, node))
            board[r][c] = player
            branch.append((r, c, player, weight))
            contexts.append(context)
            try:
                self.search.on_move(board, r, c, player, journal=journal, record_history=False)
                self._finish_extension_step(context, board, player)
                child = self._remember_position(board, branch)
                child['tactical_count'] = count
                self._save_extension_state(child, context)
                self._continue_cached_move(board, child, main_player, branch, contexts,
                                           temperature, path_weight)
            finally:
                self.search.restore(journal)
                contexts.pop()
                branch.pop()
                board[r][c] = EMPTY
            if child['move_result']['state'] == self.STATE_WIN:
                self._mark_state_cut(node, 'cached_winning_reply')
                break
        if self._aggregate_cached_move(node, main_player) is None:
            raise RuntimeError('旧子树回传时没有任何可用分支结果')

    def _expand_cached_leaf(self, board, node, main_player, temperature,
                            path_weight, branch, contexts):
        """本方刚落子的叶子从对方应手处展开，不重走此前路径或重扣旧步温度。"""
        opp = 3-main_player
        if self._temperature_exhausted(board, opp, temperature, node['tactical_count'],
                                       extension_count=self._extension_state(contexts, node)[0],
                                       depth=len(branch)):
            if node['move_result'] is not None and node['move_result']['complete']:
                node['continued_epoch'] = self._tree_epoch
                return
            cands = []
        else:
            cands = self._candidate_points(board, opp, self.CAND_N_FULL, plies=len(branch))
        if not cands:
            if self._no_defense_leaf(board, node, branch, contexts, main_player, opp,
                                     temperature, path_weight) is not None:
                return
            score = self._cutoff_leaf(board, node, branch, contexts, main_player, opp,
                                      temperature, path_weight)
            self._store_cached_move(node, score, 0, 0, 0, 1, main_player)
            return
        weights = [w for _r, _c, w in cands]
        kept = self._prune_branches([(path_weight*self._step_quality(w, weights), r, c, w)
                                    for r, c, w in cands])
        self._cache_edges(board, node, opp, kept, temperature, main_player)
        self._walk_cached_edges(board, node, main_player, branch, contexts)

    def reset_leaf_eval_stats(self):
        """重置本次决策的叶子评估统计，供界面展示实际使用路径。"""
        if self._root_pass is not None:
            return  # 同次搜索累计实际叶子计算，不重复清零。
        self.leaf_eval_stats = {
            'heuristic': 0,
            'total': 0,
            'reason': None,
        }

    def leaf_eval_status(self):
        """返回本次实际叶子计算或缓存读取状态。"""
        out = dict(self.leaf_eval_stats)
        if out['heuristic']:
            out['mode'] = 'heuristic'
        elif self.cache_stats['leaf_hits'] or self.cache_stats['subtree_hits']:
            out['mode'] = 'heuristic'
            out['reason'] = '命中纯算法搜索缓存；本次未重算叶子'
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
        """普通步耗温：DT = DT_M*((100-w)/100)^DT_POW + DT_N（纯棋型权重0~100）。
        不接受邻近/攻防一体的综合分；棋型权重高→耗温少，权重低→浅推。
        幂函数让 100 与 80 差异小、80 与 60 差异明显，形状由 DT_POW 控制。"""
        x = max(100.0 - float(w), 0.0) / 100.0
        return DeepSearch.DT_M * (x ** DeepSearch.DT_POW) + DeepSearch.DT_N

    def _has_kill_points(self, board):
        """全盘原始必杀档存在性，仅供局面诊断，不作为免温续推入口。

        状态表描述该空点再落一子后的棋型：模拟盘面的活三可对应活四点，
        冲四可对应成五点。必须使用 on_move 更新后的表，并排除已占格。
        此处只扫描双方空点0~4档，不等同候选生成的战术优先级判定。
        续推必须复用_tactical_result返回的reason，与_candidate_points的直接取点分支一致。
        """
        return any(cell == EMPTY and (0 <= black <= 4 or 0 <= white <= 4)
                   for row, black_row, white_row in zip(board, self.search.sb, self.search.sw)
                   for cell, black, white in zip(row, black_row, white_row))

    def _tactical_points(self, board, player):
        """落子前按当前攻防优先级取得必杀/防守候选，复用四三反四过滤。"""
        _reason, points = self._tactical_result(board, player)
        return points

    def _temperature_cost(self, weight, is_tactical):
        """基础耗温；普通weight须为纯棋型分，战术实际收费由路径梯度覆盖。"""
        return 0.0 if is_tactical else self._dt(weight)

    def _extension_limit_reached(self, count):
        limit = self.TACTICAL_EXTENSION_PLIES
        if limit is None:
            return False
        if type(limit) is not int or limit < 0:
            raise ValueError('TACTICAL_EXTENSION_PLIES须为None或非负整数')
        return count >= limit

    def _extension_state(self, contexts=None, node=None):
        """不可变路径记录：(耗温后战术手数, 黑新威胁数, 白新威胁数)。"""
        if contexts:
            return tuple(contexts[-1].get('extension_state', (0, 0, 0)))
        return tuple(node.get('extension_state', (0, 0, 0))) if node is not None else (0, 0, 0)

    def _active_threat_points(self, board, player):
        """按现有必杀棋型与四三反四过滤取威胁落点；一个33的两个方向仍只算一点。"""
        state = self.search.sb if player == BLACK else self.search.sw
        return frozenset((r, c) for r in range(SIZE) for c in range(SIZE)
                         if board[r][c] == EMPTY and 0 <= state[r][c] <= 4
                         and not (state[r][c] == 3
                                  and self.search._d34_is_defensible(board, r, c, player)))

    def _prepare_extension_step(self, context, board, player, temperature, tactical, previous):
        """落子前快照只在耗温后的免费战术步取得，不把纯防守直接计作己方造杀。"""
        context['extension_state'] = tuple(previous)
        if (self.TACTICAL_EXTENSION_PLIES is not None and temperature <= 0 and tactical
                and not self._extension_limit_reached(previous[0])):
            context['_extension_before'] = self._active_threat_points(board, player)

    def _finish_extension_step(self, context, board, player):
        """状态表已随模拟落子更新后，统计新出现的独立威胁；旧威胁持续不重复计数。"""
        before = context.pop('_extension_before', None)
        if before is None:
            return
        count, black, white = context['extension_state']
        created = len(self._active_threat_points(board, player) - before)
        context['extension_state'] = (count + 1, black + (created if player == BLACK else 0),
                                      white + (created if player == WHITE else 0))

    @staticmethod
    def _save_extension_state(node, context):
        if node is not None:
            node['extension_state'] = tuple(context.get('extension_state', (0, 0, 0)))

    def _extension_score(self, contexts, node, score_player):
        _count, black, white = self._extension_state(contexts, node)
        difference = black-white if score_player == BLACK else white-black
        return self.TACTICAL_THREAT_VALUE * max(-self.TACTICAL_THREAT_CLAMP,
                                               min(self.TACTICAL_THREAT_CLAMP, difference))

    def _tactical_count(self, contexts, node=None):
        """累计实际战术落子数；旧树保留原路径计数，不重新领取免费额度。"""
        if self.TACTICAL_FREE_PLIES is None:
            return 0
        if contexts:
            return int(contexts[-1].get('tactical_count', 0))
        return int(node['tactical_count']) if node is not None else 0

    def _advance_tactical_count(self, count, is_tactical):
        return count + int(is_tactical) if self.TACTICAL_FREE_PLIES is not None else 0

    def _path_temperature_cost(self, weight, is_tactical, count, temperature=1.0):
        """正温战术按已用次数走梯度；非正温能否继续由预算/延伸规则检查。"""
        if not is_tactical:
            return self._temperature_cost(weight, is_tactical)
        if float(temperature) <= 0:
            return self._temperature_cost(weight, is_tactical)
        free = self.TACTICAL_FREE_PLIES
        if free is None:
            return self._temperature_cost(weight, is_tactical)
        if type(free) is not int or free < 0:
            raise ValueError('TACTICAL_FREE_PLIES须为None或非负整数')
        if count < free:
            return 0.0
        stage = min((count - free) // self.TACTICAL_STAGE_PLIES,
                    len(self.TACTICAL_COSTS) - 1)
        return self.TACTICAL_COSTS[stage]

    def _pattern_temperature_weight(self, player, r, c):
        """落子前读取该方落下后的预测棋型，归一化后仅用于普通扣温。

        防守点同样看实际落子方，而非对方威胁方；无对应棋型基础分时为0。
        综合候选分、邻近加分和攻防一体奖励不进入耗温，原排序/剪枝仍保留。
        """
        sub = self._sub_at(player, r, c)
        raw = float(search_module.TYPE_SCORE.get(sub, 0.0))
        return max(0.0, min(100.0, raw / search_module.SCORE_MAX * 100.0))

    def _minimum_depth_pending(self, depth):
        """深度按本次搜索根后的真实落子数计算，不从每个旧叶子重新计三手。"""
        limit = self.MIN_SEARCH_PLIES
        if type(limit) is not int or not 0 <= limit <= 225:
            raise ValueError('MIN_SEARCH_PLIES须为0..225的整数')
        return depth is not None and depth < limit

    def _temperature_exhausted(self, board, turn, temperature, count, blocked=False,
                               extension_count=0, *, depth=None):
        # 正温阶段永远由梯度温度负责；不能把正温战术链误判成8手延伸。
        if temperature > 0:
            return False
        reason, points = self._tactical_result(board, turn)
        if reason in ('opp-vcf', 'opp-d33') and not points:
            return False  # 防守无解优先，不能被次数上限遮成未知。
        # 实际成五优先于延伸/温度截断。
        if reason == 'my-five':
            return False
        if self._extension_limit_reached(extension_count):
            return True
        if not blocked and self._minimum_depth_pending(depth):
            # 仅暂缓温度结束；每步仍按原公式扣温，欠温保留，其他门槛不变。
            return False
        if self.TACTICAL_FREE_PLIES is None:
            return blocked or not reason
        # None 是已有的“关闭8手延伸”内部对照：保持旧梯度截断。
        if self.TACTICAL_EXTENSION_PLIES is None:
            return blocked or not reason or self._path_temperature_cost(
                100., True, count, 1.0) > 0
        # 显式开启梯度和延伸上限时，非正温战术走限额内免费延伸。
        return blocked or not reason

    def _no_defense_leaf(self, board, node, branch, contexts, score_player, defender,
                         temperature, path_weight):
        """仅在防守reason仍在且候选为空时终止；硬态按实际落子方保存。"""
        reason, points = self._tactical_result(board, defender)
        if reason not in ('opp-vcf', 'opp-d33') or points:
            return None
        winner = 3-defender
        score = self.WIN_SCORE if winner == score_player else -self.WIN_SCORE
        mover = (node['mover'] if node is not None else None)
        if mover is None:
            mover = branch[-1][2] if branch else 3-defender
            if node is not None:
                node['mover'] = mover
        state = self.STATE_LOSE if mover == defender else self.STATE_WIN
        self._remember_frontier(node, score_player, temperature, path_weight,
            before_move=defender == score_player, depth=len(branch), score=score,
            state=state, terminal=True)
        if node is not None:
            node['selection_reason'] = reason
            node['frontier']['leaf_end_reason'] = 'no_common_defense'
            node['frontier']['to_move'] = defender
        self._store_cached_move(node, score, state, int(winner == score_player),
                                int(winner != score_player), 1, score_player)
        if self._debug_observer is not None:
            self._debug_emit('defense_end', node, turn=defender, mover=mover, score_player=score_player,
                score=score, state=state, leaf_end_reason='no_common_defense')
        return score

    def _no_defense_turn(self, board, player, reason):
        """真实根无解也保留合法败势兜底；缓存保存的是上一落子方硬胜。"""
        self._begin_search_cache(board, False)
        try:
            node = self._tree_root if self._reuse_active else None
            if node is not None:
                node['mover'] = 3-player
            score = self._no_defense_leaf(board, node, [], [], player, player, self.T0, 1.)
            if score is None:
                raise RuntimeError('无解根的防守条件已改变')
            point = self.search._defense_fallback(board, player, reason)
            rows = [] if point is None else [self._candidate_result(*point, score, self.STATE_LOSE)]
            self._refresh_root_contribution_report(rows)
            return rows
        finally:
            self._reuse_active = False
            self._cache_positions = {}
            self._cache_paths = {}
            self._cold_results = None

    def _cutoff_leaf(self, board, node, branch, contexts, score_player, turn,
                     temperature, path_weight, *, before_move=False):
        """完成本预算下的叶子评分；战术未解不等于评分计算未完成。"""
        extension_state = self._extension_state(contexts, node)
        reason = self._tactical_result(board, turn)[0]
        extension_cut = self._extension_limit_reached(extension_state[0]) and reason is not None
        tactical = self.TACTICAL_FREE_PLIES is not None and reason is not None
        status = ('tactical_extension' if extension_cut else
                  'tactical_budget:%d' % turn if tactical else 'stale')
        self._remember_frontier(node, score_player, temperature, path_weight,
                                before_move=before_move, depth=len(branch))
        if node is not None:
            node['frontier']['leaf_end_reason'] = ('tactical_extension' if extension_cut else
                                                  'tactical_budget' if tactical else 'temperature')
            node['frontier']['to_move'] = turn
        score = self._leaf_score(board, branch, status, score_player, contexts)
        self._update_frontier(node, score=score)
        if self._debug_observer is not None:
            self._debug_emit('cutoff', node, turn=turn, score_player=score_player, score=score,
                             temperature=temperature, before_move=before_move,
                             leaf_end_reason=('tactical_extension' if extension_cut else
                                              'tactical_budget' if tactical else 'temperature'),
                             extension_state=extension_state)
        return score

    def _tactical_budget_score(self, board, turn, score_player):
        """战术截断的一层静态应手值，不把opp-*直接等同于负分。

        评估当前档位所有合法候选，以原权重归一汇总。模拟后按下一行棋方
        评估，再转回score_player；没有反四奖励、新分值或递归战术延伸。
        这里只给估值，不产生新的硬胜败状态。journal必须成对恢复。
        """
        _reason, points = self._tactical_result(board, turn)
        weighted = []
        for (r, c), weight in points.items():
            if board[r][c] != EMPTY:
                continue
            journal = {}
            board[r][c] = turn
            try:
                self.search.on_move(board, r, c, turn, journal=journal, record_history=False)
                self.cache_stats['tactical_probe_moves'] += 1
                if self._check_win(board, r, c, turn):
                    score = self.WIN_SCORE if turn == score_player else -self.WIN_SCORE
                else:
                    next_turn = 3-turn
                    value = self._stale_result_score(board, next_turn)
                    score = value if next_turn == score_player else -value
                weighted.append((weight, score))
            finally:
                self.search.restore(journal)
                board[r][c] = EMPTY
        return (self._weighted_average(weighted) if weighted else
                self._stale_result_score(board, score_player))

    @staticmethod
    def _step_quality(w, cand_ws):
        """单步路径质量 = 该步候选分 / 当步最高候选分（走最优候选 → 1.0）。"""
        top = max(cand_ws) if cand_ws else 0.0
        if top <= 0:
            return 1.0
        return min(1.0, float(w) / top)

    # ---------- 胜败证据与兼容统计（硬态同时确定±1000） ----------
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
        if cls._state_forced(state):
            score = cls.WIN_SCORE * state
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
        """对本层保留分支作加权均值，避免分叉较多时放大分数绝对值。
        绝对路径权重仍供剪枝使用；分数回传只保留本层相对权重。
        先归一权重再相乘，保留离线记录器的标量表达式，不强转成 float。"""
        items = list(items)
        total = sum(w for w, _v in items)
        if total <= 0:
            return 0.0
        return sum((w / total) * v for w, v in items)

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
        只有我方成五点可安全早停；其余必须扫描完全盘，避免漏掉后续更高档。
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
        node = self._position_node(board)
        if node is not None and player in node['features']:
            self.cache_stats['feature_hits'] += 1
            my_tier, my_total, opp_tier, opp_total = node['features'][player]
        else:
            features = self._scan_leaf_tables(board, my_state, opp_state)
            if node is not None:
                node['features'][player] = features
            my_tier, my_total, opp_tier, opp_total = features
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


    def _leaf_score(self, board, branch, status, player, ctx_branch=None):
        node = self._position_node(board)
        value_key = (status, player, self._history_key(),
                     self._tactical_count(ctx_branch, node), self._extension_state(ctx_branch, node))
        if node is not None and value_key in node['values']:
            self.cache_stats['leaf_hits'] += 1
            if self._debug_observer is not None:
                self._debug_emit('leaf', node, status=status, score_player=player,
                                 score=node['values'][value_key], cached=True)
            return node['values'][value_key]
        context = self._root_pass
        if context is None or not context['memo_leaves']:
            value = self._leaf_score_impl(board, branch, status, player, ctx_branch)
        else:
            key = (tuple(branch), status, player, self._extension_state(ctx_branch, node))
            if key not in context['leaves']:
                context['leaves'][key] = self._leaf_score_impl(
                    board, branch, status, player, ctx_branch)
            value = context['leaves'][key]
        if node is not None:
            node['values'][value_key] = value
            if status in ('win', 'lose'):
                node['terminal'] = player if status == 'win' else 3 - player
        if self._debug_observer is not None:
            self._debug_emit('leaf', node, status=status, score_player=player, score=value, cached=False)
        return value

    def _leaf_score_impl(self, board, branch, status, player, ctx_branch=None):
        """纯算法叶子评估：实际成五给终局分，其他叶子读取当前状态表。"""
        if status == 'win':
            return self.WIN_SCORE
        if status == 'lose':
            return -self.WIN_SCORE
        self.leaf_eval_stats['total'] += 1
        self.leaf_eval_stats['heuristic'] += 1
        self.leaf_eval_stats['reason'] = '纯算法'
        if status == 'tactical_extension':
            self.cache_stats['tactical_cutoffs'] += 1
            self.leaf_eval_stats['reason'] = '纯算法：战术8手截断，主动新增威胁计分'
            return self._extension_score(ctx_branch, self._position_node(board), player)
        if status.startswith('tactical_budget:'):
            self.cache_stats['tactical_cutoffs'] += 1
            self.leaf_eval_stats['reason'] = '纯算法：战术预算截断，一层静态应手'
            return self._tactical_budget_score(board, int(status.split(':')[1]), player)
        return self._stale_result_score(board, player)


    # ---------- 候选点展开（供推演层） ----------

    def _ordered_candidates(self, board, player, points):
        """只排序访问次序：战术沿Search来源优先级，普通按当前方候选分。"""
        reason, tactical = self._tactical_result(board, player)
        if reason:
            order = {point: index for index, point in enumerate(tactical)}
            return sorted(points, key=lambda row: order.get((row[0], row[1]), len(order)))
        return sorted(points, key=lambda row: -float(row[2]))

    def _candidate_points(self, board, player, n=5, plies=0):
        """生成当前落子方的带权候选，普通候选不再区分根层和内部层。

        战术档先生成全部候选，实际边由_cache_edges按3点分批保留pending；
        普通候选按五档分配攻防、最多4点。战术优先级与顺序沿原规则。
        所有候选权重保留真实综合分用于排序与路径质量；普通扣温另读纯棋型分。
        """
        n = min(n, self.CAND_N_FULL)
        reason, tactical = self._tactical_result(board, player)
        if reason:
            # 当前档位的全部候选；不按深度或根层/内部截断。
            return self._debug_candidates(board, player,
                [(r, c, w) for (r, c), w in tactical.items()], reason)
        n = min(n, self.CAND_N_ORDINARY)
        node = self._position_node(board)
        candidate_key = (player, self.search._gear_context())
        if node is not None and candidate_key in node['candidates']:
            self.cache_stats['candidate_hits'] += 1
            return self._debug_candidates(board, player,
                list(node['candidates'][candidate_key][:n]), None, cached=True)
        # 候选纯局面缓存须区分历史档位；已有路径edges仍原样复用。
        fb = self.search._ordinary_candidates(board, player=player, with_score=True)
        # clamp保持原温度语义：避免超过100后平方耗温反而增大。
        w_lst = self._ordered_candidates(board, player,
            [(r, c, min(max(float(w), 1.0), 100.0)) for (r, c, w) in fb])
        if node is not None:
            node['candidates'][candidate_key] = tuple(w_lst)
        return self._debug_candidates(board, player, w_lst[:n], None)

    def _make_ctx(self, board, player, main_player, w, cand_ws, cand_pts=None, temp=None, depth=None):
        """构造一步诊断上下文，供轨迹记录及离线工具读取。
        cand_pts：与 cand_ws 对位的候选点 [(r,c,w)]（用于按状态表取 18 子类）。
        temp/depth：该步的剩余温度与已推层数。"""
        cand_ws = [float(x) for x in (cand_ws or [])]
        kill = any(cw >= 100 for cw in cand_ws)
        gear = 0 if kill else self.search._ordinary_gear(player)
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

    def _mark_state_cut(self, node=None, reason=None):
        """硬证明已经确定±1000，只记录省略分支，不使祖先分数变成未完成。"""
        self._state_cut_serial += 1
        context = self._root_pass or {}
        if self._debug_observer is not None:
            self._debug_emit('state_cut', node, root_move=context.get('root'),
                             reason=reason, pending_depth=len(context.get('stack', ())))

    def _recursive(self, board, player, opp, main_player, temp, branch, ctx_branch,
                   cache, trace=False, indent=0, path_w=1.0):
        context = self._root_pass
        node = self._remember_position(board, branch)
        tactical_count = self._tactical_count(ctx_branch, node)
        if node is not None:
            node['tactical_count'] = tactical_count
        extension_state = self._extension_state(ctx_branch, node)
        if node is not None:
            node['extension_state'] = extension_state
        result_key = self._result_key(player, main_player, temp, path_w, tactical_count, extension_state,
                                      depth=len(branch))
        if node is not None:
            if node['result_epoch'] != self._tree_epoch:
                node['results'] = {}
                node['result_epoch'] = self._tree_epoch
            if result_key in node['results']:
                self.cache_stats['subtree_hits'] += 1
                if self._debug_observer is not None:
                    self._debug_emit('cache_hit', node, reason='complete_result')
                return node['results'][result_key]
        cold_key = self._cold_result_key(board, player, opp, main_player,
                                         temp, path_w, len(branch), tactical_count, extension_state)
        if cold_key is not None:
            cached = self._cold_results.get(cold_key)
            if cached is not None:
                value, source = cached
                # 保存节点已完整定分；硬证明早停的±1000也可复用。
                if (source['result_epoch'] == self._tree_epoch
                        and source['results'].get(result_key) == value
                        and source['move_result'] is not None
                        and source['move_result']['complete']):
                    self._copy_cold_subtree(node, source)
                    if self._debug_observer is not None:
                        self._debug_emit('cache_copy', node, source=source)
                    self.cache_stats['cold_result_hits'] += 1
                    self.cache_stats['subtree_hits'] += 1
                    return value
        args = (board, player, opp, main_player, temp, branch, ctx_branch, cache, trace, indent)
        if context is None or context['mode'] == 'full':
            value = self._recursive_impl(*args, path_w=path_w)
            self._save_tree_result(node, result_key, value)
            if (cold_key is not None and node is not None
                    and node['move_result'] is not None and node['move_result']['complete']):
                self._cold_results.setdefault(cold_key, (value, node))
            return value
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
            self._save_tree_result(node, result_key, value)
        else:
            context['incomplete'].add(key)
            if context['stack']:
                context['stack'][-1] = False
        return value

    def _recursive_impl(self, board, player, opp, main_player, temp, branch, ctx_branch, cache, trace=False, indent=0, path_w=1.0):
        """递归推演：player 落一子（候选）→ opp 应一手 → 递归下一层。
        默认当前根最低三层内只暂缓温度停止，仍扣温；达到后按原战术预算停止。
        战术免费/梯度身份取落子前候选，不以落子后的潜力追溯免温；硬态优先。
        本方落子后按下一方战术reason决定续推；下一轮本方落子前同样检查，实际成五优先终局。
        模拟落子后同步更新状态表（journal 哈希表回滚），保证候选点/必杀检测
        基于当前分支的真实棋盘，而不是进入深推前的过期状态表。
        ctx_branch 与 branch 同步记录每步诊断上下文。
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
        entry_node = self._position_node(board)
        tactical_count = self._tactical_count(ctx_branch, entry_node)
        # 对方落子后先查预算；可选根最低深度只豁免温度停止，不免扣温。
        # 新的落子前末端单独标相位，沿用真实换根后的既有欠温续算公式。
        temp = self._resume_temperature(entry_node, player, temp, depth=len(branch), before_move=True)
        if self._temperature_exhausted(board, player, temp, tactical_count,
                self._resume_blocked(entry_node, player, before_move=True),
                self._extension_state(ctx_branch, entry_node)[0], depth=len(branch)):
            value = self._cutoff_leaf(board, entry_node, branch, ctx_branch, player,
                                      player, temp, path_w, before_move=True)
            cache[key] = (value, 0, 0, 0, 1, 0, ())
            return cache[key]
        if entry_node is not None:
            entry_node['frontier'] = None
        cands = self._candidate_points(board, player, 5, plies=len(branch))
        if not cands:
            score = self._no_defense_leaf(board, entry_node, branch, ctx_branch,
                                          main_player, player, temp, path_w)
            cache[key] = ((score, -1, 0, 1, 1, -1, ()) if score is not None else
                          (0.0, 0, 0, 0, 1, 0, ()))
            return cache[key]
        cand_ws = [x[2] for x in cands]
        # 我方候选分支：绝对路径权重 = path_w × 本步质量（深层连乘衰减 → 自动剪弱枝）
        raw = []
        for (r, c, w) in cands:
            q = self._step_quality(w, cand_ws)
            raw.append((path_w * q, r, c, w))
        kept = self._prune_branches(raw)
        kept = self._cache_edges(board, entry_node, player, kept, temp, main_player)
        weighted = []
        cand_states = []                     # 各实际落子节点的硬状态（player 视角）
        new_cand_states = []                 # 各实际落子节点五态；仅直接候选硬败可作孙节点证据
        cand_wins = []                       # 子子树 main 视角必胜判定节点数
        cand_losses = []
        cand_totals = []                     # 子分支全部节点数（含未判定）
        tactical_points = self._tactical_points(board, player)
        # kept首项已是绝对路径权重，不能再乘一次path_w。
        for (candidate_path_w, r, c, w) in kept:
            if board[r][c] != EMPTY:
                continue
            is_tactical = (r, c) in tactical_points
            ctx = self._make_ctx(board, player, main_player, w, cand_ws,
                                 cand_pts=cands, temp=temp, depth=len(branch))
            ctx['tactical_count'] = self._advance_tactical_count(tactical_count, is_tactical)
            self._prepare_extension_step(ctx, board, player, temp, is_tactical,
                                         self._extension_state(ctx_branch, entry_node))
            cost_weight = w if is_tactical else self._pattern_temperature_weight(player, r, c)
            board[r][c] = player
            branch.append((r, c, player, w))
            ctx_branch.append(ctx)
            j_w = {}
            self.search.on_move(board, r, c, player, journal=j_w, record_history=False)
            self._finish_extension_step(ctx, board, player)
            tree_node = self._remember_position(board, branch)
            self._save_extension_state(tree_node, ctx)
            if tree_node is not None:
                tree_node['tactical_count'] = ctx['tactical_count']
            temp_after = temp - self._path_temperature_cost(cost_weight, is_tactical, tactical_count, temp)
            self._debug_step(tree_node, temp, temp_after, w, is_tactical, ctx['tactical_count'],
                             cost_weight=cost_weight)
            temp_after = self._resume_temperature(tree_node, player, temp_after, depth=len(branch))
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
            elif self._temperature_exhausted(board, opp, temp_after, ctx['tactical_count'],
                                               self._resume_blocked(tree_node, player),
                                               ctx['extension_state'][0], depth=len(branch)):
                # 普通候选或实验战术免费额度耗尽时，在当前盘面结算预算叶子。
                ctx['fatal'] = 1 if self._has_forcing_threat(player) else 0
                score = self._cutoff_leaf(board, tree_node, branch, ctx_branch, player,
                                          opp, temp_after, candidate_path_w)
                branch_state = new_branch_state = 0
                cw = cl = 0
                c_total = 1
                if trace:
                    print('  ' * indent + '   -> 本方落子后温度尽 leaf=%g' % score)
            else:
                if tree_node is not None:
                    tree_node['frontier'] = None  # 旧末端已展开，新欠温记在新末端。
                # 仅保留实验上下文特征，不作为胜败种子的门槛。
                ctx['fatal'] = 1 if self._has_forcing_threat(player) else 0
                # 对方应手与本方规则对称：战术保留分批资格，普通候选最多4点。
                opp_pts = self._candidate_points(board, opp, 5, plies=len(branch))
                if not opp_pts:
                    proof_score = self._no_defense_leaf(board, tree_node, branch, ctx_branch,
                        main_player, opp, temp_after, candidate_path_w)
                    if proof_score is None:
                        score = self._cutoff_leaf(board, tree_node, branch, ctx_branch, player,
                                                  opp, temp_after, candidate_path_w)
                    branch_state = new_branch_state = 1 if proof_score is not None else 0
                    if proof_score is not None:
                        score = proof_score
                    cw, cl = int(proof_score is not None), 0
                    c_total = 1          # 无解叶子或普通无应手末端
                else:
                    opp_tactical_points = self._tactical_points(board, opp)
                    opp_cand_ws = [x[2] for x in opp_pts]
                    # 本方绝对路径权重只乘一次对方本步质量，不重复累计前缀。
                    opp_raw = []
                    for (or_, oc, ow) in opp_pts:
                        oq = self._step_quality(ow, opp_cand_ws)
                        opp_raw.append((candidate_path_w * oq, or_, oc, ow))
                    opp_kept = self._prune_branches(opp_raw)
                    opp_kept = self._cache_edges(board, tree_node, opp, opp_kept, temp_after, main_player)
                    opp_weighted = []
                    opp_states = []          # 直接对方应手的硬状态（opp 视角）
                    grandchild_states = []   # 本方直接下一手的状态，不收集更深层败点
                    cw = 0
                    cl = 0
                    c_total = 0
                    for (reply_path_w, or_, oc, ow) in opp_kept:
                        if board[or_][oc] != EMPTY:
                            continue
                        opp_is_tactical = (or_, oc) in opp_tactical_points
                        cost_weight = ow if opp_is_tactical else self._pattern_temperature_weight(opp, or_, oc)
                        temp2 = temp_after - self._path_temperature_cost(
                            cost_weight, opp_is_tactical, ctx['tactical_count'], temp_after)
                        ctx_b = self._make_ctx(board, opp, main_player, ow, opp_cand_ws,
                                               cand_pts=opp_pts, temp=temp2, depth=len(branch))
                        ctx_b['tactical_count'] = self._advance_tactical_count(
                            ctx['tactical_count'], opp_is_tactical)
                        self._prepare_extension_step(ctx_b, board, opp, temp_after, opp_is_tactical,
                                                     self._extension_state(ctx_branch, tree_node))
                        board[or_][oc] = opp
                        branch.append((or_, oc, opp, ow))
                        ctx_branch.append(ctx_b)
                        j_b = {}
                        self.search.on_move(board, or_, oc, opp, journal=j_b, record_history=False)
                        self._finish_extension_step(ctx_b, board, opp)
                        reply_node = self._remember_position(board, branch)
                        self._save_extension_state(reply_node, ctx_b)
                        if reply_node is not None:
                            reply_node['tactical_count'] = ctx_b['tactical_count']
                        self._debug_step(reply_node, temp_after, temp2, ow,
                                         opp_is_tactical, ctx_b['tactical_count'], cost_weight=cost_weight)
                        opp_won = self._check_win(board, or_, oc, opp)
                        ctx_b['fatal'] = 1 if (opp_won or self._has_forcing_threat(opp)) else 0
                        if opp_won:
                            child = self._leaf_score(board, branch, 'lose', player, ctx_branch)
                            st = self._leaf_forced_state(ctx_branch, opp, main_player)  # opp 视角
                            opp_states.append(st)
                            self._store_cached_move(reply_node, child, st, 0, 1, 1, main_player)
                            cw += 0
                            cl += 1 if st == 1 else 0
                            c_total += 1      # 对方应手子节点
                            if trace:
                                print('  ' * indent + '   -> B应%s%d LOSE %g' % ('ABCDEFGHIJKLMNO'[oc], or_ + 1, child))
                        else:
                            # 回本方待走入口，统一处理预算及可选最低根深度。
                            child, st, cw2, cl2, ct2, new_st, next_states = self._recursive(board, player, opp, main_player, temp2, branch, ctx_branch, cache, trace, indent + 1, path_w=reply_path_w)
                            opp_states.append(-st)   # 递归返回 player 视角 → 投影到 opp 视角
                            grandchild_states.extend(next_states)
                            cw += cw2
                            cl += cl2
                            c_total += ct2        # 递归子节点及其子树
                        self.search.restore(j_b)
                        branch.pop()
                        ctx_branch.pop()
                        board[or_][oc] = EMPTY
                        opp_weighted.append((reply_path_w, child))
                        if opp_states[-1] == 1:
                            # 对方已有硬胜应手，本方该落子必败，直接计-1000。
                            self._mark_state_cut(tree_node, 'opponent_winning_reply')
                            break
                    score = self._weighted_average(opp_weighted)
                    # 对方最优应手决定硬胜败；局部机会/风险只取直接子/孙节点。
                    opp_states = self._pending_state(tree_node, opp_states)
                    opp_layer = self._merge_fatal_states(opp_states)
                    # opp 应手层聚合判定计入"判定节点数"（统一 main 视角：opp 必胜=main 败）
                    if opp_layer == 1:
                        cl += 1
                    elif opp_layer == -1:
                        cw += 1
                    c_total += 1          # 该候选子节点本身（应手层聚合节点，含未判定）
                    new_branch_state = self._merge_child_states(opp_states, grandchild_states)
                    branch_state = self._state_forced(new_branch_state)
            # 旧温度叶子可能在父层聚合出 +2/-2；把软态回写到该盘面，
            # 下一回合换根时与 state=0 共用续推策略。
            score = self._hard_score(new_branch_state, player, main_player, score)
            self._update_frontier(tree_node, score=score, state=new_branch_state)
            self._store_cached_move(tree_node, score, new_branch_state, cw, cl, c_total,
                                    main_player)
            if tree_node is not None:
                new_branch_state = tree_node['move_result']['state']
                branch_state = self._state_forced(new_branch_state)
                score = tree_node['move_result']['score']
            self.search.restore(j_w)
            branch.pop()
            ctx_branch.pop()
            board[r][c] = EMPTY
            if trace:
                print('  ' * indent + '    branch=%g' % score)
            weighted.append((candidate_path_w, score))
            cand_states.append(branch_state)
            new_cand_states.append(new_branch_state)
            cand_wins.append(cw)
            cand_losses.append(cl)
            cand_totals.append(c_total)
            if branch_state == 1:
                # 找到本方硬胜选择即完成该选择层，无需其余候选分。
                self._mark_state_cut(entry_node, 'winning_choice')
                break
        val = self._weighted_average(weighted)
        cand_states = self._pending_state(entry_node, cand_states)
        new_cand_states = self._pending_state(entry_node, new_cand_states)
        node_state = self._merge_fatal_states(cand_states)
        # 此处是 player 主动选落点，不能再次当作对方应手层合并。
        new_node_state = node_state
        val = self._hard_score(node_state, player, main_player, val)
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
        """当前真实根上复用已完成计算；真实落子前移树，显式重建清空树。"""
        if not trace and self._root_pass is None:
            cached = self.cached_turn(board, player)
            if cached is not None:
                return cached[2]
        self._begin_search_cache(board, trace)
        try:
            node = self._tree_root if self._reuse_active else None
            if node is not None:
                node['mover'] = 3-player
                if node['main_player'] not in (None, player):
                    self._tree_root = node = self._new_tree_node(self._position_key(board))
                    node['mover'] = 3-player
                    self._cache_positions = {node['board']: node}
                    self._cache_paths = {(): node}
                weighted = points.items() if isinstance(points, dict) else ((p, 60.) for p in points)
                ordered = self._ordered_candidates(board, player,
                    [(r, c, w) for (r, c), w in weighted if board[r][c] == EMPTY])
                items = [(1.0, r, c, w) for r, c, w in ordered]
                selected = self._cache_edges(board, node, player, items, self._initial_temperature(), player)
                points = ({(r, c): w for _q, r, c, w in selected}
                          if isinstance(points, dict) else [(r, c) for _q, r, c, _w in selected])
            result = self._rank_candidates_impl(board, points, player, trace,
                                              complete_scores=complete_scores)
            if node is not None:
                self._aggregate_cached_move(node, player)
            self._refresh_root_contribution_report(result)
            return result
        finally:
            self._reuse_active = False
            self._cache_positions = {}
            self._cache_paths = {}
            self._cold_results = None

    def _rank_candidates_impl(self, board, points, player, trace=False, *, complete_scores=False):
        """单遍按序搜索：硬态直接定分，其他节点才作加权估值，不重开补分遍。

        complete_scores保留调用接口；硬态±1000本身就是完整分，不要求展开所有后代。
        批次未补齐的非硬态仍按score_complete=False返回，不冒充完整分。
        """
        if self._root_pass is not None:
            return self._rank_candidates_pass(board, points, player, trace)
        self._root_pass = {'mode': 'full', 'proof': {}, 'complete': {}, 'leaves': {},
            'incomplete': set(), 'partial_roots': set(), 'stack': [],
            'root': None, 'memo_leaves': False, 't0': self._initial_temperature()}
        try:
            return self._rank_candidates_pass(board, points, player, trace)
        finally:
            self._root_pass = None

    def _rank_candidates_pass(self, board, points, player, trace=False):
        """对本批获准候选逐个深推，按五态优先、同档分支暂估分降序返回列表
        [{'r': r, 'c': c, 'score': val, ...}, ...]；未获准候选保存在父节点待后续补推。
        已占格跳过（不打分）；空候选返回 []。
        每条结果额外带新 5 态字段 state/state_name/child_state；
        排序规则：状态档位优先，同档按候选分降序。
        旧 forced/fatal_ratio 仅作兼容字段保留，不再参与排序。
        供 analyze_turn 输出已搜索候选及分数；首个硬胜允许提前返回。"""
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
        t0 = self._initial_temperature() if context is None else context['t0']
        if trace:
            print('=== rank_candidates turn=%s T0=%g cands=%d ===' % ('W' if player == WHITE else 'B', t0, len(cands)))
        cache = (context['proof'] if context['mode'] == 'proof' else context['complete']) if context is not None else {}
        opp = 3 - player
        main_player = player
        results = []
        tactical_points = self._tactical_points(board, player)
        for (r, c) in cands:
            if context is not None:
                context['root'] = (r, c)
            if board[r][c] != EMPTY:
                continue
            w = points.get((r, c), 60.0) if isinstance(points, dict) else 60.0
            is_tactical = (r, c) in tactical_points
            entry_node = self._position_node(board)
            entry_temp = self._resume_temperature(entry_node, player, t0, depth=0, before_move=True)
            tactical_count = self._tactical_count([], entry_node)
            direct_win = (self.TACTICAL_FREE_PLIES is not None
                          and self._tactical_result(board, player)[0] == 'my-five'
                          and is_tactical)
            if (not direct_win and
                    (self._extension_limit_reached(self._extension_state([], entry_node)[0])
                     or entry_temp <= 0 and (self._resume_blocked(entry_node, player, before_move=True)
                                     or (self._path_temperature_cost(
                                         (w if is_tactical else self._pattern_temperature_weight(player, r, c)),
                                         is_tactical, tactical_count, entry_temp) > 0
                                         and not self._minimum_depth_pending(0))))):
                # 外部传入的根候选也不能靠权重100冒充零耗温战术步。
                value = self._cutoff_leaf(board, entry_node, [], [], player, player,
                                          entry_temp, 1.0, before_move=True)
                self._store_cached_move(entry_node, value, 0, 0, 0, 1, main_player)
                results.append(self._candidate_result(r, c, value))
                continue
            if entry_node is not None:
                entry_node['frontier'] = None
            root_cands = self._candidate_points(board, player, 5)
            cand_ws = [x[2] for x in root_cands]
            is_tactical = (r, c) in tactical_points
            ctx = self._make_ctx(board, player, main_player, w, cand_ws,
                                 cand_pts=root_cands, temp=entry_temp, depth=0)
            ctx['tactical_count'] = self._advance_tactical_count(tactical_count, is_tactical)
            self._prepare_extension_step(ctx, board, player, entry_temp, is_tactical,
                                         self._extension_state([], entry_node))
            cost_weight = w if is_tactical else self._pattern_temperature_weight(player, r, c)
            board[r][c] = player
            branch = [(r, c, player, w)]
            ctx_branch = [ctx]
            j_w = {}
            self.search.on_move(board, r, c, player, journal=j_w, record_history=False)
            self._finish_extension_step(ctx, board, player)
            tree_node = self._remember_position(board, branch)
            self._save_extension_state(tree_node, ctx)
            if tree_node is not None:
                tree_node['tactical_count'] = ctx['tactical_count']
            temp_after = entry_temp - self._path_temperature_cost(cost_weight, is_tactical, tactical_count, entry_temp)
            self._debug_step(tree_node, entry_temp, temp_after, w, is_tactical, ctx['tactical_count'],
                             cost_weight=cost_weight)
            temp_after = self._resume_temperature(tree_node, player, temp_after, depth=len(branch))
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
            elif self._temperature_exhausted(board, opp, temp_after, ctx['tactical_count'],
                                               self._resume_blocked(tree_node, player),
                                               ctx['extension_state'][0], depth=len(branch)):
                # 根层与递归同源：按下一方是否命中战术直接取点分支决定续推。
                ctx['fatal'] = 1 if self._has_forcing_threat(player) else 0
                val = self._cutoff_leaf(board, tree_node, branch, ctx_branch, player,
                                          opp, temp_after, 1.0)
                root_state = new_root_state = 0
                cw = cl = 0
                c_total = 1
            else:
                if tree_node is not None:
                    tree_node['frontier'] = None  # 旧末端已展开，新欠温记在新末端。
                # 仅保留实验上下文特征，不作为胜败种子的门槛。
                ctx['fatal'] = 1 if self._has_forcing_threat(player) else 0
                # 对方应手与本方规则对称：战术保留分批资格，普通候选最多4点。
                opp_pts = self._candidate_points(board, opp, 5, plies=len(branch))
                if not opp_pts:
                    proof_score = self._no_defense_leaf(board, tree_node, branch, ctx_branch,
                        main_player, opp, temp_after, 1.)
                    if proof_score is None:
                        val = self._cutoff_leaf(board, tree_node, branch, ctx_branch, player,
                                                  opp, temp_after, 1.0)
                    root_state = new_root_state = 1 if proof_score is not None else 0
                    if proof_score is not None:
                        val = proof_score
                    cw, cl = int(proof_score is not None), 0
                    c_total = 1          # 无解叶子或普通无应手末端
                else:
                    opp_tactical_points = self._tactical_points(board, opp)
                    opp_cand_ws = [x[2] for x in opp_pts]
                    # 对方应手分支：按路径质量权重剪枝 + 本层加权均值（不取 min）
                    opp_raw = []
                    for (or_, oc, ow) in opp_pts:
                        oq = self._step_quality(ow, opp_cand_ws)
                        opp_raw.append((oq, or_, oc, ow))
                    opp_kept = self._prune_branches(opp_raw)
                    opp_kept = self._cache_edges(board, tree_node, opp, opp_kept, temp_after, main_player)
                    opp_weighted = []
                    opp_states = []          # 根层直接对方应手的硬状态（opp 视角）
                    grandchild_states = []   # 根候选方下一手各落点的状态
                    cw = 0
                    cl = 0
                    c_total = 0
                    for (oq, or_, oc, ow) in opp_kept:
                        if board[or_][oc] != EMPTY:
                            continue
                        opp_is_tactical = (or_, oc) in opp_tactical_points
                        cost_weight = ow if opp_is_tactical else self._pattern_temperature_weight(opp, or_, oc)
                        temp2 = temp_after - self._path_temperature_cost(
                            cost_weight, opp_is_tactical, ctx['tactical_count'], temp_after)
                        ctx_b = self._make_ctx(board, opp, main_player, ow, opp_cand_ws,
                                               cand_pts=opp_pts, temp=temp2, depth=len(branch))
                        ctx_b['tactical_count'] = self._advance_tactical_count(
                            ctx['tactical_count'], opp_is_tactical)
                        self._prepare_extension_step(ctx_b, board, opp, temp_after, opp_is_tactical,
                                                     self._extension_state(ctx_branch, tree_node))
                        board[or_][oc] = opp
                        branch.append((or_, oc, opp, ow))
                        ctx_branch.append(ctx_b)
                        j_b = {}
                        self.search.on_move(board, or_, oc, opp, journal=j_b, record_history=False)
                        self._finish_extension_step(ctx_b, board, opp)
                        reply_node = self._remember_position(board, branch)
                        self._save_extension_state(reply_node, ctx_b)
                        if reply_node is not None:
                            reply_node['tactical_count'] = ctx_b['tactical_count']
                        self._debug_step(reply_node, temp_after, temp2, ow,
                                         opp_is_tactical, ctx_b['tactical_count'], cost_weight=cost_weight)
                        opp_won = self._check_win(board, or_, oc, opp)
                        ctx_b['fatal'] = 1 if (opp_won or self._has_forcing_threat(opp)) else 0
                        if opp_won:
                            child = self._leaf_score(board, branch, 'lose', player, ctx_branch)
                            st = self._leaf_forced_state(ctx_branch, opp, main_player)
                            opp_states.append(st)
                            self._store_cached_move(reply_node, child, st, 0, 1, 1, main_player)
                            cl += 1 if st == 1 else 0      # opp 必胜 → main 必败节点
                            c_total += 1                  # 对方应手子节点
                            if trace:
                                print('   -> B resp %s%d LOSE %g' % ('ABCDEFGHIJKLMNO'[oc], or_ + 1, child))
                        else:
                            # 回本方待走入口；最低深度不能在每个旧叶子重新领取。
                            child, st, cw2, cl2, ct2, new_st, next_states = self._recursive(board, player, opp, main_player, temp2, branch, ctx_branch, cache, trace, 1, path_w=oq)
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
                        if opp_states[-1] == 1:
                            self._mark_state_cut(tree_node, 'opponent_winning_reply')
                            break
                    val = self._weighted_average(opp_weighted)
                    opp_states = self._pending_state(tree_node, opp_states)
                    opp_layer = self._merge_fatal_states(opp_states)
                    if opp_layer == 1:
                        cl += 1              # opp 层聚合：opp 必胜 → main 必败节点
                    elif opp_layer == -1:
                        cw += 1              # opp 层聚合：opp 必败 → main 必胜节点
                    c_total += 1              # 该候选根子节点本身（应手层聚合节点，含未判定）
                    new_root_state = self._merge_child_states(opp_states, grandchild_states)
                    root_state = self._state_forced(new_root_state)
            val = self._hard_score(new_root_state, player, main_player, val)
            self._update_frontier(tree_node, score=val, state=new_root_state)
            self._store_cached_move(tree_node, val, new_root_state, cw, cl, c_total,
                                    main_player)
            if tree_node is not None:
                new_root_state = tree_node['move_result']['state']
                root_state = self._state_forced(new_root_state)
                val = tree_node['move_result']['score']
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
            row = self._candidate_result(r, c, val, new_root_state, ratio)
            if tree_node is not None:
                row['score_complete'] = tree_node['move_result']['complete']
            results.append(row)
            if new_root_state == self.STATE_WIN:
                self._mark_state_cut(entry_node, 'winning_root')
                break
        results.sort(key=lambda x: self._pick_key_state(x.get('state'), x.get('score', 0.0)),
                     reverse=True)
        if trace:
            top = results[0] if results else None
            if top:
                print('=== pick: %s%d val=%g ===' % ('ABCDEFGHIJKLMNO'[top['c']], top['r'] + 1, top['score']))
            else:
                print('=== no candidates ===')
        return results
