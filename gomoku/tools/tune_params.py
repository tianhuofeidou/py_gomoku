# -*- coding: utf-8 -*-
"""离线教师调参的参数、搜索记录与评分拟合；正式运行路径不导入本模块。

前段参数决定搜索树，叶子参数只改变已有树的数值。记录器继承正式深推器，
只为叶子和每层加权均值附加表达式，不复制候选生成、五态或回溯规则。
"""
import copy
import hashlib
import json
import math
import random
import time
from contextlib import contextmanager

from gomoku.core import search as S
from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine
from gomoku.core.utils import empty_board


GROUPS = {
    'leaf': tuple(f'{side}_{i}' for side in ('my', 'opp') for i in range(12)),
    'score': (tuple(f'type_{i}' for i in range(5, 17))
              + tuple(f'near_{mode}_{side}' for mode in ('attack', 'defend', 'dual')
                      for side in ('my', 'opp'))
              + ('distance_1', 'distance_2', 'direction_line', 'direction_other',
                 'neighbor', 'dual_bonus')),
    'gear': ('gear_a1', 'gear_a2', 'gear_b', 'gear_zero', 'gear_window',
             'threshold_0', 'threshold_1', 'threshold_2'),
    'search': ('dt_m', 'dt_pow', 'dt_n', 'prune'),
}
LEAF_KEYS = GROUPS['leaf']


def defaults():
    """读取当前源码值，不读取 GA 文件或环境开关。"""
    p = {f'my_{i}': float(v) for i, v in enumerate(DeepSearch.MY_SUB_VALUE[5:17])}
    p.update({f'opp_{i}': float(v) for i, v in enumerate(DeepSearch.OPP_SUB_VALUE[5:17])})
    p.update({f'type_{i}': float(S.TYPE_SCORE[i]) for i in range(5, 17)})
    for mode, weights in S.NEIGHBOR_W.items():
        p.update({f'near_{mode}_my': float(weights[0]), f'near_{mode}_opp': float(weights[1])})
    p.update(distance_1=S.PROX_DIST_W[1], distance_2=S.PROX_DIST_W[2],
             direction_line=S.PROX_DIR_W['line'], direction_other=S.PROX_DIR_W['other'],
             neighbor=S.W_NEIGHBOR, dual_bonus=S.DUAL_BONUS,
             gear_a1=S.GEAR_W_A1, gear_a2=S.GEAR_W_A2, gear_b=S.GEAR_W_B,
             gear_zero=S.GEAR_W_ZERO, gear_window=S.GEAR_ZERO_WINDOW,
             dt_m=DeepSearch.DT_M, dt_pow=DeepSearch.DT_POW,
             dt_n=DeepSearch.DT_N, prune=DeepSearch.PRUNE_W)
    p.update({f'threshold_{i}': v for i, v in enumerate(S.GEAR_THRESHOLDS)})
    return p


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def bounds(base, key):
    if key == 'gear_window':
        return 2, 30
    if key.startswith('threshold_'):
        return -12.0, 12.0
    if key == 'prune':
        return 0.005, 0.08
    if key == 'dt_pow':
        return 1.0, 4.0
    if key == 'dt_n':
        return 1.0, 9.0
    value = base[key]
    return (value * 4, value * .25) if value < 0 else (max(.001, value * .25), value * 4)


def project(params, base):
    """限制尺度和组内次序；参数只是实验候选，不改正式默认值。"""
    if set(params) != set(base):
        raise ValueError('参数字段不完整或包含未知字段')
    p = {}
    for k, v in params.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f'参数不是有限数值: {k}')
        low, high = bounds(base, k)
        p[k] = min(high, max(low, float(v)))
    p['gear_window'] = int(round(p['gear_window']))
    ordered = [tuple(f'{side}_{i}' for i in range(start, start + 4))
               for side in ('my', 'opp') for start in (0, 4, 8)]
    ordered += [tuple(f'type_{i}' for i in range(start, start + 4)) for start in (5, 9, 13)]
    ordered += [('gear_a1', 'gear_a2', 'gear_b'),
                ('threshold_0', 'threshold_1', 'threshold_2')]
    for keys in ordered:
        # 各字段的原始界限也按相同方向递减；投影后仍满足界限。
        for k, v in zip(keys, sorted((p[k] for k in keys), reverse=True)):
            lo, hi = bounds(base, k)
            p[k] = min(hi, max(lo, v))
    total = sum(p[k] for k in LEAF_KEYS)
    base_total = sum(base[k] for k in LEAF_KEYS)
    target = min(1.2 * base_total, max(.8 * base_total, total))
    if total:
        # 在共同倍率下投影到盒约束，同时保持棋型组内次序和总尺度。
        lo, hi = 0.0, 10.0
        for _ in range(45):
            factor = (lo + hi) / 2
            s = sum(min(bounds(base, k)[1], max(bounds(base, k)[0], p[k] * factor))
                    for k in LEAF_KEYS)
            if s < target:
                lo = factor
            else:
                hi = factor
        for k in LEAF_KEYS:
            a, b = bounds(base, k)
            p[k] = min(b, max(a, p[k] * (lo + hi) / 2))
    return p


def validate(params, base):
    p = project(params, base)
    if any(not math.isclose(float(p[k]), float(params[k]), rel_tol=1e-10, abs_tol=1e-8)
           for k in base):
        raise ValueError('参数不满足范围、尺度或棋型次序约束')


@contextmanager
def parameter_scope(params, t0=80):
    """单线程离线参数与空棋谱记忆上下文；异常退出也还原注入内容。

    训练搜索和显式实验协议入口共用此上下文，避免日常棋谱的经验分
    改变根温度而未进入缓存身份。作用域外的正式引擎保持原记忆行为。
    """
    search_values = dict(
        TYPE_SCORE={i: params[f'type_{i}'] for i in range(5, 17)},
        NEIGHBOR_W={m: (params[f'near_{m}_my'], params[f'near_{m}_opp'])
                    for m in ('attack', 'defend', 'dual')},
        PROX_DIST_W={i: params[f'distance_{i}'] for i in (1, 2)},
        PROX_DIR_W={d: params[f'direction_{d}'] for d in ('line', 'other')},
        W_NEIGHBOR=params['neighbor'], DUAL_BONUS=params['dual_bonus'],
        GEAR_W_A1=params['gear_a1'], GEAR_W_A2=params['gear_a2'], GEAR_W_B=params['gear_b'],
        GEAR_W_ZERO=params['gear_zero'], GEAR_ZERO_WINDOW=int(params['gear_window']),
        GEAR_THRESHOLDS=tuple(params[f'threshold_{i}'] for i in range(3)))
    search_values['SCORE_MAX'] = max(search_values['TYPE_SCORE'].values()) + params['neighbor'] + params['dual_bonus']
    deep_values = dict(
        MY_SUB_VALUE=(0.0,) * 5 + tuple(params[f'my_{i}'] for i in range(12)) + (0.0,),
        OPP_SUB_VALUE=(0.0,) * 5 + tuple(params[f'opp_{i}'] for i in range(12)) + (0.0,),
        T0=t0, DT_M=params['dt_m'], DT_POW=params['dt_pow'], DT_N=params['dt_n'],
        PRUNE_W=params['prune'])
    scoped_values = ((S, search_values), (DeepSearch, deep_values),
                     (Engine, {'_memory_bonus': lambda self: {}}))
    saved = [(obj, k, getattr(obj, k)) for obj, values in scoped_values
             for k in values]
    try:
        for obj, values in scoped_values:
            for k, v in values.items():
                setattr(obj, k, v)
        yield
    finally:
        for obj, k, v in saved:
            setattr(obj, k, v)


class SearchLimit(RuntimeError):
    pass


class Expression(float):
    """保留原浮点计算顺序，同时记录线性回溯表达式；不参与五态判断。"""
    def __new__(cls, value, node):
        obj = super().__new__(cls, value)
        obj.node = node
        return obj

    def __add__(self, other):
        node = other.node if isinstance(other, Expression) else ('constant', float(other))
        return Expression(float(self) + float(other), ('sum', self.node, node))

    __radd__ = __add__

    def __mul__(self, other):
        return Expression(float(self) * other, ('scale', float(other), self.node))

    __rmul__ = __mul__


def flatten(value):
    """把共享表达式展开成叶子特征的累计系数；每层归一系数已由正式回传写入表达式。"""
    if not isinstance(value, Expression):
        return float(value), []
    constant, terms = 0.0, {}
    stack = [(1.0, value.node)]
    while stack:
        weight, node = stack.pop()
        if node[0] == 'constant':
            constant += weight * node[1]
        elif node[0] == 'leaf':
            feature = node[1]
            terms[feature] = terms.get(feature, 0.0) + weight
        elif node[0] == 'sum':
            stack.extend(((weight, node[1]), (weight, node[2])))
        else:
            stack.append((weight * node[1], node[2]))
    return constant, [[w, list(f)] for f, w in sorted(terms.items())]


class RecordingSearch(DeepSearch):
    """记录器只在离线工具中启用；超限进程的引擎实例直接丢弃，不再复用。"""
    def __init__(self, search, record=False, node_limit=50000):
        super().__init__(search)
        self.record = record
        self.node_limit = node_limit
        self.nodes = 0
        self.leaves = 0

    def _make_ctx(self, *args, **kwargs):
        self.nodes += 1
        if self.nodes > self.node_limit:
            raise SearchLimit('搜索节点超过实验上限')
        return super()._make_ctx(*args, **kwargs)

    def _leaf_score_impl(self, board, branch, status, player, ctx_branch=None):
        self.leaves += 1
        value = super()._leaf_score_impl(board, branch, status, player, ctx_branch)
        if not self.record or status != 'stale':
            return value
        my = self.search.sb if player == 1 else self.search.sw
        opp = self.search.sw if player == 1 else self.search.sb
        mt, _, ot, _ = self._scan_leaf_tables(board, my, opp)
        if mt or ot:
            return value  # 战术叶子恒值，不拟合。
        counts = [0] * 24
        for r in range(15):
            for c in range(15):
                if board[r][c] == 0:
                    for offset, state in ((0, my), (12, opp)):
                        v = state[r][c]
                        if 5 <= v <= 16:
                            counts[offset + v - 5] += 1
        coef = self.GEAR_OPP_COEF[self.search._target_gear(board, player)]
        return Expression(value, ('leaf', tuple(counts) + (coef,)))


def replay_row(row, params, gradient=False):
    score = row['constant']
    grad = [0.0] * 24
    theta = [params[k] for k in LEAF_KEYS]
    for weight, feature in row['terms']:
        coef = feature[24]
        diff = sum(theta[i] * feature[i] for i in range(12))
        diff -= coef * sum(theta[i] * feature[i] for i in range(12, 24))
        x = abs(diff)
        r, scale = DeepSearch.LEAF_TAX_R, DeepSearch.LEAF_TAX_S
        exp = math.exp(-x / scale)
        val = r * x + (1 - r) * scale * (1 - exp)
        score += weight * (val if diff >= 0 else -val)
        if gradient:
            deriv = weight * (r + (1 - r) * exp)
            for i in range(24):
                grad[i] += deriv * feature[i] * (1 if i < 12 else -coef)
    return score, grad


def choose(rows, params):
    if not rows:
        raise ValueError('无候选，不能参与拟合')
    scored = [(row, replay_row(row, params)[0]) for row in rows]
    return max(scored, key=lambda item: DeepSearch._pick_key_state(item[0]['state'], item[1]))[0]


def _restore_snapshot(engine, board):
    mutable = {k: v for k, v in engine.search.__dict__.items() if isinstance(v, (dict, list))}
    return copy.deepcopy((board, mutable, engine.moves))


def search_position(position, params, t0=80, node_limit=50000, record=True):
    """复用 analyze_turn；保留历史、固定开局方向，离线搜索不叠加本机棋谱记忆。

    记忆会改变根候选权重和温度，不能在未绑定它的树缓存中隐式读取。
    使用 parameter_scope 隔离经验入口，正式引擎及存档保持原行为。
    """
    with parameter_scope(params, t0):
        engine = Engine()
        engine._opening_symmetry = 0
        ds = RecordingSearch(engine.search, record=record, node_limit=node_limit)
        engine.deep_search = ds
        board = empty_board()
        for r, c, player in position['moves']:
            reason, points = engine.search.tactical_candidates(board, player)
            # 实际每手只在普通候选路径选择档位；重放同样的历史状态转移，不深搜旧手。
            if not reason and engine._opening_candidates(board, player, (reason, points)) is None:
                engine.search._choose_gear(board, player)
            board[r][c] = player
            engine.on_move(board, r, c, player)
        before_board = copy.deepcopy(board)
        before_search = _restore_snapshot(engine, board)
        start = time.perf_counter()
        result = engine.analyze_turn(board, position['player'], complete_scores=record)
        elapsed = time.perf_counter() - start
        # analyze_turn 允许更新根档位，模拟搜索不能污染其他状态。
        after = _restore_snapshot(engine, board)
        before_search[1]['gear'] = copy.deepcopy(after[1]['gear'])
        if after != before_search or board != before_board:
            raise AssertionError('深推未恢复棋盘、状态表或真实历史')
        rows = []
        for item in result['part4_result']['ranked']:
            if item['score'] is None:
                # 普通性能探针不需要被淘汰候选的评分；完整记录路径不允许 None。
                if record:
                    raise AssertionError('完整记录仍有未补算分')
                continue
            constant, terms = flatten(item['score'])
            row = dict(r=item['r'], c=item['c'], state=item['state'],
                       score=float(item['score']), constant=constant, terms=terms)
            score = replay_row(row, params)[0]
            if not math.isclose(score, row['score'], rel_tol=1e-10, abs_tol=1e-7):
                raise AssertionError('叶子缓存无法复现原始候选分')
            rows.append(row)
        move = result['part4_result']['move']
        if move is None:
            raise ValueError('没有合法着法')
        if record and (choose(rows, params)['r'], choose(rows, params)['c']) != tuple(move):
            raise AssertionError('缓存无法复现正式选点')
        return dict(rows=rows, move=list(move), nodes=ds.nodes, leaves=ds.leaves,
                    seconds=elapsed, reason=result['part3_candidates']['reason'])


def fit_leaf(cases, initial, base, steps=300, seed=1, margin=.08, deadline=None):
    """固定树上的平滑成对排序优化，手工求导 + Adam，不使用神经网络或第三方库。

    仅比较相同最高五态档内、教师差距超过 margin 的候选；不同硬/软档不可
    通过数值分跨越。无有效比较时原样返回，并报告零个可训练比较。
    """
    p = dict(initial)
    eligible = []
    for case in cases:
        rows = case['tree']['rows']
        tier = max(DeepSearch.STATE_ORDER[row['state']] for row in rows)
        rows = [row for row in rows if DeepSearch.STATE_ORDER[row['state']] == tier]
        pairs = []
        for i, a in enumerate(rows):
            for j, b in enumerate(rows):
                gap = case['utilities'][f'{a["r"]},{a["c"]}'] - case['utilities'][f'{b["r"]},{b["c"]}']
                if gap > margin:
                    pairs.append((i, j, gap))
        if pairs:
            eligible.append((rows, pairs))
    if not eligible:
        return p, {'trainable_positions': 0, 'pairs': 0, 'steps': 0}
    rng = random.Random(seed)
    m, v = [0.0] * 24, [0.0] * 24
    for step in range(1, steps + 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise SearchLimit('叶子拟合达到本轮总时长预算')
        rows, pairs = rng.choice(eligible)
        values = [replay_row(row, p, gradient=True) for row in rows]
        grad = [0.0] * 24
        for i, j, gap in pairs:
            z = (values[i][0] - values[j][0]) / 500.0
            sig = 1 / (1 + math.exp(max(-60, min(60, z))))
            for k, name in enumerate(LEAF_KEYS):
                grad[k] -= gap * sig * (values[i][1][k] - values[j][1][k]) * base[name] / 500 / len(pairs)
        for k, name in enumerate(LEAF_KEYS):
            # 在相对原始权重的坐标中优化，并惩罚远离基线的无必要漂移。
            grad[k] += .002 * (p[name] / base[name] - 1)
            m[k] = .9 * m[k] + .1 * grad[k]
            v[k] = .999 * v[k] + .001 * grad[k] ** 2
            update = .015 * (m[k] / (1 - .9 ** step)) / (math.sqrt(v[k] / (1 - .999 ** step)) + 1e-8)
            p[name] -= base[name] * update
        p = project(p, base)
    return p, dict(trainable_positions=len(eligible), pairs=sum(len(x[1]) for x in eligible), steps=steps)


def perturb(params, base, group, seed):
    rng = random.Random(seed)
    p = dict(params)
    for k in rng.sample(GROUPS[group], min(3, len(GROUPS[group]))):
        if k.startswith('threshold_'):
            p[k] += rng.gauss(0, .8)
        elif k == 'gear_window':
            p[k] += rng.choice((-2, -1, 1, 2))
        else:
            p[k] *= math.exp(rng.gauss(0, .16))
    return project(p, base)
