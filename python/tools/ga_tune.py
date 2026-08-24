# ============================================================
# ga_tune.py —— 遗传算法参数寻优（候选点评分 + 动态攻防配比 + 深推温度）
#
# 优化对象（monkey-patch 注入，零侵入）：
#   组1 score（24 个）：TYPE_SCORE(12) NEIGHBOR_W(6) PROX_DIST_W(2)
#                       PROX_DIR_W(2) W_NEIGHBOR(1) DUAL_BONUS(1)
#   组2 gear （8 个）：GEAR_W_A1/A2/B/ZERO GEAR_ZERO_WINDOW GEAR_THRESHOLDS×3
#   组3 deep （5 个）：T0 DT_M DT_POW DT_N WIN_SCORE
#   共 37 维；通过 --groups 可只优化其中若干组，未选组固定为基线。
#
# 约束（罚函数）：
#   C1-C4 棋型分大小关系（a2>a1>b 系内递减）→ 强罚
#   C5  1格距离权 >= 2格距离权
#   C6  同线方向权 >= 异线方向权
#   C7  attack: 自家>0 且 对家<0
#   C8  defend/dual: 自家、对家均 >0
#   C9  W_NEIGHBOR/DUAL_BONUS 在值域内（初始化 clamp 保证）
#   C10 gear 权重递减且为正：a1 > a2 > b > 0
#   C11 deep 参数为正
#   C12 gear_zero_window >= 2
#   gear_thresholds 顺序在 _vec_to_genome/_fix_genome 中自动保证 t1>=t2>=t3
#
# 适应度：与基准引擎自对弈胜率
#   - 黑白轮换（先手抵消） + 固定开局4手（聚焦中盘）+ 僵持判平 0.5
#   - 每体 games 局，取均值
#   - 若 deep 组未参与，评估用 FAST_T0 浅推加速；若参与，则使用个体自己的 T0
#
# 存档：每代结束保存 checkpoint（种群/rng/历史/args），可 --resume 续跑。
#
# 运行：
#   python ga_tune.py [--pop 40 --gens 40 --games 8 --workers 4 --seed 1]
#                    [--groups score,gear,deep] [--ckpt ga_ckpt.json] [--resume]
# 输出：best_params.json（最优参数）+ 代际日志
# ============================================================

import argparse
import json
import math
import os
import random
import time
from multiprocessing import Pool

from python.core.utils import empty_board, place, EMPTY, BLACK, WHITE
from python.core.engine import Engine
from python.core.deep_search import DeepSearch
from python.core import search as S

# ---------- 基线参数（当前默认，search.py / deep_search.py 原值） ----------

def _baseline():
    return {
        'type_score': {int(k): float(v) for k, v in S.TYPE_SCORE.items()},
        'neighbor_w': {k: (float(v[0]), float(v[1])) for k, v in S.NEIGHBOR_W.items()},
        'prox_dist': [float(S.PROX_DIST_W[1]), float(S.PROX_DIST_W[2])],
        'prox_dir': [float(S.PROX_DIR_W['line']), float(S.PROX_DIR_W['other'])],
        'w_neighbor': float(S.W_NEIGHBOR),
        'dual_bonus': float(S.DUAL_BONUS),
        # 动态攻防配比
        'gear_w_a1': float(S.GEAR_W_A1),
        'gear_w_a2': float(S.GEAR_W_A2),
        'gear_w_b': float(S.GEAR_W_B),
        'gear_w_zero': float(S.GEAR_W_ZERO),
        'gear_zero_window': int(S.GEAR_ZERO_WINDOW),
        'gear_thresholds': [float(x) for x in S.GEAR_THRESHOLDS],
        # 深推温度
        'deep_t0': float(DeepSearch.T0),
        'deep_dt_m': float(DeepSearch.DT_M),
        'deep_dt_pow': float(DeepSearch.DT_POW),
        'deep_dt_n': float(DeepSearch.DT_N),
        'deep_win_score': float(DeepSearch.WIN_SCORE),
    }

BASELINE = _baseline()

# ---------- 基因分组与平铺顺序 ----------
# 向量顺序：score(24) + gear(8) + deep(5) = 37
GENE_GROUPS = {
    'score': (
        'ts_a2_bbb', 'ts_a2_bbc', 'ts_a2_bcc', 'ts_a2_ccc',
        'ts_a1_bbb', 'ts_a1_bbc', 'ts_a1_bcc', 'ts_a1_ccc',
        'ts_b_bbb', 'ts_b_bbc', 'ts_b_bcc', 'ts_b_ccc',
        'nw_attack_self', 'nw_attack_opp',
        'nw_defend_self', 'nw_defend_opp',
        'nw_dual_self', 'nw_dual_opp',
        'pd_1', 'pd_2', 'pr_line', 'pr_other',
        'w_neighbor', 'dual_bonus',
    ),
    'gear': (
        'gear_w_a1', 'gear_w_a2', 'gear_w_b', 'gear_w_zero',
        'gear_zero_window', 'gt_t1', 'gt_t2', 'gt_t3',
    ),
    'deep': (
        'deep_t0', 'deep_dt_m', 'deep_dt_pow', 'deep_dt_n', 'deep_win_score',
    ),
}

GENE_KEYS = GENE_GROUPS['score'] + GENE_GROUPS['gear'] + GENE_GROUPS['deep']
GENE_KEY_TO_GROUP = {key: group for group, keys in GENE_GROUPS.items() for key in keys}

# 值域（clamp 用）
RANGES = {
    'ts_a2_bbb': (0, 400), 'ts_a2_bbc': (0, 400), 'ts_a2_bcc': (0, 400), 'ts_a2_ccc': (0, 400),
    'ts_a1_bbb': (0, 400), 'ts_a1_bbc': (0, 400), 'ts_a1_bcc': (0, 400), 'ts_a1_ccc': (0, 400),
    'ts_b_bbb': (0, 400), 'ts_b_bbc': (0, 400), 'ts_b_bcc': (0, 400), 'ts_b_ccc': (0, 400),
    'nw_attack_self': (10, 300), 'nw_attack_opp': (-200, -5),
    'nw_defend_self': (0, 300), 'nw_defend_opp': (0, 300),
    'nw_dual_self': (0, 300), 'nw_dual_opp': (0, 300),
    'pd_1': (0.1, 5.0), 'pd_2': (0.1, 5.0),
    'pr_line': (0.1, 5.0), 'pr_other': (0.1, 5.0),
    'w_neighbor': (0, 100), 'dual_bonus': (0, 100),
    # gear
    'gear_w_a1': (0.5, 12.0),
    'gear_w_a2': (0.2, 8.0),
    'gear_w_b': (0.0, 3.0),
    'gear_w_zero': (0.0, 20.0),
    'gear_zero_window': (2, 30),
    'gt_t1': (-10.0, 20.0),
    'gt_t2': (-15.0, 10.0),
    'gt_t3': (-20.0, 5.0),
    # deep
    'deep_t0': (10.0, 40.0),
    'deep_dt_m': (100.0, 3000.0),
    'deep_dt_pow': (0.2, 4.0),
    'deep_dt_n': (0.0, 20.0),
    'deep_win_score': (100.0, 5000.0),
}

def _genome_to_vec(g):
    ts = g['type_score']
    nw = g['neighbor_w']
    return [
        ts[9], ts[10], ts[11], ts[12],
        ts[5], ts[6], ts[7], ts[8],
        ts[13], ts[14], ts[15], ts[16],
        nw['attack'][0], nw['attack'][1],
        nw['defend'][0], nw['defend'][1],
        nw['dual'][0], nw['dual'][1],
        g['prox_dist'][0], g['prox_dist'][1],
        g['prox_dir'][0], g['prox_dir'][1],
        g['w_neighbor'], g['dual_bonus'],
        g['gear_w_a1'], g['gear_w_a2'], g['gear_w_b'], g['gear_w_zero'],
        float(g['gear_zero_window']),
        g['gear_thresholds'][0], g['gear_thresholds'][1], g['gear_thresholds'][2],
        g['deep_t0'], g['deep_dt_m'], g['deep_dt_pow'], g['deep_dt_n'], g['deep_win_score'],
    ]

def _vec_to_genome(vec):
    # 阈值顺序统一降序：t1 >= t2 >= t3
    gt = sorted([float(vec[29]), float(vec[30]), float(vec[31])], reverse=True)
    return {
        'type_score': {9: vec[0], 10: vec[1], 11: vec[2], 12: vec[3],
                       5: vec[4], 6: vec[5], 7: vec[6], 8: vec[7],
                       13: vec[8], 14: vec[9], 15: vec[10], 16: vec[11]},
        'neighbor_w': {'attack': (vec[12], vec[13]),
                       'defend': (vec[14], vec[15]),
                       'dual': (vec[16], vec[17])},
        'prox_dist': [vec[18], vec[19]],
        'prox_dir': [vec[20], vec[21]],
        'w_neighbor': vec[22],
        'dual_bonus': vec[23],
        # gear
        'gear_w_a1': float(vec[24]),
        'gear_w_a2': float(vec[25]),
        'gear_w_b': float(vec[26]),
        'gear_w_zero': float(vec[27]),
        'gear_zero_window': int(round(float(vec[28]))),
        'gear_thresholds': gt,
        # deep
        'deep_t0': float(vec[32]),
        'deep_dt_m': float(vec[33]),
        'deep_dt_pow': float(vec[34]),
        'deep_dt_n': float(vec[35]),
        'deep_win_score': float(vec[36]),
    }

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _parse_groups(text):
    """解析 --groups 字符串，返回规范分组列表。"""
    return [s.strip() for s in text.split(',') if s.strip()]

def _active_keys(groups):
    active = set()
    for g in (groups or GENE_GROUPS.keys()):
        active.update(GENE_GROUPS[g])
    return active

# ---------- 约束罚分 ----------

PEN_STRONG = 2000.0   # C1-C4/C7-C8/C10-C12 违反（几乎淘汰）
PEN_SCALE = 0.01      # 罚分 → 适应度扣减系数

def constraint_penalty(g):
    vec = _genome_to_vec(g)
    nw = g['neighbor_w']
    pen = 0.0
    # C1-C3 系内严格递减（vec 顺序：0-3 a2、4-7 a1、8-11 b）
    for base in (0, 4, 8):
        for i in range(3):
            if vec[base + i] <= vec[base + i + 1]:
                pen += PEN_STRONG
    # C4 大类首档 a2 > a1 > b
    if not (vec[0] > vec[4] > vec[8]):
        pen += PEN_STRONG
    # C5 距离 1格 >= 2格（弱罚）
    if g['prox_dist'][0] < g['prox_dist'][1]:
        pen += (g['prox_dist'][1] - g['prox_dist'][0]) * 10
    # C6 同线 >= 异线（弱罚）
    if g['prox_dir'][0] < g['prox_dir'][1]:
        pen += (g['prox_dir'][1] - g['prox_dir'][0]) * 10
    # C7 attack 符号
    if nw['attack'][0] <= 0 or nw['attack'][1] >= 0:
        pen += PEN_STRONG
    # C8 defend/dual 双正
    for k in ('defend', 'dual'):
        if nw[k][0] <= 0 or nw[k][1] <= 0:
            pen += PEN_STRONG
    # C10 gear 权重递减且为正
    if not (g['gear_w_a1'] > g['gear_w_a2'] > g['gear_w_b'] > 0):
        pen += PEN_STRONG
    # C11 deep 参数为正
    if g['deep_t0'] <= 0 or g['deep_dt_m'] <= 0 or g['deep_dt_pow'] <= 0 or g['deep_dt_n'] < 0 or g['deep_win_score'] <= 0:
        pen += PEN_STRONG
    # C12 窗口至少 2 步
    if g['gear_zero_window'] < 2:
        pen += PEN_STRONG
    return pen

# ---------- 参数注入（monkey-patch search / deep_search 模块） ----------

def apply_params(g):
    ts = {int(k): float(v) for k, v in g['type_score'].items()}
    S.TYPE_SCORE = ts
    S.NEIGHBOR_W = {k: (float(v[0]), float(v[1])) for k, v in g['neighbor_w'].items()}
    S.PROX_DIST_W = {1: float(g['prox_dist'][0]), 2: float(g['prox_dist'][1])}
    S.PROX_DIR_W = {'line': float(g['prox_dir'][0]), 'other': float(g['prox_dir'][1])}
    S.W_NEIGHBOR = float(g['w_neighbor'])
    S.DUAL_BONUS = float(g['dual_bonus'])
    # SCORE_MAX 动态联动
    S.SCORE_MAX = max(S.TYPE_SCORE.values()) + S.W_NEIGHBOR + S.DUAL_BONUS
    # 动态攻防配比
    S.GEAR_W_A1 = float(g['gear_w_a1'])
    S.GEAR_W_A2 = float(g['gear_w_a2'])
    S.GEAR_W_B = float(g['gear_w_b'])
    S.GEAR_W_ZERO = float(g['gear_w_zero'])
    S.GEAR_ZERO_WINDOW = int(round(float(g['gear_zero_window'])))
    S.GEAR_THRESHOLDS = tuple(float(x) for x in g['gear_thresholds'])
    # 深推温度
    DeepSearch.T0 = float(g['deep_t0'])
    DeepSearch.DT_M = float(g['deep_dt_m'])
    DeepSearch.DT_POW = float(g['deep_dt_pow'])
    DeepSearch.DT_N = float(g['deep_dt_n'])
    DeepSearch.WIN_SCORE = float(g['deep_win_score'])

def _fix_genome(g):
    """JSON 往返后的基因组规范化：int 键、tuple 值还原，旧存档缺字段补默认。"""
    return {
        'type_score': {int(k): float(v) for k, v in g['type_score'].items()},
        'neighbor_w': {k: (float(v[0]), float(v[1])) for k, v in g['neighbor_w'].items()},
        'prox_dist': [float(x) for x in g['prox_dist']],
        'prox_dir': [float(x) for x in g['prox_dir']],
        'w_neighbor': float(g['w_neighbor']),
        'dual_bonus': float(g['dual_bonus']),
        'gear_w_a1': float(g.get('gear_w_a1', BASELINE['gear_w_a1'])),
        'gear_w_a2': float(g.get('gear_w_a2', BASELINE['gear_w_a2'])),
        'gear_w_b': float(g.get('gear_w_b', BASELINE['gear_w_b'])),
        'gear_w_zero': float(g.get('gear_w_zero', BASELINE['gear_w_zero'])),
        'gear_zero_window': int(round(float(g.get('gear_zero_window', BASELINE['gear_zero_window'])))),
        'gear_thresholds': sorted([float(x) for x in g.get('gear_thresholds', BASELINE['gear_thresholds'])], reverse=True),
        'deep_t0': float(g.get('deep_t0', BASELINE['deep_t0'])),
        'deep_dt_m': float(g.get('deep_dt_m', BASELINE['deep_dt_m'])),
        'deep_dt_pow': float(g.get('deep_dt_pow', BASELINE['deep_dt_pow'])),
        'deep_dt_n': float(g.get('deep_dt_n', BASELINE['deep_dt_n'])),
        'deep_win_score': float(g.get('deep_win_score', BASELINE['deep_win_score'])),
    }

def _to_tuples(x):
    """JSON list → tuple（rng state 恢复用）"""
    if isinstance(x, list):
        return tuple(_to_tuples(i) for i in x)
    if isinstance(x, dict):
        return {k: _to_tuples(v) for k, v in x.items()}
    return x

# ---------- 对弈 ----------

# 固定开局（4 手：黑1 白1 黑2 白2，均为常见合理走法）
OPENINGS = [
    [(7, 7, BLACK), (6, 6, WHITE), (5, 7, BLACK), (6, 8, WHITE)],   # H8 G7 H6 I7
    [(7, 7, BLACK), (8, 8, WHITE), (6, 6, BLACK), (8, 6, WHITE)],   # H8 I9 G7 G9
    [(7, 7, BLACK), (7, 8, WHITE), (6, 6, BLACK), (8, 6, WHITE)],   # H8 H9 G7 G9
]

FAST_T0 = 40.0   # 评估温度：deep 组未参与时用（越高越准越慢）

def play_game(params_black, params_white, opening, max_moves=30, use_fast_t0=True):
    """一场自对弈：黑用 params_black 参数、白用 params_white 参数。
    返回 'B' / 'W' / None（僵持平局）。"""
    if use_fast_t0:
        DeepSearch.T0 = FAST_T0
    e = Engine()
    b = empty_board()
    for r, c, p in opening:
        place(b, r, c, p)
        e.on_move(b, r, c, p)
    turn = BLACK
    for _ in range(max_moves):
        params = params_black if turn == BLACK else params_white
        apply_params(params)
        if use_fast_t0:
            # deep 组未参与寻优：固定用浅推温度加速评估
            DeepSearch.T0 = FAST_T0
        res = e.analyze_turn(b, turn)
        move = res['part4_result']['move']
        if move is None:
            break
        r, c = move
        if not (0 <= r < 15 and 0 <= c < 15) or b[r][c] != EMPTY:
            break
        place(b, r, c, turn)
        e.on_move(b, r, c, turn)
        if DeepSearch._check_win(b, r, c, turn):
            return 'B' if turn == BLACK else 'W'
        turn = 3 - turn
    return None

# ---------- 个体评估（Pool worker 顶层函数） ----------

def evaluate_one(task):
    """task: (index, genome, cfg) -> (index, 胜率)"""
    idx, genome, cfg = task
    wins = 0.0
    n = 0
    for gi in range(cfg['games']):
        opening = cfg['openings'][gi % len(cfg['openings'])]
        black_is_trial = (gi % 2 == 0)      # 黑白轮换
        pb = genome if black_is_trial else cfg['baseline']
        pw = cfg['baseline'] if black_is_trial else genome
        w = play_game(pb, pw, opening, cfg['max_moves'], cfg.get('use_fast_t0', True))
        if w == 'B' and black_is_trial:
            wins += 1.0
        elif w == 'W' and not black_is_trial:
            wins += 1.0
        elif w is None:
            wins += 0.5                     # 僵持判平
        n += 1
    return idx, wins / max(n, 1)

# ---------- 遗传操作 ----------

def init_individual(seed_rng, groups=None):
    rng = seed_rng
    active = _active_keys(groups)
    vec = []
    for i, key in enumerate(GENE_KEYS):
        lo, hi = RANGES[key]
        base = _genome_to_vec(BASELINE)[i]
        if key not in active:
            v = base
        elif key.startswith('ts_'):
            # 以默认值为中心 × [0.5, 2]，保证初代质量
            v = base * rng.uniform(0.5, 2.0)
        elif key in ('pd_1', 'pd_2', 'pr_line', 'pr_other'):
            v = base * rng.uniform(0.3, 3.0)
        else:
            v = base * rng.uniform(0.5, 2.0)
        vec.append(_clamp(v, lo, hi))
    return _vec_to_genome(vec)

def blx_crossover(p1v, p2v, rng, alpha=0.25):
    """BLX-alpha 交叉（实数编码标准做法）"""
    c = []
    for a, b in zip(p1v, p2v):
        lo, hi = min(a, b), max(a, b)
        ext = (hi - lo) * alpha
        c.append(_clamp(lo - ext + rng.random() * (hi - lo + 2 * ext), 0.0, 1e6))
    return c

def mutate(vec, rng, sigma_scale=0.15, pm=0.15, groups=None):
    active = _active_keys(groups)
    out = []
    for i, v in enumerate(vec):
        lo, hi = RANGES[GENE_KEYS[i]]
        if GENE_KEYS[i] not in active:
            out.append(v)
            continue
        if rng.random() < pm:
            span = hi - lo
            v = v + rng.gauss(0, span * sigma_scale)
        out.append(_clamp(v, lo, hi))
    return out

def tournament(pop_fits, k, rng):
    """锦标赛选择：随机抓 k 个，取适应度最高者，返回其索引"""
    best_i = rng.randrange(len(pop_fits))
    best_f = pop_fits[best_i][1]
    for _ in range(k - 1):
        i = rng.randrange(len(pop_fits))
        if pop_fits[i][1] > best_f:
            best_i, best_f = i, pop_fits[i][1]
    return best_i

# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pop', type=int, default=40, help='population size')
    ap.add_argument('--gens', type=int, default=40, help='generations')
    ap.add_argument('--games', type=int, default=10, help='games per individual')
    ap.add_argument('--workers', type=int, default=4, help='parallel workers')
    ap.add_argument('--seed', type=int, default=1, help='random seed')
    ap.add_argument('--groups', default=None,
                    help='comma separated groups to optimize: score,gear,deep (default: all; resume uses saved groups if omitted)')
    ap.add_argument('--out', default='best_params.json', help='output json')
    ap.add_argument('--max-moves', type=int, default=30, help='max moves per game')
    ap.add_argument('--ckpt', default='ga_ckpt.json', help='checkpoint file (per-generation save)')
    ap.add_argument('--resume', action='store_true', help='resume from checkpoint file')
    args = ap.parse_args()

    if args.groups:
        groups = _parse_groups(args.groups)
    else:
        groups = None  # 稍后根据 resume/默认决定
    unknown = [g for g in (groups or []) if g not in GENE_GROUPS]
    if unknown:
        ap.error('unknown groups: %s' % unknown)

    rng = random.Random(args.seed)
    cfg = {
        'games': args.games,
        'openings': OPENINGS,
        'baseline': BASELINE,
        'max_moves': args.max_moves,
        'use_fast_t0': True,
    }
    pop = args.pop
    gens = args.gens
    elite_n = max(2, pop // 20)

    t0 = time.time()
    print('[ga] pop=%d gens=%d games=%d workers=%d seed=%d' % (
        pop, gens, args.games, args.workers, args.seed), flush=True)

    # 初代 或 从 checkpoint 续跑
    start_gen = 0
    best_ever = None
    best_ever_fit = -1.0
    no_improve = 0
    history = []
    if args.resume and os.path.exists(args.ckpt):
        with open(args.ckpt, 'r', encoding='utf-8') as f:
            ck = json.load(f)
        if groups is None:
            saved_groups = ck.get('args', {}).get('groups')
            groups = _parse_groups(saved_groups) if saved_groups else list(GENE_GROUPS.keys())
        population = [_fix_genome(g) for g in ck['population']]
        best_ever = _fix_genome(ck['best_ever'])
        best_ever_fit = ck['best_ever_fit']
        no_improve = ck['no_improve']
        history = ck['history']
        start_gen = ck['gen']
        rng.setstate(_to_tuples(ck['rng_state']))
        print('[ga] RESUME from gen %d (ckpt=%s) groups=%s' % (
            start_gen, args.ckpt, ','.join(groups)), flush=True)
    else:
        if groups is None:
            groups = list(GENE_GROUPS.keys())
        # 初代：含 1 个基线个体（保证不退化）
        population = [init_individual(rng, groups) for _ in range(pop - 1)] + [BASELINE]
        print('[ga] groups=%s' % ','.join(groups), flush=True)

    cfg['use_fast_t0'] = 'deep' not in groups

    def save_ckpt(gen_done):
        ck = {
            'gen': gen_done,
            'population': population,
            'best_ever': best_ever if best_ever is not None else BASELINE,
            'best_ever_fit': best_ever_fit,
            'no_improve': no_improve,
            'history': history,
            'rng_state': rng.getstate(),
            'args': vars(args),
        }
        ckpt_dir = os.path.dirname(args.ckpt)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        with open(args.ckpt, 'w', encoding='utf-8') as f:
            json.dump(ck, f, ensure_ascii=False)

    pool = Pool(args.workers)
    try:
        for gen in range(start_gen, gens):
            g0 = time.time()
            tasks = [(i, population[i], cfg) for i in range(pop)]
            results = pool.map(evaluate_one, tasks)
            results.sort(key=lambda x: x[0])
            fits = []
            for i in range(pop):
                base_fit = results[i][1]
                pen = constraint_penalty(population[i])
                fit = base_fit - pen * PEN_SCALE
                fits.append((i, fit, base_fit))
            fits.sort(key=lambda x: x[1], reverse=True)
            best_i, best_fit, best_base = fits[0]
            avg = sum(f[1] for f in fits) / pop

            # 精英
            elite = [population[f[0]] for f in fits[:elite_n]]

            if best_fit > best_ever_fit:
                best_ever_fit = best_fit
                best_ever = population[best_i]
                no_improve = 0
            else:
                no_improve += 1

            # 生成下一代（必须用旧种群选父母；best_ever 也须在替换前保存）
            sigma_decay = 1.0 - (gen / max(gens, 1)) * 0.5
            new_pop = list(elite)
            while len(new_pop) < pop:
                p1i = tournament(fits, 3, rng)
                p2i = tournament(fits, 3, rng)
                v1 = _genome_to_vec(population[p1i])
                v2 = _genome_to_vec(population[p2i])
                if rng.random() < 0.8:
                    child_v = blx_crossover(v1, v2, rng)
                else:
                    child_v = list(v1)
                child_v = mutate(child_v, rng, sigma_scale=0.15 * sigma_decay, groups=groups)
                new_pop.append(_vec_to_genome(child_v))
            population = new_pop

            history.append({'gen': gen, 'best': best_fit, 'best_base': best_base, 'avg': avg})
            print('[gen %d] best=%.4f (base=%.4f) avg=%.4f elapsed=%ds total=%ds no_improve=%d' % (
                gen, best_fit, best_base, avg, int(time.time() - g0), int(time.time() - t0), no_improve), flush=True)
            save_ckpt(gen + 1)   # 每代结束落盘，可断点续跑

            # 不启用 5 代无提升早停：按用户要求一直跑满 gens
    finally:
        pool.close()
        pool.join()

    # 输出
    if best_ever is None:
        best_ever = BASELINE
    out = {'fitness': best_ever_fit, 'params': best_ever, 'history': history,
           'config': vars(args)}
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('[ga] done. best fitness=%.4f -> %s' % (best_ever_fit, args.out), flush=True)

if __name__ == '__main__':
    main()
