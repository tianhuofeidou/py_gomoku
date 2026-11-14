"""普通候选D：原生初选、单轴潜力扫描和最多四点的合榜精筛。

潜力是假设同方再落一子的结构特征，不代表对手没有应手或真实杀棋。
只在局部棋盘副本落子；Search状态表、方向缓存和档位均只读。
"""
from .utils import SIZE, EMPTY, BLACK, DIRECTIONS


def _axis(r, c, rr, cc):
    dr, dc = rr - r, cc - c
    if dr == 0:
        return 0
    if dc == 0:
        return 1
    if dr == dc:
        return 2
    if dr == -dc:
        return 3
    raise ValueError('潜力续点必须与候选位于同一轴线')


def _potential_strength(search, board, r, c, player, threat_points):
    """沿受影响轴重扫续点，其余三轴读取现有缓存，只计新生威胁。"""
    local = [row[:] for row in board]
    if local[r][c] != EMPTY:
        raise ValueError('普通候选必须为空点')
    local[r][c] = player
    table = search.sb if player == BLACK else search.sw
    cache = search._direction_levels[player]
    strength = 0
    for rr, cc in sorted(search.scope._segment_scope(local, r, c, player)):
        if local[rr][cc] != EMPTY or table[rr][cc] in threat_points:
            continue
        axis = _axis(r, c, rr, cc)
        levels = list(cache[rr * SIZE + cc])
        local[rr][cc] = player
        try:
            levels[axis] = search._direction_level(
                local, rr, cc, player, *DIRECTIONS[axis])
            after = search._classify_levels(levels)
        finally:
            local[rr][cc] = EMPTY
        if after in threat_points:
            strength = max(strength, threat_points[after])
    return strength


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(map(_freeze, value))
    return value


def _choose(rows, quota, limit=4):
    """按强侧分合榜；五档只参与双侧点的角色/去重视角分配。"""
    selected, used, seen = [], set(), set()
    counts = {'a': 0, 'd': 0}
    qa, qd = quota
    for _ in range(limit):
        ranked = []
        for row in rows:
            point = row['r'], row['c']
            if point in used:
                continue
            a, d = row['ascore'], row['dscore']
            if a is None and d is None:
                continue
            if a is None:
                role = 'd'
            elif d is None:
                role = 'a'
            else:
                da, dd = max(0, qa - counts['a']), max(0, qd - counts['d'])
                role = 'a' if da > dd or (da == dd and a >= d) else 'd'
            signature = role, _freeze(row['sig_' + role])
            if signature in seen:
                continue
            score = min(100.0, max(x for x in (a, d) if x is not None))
            oldbase = max(x for x in (row['a'], row['d']) if x is not None)
            # 与冻结D一致：同潜力分时先比较未加潜力的分，再按行列稳定排序。
            key = -score, -score, -oldbase, *point
            ranked.append((key, row, role, score, signature))
        if not ranked:
            break
        _, row, role, score, signature = min(ranked, key=lambda item: item[0])
        point = row['r'], row['c']
        used.add(point)
        seen.add(signature)
        counts[role] += 1
        selected.append((*point, max(1.0, score)))
    return selected


def select_candidates(search, board, player):
    """根层与递归共享普通入口；参数在search模块中，纳入既有缓存身份。"""
    from . import search as policy

    quota = policy.ORDINARY_GEAR_PROFILE[search._ordinary_gear(player)]
    ordered = {}
    for side, mode in ((player, 'attack'), (3 - player, 'defend')):
        table = search.sb if side == BLACK else search.sw
        points = []
        for r in range(SIZE):
            for c in range(SIZE):
                if (board[r][c] != EMPTY
                        or policy.SUBCLASS_PARENT[table[r][c]] not in (policy.THREAT, policy.POTEN)):
                    continue
                actual_mode = 'dual' if search._dual_score(board, r, c) > 0 else mode
                points.append((r, c, search._candidate_score(board, r, c, side, actual_mode)))
        points.sort(key=lambda point: point[2], reverse=True)
        ordered[side] = points

    # 原生五档名单额外入池，避免每侧前八的预筛丢掉原生去重后的代表。
    original = search._fallback_by_count(board, *quota, player, True)
    pool = {point[:2] for point in original}
    for points in ordered.values():
        pool.update(point[:2] for point in points[:policy.ORDINARY_POOL_SIDE])
    bases = {side: {(r, c): score for r, c, score in points}
             for side, points in ordered.items()}
    symmetries = search._board_symmetries(board)
    rows = []
    for r, c in sorted(pool):
        row = dict(r=r, c=c, a=None, d=None, ascore=None, dscore=None,
                   sig_a=None, sig_d=None)
        for role, side in (('a', player), ('d', 3 - player)):
            if (r, c) not in bases[side]:
                continue
            table = search.sb if side == BLACK else search.sw
            subtype = table[r][c]
            multiplier = (policy.ORDINARY_A1_MULT if subtype in (5, 6, 7, 8)
                          else policy.ORDINARY_A2_MULT if subtype in (9, 10, 11, 12)
                          else 1.0)
            base = bases[side][r, c]
            base += ((multiplier - 1.0) * policy.TYPE_SCORE.get(subtype, 0.0)
                     / policy.SCORE_MAX * 100.0)
            strength = _potential_strength(search, board, r, c, side,
                                           policy.GEAR_THREAT_POINTS)
            bonus = policy.ORDINARY_POTENTIAL_WEIGHT * strength / 100.0
            row[role] = base
            row[role + 'score'] = min(100.0, base + bonus / policy.SCORE_MAX * 100.0)
            row['sig_' + role] = search._dedup_sig(board, symmetries, r, c, side)
        rows.append(row)
    return _choose(rows, quota)
