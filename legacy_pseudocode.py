# ============================================================
# 单向扫描规则（假代码）
# gap 编号：1 = 贴身（x x 相邻），2 = 隔一个空（x _ x）
# ============================================================

SIZE = 15
EMPTY, BLACK, WHITE = 0, 1, 2

def in_board(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE

def scan_two(board, r, c, player, dr, dc):
    stones = 1
    l_gap_count = 0
    r_gap_count = 0
    l_gap_location=0
    r_gap_location=0

    i = 1

    r_bull, r_gap = True, False
    l_bull, l_gap = True, False

    # [位置, 类型]
    # 1 = __
    # 2 = |
    # 3 = _|
    l_edge = [0, 0]
    r_edge = [0, 0]

    while (r_bull or l_bull):
        rr, cr = r + i * dr, c + i * dc
        rl, cl = r - i * dr, c - i * dc

        # 向右扫描
        if r_bull and r_gap_count<2:
            if (rr, cr) == 'stone':
                stones += 1

                if r_gap:
                    r_gap_count += 1
                    if r_gap_location==0:
                        r_gap_location=i - 1
                    r_gap = False
                    if r_gap_count >= 2:
                        r_edge[0]=i-1
                        r_edge[1]=4
                        r_gap_count=1
                        stones-=1
                        r_bull = False

            elif (rr, cr) == 'open':
                if r_gap:
                    # __
                    r_bull = False
                    r_edge[0] = i - 1
                    r_edge[1] = 1
                else:
                    r_gap = True

            elif (rr, cr) == 'close':
                r_bull = False

                if r_gap:
                    # _|
                    r_edge[0] = i - 1
                    r_edge[1] = 3
                else:
                    # |
                    r_edge[0] = i
                    r_edge[1] = 2

        # 向左扫描
        if l_bull and l_gap_count<2:
            if (rl, cl) == 'stone':
                stones += 1

                if l_gap:
                    l_gap_count += 1
                    if l_gap_location==0:
                        l_gap_location=i - 1
                    l_gap = False
                    if l_gap_count >= 2:
                        l_edge[0] = i - 1
                        l_edge[1] = 4
                        l_gap_count = 1
                        stones -= 1
                        l_bull = False

            elif (rl, cl) == 'open':
                if l_gap:
                    # __
                    l_bull = False
                    l_edge[0] = i - 1
                    l_edge[1] = 1
                else:
                    l_gap = True

            elif (rl, cl) == 'close':
                l_bull = False

                if l_gap:
                    # _|
                    l_edge[0] = i - 1
                    l_edge[1] = 3
                else:
                    # |
                    l_edge[0] = i
                    l_edge[1] = 2

        i += 1

    result = {
        'stones': stones,
        'l_gap': l_gap_count,
        'r_gap': r_gap_count,
        'l_gap_locations': l_gap_location,
        'r_gap_locations': r_gap_location,
        'l_edge': l_edge,
        'r_edge': r_edge,
    }

    return result


# ============ 1.5 整体判定：双缝跳三（normalize 之前调用） ============

def overall_judge(res):
    """双缝 stones=3 的 x_X_x 跳三：填任一缝只成冲四（SLEEP4），
    整体恒为眠三，与两端类型无关。
    命中 → {'overall': 'SLEEP3'}；未命中 → None（走 normalize → classify）"""
    stones = res['stones']
    l_gap, r_gap = res['l_gap'], res['r_gap']

    # 只拦截：双缝 + 3 子（x_X_x），其余形状不拦
    if not (stones == 3 and l_gap == 1 and r_gap == 1):
        return None

    return {'overall': 'SLEEP3'}


# ============ 1. 标准化：scan_two 输出 → 1~2 条棋型记录 ============

def normalize(res):
    """scan_two 输出 → [{'stones','gap_location','l_edge','r_edge'}, ...]
    总 gap<=1 → 1 条；总 gap=2 → 拆 2 条（左段+中心段 / 中心段+右段）"""
    stones = res['stones']
    l_gap, r_gap = res['l_gap'], res['r_gap']
    l_loc, r_loc = res['l_gap_locations'], res['r_gap_locations']
    l_edge, r_edge = res['l_edge'], res['r_edge']

    if l_gap + r_gap <= 1:
        gap_location = l_loc if l_gap else (r_loc if r_gap else 0)
        return [{'stones': stones, 'gap_location': gap_location,
                 'l_edge': l_edge, 'r_edge': r_edge}]

    # 双缝：拆两份，基子端=缝（断裂端，标记为 4）
    left_seg  = (l_edge[0] - 1) - l_loc   # 左边缘段子数
    right_seg = (r_edge[0] - 1) - r_loc   # 右边缘段子数
    mid       = stones - left_seg - right_seg  # 中心段（含基子）

    return [
        {'stones': left_seg + mid, 'gap_location': l_loc,
         'l_edge': l_edge, 'r_edge': [r_loc, 4]},
        {'stones': mid + right_seg, 'gap_location': r_loc,
         'l_edge': [l_loc, 4], 'r_edge': r_edge},
    ]


# ============ 2. 父类判定：标准化记录 → 10 类 ============

def classify(item):
    """item: {'stones','gap_location','l_edge','r_edge'}
    端类型: 1=活(__/_x) 2=堵(|) 3=半活(_|) 4=断裂端(第二个_x，等价1活)"""
    stones = item['stones']
    gap = 1 if item['gap_location'] != 0 else 0
    l0, lt = item['l_edge']
    r0, rt = item['r_edge']

    # 段长（墙到墙，类型3的墙在空后+1）
    wall_l = l0 + (1 if lt == 3 else 0)
    wall_r = r0 + (1 if rt == 3 else 0)
    seg_len = wall_l + wall_r - 1

    # 活端候选集：1(活) 3(半活) 4(断裂=活)
    OPEN_END = (1, 3, 4)

    # 1. DEAD（4 不是堵端，不会被误判死）
    if lt in (2, 3) and rt in (2, 3) and seg_len < 5:
        return 'DEAD'

    # 2. SINGLE
    if stones == 1:
        return 'SINGLE'

    # 3. FIVE
    if stones >= 5 and gap == 0:
        return 'FIVE'

    # 4/5. 四连级
    if stones == 4:
        if gap == 0 and lt in OPEN_END and rt in OPEN_END:
            return 'LIVE4'
        return 'SLEEP4'
    if stones > 4:          # 带缝 5 子+：缝唯一补点 → 眠四
        return 'SLEEP4'

    # 6/7. 三连级
    if stones == 3:
        if gap == 0:
            if (lt in (1, 4) and rt in OPEN_END) or (rt in (1, 4) and lt in OPEN_END):
                return 'LIVE3'          # 1·1 / 1·3 / 4·1 / 4·3，排除 3·3
        elif lt in OPEN_END and rt in OPEN_END:
            return 'LIVE3'              # gap=1 时 3·3 也算活，4 同样算
        return 'SLEEP3'

    # 8/9. 二连级
    if stones == 2:
        if (lt in (1, 4) and rt in OPEN_END) or (rt in (1, 4) and lt in OPEN_END):
            return 'LIVE2'              # 1·1 / 1·3 / 4·1 / 4·3，排除 3·3
        return 'SLEEP2'

    # 10. NONE
    return 'NONE'