# ============================================================
# utils.py —— 纯工具层：棋盘常量与访问（无业务逻辑）
# 被 pattern / search / evaluate / deep_search 依赖
# ============================================================

SIZE = 15
EMPTY, BLACK, WHITE = 0, 1, 2

# 四个扫描方向：(dr, dc)
# 0 = 横, 1 = 竖, 2 = 左斜(\)... 注：方向编号仅作约定，扫描本身对称
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# 棋盘列名（engine.py JSON 接口用）
COLUMNS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def in_board(r, c):
    """是否在棋盘内"""
    return 0 <= r < SIZE and 0 <= c < SIZE


def cell_state(board, r, c, player):
    """扫描状态：
    - 'stone'  : 该位置是 player 的己方子
    - 'open'   : 空位
    - 'close'  : 越界 或 对方子（对己方棋型都是墙）
    """
    if not in_board(r, c):
        return 'close'
    v = board[r][c]
    if v == player:
        return 'stone'
    if v == EMPTY:
        return 'open'
    return 'close'


def empty_board():
    """生成空棋盘（15x15 全 0）"""
    return [[EMPTY] * SIZE for _ in range(SIZE)]


def place(board, r, c, player):
    """落子（直接修改 board，返回是否成功）"""
    if not in_board(r, c) or board[r][c] != EMPTY:
        return False
    board[r][c] = player
    return True


def neighbors(r, c, radius=2):
    """返回 (r,c) 周围 radius 格内的所有合法坐标（含自身？不含自身）"""
    res = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if in_board(rr, cc):
                res.append((rr, cc))
    return res


def has_stone_nearby(board, r, c, radius=2):
    """该点周围 radius 格内是否有任何棋子（粗筛触发条件）"""
    for (rr, cc) in neighbors(r, c, radius):
        if board[rr][cc] != EMPTY:
            return True
    return False
