# -*- coding: utf-8 -*-
"""完整盘面开局库与前三手的谱外规则；不调用深推、不产生五态结论。"""
from .utils import BLACK, WHITE, EMPTY, SIZE
from .opening_book_data import RECORDS


def transform_point(r, c, symmetry):
    """棋盘的八种旋转/镜像，不交换颜色、不平移边界。"""
    if symmetry >= 4:
        c = SIZE - 1 - c
    for _ in range(symmetry % 4):
        r, c = c, SIZE - 1 - r
    return r, c


def inverse_point(r, c, symmetry):
    for _ in range(symmetry % 4):
        r, c = SIZE - 1 - c, r
    if symmetry >= 4:
        c = SIZE - 1 - c
    return r, c


def canonical_position(board, player):
    """盘面及行棋方作为键，避免 BOARD 的输入次序影响查谱。"""
    if player not in (BLACK, WHITE):
        raise ValueError('bad player')
    stones = [(r, c, board[r][c]) for r in range(SIZE) for c in range(SIZE)
              if board[r][c] != EMPTY]
    choices = []
    for symmetry in range(8):
        colors = {BLACK: [], WHITE: []}
        for r, c, color in stones:
            rr, cc = transform_point(r, c, symmetry)
            colors[color].append(rr * SIZE + cc)
        key = (player, tuple(sorted(colors[BLACK])), tuple(sorted(colors[WHITE])))
        choices.append((key, symmetry))
    return min(choices)


class OpeningBook:
    """只读着法数据。离线分析的评价、深度不混入正式搜索分或必胜标签。"""
    def __init__(self, records=RECORDS):
        self._moves = {}
        for player, black, white, r, c in records:
            key = (player, tuple(black), tuple(white))
            if (player not in (BLACK, WHITE) or not (0 <= r < SIZE and 0 <= c < SIZE)
                    or r * SIZE + c in black or r * SIZE + c in white
                    or set(black) & set(white)):
                raise ValueError('invalid opening record')
            move = (r, c)
            if key in self._moves and self._moves[key] != move:
                raise ValueError('conflicting opening records')
            self._moves[key] = move
        self._max_stones = max((len(k[1]) + len(k[2]) for k in self._moves), default=-1)

    def lookup(self, board, player):
        if sum(cell != EMPTY for row in board for cell in row) > self._max_stones:
            return None
        key, symmetry = canonical_position(board, player)
        move = self._moves.get(key)
        if move is None:
            return None
        r, c = inverse_point(*move, symmetry)
        if board[r][c] != EMPTY:
            raise ValueError('occupied opening move')
        return r, c


def _opening_move(book, board, player, tactical, symmetry):
    """仅开局接管阶段使用每局固定视角；真实盘面与搜索状态不作修改。

    在旋转后的完整盘面查谱，再逆变换落点。盘面不对称时仍只能得到原应手，
    不能把谱着单独旋转到与现有棋子不匹配的位置。出谱后仍用真实盘面的原规则，
    不把仅几何评分相同的点当作对称等价；重复分析不消耗随机数。
    """
    view = [[EMPTY] * SIZE for _ in range(SIZE)]
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != EMPTY:
                rr, cc = transform_point(r, c, symmetry)
                view[rr][cc] = board[r][c]
    move = book.lookup(view, player)
    if move is None:
        return opening_fallback(board, player, tactical), 'rules'
    return inverse_point(*move, symmetry), 'book'


def opening_fallback(board, player, tactical=()):
    """前三手出谱时的确定性规则，复用紧急棋型候选但不进入温度银行。

    有战术候选时优先其原始权重；其余按邻接双方、局部连片和靠近中心选点。
    这是谱外规则，不冒充外部引擎分析结果；仅用于本方已落子数不足三的阶段。
    """
    occupied = [(r, c, board[r][c]) for r in range(SIZE) for c in range(SIZE)
                if board[r][c] != EMPTY]
    if not occupied:
        return SIZE // 2, SIZE // 2
    mine = [(r, c) for r, c, p in occupied if p == player]
    other = [(r, c) for r, c, p in occupied if p != player]

    def geometry(point):
        r, c = point
        near_mine = sum(max(abs(r - rr), abs(c - cc)) == 1 for rr, cc in mine)
        near_other = sum(max(abs(r - rr), abs(c - cc)) == 1 for rr, cc in other)
        distance = min(max(abs(r - rr), abs(c - cc)) for rr, cc, _p in occupied)
        return (bool(near_mine and near_other), near_mine + near_other,
                -distance, -max(abs(r - SIZE // 2), abs(c - SIZE // 2)), -r, -c)

    if tactical:
        return max(tactical, key=lambda point: (tactical[point], geometry(point)))
    empty = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == EMPTY]
    return max(empty, key=geometry) if empty else None
