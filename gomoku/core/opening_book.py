# -*- coding: utf-8 -*-
"""完整盘面的人类开局应手与研究线索；谱外交回引擎，不产生五态结论。"""
from .utils import BLACK, WHITE, EMPTY, SIZE
from .opening_book_data import RECORDS, ALTERNATIVES, CANDIDATES


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
    """推荐与未评级研究线分开索引；研究线不能用于前三手直接接管。"""
    def __init__(self, records=RECORDS, alternatives=None):
        self._moves = self._index_records(records)
        self._candidates = self._index_records(CANDIDATES if records is RECORDS else ())
        if any(len(k[1]) + len(k[2]) < 6 for k in self._candidates):
            raise ValueError('study candidates must follow both players first three moves')
        if alternatives is None:
            alternatives = ALTERNATIVES if records is RECORDS else ()
        self._alternatives = self._index_records(alternatives)
        for key, move in self._alternatives.items():
            if key not in self._moves or move == self._moves[key]:
                raise ValueError('alternative must have a distinct primary opening move')
        self._max_stones = max((len(k[1]) + len(k[2]) for k in
                                self._moves.keys() | self._candidates.keys()), default=-1)

    @staticmethod
    def _index_records(records):
        moves = {}
        for player, black, white, r, c in records:
            key = (player, tuple(black), tuple(white))
            if (player not in (BLACK, WHITE) or not (0 <= r < SIZE and 0 <= c < SIZE)
                    or r * SIZE + c in black or r * SIZE + c in white
                    or set(black) & set(white)):
                raise ValueError('invalid opening record')
            move = (r, c)
            if key in moves and moves[key] != move:
                raise ValueError('conflicting opening records')
            moves[key] = move
        return moves

    def lookup(self, board, player):
        """保持单点返回；推荐优先，未评级研究线仅在本方第四手起提供根候选。"""
        choices = self._lookup_choices(board, player)
        if choices:
            return choices[0]
        if (sum(row.count(player) for row in board) < 3
                or sum(cell != EMPTY for row in board for cell in row) > self._max_stones):
            return None
        key, symmetry = canonical_position(board, player)
        move = self._candidates.get(key)
        if move is None:
            return None
        r, c = inverse_point(*move, symmetry)
        if board[r][c] != EMPTY:
            raise ValueError('occupied study move')
        return r, c

    def _lookup_choices(self, board, player):
        if sum(cell != EMPTY for row in board for cell in row) > self._max_stones:
            return ()
        key, symmetry = canonical_position(board, player)
        move = self._moves.get(key)
        if move is None:
            return ()
        choices = [move]
        if key in self._alternatives:
            choices.append(self._alternatives[key])
        result = tuple(inverse_point(*point, symmetry) for point in choices)
        if any(board[r][c] != EMPTY for r, c in result):
            raise ValueError('occupied opening move')
        return result


def _opening_move(book, board, player, tactical, symmetry):
    """仅开局接管阶段使用每局固定视角；真实盘面与搜索状态不作修改。

    在旋转后的完整盘面查谱，再逆变换落点。原有方向的高位同时选择主着或备选，
    当前仅白二有正邻/斜邻两个备选：前四种视角取正邻，后四种取斜邻，
    各自在四次旋转中遍历四个方向，不新增随机状态或重新抽签。
    所有备选均须匹配完整盘面；出谱返回空，由同一个引擎继续战术分析及深推。
    """
    view = [[EMPTY] * SIZE for _ in range(SIZE)]
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != EMPTY:
                rr, cc = transform_point(r, c, symmetry)
                view[rr][cc] = board[r][c]
    choices = book._lookup_choices(view, player)
    if not choices:
        return None, None
    move = choices[(symmetry // 4) % len(choices)]
    return inverse_point(*move, symmetry), 'book'


def opening_fallback(board, player, tactical=()):
    """候选池为空时的合法点工具；不负责开局接管或替代正常多候选搜索。

    有战术候选时优先其原始权重；其余按邻接双方、局部连片和靠近中心选点。
    正式引擎仅在原算法无候选时调用，作为普通唯一候选继续处理，不提供开局标签。
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
