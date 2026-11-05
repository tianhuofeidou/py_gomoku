# -*- coding: utf-8 -*-
"""完整盘面的人类开局应手与研究线索；谱外交回引擎，不产生五态结论。

三套数据分工：
- 整盘前5手稀疏库（`opening_book_front`）：唯一可用于"固定根落点接管"的来源，
  每个完整盘面保留全部确认候选、暂定主着、评分视角与来源；
- 人类经典推荐（`RECORDS`/`ALTERNATIVES`）：本方第四次落子起只补一个根候选；
- 后续研究线（`CANDIDATES`）：同样只补根候选，不提供胜败证明。

接管只在整盘第1-5手生效（着手前盘面最多4子）。第6手起新库不再接管，
也不会回落到旧库接管；出谱立即交回原候选与深推。
"""
from .utils import BLACK, WHITE, EMPTY, SIZE
from .opening_book_data import RECORDS, ALTERNATIVES, CANDIDATES
from .opening_book_front import PROVENANCE as FRONT_PROVENANCE, RECORDS as FRONT_RECORDS

# 整盘前5手指双方合计的第1-5手：第5手落子前盘面最多4子。
FRONT_STONES_BEFORE = 4

# 候选种类。primary=暂定主着；policy=白二每局视角策略；coverage=已评分保留候选；
# unrated=待评候选（无同口径评分，不得当作最高分或零分）。
FRONT_KINDS = ('primary', 'policy', 'coverage', 'unrated')


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


def _front_rows(records):
    """把前5手记录按完整盘面键索引；多候选与来源全部保留，不做单值截断。"""
    data = {}
    for row in records:
        player, black, white, r, c, kind, score, score_kind, view, source = row
        if player not in (BLACK, WHITE):
            raise ValueError('invalid front player')
        if kind not in FRONT_KINDS:
            raise ValueError('invalid front kind')
        if not (0 <= r < SIZE and 0 <= c < SIZE):
            raise ValueError('invalid front move')
        if r * SIZE + c in black or r * SIZE + c in white or set(black) & set(white):
            raise ValueError('occupied front move')
        # 已评分覆盖必须带同口径分数；主着、白二策略和待评候选可以没有评分
        # （人工规则或尚无同口径评估），不得因此被当作最高分或零分。
        if kind == 'coverage' and score is None:
            raise ValueError('scored coverage candidate must carry its score')
        if score is not None and (score_kind not in ('cp', 'mate') or view not in ('black', 'white')):
            raise ValueError('invalid front evaluation')
        if not source:
            raise ValueError('front candidate must carry its source')
        key = (player, tuple(black), tuple(white))
        entry = {'r': r, 'c': c, 'kind': kind, 'score': score,
                 'score_kind': score_kind, 'view': view, 'source': source}
        bucket = data.setdefault(key, [])
        for existing in bucket:
            if (existing['r'], existing['c']) == (r, c):
                raise ValueError('conflicting front candidates')
        bucket.append(entry)
    return {key: tuple(bucket) for key, bucket in data.items()}


class OpeningBook:
    """推荐、研究线与整盘前5手库分开索引；研究线不能用于接管。

    `front` 缺省按打包数据取值；传入自定义 `records` 时，前5手索引退化为
    每条记录一个暂定主着，用于保持既有单记录测试入口的可接管语义。
    """
    def __init__(self, records=RECORDS, alternatives=None, front=None):
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
        if front is None:
            front = ((player, black, white, r, c, 'primary', None, None, None, 'custom-book')
                     for player, black, white, r, c in records) if records is not RECORDS \
                else FRONT_RECORDS
        self._front = _front_rows(tuple(front))
        self._max_stones = max((len(k[1]) + len(k[2]) for k in
                                self._moves.keys() | self._candidates.keys()), default=-1)
        self.front_provenance = FRONT_PROVENANCE if records is RECORDS else {}

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

    # ---------- 整盘前5手稀疏库 ----------

    def front_choices(self, board, player):
        """返回该完整盘面保留的全部前5手候选（盘面坐标），只含第1-5手。

        顺序为记录顺序：主着/策略着在前，其余按分数与坐标稳定排序。
        超过第5手或不在库内返回空；同盘面键的多候选不会被截断成单点。
        """
        if sum(cell != EMPTY for row in board for cell in row) > FRONT_STONES_BEFORE:
            return ()
        key, symmetry = canonical_position(board, player)
        rows = self._front.get(key)
        if not rows:
            return ()
        result = []
        for row in rows:
            r, c = inverse_point(row['r'], row['c'], symmetry)
            if board[r][c] != EMPTY:
                raise ValueError('occupied front move')
            result.append(((r, c), row['kind'], row['score'], row['score_kind'],
                           row['view'], row['source']))
        return tuple(result)

    def front_move(self, board, player, symmetry):
        """整盘前5手的接管落点；返回 (落点, 来源) 或 (None, None)。

        白二位置保留既有正邻/斜邻选择方式：八种视角前半取正邻、后半取斜邻，
        各自在四次旋转中遍历方向。其他局面只取暂定主着，不随机抽取候选，
        也不按 `symmetry // 4` 只在候选前两个里打转。
        """
        if sum(cell != EMPTY for row in board for cell in row) > FRONT_STONES_BEFORE:
            return None, None
        key, canonical = canonical_position(board, player)
        rows = self._front.get(key)
        if not rows:
            return None, None
        policy = [row for row in rows if row['kind'] == 'policy']
        if policy:
            chosen = policy[(symmetry // 4) % len(policy)]
        else:
            primary = [row for row in rows if row['kind'] == 'primary']
            if not primary:
                return None, None
            chosen = primary[0]
        move = inverse_point(chosen['r'], chosen['c'], canonical)
        if board[move[0]][move[1]] != EMPTY:
            raise ValueError('occupied front move')
        return move, chosen['source']

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

    在旋转后的完整盘面查前5手库，再逆变换落点。白二保留正邻/斜邻策略；
    其余局面只取暂定主着，未评分候选仅作覆盖，不参与选择。
    谱点须匹配完整盘面且只覆盖第1-5手；出谱返回空，由同一个引擎继续战术
    分析及深推。
    """
    view = [[EMPTY] * SIZE for _ in range(SIZE)]
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != EMPTY:
                rr, cc = transform_point(r, c, symmetry)
                view[rr][cc] = board[r][c]
    move, source = book.front_move(view, player, symmetry)
    if move is None:
        return None, None
    return inverse_point(*move, symmetry), source


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
