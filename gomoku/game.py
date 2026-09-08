# -*- coding: utf-8 -*-
"""对局状态机与持久化（独立版，复刻插件的 state.js）。

负责棋局对象生命周期、落子/胜负/悔棋/提议协议、磁盘存档。
规则用标准库即可，不加载任何算法模块（算法只在 service.py 里被界面按需调用）。
"""
import json
import os
import time

from gomoku import paths

SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2
COLUMNS = 'ABCDEFGHIJKLMNO'  # 15 列

DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]


def check_win(board, row, col, player):
    """落子 (row,col) 后是否成五（四方向）。"""
    for dr, dc in DIRECTIONS:
        cnt = 1
        for step in range(1, 5):
            r, c = row + dr * step, col + dc * step
            if r < 0 or r >= SIZE or c < 0 or c >= SIZE or board[r][c] != player:
                break
            cnt += 1
        for step in range(1, 5):
            r, c = row - dr * step, col - dc * step
            if r < 0 or r >= SIZE or c < 0 or c >= SIZE or board[r][c] != player:
                break
            cnt += 1
        if cnt >= 5:
            return True
    return False


class Game:
    """一场对局。字段与插件的 emptyGame 对齐，便于迁移。"""

    def __init__(self, human_player=BLACK, engine='python', mode='ai',
                 ai_mode='engine'):
        self.moves = []               # [{r,c,player}, ...]
        self.winner = None            # None | BLACK | WHITE | 0（0=和棋）
        self.seq = 0
        self.human_player = human_player   # 1/2，谁是人类（先手恒黑）
        self.engine = engine
        self.mode = mode              # 'ai' | 'pvp' | 'vs'
        self.ai_mode = ai_mode        # 'engine' | 'llm'
        self.started_at = None
        self.history = []
        self.experience = [0, 0, 0, 0]
        self.pending = None           # {type, from, at}
        self.archived = False
        self.solidified = False
        self.solidify_info = None
        self.sid = None

    # ---------- 序列化 ----------
    def to_dict(self):
        return {
            'moves': self.moves, 'winner': self.winner, 'seq': self.seq,
            'humanPlayer': self.human_player, 'engine': self.engine,
            'mode': self.mode, 'aiMode': self.ai_mode,
            'startedAt': self.started_at, 'history': self.history,
            'experience': self.experience, 'pending': self.pending,
            'archived': self.archived, 'solidified': self.solidified,
            'solidifyInfo': self.solidify_info,
        }

    @classmethod
    def from_dict(cls, g):
        if not isinstance(g, dict):
            return None
        obj = cls(
            human_player=g.get('humanPlayer', BLACK),
            engine=g.get('engine', 'python'),
            mode=g.get('mode', 'ai'),
            ai_mode=g.get('aiMode', 'engine'),
        )
        obj.moves = g.get('moves') or []
        obj.winner = g.get('winner')
        obj.seq = g.get('seq', 0)
        obj.started_at = g.get('startedAt')
        obj.history = g.get('history') or []
        obj.experience = g.get('experience') or [0, 0, 0, 0]
        obj.pending = g.get('pending')
        obj.archived = g.get('archived', False)
        obj.solidified = g.get('solidified', False)
        obj.solidify_info = g.get('solidifyInfo')
        return obj

    # ---------- 持久化 ----------
    def register(self, sid):
        """绑定存档键并立即入档。"""
        self.sid = sid
        games[sid] = self
        persist()

    def persist(self):
        if self.sid:
            games[self.sid] = self
        from gomoku import game as game_mod
        game_mod.persist()
    # ---------- 基本查询 ----------
    def board(self):
        b = [[0] * SIZE for _ in range(SIZE)]
        for m in self.moves:
            b[m['r']][m['c']] = m['player']
        return b

    def current_player(self):
        return BLACK if len(self.moves) % 2 == 0 else WHITE

    def human_player_(self):
        return WHITE if self.human_player == WHITE else BLACK

    def ai_player(self):
        return WHITE if self.human_player_() == BLACK else BLACK

    def is_human_turn(self):
        return self.current_player() == self.human_player_()

    def coord(self, r, c):
        return COLUMNS[c] + str(r + 1)

    # ---------- 落子 ----------
    def place(self, row, col, player):
        if self.winner is not None:
            return {'ok': False, 'error': '棋局已结束'}
        if not isinstance(row, int) or not isinstance(col, int) \
                or row < 0 or row >= SIZE or col < 0 or col >= SIZE:
            return {'ok': False, 'error': '坐标越界'}
        if self.current_player() != player:
            return {'ok': False, 'error': '还没轮到该方落子'}
        b = self.board()
        if b[row][col] != 0:
            return {'ok': False, 'error': '该位置已有棋子'}
        if len(self.moves) == 0:
            self.started_at = int(time.time() * 1000)
        self.moves.append({'r': row, 'c': col, 'player': player})
        self.seq += 1
        if check_win(b, row, col, player):
            self.winner = player
        elif len(self.moves) >= SIZE * SIZE:
            self.winner = 0
        if self.winner is not None:
            self._archive_finished()
        self.persist()
        return {'ok': True}

    # ---------- 悔棋 / 认输 / 求和 ----------
    def _rollback_solidify(self):
        if self.archived and self.history:
            entry = self.history[-1]
            if entry['moves'] == self.moves and entry['winner'] == self.winner:
                from gomoku import memory
                memory.rollback_result(entry.get('memoryReceipt'), entry)
                self.history.pop()
        self.archived = False
        self.solidified = False
        self.solidify_info = None

    def undo_human_move(self):
        hp = self.human_player_()
        last = -1
        for i, m in enumerate(self.moves):
            if m['player'] == hp:
                last = i
        if last < 0:
            return
        self._rollback_solidify()
        self.moves = self.moves[:last]
        self.winner = None
        self.seq += 1
        self.pending = None
        self.persist()

    def undo_ai_move(self):
        ai = self.ai_player()
        last = -1
        for i, m in enumerate(self.moves):
            if m['player'] == ai:
                last = i
        if last < 0:
            return
        self._rollback_solidify()
        self.moves = self.moves[:last]
        self.winner = None
        self.seq += 1
        self.pending = None
        self.persist()

    def undo_last_move(self):
        if len(self.moves) == 0:
            return
        self._rollback_solidify()
        self.moves = self.moves[:-1]
        self.winner = None
        self.seq += 1
        self.pending = None
        self.persist()

    # ---------- 提议协议 ----------
    def propose(self, type_, from_):
        if self.winner is not None:
            return {'ok': False, 'error': '对局已结束'}
        if self.pending:
            return {'ok': False,
                    'error': '已有未决提议（%s，来自%s），先处理它' % (self.pending['type'], self.pending['from'])}
        self.pending = {'type': type_, 'from': from_, 'at': int(time.time() * 1000)}
        self.persist()
        return {'ok': True, 'pending': self.pending}

    def respond(self, accept):
        if not self.pending:
            return {'ok': False, 'error': '无未决提议'}
        pend = self.pending
        self.pending = None
        if not accept:
            self.persist()
            return {'ok': True, 'accepted': False, 'winner': self.winner}
        if pend['type'] == 'undo':
            if pend['from'] == 'ai':
                self.undo_ai_move()
            else:
                self.undo_human_move()
        elif pend['type'] == 'resign':
            if self.winner is None:
                self.winner = self.human_player_() if pend['from'] == 'ai' else self.ai_player()
                self.seq += 1
                self._archive_finished()
        elif pend['type'] == 'draw':
            if self.winner is None:
                self.winner = 0
                self.seq += 1
                self._archive_finished()
        return {'ok': True, 'accepted': True, 'winner': self.winner}

    # 对话触发式直接执行：by=human 处理人类请求（AI 已同意）；by=ai 确认 AI 提议。
    def exec(self, type_, by):
        if self.winner is not None:
            return {'ok': False, 'error': '对局已结束'}
        self.pending = None
        if type_ == 'undo':
            if by == 'ai':
                self.undo_ai_move()
            else:
                self.undo_human_move()
        elif type_ == 'resign':
            self.winner = self.human_player_() if by == 'ai' else self.ai_player()
            self.seq += 1
            self._archive_finished()
        elif type_ == 'draw':
            self.winner = 0
            self.seq += 1
            self._archive_finished()
        return {'ok': True, 'winner': self.winner}

    # ---------- 归档 / 结算 ----------
    def _archive_finished(self):
        if self.winner is None or len(self.moves) == 0 or self.archived:
            return
        self.archived = True
        # 记账闭环：胜负写入全局记忆（memo 里再 import，避免循环依赖）
        receipt = None
        try:
            from gomoku import memory
            receipt = memory.record_result(self)
        except Exception:
            pass
        self.history.append({
            'moves': [dict(m) for m in self.moves],
            'winner': self.winner,
            'humanPlayer': self.human_player,
            'engine': self.engine,
            'mode': self.mode or 'ai',
            'startedAt': self.started_at,
            'endedAt': int(time.time() * 1000),
            'memoryReceipt': receipt,
        })
        self._solidify()
        self.persist()

    def _solidify(self):
        if self.winner is None or len(self.moves) == 0 or self.solidified:
            return
        self.solidified = True
        hp = self.human_player_()
        ai = self.ai_player()
        result = 'loss' if self.winner == hp else 'win' if self.winner == ai else 'draw'
        self.solidify_info = {'result': result, 'mode': self.mode or 'ai'}


# ---------------------------------------------------------------------------
# 持久化：sid -> Game（内存权威；磁盘是副本）
# ---------------------------------------------------------------------------
games = {}


def load_all():
    try:
        with open(paths.games_file(), 'r', encoding='utf-8') as f:
            parsed = json.load(f)
        if isinstance(parsed, dict):
            for sid, g in parsed.items():
                obj = Game.from_dict(g)
                if obj is not None:
                    obj.sid = sid
                    games[sid] = obj
    except Exception:
        pass


def persist():
    try:
        paths.ensure_base()
        obj = {sid: g.to_dict() for sid, g in games.items()}
        with open(paths.games_file(), 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def game_of(sid):
    if not sid:
        return None
    g = games.get(sid)
    if g is None:
        g = Game()
        g.sid = sid
        games[sid] = g
    return g


load_all()
