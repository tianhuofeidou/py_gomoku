# -*- coding: utf-8 -*-
"""self_memory.py —— 自我对弈记忆（记忆系统 B 套：引擎自己下棋的经验）。

与人机对弈记忆（global-memory.json 的 badLines/goodLines）分开存放：
  DSH_HOME/storages/gomoku/self-memory.json
  selfGoodLines = [{opp, ai, killType, winLine, at}, ...]  胜方视角（opp=负方, ai=胜方）
  selfBadLines  = [{opp, ai, killType, winLine, at}, ...]  负方视角（opp=胜方, ai=负方）

每局写两条（胜方视角 + 负方视角），保证决策网络无论执黑执白、无论当前方是
人机对弈的 AI 方还是自我对弈的任一方，都能对称匹配到历史经验。
"""
import json
import os
import time
from os.path import join, expanduser

from gomoku import paths
from gomoku.core.utils import empty_board, place, in_board, EMPTY

DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def default_self_memory_file():
    return paths.self_memory_file()


def loss_type_of(board, r, c, player):
    """player 最后一手 (r,c) 成五的杀法类型：跳四 / 活四 / 冲四 / 其他。
    与 Node memory.js lossTypeOf 同逻辑：移除最后一手后沿其五连线检查。"""
    board[r][c] = EMPTY
    best = None
    try:
        for (dr, dc) in DIRECTIONS:
            count = 1
            gap = 0
            rr, cc = r + dr, c + dc
            while in_board(rr, cc) and board[rr][cc] == player:
                count += 1
                rr += dr
                cc += dc
            end1 = 1 if (in_board(rr, cc) and board[rr][cc] == EMPTY) else 0
            gr, gc = rr + dr, cc + dc
            if (in_board(rr, cc) and board[rr][cc] == EMPTY
                    and in_board(gr, gc) and board[gr][gc] == player):
                while in_board(gr, gc) and board[gr][gc] == player:
                    gap += 1
                    gr += dr
                    gc += dc
            rr, cc = r - dr, c - dc
            while in_board(rr, cc) and board[rr][cc] == player:
                count += 1
                rr -= dr
                cc -= dc
            end2 = 1 if (in_board(rr, cc) and board[rr][cc] == EMPTY) else 0
            gr, gc = rr - dr, cc - dc
            if (in_board(rr, cc) and board[rr][cc] == EMPTY
                    and in_board(gr, gc) and board[gr][gc] == player):
                while in_board(gr, gc) and board[gr][gc] == player:
                    gap += 1
                    gr -= dr
                    gc -= dc
            if count + gap >= 5:
                best = {'count': count, 'gap': gap, 'end1': end1, 'end2': end2}
                break
    finally:
        board[r][c] = player
    if not best:
        return '其他'
    if best['gap'] > 0:
        return '跳四'
    if best['count'] >= 5:
        if best['end1'] and best['end2']:
            return '活四'
        if best['end1'] or best['end2']:
            return '冲四'
    return '其他'


def win_line_of(board, r, c, player):
    """player 最后一手 (r,c) 成五的那条线（5 个坐标 [(r,c),...]）。"""
    for (dr, dc) in DIRECTIONS:
        cells = [(r, c)]
        rr, cc = r + dr, c + dc
        while in_board(rr, cc) and board[rr][cc] == player:
            cells.append((rr, cc))
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while in_board(rr, cc) and board[rr][cc] == player:
            cells.insert(0, (rr, cc))
            rr -= dr
            cc -= dc
        if len(cells) >= 5:
            return cells[:5]
    return []


def _to_moves(arr):
    return [{'r': x[0], 'c': x[1]} for x in arr]


class SelfMemory:
    """自我对弈记忆：只追加写 self-memory.json（读由 memory_lookup 负责）。"""

    def __init__(self, path=None):
        self.path = path or default_self_memory_file()
        self.good_lines = []
        self.bad_lines = []
        try:
            with open(self.path, encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                if isinstance(data.get('selfGoodLines'), list):
                    self.good_lines = data['selfGoodLines']
                if isinstance(data.get('selfBadLines'), list):
                    self.bad_lines = data['selfBadLines']
        except Exception:
            pass

    def record_game(self, moves, winner):
        """记录一局自对弈。
        moves: [(r, c, player), ...] 完整对局；winner: 1/2（和棋 0 不记）。
        写两条：胜方视角 → good；负方视角 → bad（killType/winLine 按胜方杀法）。"""
        if winner not in (1, 2) or not moves:
            return False
        loser = 3 - winner
        win_moves = [(m[0], m[1]) for m in moves if m[2] == winner]
        lose_moves = [(m[0], m[1]) for m in moves if m[2] == loser]
        # 胜方最后一手 = 成五手
        last = None
        for m in reversed(moves):
            if m[2] == winner:
                last = (m[0], m[1])
                break
        if last is None:
            return False
        b = empty_board()
        for (r, c, p) in moves:
            place(b, r, c, p)
        kt = loss_type_of(b, last[0], last[1], winner)
        wl = _to_moves(win_line_of(b, last[0], last[1], winner))
        at = int(time.time() * 1000)
        self.good_lines.append({
            'opp': _to_moves(lose_moves), 'ai': _to_moves(win_moves),
            'killType': kt, 'winLine': wl, 'at': at,
        })
        self.bad_lines.append({
            'opp': _to_moves(win_moves), 'ai': _to_moves(lose_moves),
            'killType': kt, 'winLine': wl, 'at': at,
        })
        self.save()
        return True

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump({'selfGoodLines': self.good_lines, 'selfBadLines': self.bad_lines},
                          f, ensure_ascii=False)
        except Exception as e:
            print('[self-memory] 写盘失败:', e)
