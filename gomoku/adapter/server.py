# -*- coding: utf-8 -*-
"""常驻 Python 引擎服务（供 Node 插件通过 stdio JSON 调用）。

协议：每行一个 JSON 请求，每行一个 JSON 响应。

请求：
  {"id": 1, "type": "move", "session": "s1", "moves": [...], "player": 2}
  {"id": 2, "type": "ping"}
  {"id": 3, "type": "reset", "session": "s1"}

响应：
  {"id": 1, "ok": true, "row": 5, "col": 6}
  {"id": 1, "ok": false, "error": "..."}

特性：
  - 仅 DSH_GOMOKU_GA=1 时加载包内 GA 参数
  - 按 session 缓存 Engine 与棋盘，只增量应用新增 moves，避免每次重放全盘
"""
import json
import sys

from gomoku.core.utils import empty_board, place, BLACK, WHITE
from gomoku import service, memory

# 兼容已有调用者；配置和开局只保留一份实现。
load_ga_params = service.load_ga_params


def compute_move(engine, board, moves, player):
    move, ranked, net_used = service.ai_move(board, moves, player, engine)
    return (move[0], move[1], ranked, net_used) if move else (None, None, ranked, net_used)


class Session:
    def __init__(self):
        self.engine = service.create_engine()
        self.board = empty_board()
        self.moves = []

    @property
    def count(self):
        return len(self.moves)


sessions = {}


def handle(data):
    req_id = data.get('id')
    req_type = data.get('type')

    if req_type == 'ping':
        return {'id': req_id, 'ok': True}

    if req_type == 'reset':
        sessions.pop(data.get('session'), None)
        return {'id': req_id, 'ok': True}

    if req_type != 'move':
        return {'id': req_id, 'ok': False, 'error': 'unknown type: ' + str(req_type)}

    sid = data.get('session') or 'default'
    moves = data.get('moves') or []
    try:
        player = int(data.get('player', WHITE))
    except (TypeError, ValueError):
        return {'id': req_id, 'ok': False, 'error': 'bad player'}

    if player not in (BLACK, WHITE):
        return {'id': req_id, 'ok': False, 'error': 'bad player'}
    # 先完整校验，非法请求不得部分修改常驻会话。
    try:
        normalized = []
        seen = set()
        for i, move in enumerate(moves):
            r, c, p = move['r'], move['c'], move['player']
            if (type(r) is not int or type(c) is not int or type(p) is not int
                    or not (0 <= r < 15 and 0 <= c < 15)
                    or p != 1 + i % 2 or (r, c) in seen):
                raise ValueError('invalid move history')
            normalized.append({'r': r, 'c': c, 'player': p})
            seen.add((r, c))
    except (KeyError, TypeError, ValueError) as e:
        return {'id': req_id, 'ok': False, 'error': str(e)}
    moves = normalized
    s = sessions.get(sid)
    if s is None or moves[:s.count] != s.moves:
        s = Session()
        sessions[sid] = s
    for m in moves[s.count:]:
        r, c, p = m['r'], m['c'], m['player']
        place(s.board, r, c, p)
        s.engine.on_move(s.board, r, c, p)
        s.moves.append(dict(m))
    # Node 在进程外写入战绩：每次真实请求重新读取，避免缓存一直看旧记忆。
    memory._global_memory = None
    s.engine._memory = None

    try:
        row, col, ranked, net_used = compute_move(s.engine, s.board, moves, player)
        return {'id': req_id, 'ok': True, 'row': row, 'col': col, 'ranked': ranked, 'net_used': bool(net_used)}
    except Exception as e:
        sessions.pop(sid, None)
        return {'id': req_id, 'ok': False, 'error': str(e)}


def main():
    load_ga_params()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            resp = handle(data)
        except Exception as e:
            resp = {'id': None, 'ok': False, 'error': str(e)}
        sys.stdout.write(json.dumps(resp) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
