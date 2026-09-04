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
  - 启动时加载 GA 最优参数（data/ga_best_params.json）
  - 按 session 缓存 Engine 与棋盘，只增量应用新增 moves，避免每次重放全盘
"""
import json
import os
import sys

from gomoku.core.utils import empty_board, place, BLACK, WHITE
from gomoku.core.engine import Engine
from gomoku.tools import ga_tune as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_ga_params():
    """加载 GA 最优参数；找不到/损坏时静默回退默认。"""
    path = os.path.join(ROOT, 'data', 'ga_best_params.json')
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        params = data.get('params', data)
        G.apply_params(G._fix_genome(params))
    except Exception:
        pass


def compute_move(engine, board, moves, player):
    """根据当前局面为 player 计算落子（与 ai_move.py 单次逻辑一致）。
    返回 (row, col, ranked, net_used)：
      - 黑第一手/白第二手：net_used=False（固定开局，不进决策网络）
      - analyze_turn：net_used = 引擎本次落子是否为决策网络选出（唯一候选=false，
        多候选且网络启用成功=true，纯算法回退=false）"""
    # 黑第一手：天元
    if player == BLACK and len(moves) == 0:
        return 7, 7, [], False
    # 白第二手：开局候选列表取第一个
    if player == WHITE and len(moves) == 1:
        cand = engine.compute_move(board)
        if isinstance(cand, list) and cand:
            return cand[0][0], cand[0][1], [], False
    res = engine.analyze_turn(board, player)
    mv = res['part4_result']['move']
    # 全候选 + 原始分（决策网络/大模型辅助用）：唯一候选=超大分 1e9，多候选=深推分支分降序
    ranked = res['part4_result'].get('ranked') or []
    # 本步是否网络选出：多候选且引擎的决策网络实际参与（_decision_pick 用网络返回第一名）。
    # _decision_pick 内部对"len==1"直取、对回退返回第一名；这里用 ranked>-1 判断"有候选可比"。
    net_used = len(ranked) >= 2 and engine.use_decision_net and engine.decision_net is not None
    if isinstance(mv, list):
        mv = mv[0] if mv else None
    if mv is None:
        r, c = 7, 7
        if board[r][c] != 0:
            for rr in range(15):
                for cc in range(15):
                    if board[rr][cc] == 0:
                        r, c = rr, cc
                        break
                else:
                    continue
                break
        return r, c, [], net_used
    return mv[0], mv[1], ranked, net_used


class Session:
    def __init__(self):
        # 决策网络开关：默认【关闭】（上纯算法）；环境变量 DSH_GOMOKU_USE_NET=1 才启用。
        # 注意：决策网络尚未可靠（只会"跟上纯算法"，防守死活未学好），默认用纯算法。
        env_net = os.environ.get('DSH_GOMOKU_USE_NET')
        use_net = env_net == '1'
        self.engine = Engine(use_decision_net=use_net)
        self.board = empty_board()
        self.count = 0


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

    s = sessions.get(sid)
    if s is None or len(moves) < s.count:
        # 新建或回退：重建后重放全量
        s = Session()
        sessions[sid] = s

    # 增量应用新增 moves
    for m in moves[s.count:]:
        r = int(m['r'])
        c = int(m['c'])
        p = int(m['player'])
        place(s.board, r, c, p)
        s.engine.on_move(s.board, r, c, p)
        s.count += 1

    try:
        row, col, ranked, net_used = compute_move(s.engine, s.board, moves, player)
        return {'id': req_id, 'ok': True, 'row': row, 'col': col, 'ranked': ranked, 'net_used': bool(net_used)}
    except Exception as e:
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
