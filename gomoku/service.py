# -*- coding: utf-8 -*-
"""引擎服务：封装核心引擎，对外提供规范化 AI 落子。

与插件 adapter/server.py 的 compute_move 保持一致：
  - 黑第一手 → 天元 (7,7)
  - 白第二手 → 开局候选列表第一个
  - 其余 → analyze_turn 深推选点
返回 ((row, col), ranked, net_used)。ranked 供界面展示全候选（按分降序）。
"""
import json
import os

from gomoku.core.utils import SIZE, EMPTY, BLACK, WHITE
from gomoku.core.engine import Engine
from gomoku.tools import ga_tune as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_engine = None


def load_ga_params():
    """DSH_GOMOKU_GA=1 时加载包内 GA 参数，缺失或损坏时回退默认。"""
    if os.environ.get('DSH_GOMOKU_GA') != '1':
        return
    path = os.path.join(ROOT, 'gomoku', 'data', 'ga_best_params.json')
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        params = data.get('params', data)
        G.apply_params(G._fix_genome(params))
    except Exception:
        pass


def create_engine(use_decision_net=None):
    """新会话拥有独立引擎，所有运行入口使用同一配置。"""
    load_ga_params()
    if use_decision_net is None:
        use_decision_net = os.environ.get('DSH_GOMOKU_USE_NET') == '1'
    return Engine(use_decision_net=use_decision_net)


def get_engine(use_decision_net=None):
    """兼容无会话调用的单例；正常会话使用 create_engine 隔离状态。"""
    global _engine
    if _engine is None:
        _engine = create_engine(use_decision_net=use_decision_net)
    elif use_decision_net is not None:
        _engine.use_decision_net = use_decision_net
    return _engine


def ai_move(board, moves, player, engine=None):
    """规范化 AI 落子。engine 可传入已与 board 同步的实例（界面用）；缺省用单例。

    返回 ((row, col), ranked, net_used)。
    """
    if engine is None:
        engine = get_engine()
        engine.reset(moves)
    if player not in (BLACK, WHITE):
        raise ValueError('bad player')
    if not any(EMPTY in row for row in board):
        return None, [], False

    # 黑第一手：天元
    if player == BLACK and len(moves) == 0:
        return (7, 7), [], False

    # 白第二手：开局候选列表取第一个
    if player == WHITE and len(moves) == 1:
        cand = engine.compute_move(board)
        if isinstance(cand, list) and cand:
            return (cand[0][0], cand[0][1]), [], False

    # 黑第二手（人类执白时 AI 的第二手，moves=[黑天元, 白1]）：
    # 贴白子开局（白 8 邻域空点中取离天元最近者），不深推——
    # 空旷 2 子局面深推候选质量差会乱走（与白第二手对称的开局特例）。
    if player == BLACK and len(moves) == 2:
        wr, wc = moves[-1]['r'], moves[-1]['c']
        best = None
        best_d = 99
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = wr + dr, wc + dc
                if not (0 <= rr < SIZE and 0 <= cc < SIZE) or board[rr][cc] != EMPTY:
                    continue
                d = max(abs(rr - 7), abs(cc - 7))   # 距天元（切比雪夫）
                if d < best_d:
                    best_d = d
                    best = (rr, cc)
        if best:
            return (best[0], best[1]), [], False

    res = engine.analyze_turn(board, player)
    mv = res['part4_result']['move']
    ranked = res['part4_result'].get('ranked') or []
    net_used = engine.decision_net_used

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
        return (r, c), ranked, net_used
    return (mv[0], mv[1]), ranked, net_used
