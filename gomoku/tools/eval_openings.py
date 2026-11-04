# -*- coding: utf-8 -*-
"""从已有中盘局面评估纯算法引擎；rebuild_states 也供 GA 工具调用。

用法：python -m gomoku.tools.eval_openings --t0 80
"""
import argparse
import json

from gomoku.core.utils import empty_board, place, SIZE, EMPTY, BLACK, WHITE
from gomoku.core.engine import Engine
from gomoku.core.deep_search import DeepSearch


def rebuild_states(search, board):
    """从静态局面全盘重建黑白状态表（无着法历史时的兜底）。"""
    search.rebuild_all(board)


def play_from(board, to_move, e_turn, e_opp, t0, max_moves=60):
    """从给定局面开始：to_move 方用 e_turn，对方用 e_opp。返回胜负视角。"""
    DeepSearch.T0 = t0
    b = [row[:] for row in board]
    e_turn_board = [row[:] for row in board]
    e_opp_board = [row[:] for row in board]
    rebuild_states(e_turn.search, e_turn_board)
    rebuild_states(e_opp.search, e_opp_board)
    turn = to_move
    for _ in range(max_moves):
        e = e_turn if turn == to_move else e_opp
        cb = e_turn_board if e is e_turn else e_opp_board
        res = e.analyze_turn(cb, turn)
        mv = res['part4_result']['move']
        if isinstance(mv, list):
            mv = mv[0] if mv else None
        if mv is None:
            return None
        r, c = mv
        if b[r][c] != EMPTY:
            return None
        place(b, r, c, turn)
        place(e_turn_board, r, c, turn)
        place(e_opp_board, r, c, turn)
        e_turn.on_move(e_turn_board, r, c, turn)
        e_opp.on_move(e_opp_board, r, c, turn)
        if DeepSearch._check_win(b, r, c, turn):
            return turn == to_move
        turn = 3 - turn
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t0', type=float, default=80.0)
    ap.add_argument('--openings', default='gomoku/data/eval_openings.json')
    ap.add_argument('--max-moves', type=int, default=60)
    args = ap.parse_args()

    opens = json.load(open(args.openings, encoding='utf-8'))
    wins = 0
    n = 0
    for i, op in enumerate(opens):
        board, to_move = op['board'], op['to_move']
        e_turn = Engine()
        e_opp = Engine()
        w = play_from(board, to_move, e_turn, e_opp, args.t0, args.max_moves)
        tag = '纯算法'
        if w is True:
            wins += 1
            n += 1
            print('open%2d: %s方 WIN' % (i, tag), flush=True)
        elif w is False:
            n += 1
            print('open%2d: %s方 LOSS' % (i, tag), flush=True)
        else:
            print('open%2d: 僵持' % i, flush=True)
    print('=== %s: 轮到方胜率 %d/%d = %.0f%% ===' % (
        '纯算法', wins, n,
        wins / max(n, 1) * 100))


if __name__ == '__main__':
    main()
