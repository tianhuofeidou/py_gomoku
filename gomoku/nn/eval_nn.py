# -*- coding: utf-8 -*-
"""NN 版 vs 规则版自对弈评估（Python 3.14 + CUDA 运行）。

用法：
  py -3.14 -m gomoku.nn.eval_nn --games 20 --max-moves 40 --t0 30 --model gomoku/nn/value_rl.pt
"""
import argparse
import random

from gomoku.core.utils import empty_board, place, BLACK, WHITE
from gomoku.core.engine import Engine
from gomoku.core.deep_search import DeepSearch


def play_game(nn_player, opening, max_moves=40, t0=30.0, model_path='gomoku/nn/value.pt'):
    """一局：nn_player 一方用 NN 评估，另一方用规则评估。
    返回胜者（1=黑,2=白,0=平/超步）。"""
    DeepSearch.T0 = t0
    e_nn = Engine()
    e_rule = Engine()
    e_nn.deep_search.use_nn = True
    e_nn.deep_search.model_path = model_path
    e_rule.deep_search.use_nn = False

    b = empty_board()
    for r, c, p in opening:
        place(b, r, c, p)
        e_nn.on_move(b, r, c, p)
        e_rule.on_move(b, r, c, p)
    turn = BLACK
    for _ in range(max_moves):
        engine = e_nn if turn == nn_player else e_rule
        res = engine.analyze_turn(b, turn)
        mv = res['part4_result']['move']
        if isinstance(mv, list):
            mv = mv[0] if mv else None
        if mv is None:
            break
        r, c = mv
        if b[r][c] != 0:
            break
        place(b, r, c, turn)
        e_nn.on_move(b, r, c, turn)
        e_rule.on_move(b, r, c, turn)
        if DeepSearch._check_win(b, r, c, turn):
            return turn
        turn = 3 - turn
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=20)
    ap.add_argument('--max-moves', type=int, default=40)
    ap.add_argument('--t0', type=float, default=30.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--model', default='gomoku/nn/value.pt')
    args = ap.parse_args()

    from gomoku.nn.data import OPENINGS
    rng = random.Random(args.seed)
    nn_wins = 0
    rule_wins = 0
    draws = 0
    nn_black_wins = 0
    nn_black_games = 0
    nn_white_wins = 0
    nn_white_games = 0
    for gi in range(args.games):
        opening = rng.choice(OPENINGS)
        # 黑白轮换让 NN 执黑/执白各一半
        nn_player = BLACK if gi % 2 == 0 else WHITE
        winner = play_game(nn_player, opening, args.max_moves, args.t0, args.model)
        if winner == 0:
            draws += 1
        elif winner == nn_player:
            nn_wins += 1
            if nn_player == BLACK:
                nn_black_wins += 1
            else:
                nn_white_wins += 1
        else:
            rule_wins += 1
        if nn_player == BLACK:
            nn_black_games += 1
        else:
            nn_white_games += 1
        print('game %d: nn=%s winner=%s' % (
            gi, 'B' if nn_player == BLACK else 'W', winner), flush=True)
    total = args.games
    print('NN wins=%d rule wins=%d draws=%d' % (nn_wins, rule_wins, draws))
    print('NN win rate=%.2f%%' % (100.0 * nn_wins / max(total, 1)))
    print('NN as black: %d/%d (%.2f%%)  as white: %d/%d (%.2f%%)' % (
        nn_black_wins, nn_black_games, 100.0 * nn_black_wins / max(nn_black_games, 1),
        nn_white_wins, nn_white_games, 100.0 * nn_white_wins / max(nn_white_games, 1)))


if __name__ == '__main__':
    main()
