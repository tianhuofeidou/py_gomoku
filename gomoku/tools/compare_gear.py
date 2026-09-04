# -*- coding: utf-8 -*-
"""动态攻防配比自对弈对比测试。

比较两种配置（黑方始终固定 3攻2防，AI 执白）：
  - fixed   : 白方也固定 3攻2防（GEAR_ENABLED=False）
  - dynamic : 白方按局势动态 1/2/3/4 档

运行: python compare_gear.py [--games 6] [--max-moves 24] [--t0 40] [--seed 1]
"""
import argparse
import os
import random
import time
from multiprocessing import Pool

from gomoku.tools import ga_tune as G
from gomoku.core import search as S
from gomoku.core.utils import empty_board, place, EMPTY, BLACK, WHITE
from gomoku.core.engine import Engine

OPENINGS = G.OPENINGS
FAST_T0 = G.FAST_T0


def play_one(opening, max_moves, dynamic_white, t0):
    """跑一局，返回 (winner, moves, gear_log)。
    winner: 'B'/'W'/None；gear_log 只记录白方回合的档位。"""
    S.GEAR_ENABLED = dynamic_white
    G.FAST_T0 = t0
    e = Engine()
    b = empty_board()
    for r, c, p in opening:
        place(b, r, c, p)
        e.on_move(b, r, c, p)
    turn = BLACK
    gear_log = []
    for step in range(max_moves):
        res = e.analyze_turn(b, turn)
        move = res['part4_result']['move']
        if move is None:
            break
        r, c = move
        if not (0 <= r < 15 and 0 <= c < 15) or b[r][c] != EMPTY:
            break
        if turn == WHITE and dynamic_white:
            gear_log.append(e.search.gear[WHITE])
        place(b, r, c, turn)
        e.on_move(b, r, c, turn)
        if G.DeepSearch._check_win(b, r, c, turn):
            return ('B' if turn == BLACK else 'W', step + 1, gear_log)
        turn = 3 - turn
    return (None, max_moves, gear_log)


def summarize(name, results):
    n = len(results)
    wins = sum(1 for w, _, _ in results if w == 'W')
    losses = sum(1 for w, _, _ in results if w == 'B')
    draws = n - wins - losses
    avg_moves = sum(m for _, m, _ in results) / max(n, 1)
    all_gears = []
    switches = 0
    for _, _, gl in results:
        all_gears.extend(gl)
        switches += sum(1 for i in range(1, len(gl)) if gl[i] != gl[i - 1])
    print('%s: W=%d B=%d draw=%d | avg_moves=%.1f | gear_hist=%s | switches=%d' % (
        name, wins, losses, draws, avg_moves,
        {g: all_gears.count(g) for g in range(1, 5)},
        switches))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=6)
    ap.add_argument('--max-moves', type=int, default=24)
    ap.add_argument('--t0', type=float, default=40.0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # 每个配置跑相同数量的开局，且保证黑白/开局多样性
    fixed_tasks = []
    dyn_tasks = []
    for i in range(args.games):
        op = OPENINGS[i % len(OPENINGS)]
        fixed_tasks.append((op, args.max_moves, False, args.t0))
        dyn_tasks.append((op, args.max_moves, True, args.t0))

    print('running %d games per config (max_moves=%d, t0=%g, workers=%d)...' % (
        args.games, args.max_moves, args.t0, args.workers), flush=True)

    t0 = time.time()
    with Pool(args.workers) as pool:
        fixed_results = pool.starmap(play_one, fixed_tasks)
    print('fixed done in %.1fs' % (time.time() - t0), flush=True)
    with Pool(args.workers) as pool:
        dyn_results = pool.starmap(play_one, dyn_tasks)
    print('dynamic done in %.1fs' % (time.time() - t0), flush=True)

    print()
    summarize('fixed 3攻2防  ', fixed_results)
    summarize('dynamic 配比   ', dyn_results)

    # 动态局档位切换详情（最多打印前 3 局）
    print()
    print('dynamic gear logs (first 3):')
    for i, (_, _, gl) in enumerate(dyn_results[:3]):
        print('  game%d: %s' % (i, gl))


if __name__ == '__main__':
    main()
