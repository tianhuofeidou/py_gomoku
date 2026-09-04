# -*- coding: utf-8 -*-
"""GA 单局耗时基准：8 进程跑 N 局固定基线自对弈，统计平均耗时。"""
import os
import time
from multiprocessing import Pool

from gomoku.tools import ga_tune as G


def one_game(task):
    opening, max_moves, t0 = task
    G.FAST_T0 = t0
    return G.play_game(G.BASELINE, G.BASELINE, opening, max_moves, use_fast_t0=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=8)
    ap.add_argument('--max-moves', type=int, default=20)
    ap.add_argument('--t0', type=float, default=40.0)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    tasks = [(G.OPENINGS[i % len(G.OPENINGS)], args.max_moves, args.t0)
             for i in range(args.games)]
    print('bench: games=%d max_moves=%d t0=%g workers=%d' % (
        args.games, args.max_moves, args.t0, args.workers), flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(one_game, tasks)
    elapsed = time.time() - t0
    print('total=%.1fs avg_per_game_wall=%.1fs results=%s' % (
        elapsed, elapsed / max(args.games, 1),
        {k: results.count(k) for k in ('B', 'W', None)}), flush=True)


if __name__ == '__main__':
    main()
