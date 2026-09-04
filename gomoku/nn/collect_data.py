# -*- coding: utf-8 -*-
"""从深推搜索树收集训练样本（与实战评估分布一致）。

每个样本 = 深推某个叶子分支的 ctx 序列 + 标签：
  win=1, lose=-1, stale=0

--min-step 可只保留对局后半段（离终局更近）的样本，降低整局胜负标签的噪声。
"""
import argparse
import json
import os
import random

from gomoku.core.utils import empty_board, place, BLACK, WHITE
from gomoku.core.engine import Engine
from gomoku.core.deep_search import DeepSearch

from gomoku.nn.data import OPENINGS


def play_one_game(opening, max_moves=40, t0=30.0, model_path=None):
    """跑一局，收集每次深推产生的叶子分支原始样本 (ctx, player, status, step)。
    返回 (raw_samples, winner)。step 是该手在对局中的序号（从 0 开始）。"""
    DeepSearch.T0 = t0
    e = Engine()
    e.deep_search.collect_samples = True
    if model_path:
        e.deep_search.model_path = model_path
    b = empty_board()
    raw_samples = []
    winner = None

    for r, c, p in opening:
        place(b, r, c, p)
        e.on_move(b, r, c, p)
        if DeepSearch._check_win(b, r, c, p):
            return raw_samples, p

    turn = BLACK
    step = 0
    for _ in range(max_moves):
        e.deep_search.samples = []
        res = e.analyze_turn(b, turn)
        raw_samples.extend((ctx, p, status, step)
                           for (ctx, p, status) in e.deep_search.samples)
        mv = res['part4_result']['move']
        if isinstance(mv, list):
            mv = mv[0] if mv else None
        if mv is None:
            break
        r, c = mv
        if b[r][c] != 0:
            break
        place(b, r, c, turn)
        e.on_move(b, r, c, turn)
        if DeepSearch._check_win(b, r, c, turn):
            winner = turn
            break
        turn = 3 - turn
        step += 1
    return raw_samples, winner


def generate_dataset(num_games=100, max_moves=40, t0=30.0, seed=1,
                     out_path='gomoku/data/nn_samples_deep.json', model_path=None,
                     min_step=0):
    rng = random.Random(seed)
    all_samples = []
    kept = 0
    for gi in range(num_games):
        opening = rng.choice(OPENINGS)
        raw_samples, winner = play_one_game(opening, max_moves=max_moves, t0=t0, model_path=model_path)
        # 用真实胜负回传标签：win=1, lose=-1, stale 按视角方最终胜负
        for ctx, p, status, step in raw_samples:
            if step < min_step:
                continue
            if status == 'win':
                label = 1.0
            elif status == 'lose':
                label = -1.0
            else:
                if winner is None:
                    label = 0.0
                else:
                    label = 1.0 if p == winner else -1.0
            all_samples.append((ctx, label))
            kept += 1
        print('game %d: collected=%d total=%d kept=%d' % (
            gi, len(raw_samples), len(all_samples), kept), flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False)
    print('saved %d samples -> %s' % (len(all_samples), out_path))
    return all_samples


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=100)
    ap.add_argument('--max-moves', type=int, default=40)
    ap.add_argument('--t0', type=float, default=30.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='gomoku/data/nn_samples_deep.json')
    ap.add_argument('--model', default=None)
    ap.add_argument('--min-step', type=int, default=0)
    args = ap.parse_args()
    generate_dataset(args.games, args.max_moves, args.t0, args.seed, args.out,
                     args.model, args.min_step)
