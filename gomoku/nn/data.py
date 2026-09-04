# -*- coding: utf-8 -*-
"""训练数据生成（简化版）：
用现有引擎自对弈，把每一步当作一个样本：
  - 输入：从开局到该步的 ctx 序列（分支矩阵）
  - 标签：该步落子方最终胜负（赢=1，输=-1，平=0）

ctx 每步字段：
  camp    : 该步是否为本样本视角方（=该步落子方）
  move_w  : 该步落子候选点权重
  cand_ws : 当前局面全候选点权重
  kill    : 是否触发必杀
  gear    : 攻防比 0~4
"""
import json
import os
import random

from gomoku.core.utils import empty_board, place, BLACK, WHITE
from gomoku.core.engine import Engine
from gomoku.core.deep_search import DeepSearch
from gomoku.nn.self_memory import SelfMemory


def _step_ctx(engine, board, player, move_w):
    """构造一步上下文。"""
    cand_pts = engine.deep_search._candidate_points(board, player, 5)
    cand_ws = [x[2] for x in cand_pts]
    kill = any(w >= 100 for w in cand_ws)
    gear = 0 if kill else engine.search._target_gear(board, player)
    return {
        'player': player,
        'camp': 1,   # 占位，构造样本时按样本视角方重算
        'move_w': float(move_w),
        'cand_ws': cand_ws,
        'kill': 1 if kill else 0,
        'gear': gear,
    }


# 多种固定开局，避免单一开局偏差
OPENINGS = [
    [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (5, 6, WHITE)],
    [(7, 7, BLACK), (7, 8, WHITE), (6, 6, BLACK), (8, 6, WHITE)],
    [(7, 7, BLACK), (8, 8, WHITE), (6, 6, BLACK), (8, 6, WHITE)],
    [(7, 7, BLACK), (6, 7, WHITE), (5, 5, BLACK), (5, 6, WHITE)],
    [(7, 7, BLACK), (8, 7, WHITE), (6, 6, BLACK), (6, 8, WHITE)],
]


def play_one_game(max_moves=30, t0=40.0, opening=None, record_self=False):
    """跑一局自对弈，返回 (samples, winner)。
    samples: [(ctx_list, label), ...]，label 是“该步落子方”最终胜负。
    使用固定开局，避免空盘无候选。
    record_self=True 时对局结束写入自我对弈记忆（self-memory.json，胜/负方视角各一条）。"""
    DeepSearch.T0 = t0
    e = Engine()
    b = empty_board()
    steps = []   # 每步 (player, ctx)
    moves = []   # 完整着法 [(r, c, player)]（记忆记账用）
    winner = None

    opening = opening or OPENINGS[0]
    for r, c, p in opening:
        ctx = _step_ctx(e, b, p, 60.0)
        steps.append((p, ctx))
        moves.append((r, c, p))
        place(b, r, c, p)
        e.on_move(b, r, c, p)
        if DeepSearch._check_win(b, r, c, p):
            winner = p
            if record_self:
                SelfMemory().record_game(moves, winner)
            return [], winner
    turn = BLACK
    for _ in range(max_moves):
        cand_pts = e.deep_search._candidate_points(b, turn, 5)
        # 用 analyze_turn 决策
        res = e.analyze_turn(b, turn)
        mv = res['part4_result']['move']
        if isinstance(mv, list):
            mv = mv[0] if mv else None
        if mv is None:
            break
        r, c = mv
        if b[r][c] != 0:
            break
        move_w = 60.0
        for (pr, pc, pw) in cand_pts:
            if pr == r and pc == c:
                move_w = pw
                break
        ctx = _step_ctx(e, b, turn, move_w)
        steps.append((turn, ctx))
        moves.append((r, c, turn))
        place(b, r, c, turn)
        e.on_move(b, r, c, turn)
        if DeepSearch._check_win(b, r, c, turn):
            winner = turn
            break
        turn = 3 - turn
    if record_self and winner is not None:
        SelfMemory().record_game(moves, winner)
    # 构造样本：每一步前缀为输入，标签为该步方最终胜负
    # camp 必须相对于样本视角方（该步落子方）重算
    samples = []
    for i, (p, ctx) in enumerate(steps):
        prefix = []
        for (_p, c) in steps[:i + 1]:
            cc = dict(c)
            cc['camp'] = 1 if _p == p else 0
            prefix.append(cc)
        if winner is None:
            label = 0.0
        else:
            label = 1.0 if p == winner else -1.0
        samples.append((prefix, label))
    return samples, winner


def generate_dataset(num_games=50, max_moves=30, t0=40.0, seed=1, out_path='gomoku/data/nn_samples.json'):
    """生成并保存训练样本。"""
    rng = random.Random(seed)
    all_samples = []
    for gi in range(num_games):
        opening = rng.choice(OPENINGS)
        samples, winner = play_one_game(max_moves=max_moves, t0=t0, opening=opening)
        all_samples.extend(samples)
        print('game %d: winner=%s samples=%d' % (gi, winner, len(samples)), flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False)
    print('saved %d samples -> %s' % (len(all_samples), out_path))
    return all_samples


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=50)
    ap.add_argument('--max-moves', type=int, default=30)
    ap.add_argument('--t0', type=float, default=40.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='gomoku/data/nn_samples.json')
    args = ap.parse_args()
    generate_dataset(args.games, args.max_moves, args.t0, args.seed, args.out)
