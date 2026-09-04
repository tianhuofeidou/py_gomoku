# -*- coding: utf-8 -*-
"""decision_data.py —— 决策网络训练样本生成（两路数据源）。

样本 = 每步决策：
  输入：xb [8,15,15] 棋盘分层 + xc [16,8] 候选清单（坐标/深推分/排名/人机胜/负/自我胜/负）+ mask
  标签：deep 深推分（阶段1 监督模仿目标）、win 该步方最终胜负（阶段2 记忆修正目标）、
        actual_idx 实际着法在候选中的索引

来源：
  - self：引擎自对弈（每局结束写 self-memory，数据随生成增长）
  - human：重放 global-memory.json 的 badLines/goodLines 人机对局（AI 视角）

输出：npz 数组（xb/xc/mask/deep/win/actual_idx/src）
"""
import os
import random

import numpy as np

from gomoku.core.utils import empty_board, place, BLACK, WHITE
from gomoku.core.engine import Engine
from gomoku.core.deep_search import DeepSearch
from gomoku.nn.decision_input import encode_decision, MAX_CANDS, FEAT_BOARD_CH, FEAT_CAND
from gomoku.nn.memory_lookup import MemoryLookup, default_memory_file
from gomoku.nn.self_memory import SelfMemory

# 多种固定开局，避免单一开局偏差（与 data.py 一致）
OPENINGS = [
    [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (5, 6, WHITE)],
    [(7, 7, BLACK), (7, 8, WHITE), (6, 6, BLACK), (8, 6, WHITE)],
    [(7, 7, BLACK), (8, 8, WHITE), (6, 6, BLACK), (8, 6, WHITE)],
    [(7, 7, BLACK), (6, 7, WHITE), (5, 5, BLACK), (5, 6, WHITE)],
    [(7, 7, BLACK), (8, 7, WHITE), (6, 6, BLACK), (6, 8, WHITE)],
]

SRC_SELF = 0
SRC_HUMAN = 1


def _encode_step(e, b, turn, moves, ranked):
    """编码一步样本（记忆特征由 MemoryLookup 现查）。"""
    wins, losses, sw, sl = moves[0].batch_stats(moves[1], turn, ranked)
    xb, xc, mask = encode_decision(b, e.search, ranked, wins, losses, sw, sl)
    deep = np.zeros(MAX_CANDS, dtype=np.float32)
    for i, c in enumerate(ranked[:MAX_CANDS]):
        deep[i] = float(c['score'])
    return xb, xc, mask, deep


def play_decision_game(mem, max_moves=30, t0=40.0, opening=None, record_self=True):
    """自对弈一局 → (samples, winner)。
    samples: [{'xb','xc','mask','deep','actual_idx','win','player'}, ...]"""
    DeepSearch.T0 = t0
    e = Engine()
    b = empty_board()
    samples = []
    moves = []
    winner = None

    opening = opening or OPENINGS[0]
    for r, c, p in opening:
        moves.append((r, c, p))
        place(b, r, c, p)
        e.on_move(b, r, c, p)
        if DeepSearch._check_win(b, r, c, p):
            winner = p
            break
    if winner is not None:
        if record_self:
            SelfMemory().record_game(moves, winner)
        return [], winner

    turn = BLACK
    for _ in range(max_moves):
        res = e.analyze_turn(b, turn)
        r4 = res['part4_result']
        ranked = r4['ranked']
        mv = r4['move']
        if not ranked or mv is None:
            break
        actual_idx = 0
        for i, c in enumerate(ranked):
            if c['r'] == mv[0] and c['c'] == mv[1]:
                actual_idx = i
                break
        xb, xc, mask, deep = _encode_step(e, b, turn, (mem, moves), ranked)
        samples.append({'xb': xb, 'xc': xc, 'mask': mask, 'deep': deep,
                        'actual_idx': actual_idx, 'player': turn, 'mv': mv})
        moves.append((mv[0], mv[1], turn))
        place(b, mv[0], mv[1], turn)
        e.on_move(b, mv[0], mv[1], turn)
        if DeepSearch._check_win(b, mv[0], mv[1], turn):
            winner = turn
            break
        turn = 3 - turn
    # 胜负标签回填
    for s in samples:
        if winner is None:
            s['win'] = 0.0
        else:
            s['win'] = 1.0 if s['player'] == winner else -1.0
    if record_self and winner is not None:
        SelfMemory().record_game(moves, winner)
    return samples, winner


def generate_self_samples(num_games, t0=40.0, max_moves=30, seed=1):
    """批量自对弈 → 样本列表（src=SELF）。"""
    rng = random.Random(seed)
    mem = MemoryLookup()
    all_s = []
    wins = 0
    for gi in range(num_games):
        opening = rng.choice(OPENINGS)
        samples, winner = play_decision_game(mem, max_moves=max_moves, t0=t0, opening=opening)
        if winner is not None:
            wins += 1
        for s in samples:
            s['src'] = SRC_SELF
        all_s.extend(samples)
        if (gi + 1) % 50 == 0:
            print('  self game %d/%d  winner=%s  samples=%d' % (gi + 1, num_games, winner, len(all_s)), flush=True)
    print('self: %d 局完成（%d 局分出胜负）→ %d 样本' % (num_games, wins, len(all_s)))
    return all_s


def _interleave(opp, ai):
    """badLines/goodLines 的 opp/ai 着法 → 交替恢复全局着法序列 [(r,c,player),...]。
    opp=人类、ai=AI；返回 [(r, c, player)] 全局顺序。"""
    out = []
    n = max(len(opp), len(ai))
    for i in range(n):
        if i < len(opp):
            out.append((opp[i]['r'], opp[i]['c'], 1))   # 人类执黑（recordResult 约定）
        if i < len(ai):
            out.append((ai[i]['r'], ai[i]['c'], 2))     # AI 执白
    return out


def generate_human_samples(max_moves=40):
    """重放 global-memory.json 人机对局 → 样本列表（src=HUMAN）。
    badLines（AI 输）→ win=-1；goodLines（AI 赢）→ win=+1。"""
    mem = MemoryLookup()
    lines = []
    for ln in mem.bad_lines:
        lines.append((ln, -1.0))
    for ln in mem.good_lines:
        lines.append((ln, 1.0))
    all_s = []
    for ln, win in lines:
        opp = ln.get('opp') or []
        ai = ln.get('ai') or []
        if not opp or not ai:
            continue
        moves = _interleave(opp, ai)
        # 逐手重放：在 AI（白）落子前分析当前局面 → 该步决策样本
        e2 = Engine()
        b2 = empty_board()
        cur = []
        for i, (r, c, p) in enumerate(moves):
            if p == 2 and i >= 2:
                # AI 决策前分析
                res = e2.analyze_turn(b2, 2)
                r4 = res['part4_result']
                ranked = r4['ranked']
                if ranked:
                    actual_idx = 0
                    for j, cd in enumerate(ranked):
                        if cd['r'] == r and cd['c'] == c:
                            actual_idx = j
                            break
                    xb, xc, mask, deep = _encode_step(e2, b2, 2, (mem, cur), ranked)
                    all_s.append({'xb': xb, 'xc': xc, 'mask': mask, 'deep': deep,
                                  'actual_idx': actual_idx, 'win': win, 'player': 2,
                                  'mv': (r, c), 'src': SRC_HUMAN})
                    if len(all_s) >= max_moves:
                        break
            cur.append((r, c, p))
            place(b2, r, c, p)
            e2.on_move(b2, r, c, p)
        if len(all_s) > 4000:
            break
    print('human: %d 局重放 → %d 样本' % (len(lines), len(all_s)))
    return all_s


def save_samples(samples, out_path='gomoku/data/decision_samples.npz'):
    """样本列表 → npz 文件（统一数组）。"""
    n = len(samples)
    xb = np.stack([s['xb'][0] for s in samples]).astype(np.float32)
    xc = np.stack([s['xc'][0] for s in samples]).astype(np.float32)
    mask = np.stack([s['mask'][0] for s in samples]).astype(np.float32)
    deep = np.stack([s['deep'] for s in samples]).astype(np.float32)
    win = np.array([s['win'] for s in samples], dtype=np.float32)
    actual = np.array([s['actual_idx'] for s in samples], dtype=np.int64)
    src = np.array([s.get('src', SRC_SELF) for s in samples], dtype=np.int64)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, xb=xb, xc=xc, mask=mask, deep=deep, win=win, actual=actual, src=src)
    print('saved %d samples → %s' % (n, out_path))
    return out_path


def load_samples(path='gomoku/data/decision_samples.npz'):
    d = np.load(path)
    return {k: d[k] for k in ('xb', 'xc', 'mask', 'deep', 'win', 'actual', 'src')}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=50)
    ap.add_argument('--max-moves', type=int, default=30)
    ap.add_argument('--t0', type=float, default=40.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--human', action='store_true', help='同时重放人机对局样本')
    ap.add_argument('--out', default='gomoku/data/decision_samples.npz')
    args = ap.parse_args()
    samples = generate_self_samples(args.games, t0=args.t0, max_moves=args.max_moves, seed=args.seed)
    if args.human:
        samples += generate_human_samples()
    save_samples(samples, args.out)
