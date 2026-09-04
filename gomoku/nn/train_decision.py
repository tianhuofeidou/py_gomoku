# -*- coding: utf-8 -*-
"""train_decision.py —— 决策网络训练（阶段1+阶段2 混合）与评估。

阶段1（监督模仿，让网络先跟深推一样好）：
  网络对候选打分 → softmax 与「深推分 softmax(deep/τ)」交叉熵。学的是"深推觉得哪个好"。

阶段2（记忆修正，超越深推）：
  每局最终胜负当老师：胜局 → 实际着法的候选 logit 必须显著高于其他候选（margin）；
  负局 → 实际着法（走输的那手）logit 压低（换路）。网络从实战中学会
  "90 分的候选历史胜率低 → 改选 85 分那个"。

防过拟合：SamplePool 按人类胜率 R 调节人机样本重复上限（≤20% 占比），
自我对弈样本无限补量。

评估：网络引擎 vs 纯算法引擎对弈（黑白各半），统计胜率。
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from gomoku.core.utils import empty_board, place, BLACK, WHITE, SIZE
from gomoku.core.engine import Engine
from gomoku.core.deep_search import DeepSearch
from gomoku.nn.decision_net import DecisionNet
from gomoku.nn.decision_input import MAX_CANDS, FEAT_BOARD_CH, FEAT_CAND
from gomoku.nn.sample_pool import SamplePool
from gomoku.nn import decision_data as dd


def human_winrate_from_memory(mem):
    """人类胜率 R = AI 输的局数 / 有胜负的局数（从人机记忆 totals 读）。"""
    try:
        import json
        from gomoku.nn.memory_lookup import default_memory_file
        with open(default_memory_file(), encoding='utf-8') as f:
            data = json.load(f)
        t = data.get('totals') or {}
        losses = t.get('losses') or 0
        wins = t.get('wins') or 0
        total = losses + wins
        if total == 0:
            return 0.5
        return losses / total
    except Exception:
        return 0.5


def softmax_target(deep, temp=0.3):
    """深推分 → softmax 概率目标（处理 1e9 超大分：唯一候选 one-hot）。
    deep: [B, N]；返回 [B, N]（mask 外为 0，由 loss 忽略）。"""
    x = deep.clone()
    huge = x > 1e8
    x = x / max(temp, 1e-6)
    x = x - x.max(dim=1, keepdim=True).values
    exp = torch.exp(x)
    exp[huge] = 0.0
    # 超大分候选：独占概率
    if huge.any():
        exp[huge] = 1.0
        exp[~huge] = 0.0
    denom = exp.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return exp / denom


def stage1_loss(logits, mask, deep, temp):
    """阶段1：网络 softmax 与深推 softmax 的交叉熵（mask 内）。"""
    target = softmax_target(deep, temp)
    logp = torch.log_softmax(logits, dim=1)
    loss = -(target * logp).sum(dim=1)
    valid = mask.sum(dim=1).clamp(min=1)
    loss = loss / valid
    return loss.mean()


def stage2_loss(logits, mask, win, actual_idx, margin=1.0, lam=0.5):
    """阶段2：胜负 margin 损失。
    胜局（win=1）：实际着法候选 logit ≥ 其他候选最高 logit + margin → 0，否则惩罚。
    负局（win=-1）：实际着法候选 logit ≤ 其他候选最高 logit - margin → 0，否则惩罚。
    无胜负（win=0）或候选数 <2（没有"其他候选"可比）→ 不参与。"""
    B, N = logits.shape
    cand_count = mask.sum(dim=1)
    active = (win.abs() > 0) & (cand_count >= 2)
    if not active.any():
        return torch.zeros((), device=logits.device)
    idx = actual_idx.clamp(0, N - 1)
    actual = logits[torch.arange(B), idx]                      # [B]
    others = logits.masked_fill(mask < 0.5, -1e9)              # padding 行排除
    others = others.scatter(1, idx.unsqueeze(1), -1e9)         # 排除自身
    best_other = others.max(dim=1).values                      # [B]
    margin_loss = torch.zeros_like(actual)
    win_mask = active & (win > 0)
    lose_mask = active & (win < 0)
    margin_loss[win_mask] = torch.clamp(margin - (actual[win_mask] - best_other[win_mask]), min=0)
    margin_loss[lose_mask] = torch.clamp(margin - (best_other[lose_mask] - actual[lose_mask]), min=0)
    return lam * margin_loss.mean()


def build_tensors(batch, device='cpu'):
    xb = torch.from_numpy(np.stack([s['xb'][0] for s in batch])).to(device)
    xc = torch.from_numpy(np.stack([s['xc'][0] for s in batch])).to(device)
    mask = torch.from_numpy(np.stack([s['mask'][0] for s in batch])).to(device)
    deep = torch.from_numpy(np.stack([s['deep'] for s in batch])).to(device)
    win = torch.tensor([s['win'] for s in batch], dtype=torch.float32, device=device)
    actual = torch.tensor([s['actual_idx'] for s in batch], dtype=torch.long, device=device)
    return xb, xc, mask, deep, win, actual


def train(args):
    rng = random.Random(args.seed)
    t0 = time.time()
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA 不可用，回退 CPU')
        device = 'cpu'
    print('device:', device)
    # 1. 样本：自对弈 + 人机重放
    print('[1/5] 生成样本...')
    samples = dd.generate_self_samples(args.games, t0=args.t0, max_moves=args.max_moves, seed=args.seed)
    if args.human:
        samples += dd.generate_human_samples()
    if not samples:
        print('无样本，退出'); return
    # 2. 采样池（防过拟合）
    print('[2/5] 采样池：人类胜率 R 动态调节...')
    r = human_winrate_from_memory(None)
    pool = SamplePool(human_winrate=r, rng=rng)
    for s in samples:
        pool.add(s, 'human' if s.get('src') == dd.SRC_HUMAN else 'self')
    print('  池统计:', pool.stats(), 'R=%.2f' % r)
    # 3. 网络
    print('[3/5] 网络初始化...')
    net = DecisionNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # 4. 训练
    print('[4/5] 训练 %d epoch（阶段1 CE + 阶段2 margin λ=%.1f）...' % (args.epochs, args.lam))
    net.train()
    for ep in range(args.epochs):
        total1 = 0.0
        total2 = 0.0
        nbatch = 0
        for _ in range(args.steps):
            batch = pool.sample_batch(args.batch)
            if len(batch) < 2:
                break
            xb, xc, mask, deep, win, actual = build_tensors(batch, device)
            logits = net(xb, xc, mask)
            l1 = stage1_loss(logits, mask, deep, args.temp)
            l2 = stage2_loss(logits, mask, win, actual, margin=args.margin, lam=args.lam)
            loss = l1 + l2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total1 += l1.item()
            total2 += l2.item() if isinstance(l2, torch.Tensor) and l2.requires_grad else 0.0
            nbatch += 1
        if nbatch:
            print('  epoch %2d/%d  loss1=%.6f  loss2=%.6f  (%.1fs)' % (
                ep + 1, args.epochs, total1 / nbatch, total2 / nbatch, time.time() - t0), flush=True)
    # 5. 保存（评估可选——单独 eval_decision.py 跑，避免训练任务过长）
    print('[5/5] 保存 decision.pt ...')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print('已保存 →', args.out)
    if args.eval_games and not args.no_eval:
        net.eval()
        wr = evaluate(net, games=args.eval_games, t0=args.t0, device=device)
        print('评估：网络 vs 纯算法 %d 局 → 网络胜率 %.1f%%' % (args.eval_games, wr * 100))


def evaluate(net, games=20, t0=40.0, max_moves=60, device='cpu'):
    """网络引擎 vs 纯算法引擎对弈（网络黑白各半），返回网络胜率。"""
    DeepSearch.T0 = t0
    net.eval()
    net_wins = 0
    finished = 0
    for gi in range(games):
        net_is_black = (gi % 2 == 0)
        e_net = Engine(decision_net=net, use_decision_net=True)
        e_alg = Engine()
        b = empty_board()
        turn = BLACK
        winner = None
        for _ in range(max_moves):
            e = e_net if (turn == BLACK) == net_is_black else e_alg
            res = e.analyze_turn(b, turn)
            mv = res['part4_result']['move']
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
        if winner is not None:
            finished += 1
            net_win = (winner == BLACK) == net_is_black
            if net_win:
                net_wins += 1
    if finished == 0:
        return 0.5
    return net_wins / finished


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--max-moves', type=int, default=30)
    ap.add_argument('--t0', type=float, default=40.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--human', action='store_true', help='混入人机对弈样本')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--temp', type=float, default=0.3)
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--margin', type=float, default=1.0)
    ap.add_argument('--eval-games', type=int, default=20)
    ap.add_argument('--no-eval', action='store_true', help='训练完跳过评估')
    ap.add_argument('--device', default='cuda', help='cuda / cpu')
    ap.add_argument('--out', default='gomoku/nn/decision.pt')
    args = ap.parse_args()
    train(args)
