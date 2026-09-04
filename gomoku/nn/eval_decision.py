# -*- coding: utf-8 -*-
"""eval_decision.py —— 决策网络 vs 纯算法对弈评估（可独立运行）。

用法：py -3.14 -m gomoku.nn.eval_decision --games 30 [--model gomoku/nn/decision.pt] [--t0 40]
输出：网络胜率（黑白各半）。
"""
import argparse

import torch

from gomoku.nn.decision_net import DecisionNet
from gomoku.nn.train_decision import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=30)
    ap.add_argument('--model', default='gomoku/nn/decision.pt')
    ap.add_argument('--t0', type=float, default=40.0)
    ap.add_argument('--max-moves', type=int, default=60)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = DecisionNet().to(device)
    net.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    net.eval()
    print('model:', args.model, 'device:', device)
    wr = evaluate(net, games=args.games, t0=args.t0, max_moves=args.max_moves, device=device)
    print('评估：网络 vs 纯算法 %d 局 → 网络胜率 %.1f%%' % (args.games, wr * 100))


if __name__ == '__main__':
    main()
