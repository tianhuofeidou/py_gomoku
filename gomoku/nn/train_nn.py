# -*- coding: utf-8 -*-
"""训练分支价值网络（PyTorch，建议 Python 3.14 运行）。

用法：
  py -3.14 -m gomoku.nn.train_nn --data gomoku/data/nn_samples.json --out gomoku/nn/value.pt
  py -3.14 -m gomoku.nn.train_nn --data gomoku/data/nn_samples_rl.json --init gomoku/nn/value.pt --out gomoku/nn/value_rl_ft.pt
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

from gomoku.nn.features import branch_matrix, DEFAULT_MAX_STEPS
from gomoku.nn.model import BranchValueNet


def load_samples(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def flip_camp(ctx_list):
    """颜色对称增强：交换视角方颜色（camp 0<->1），label 不变。
    其他特征（候选权重/攻防比）与具体颜色无关，保持原值。"""
    return [{**ctx, 'camp': 1 - int(ctx.get('camp', 0))} for ctx in ctx_list]


def encode_batch(samples, max_steps=DEFAULT_MAX_STEPS, device='cpu'):
    """把样本列表编码成 batch 张量。"""
    xs = []
    ls = []
    ys = []
    for ctx_list, label in samples:
        m, length = branch_matrix(ctx_list, max_steps=max_steps)
        xs.append(m)
        ls.append(length)
        ys.append(float(label))
    x = torch.tensor(np.array(xs), dtype=torch.float32, device=device)
    lengths = torch.tensor(np.array(ls), dtype=torch.long, device=device)
    y = torch.tensor(np.array(ys), dtype=torch.float32, device=device).unsqueeze(1)
    return x, lengths, y


def make_batches(samples, batch_size, max_steps, device):
    for i in range(0, len(samples), batch_size):
        yield encode_batch(samples[i:i + batch_size], max_steps, device)


def train(data_path, out_path, epochs=50, batch_size=128, lr=1e-3,
          max_steps=DEFAULT_MAX_STEPS, seed=1, val_ratio=0.1, patience=10,
          init_path=None):
    random.seed(seed)
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    samples = load_samples(data_path)
    print('samples:', len(samples))
    if not samples:
        raise SystemExit('no samples')
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_ratio))
    val_data = samples[:n_val]
    train_data = samples[n_val:]
    print('train:', len(train_data), 'val:', len(val_data))

    model = BranchValueNet().to(device)
    if init_path:
        model.load_state_dict(torch.load(init_path, map_location=device, weights_only=True))
        print('init from:', init_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for x, lengths, y in make_batches(train_data, batch_size, max_steps, device):
            pred = model(x, lengths)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
        train_loss = total / max(n, 1)

        model.eval()
        vtotal = 0.0
        vn = 0
        with torch.no_grad():
            for x, lengths, y in make_batches(val_data, batch_size, max_steps, device):
                pred = model(x, lengths)
                vtotal += loss_fn(pred, y).item() * len(y)
                vn += len(y)
        val_loss = vtotal / max(vn, 1)
        scheduler.step(val_loss)

        print('[epoch %d] train=%.4f val=%.4f lr=%g' % (
            epoch, train_loss, val_loss, optimizer.param_groups[0]['lr']), flush=True)

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print('early stop at epoch %d' % epoch, flush=True)
                break

    if best_state is None:
        best_state = model.state_dict()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(best_state, out_path)
    print('best val loss=%.4f saved -> %s' % (best_loss, out_path))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='gomoku/data/nn_samples.json')
    ap.add_argument('--out', default='gomoku/nn/value.pt')
    ap.add_argument('--init', default=None)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--patience', type=int, default=10)
    args = ap.parse_args()
    train(args.data, args.out, args.epochs, args.batch_size, args.lr,
          args.max_steps, args.seed, args.val_ratio, args.patience, args.init)
