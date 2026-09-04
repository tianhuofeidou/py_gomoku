# -*- coding: utf-8 -*-
"""KataGo 蒸馏数据训练：全盘落点网络（policy + value 双头，监督学习）。

数据：katago_data/fs15x_proc/dataN.npz（katago_convert.py 产物）
  x [N,2,15,15] uint8   己方/对方棋子
  p [N,225] float32     soft policy 标签（行和=1，KataGo/MCTS 落子概率）
  v [N,2] float32       胜率标签（己方胜概率, 对方胜概率）

模型：小 CNN（<1M 参数，符合数据集 README 建议的 <5M 参数），双头：
  policy 头：225 格 softmax（交叉熵，soft label）
  value 头 ：2 类 softmax（胜/负，soft label 交叉熵）

多进程：multiprocessing.Pool(--workers) 并行加载所选 npz 文件（默认 8）。

用法：
  py -3.14 -m gomoku.tools.train_katago --data katago_data/fs15x_proc \
      --out gomoku/nn/katago_policy.pt --epochs 5 --files-per-epoch 10 \
      --workers 8 --val-files 2
  冒烟测试：--files-per-epoch 1 --limit 5000 --epochs 1
"""
import argparse
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H = W = 15
NUM_CELLS = H * W


def load_npz(path):
    """加载一个转换后的 npz → (x, p, v)。"""
    d = np.load(path)
    return d['x'], d['p'], d['v']


class KatagoNet(nn.Module):
    """全盘落点网络：输入 [B,2,15,15] → policy [B,225] + value [B,2]。"""

    def __init__(self, ch=(32, 64, 128)):
        super().__init__()
        c0, c1, c2 = ch
        self.body = nn.Sequential(
            nn.Conv2d(2, c0, 3, padding=1), nn.BatchNorm2d(c0), nn.ReLU(),
            nn.Conv2d(c0, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(c2, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(c2, 16, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.value_fc = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        h = self.body(x)                                     # [B,C,15,15]
        p = self.policy_head(h).flatten(1)                   # [B,225]
        v = self.value_fc(self.value_head(h).flatten(1))     # [B,2]
        return p, v

    def param_count(self):
        return sum(w.numel() for w in self.parameters())


def train_epoch(model, opt, files, device, batch_size, limit=0, pool=None):
    """一个 epoch：并行加载文件 → 拼接 → 随机 batch 训练。
    返回 (平均 loss, policy top1 命中率, 样本数)。"""
    model.train()
    total_loss = 0.0
    total_top1 = 0
    total_n = 0
    for path in files:
        if pool is not None:
            x, p, v = pool.apply(load_npz, (path,))
        else:
            x, p, v = load_npz(path)
        if limit > 0:
            x, p, v = x[:limit], p[:limit], v[:limit]
        n = x.shape[0]
        perm = np.random.permutation(n)
        x, p, v = x[perm], p[perm], v[perm]
        x = torch.from_numpy(x).float().to(device)
        p = torch.from_numpy(p).float().to(device)
        v = torch.from_numpy(v).float().to(device)
        for i in range(0, n, batch_size):
            xb, pb, vb = x[i:i + batch_size], p[i:i + batch_size], v[i:i + batch_size]
            opt.zero_grad()
            p_out, v_out = model(xb)
            loss_p = -torch.sum(pb * F.log_softmax(p_out, dim=1), dim=1).mean()
            loss_v = -torch.sum(vb * F.log_softmax(v_out, dim=1), dim=1).mean()
            loss = loss_p + loss_v
            loss.backward()
            opt.step()
            # policy top1 命中：预测 argmax == 标签 argmax
            with torch.no_grad():
                hit = (p_out.argmax(1) == pb.argmax(1)).sum().item()
                total_top1 += hit
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)
        del x, p, v
    return total_loss / max(total_n, 1), total_top1 / max(total_n, 1), total_n


@torch.no_grad()
def evaluate(model, files, device, pool=None, limit=5000):
    """验证：policy top1 命中率 + value 预测与标签的相关性。"""
    model.eval()
    hits = 0
    n = 0
    v_err = 0.0
    for path in files:
        if pool is not None:
            x, p, v = pool.apply(load_npz, (path,))
        else:
            x, p, v = load_npz(path)
        x, p, v = x[:limit], p[:limit], v[:limit]
        x = torch.from_numpy(x).float().to(device)
        p = torch.from_numpy(p).float().to(device)
        v = torch.from_numpy(v).float().to(device)
        p_out, v_out = model(x)
        hits += (p_out.argmax(1) == p.argmax(1)).sum().item()
        # value：预测己方胜率（softmax 后第一维）与标签第一维的 MAE
        vp = F.softmax(v_out, dim=1)[:, 0]
        v_err += (vp - v[:, 0]).abs().sum().item()
        n += x.size(0)
    return hits / max(n, 1), v_err / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='katago_data/fs15x_proc')
    ap.add_argument('--out', default='gomoku/nn/katago_policy.pt')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--files-per-epoch', type=int, default=10)
    ap.add_argument('--val-files', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='每文件样本上限（冒烟测试用）')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    print('device:', device)

    all_files = sorted(os.path.join(args.data, f)
                       for f in os.listdir(args.data) if f.endswith('.npz'))
    if len(all_files) < args.val_files + 1:
        raise SystemExit('数据文件不足（至少 %d 个）' % (args.val_files + 1))
    val_files = all_files[-args.val_files:]
    train_files = all_files[:-args.val_files]
    print('训练文件 %d 个，验证文件 %d 个（%s）' % (len(train_files), len(val_files), val_files))

    model = KatagoNet().to(device)
    print('模型参数量: %d' % model.param_count())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    pool = mp.Pool(args.workers) if args.workers > 1 else None
    best_acc = -1.0
    try:
        for ep in range(args.epochs):
            files = random.sample(train_files, min(args.files_per_epoch, len(train_files)))
            t0 = time.time()
            loss, acc, n = train_epoch(model, opt, files, device, args.batch_size,
                                       limit=args.limit, pool=pool)
            val_acc, val_err = evaluate(model, val_files, device, pool=pool,
                                        limit=args.limit or 5000)
            print('[epoch %2d/%d] loss=%.4f top1=%.3f  val_top1=%.3f val_vMAE=%.3f  (%d样本 %.1fs)'
                  % (ep + 1, args.epochs, loss, acc, val_acc, val_err, n, time.time() - t0),
                  flush=True)
            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                torch.save(model.state_dict(), args.out)
                print('  save best -> %s (val_top1=%.3f)' % (args.out, val_acc), flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print('DONE best val_top1=%.3f -> %s' % (best_acc, args.out))


if __name__ == '__main__':
    main()
