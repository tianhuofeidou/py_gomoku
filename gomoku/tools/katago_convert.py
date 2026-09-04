# -*- coding: utf-8 -*-
"""KataGo 蒸馏数据转换：npz → 紧凑训练样本。

输入：katago_data/fs15x/dataN.npz（KataGo shuffle 后格式）
输出：katago_data/fs15x_proc/dataN.npz，每文件含：
  x : [N, 2, 15, 15] uint8   己方棋子 / 对方棋子（0/1）
  p : [N, 225] float32       落子策略 soft 标签（行和=1，int16 原始值归一化）
  v : [N, 2] float32         胜率标签（己方胜概率, 对方胜概率）

README 要点：fs15x = 无禁手 15x15，输入只需通道 1（己方）、2（对方）；
policyTargetsNCMove[:,0,:225] 是 int16 未归一化；globalTargetsNC[:,0:2] 是胜率。

用法：
  py -m gomoku.tools.katago_convert --src katago_data/fs15x --out katago_data/fs15x_proc
  （加 --max-files N 只转前 N 个文件）
"""
import argparse
import os

import numpy as np

H = W = 15
NCH = 2


def load_npz(path):
    """读取一个 KataGo npz → (x, p, v) 紧凑样本。"""
    d = np.load(path)
    n = d['binaryInputNCHWPacked'].shape[0]
    # 22 通道位打包 → [N,22,15,15]
    bits = np.unpackbits(d['binaryInputNCHWPacked'], axis=2)[:, :, :H * W]
    board = bits.reshape(n, -1, H, W)
    x = np.stack([board[:, 1], board[:, 2]], axis=1).astype(np.uint8)  # [N,2,15,15]
    # policy：int16 未归一化 → float32 行归一化
    p_raw = d['policyTargetsNCMove'][:, 0, :H * W].astype(np.float64)
    row_sum = p_raw.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0          # 防御：全 0 行（理论上不会）
    p = (p_raw / row_sum).astype(np.float32)
    # value：胜/负/和 → 取前两维（胜、负）；和棋概率并入其余
    g = d['globalTargetsNC']
    v = g[:, 0:2].astype(np.float32)
    return x, p, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='katago_data/fs15x')
    ap.add_argument('--out', default='katago_data/fs15x_proc')
    ap.add_argument('--max-files', type=int, default=0)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.src) if f.endswith('.npz'))
    if args.max_files > 0:
        files = files[:args.max_files]
    os.makedirs(args.out, exist_ok=True)

    total = 0
    for i, fn in enumerate(files):
        x, p, v = load_npz(os.path.join(args.src, fn))
        out_path = os.path.join(args.out, fn)
        np.savez_compressed(out_path, x=x, p=p, v=v)
        total += x.shape[0]
        print('%-12s -> %s  samples=%d  (total=%d)' % (fn, out_path, x.shape[0], total), flush=True)
    print('DONE  files=%d  samples=%d' % (len(files), total))


if __name__ == '__main__':
    main()
