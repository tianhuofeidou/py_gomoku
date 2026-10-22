# -*- coding: utf-8 -*-
"""离线开局研究统计，不参与正式引擎运行，不随仓库分发原始棋谱。

用法:
    python scripts/verify_opening_research.py <renjunet.xml> [--section sym|313|317|195]

设计要点（这两条是前两版错误的根源，务必保持）:
  1. 对称变换必须作用在【绝对坐标】上（绕棋盘中心 (7,7)），
     不能把相对坐标 (0,0) 直接喂给 tf()。
  2. 比较点集时必须【保留颜色】——黑集合与白集合分开比，
     否则"交换黑白"的变换会被误判为有效对称。
  3. 同一盘面仍有残余对称时，用应手打破平局，合并等价应手；
     统计相对坐标时，锚点与应手必须使用同一次变换。
"""
import argparse
import collections
import math
import re

SIZE = 15

def tf(r, c, s):
    """D4 的 8 个变换，绕棋盘中心 (7,7)。"""
    if s >= 4:
        c = SIZE - 1 - c
    for _ in range(s % 4):
        r, c = c, SIZE - 1 - r
    return r, c

def color_key(black, white, s):
    """对黑/白分别变换，返回**带颜色**的键。"""
    b = tuple(sorted(tf(r, c, s) for r, c in black))
    w = tuple(sorted(tf(r, c, s) for r, c in white))
    return (b, w)

def valid_syms(black, white):
    """返回能保持（黑集合, 白集合）都不变的变换列表。"""
    base = (tuple(sorted(black)), tuple(sorted(white)))
    return [s for s in range(8) if color_key(black, white, s) == base]

def canon_move(black, white, move):
    """先归一化落子前的有色盘面，再合并残余对称下的等价应手。

    应手只用于最小盘面键的平局裁决，不加入盘面，避免改变统计对象。
    """
    _key, transformed, symmetry = min(
        (color_key(black, white, s), tf(*move, s), s) for s in range(8))
    return transformed, symmetry

def parse_moves(text):
    out = []
    for t in text.split():
        t = t.strip().lower()
        if len(t) < 2:
            continue
        c = ord(t[0]) - 97
        try:
            r = int(t[1:]) - 1
        except ValueError:
            continue
        out.append((r, c))
    return out

def load(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    ob = txt[txt.find('<openings>'):txt.find('</openings>')]
    ops = dict(re.findall(r'<opening id="(\d+)" abbr="([^"]+)"', ob))
    rb = txt[txt.find('<rules>'):txt.find('</rules>')]
    rules = {m[0]: (m[1], int(m[2])) for m in
             re.findall(r'<rule id="(\d+)" name="([^"]+)" category="(\d)"', rb)}
    games = re.findall(r'<game ([^>]*?)>\s*<move>([^<]*)</move>', txt)
    return ops, rules, games

def attr(a, k):
    m = re.search(k + r'="([^"]*)"', a)
    return m.group(1) if m else None

def ci95(p, n):
    return 196 * math.sqrt(p * (1 - p) / n) if n else 0

# ---------- 各节复核 ----------

def sec_313(ops, games):
    """§3.3.13a 白4 集中度：整盘归一化后统计。"""
    st = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    for a, mv in games:
        oid = attr(a, 'opening')
        ms = parse_moves(mv)
        if not oid or len(ms) < 4:
            continue
        b1, w2, b3, w4 = ms[0], ms[1], ms[2], ms[3]
        w4t, symmetry = canon_move([b1, b3], [w2], w4)
        b1t = tf(*b1, symmetry)
        st[oid][(w4t[0]-b1t[0], w4t[1]-b1t[1])] += 1
        tot[oid] += 1
    print('%-6s %-8s %-14s %-8s %s' % ('opening','样本','白4众数','占比','CI'))
    for oid in sorted(st, key=lambda x: int(x) if x.isdigit() else 999):
        if oid == '27' or tot[oid] < 100: continue
        k, n = st[oid].most_common(1)[0]
        p = n / tot[oid]
        print('%-6s %-8d %-14s %6.1f%%  ±%.2f' % (
            ops.get(oid, '?'), tot[oid], str(k), 100*p, ci95(p, tot[oid])))

def sec_sym(ops, games):
    """§3.3.13b 对称群：哪些珠型有非恒等对称。"""
    tri = collections.defaultdict(collections.Counter)
    for a, mv in games:
        oid = attr(a, 'opening')
        ms = parse_moves(mv)
        if not oid or len(ms) < 3: continue
        tri[oid][(ms[0], ms[1], ms[2])] += 1
    print('%-6s %-30s %s' % ('opening','三手','有效对称'))
    multi = []
    for oid in sorted(tri, key=lambda x: int(x) if x.isdigit() else 999):
        if oid == '27': continue
        b1, w2, b3 = tri[oid].most_common(1)[0][0]
        syms = valid_syms([b1, b3], [w2])
        if len(syms) > 1: multi.append(ops.get(oid,'?'))
        print('%-6s %-30s %s' % (ops.get(oid,'?'), '%s %s %s'%(b1,w2,b3), syms))
    print('\n有非恒等对称的珠型: %d 个 -> %s' % (len(multi), ', '.join(multi)))

def sec_317(ops, rules, games):
    """§3.3.17 白胜局 / 有效结果局数；和棋计入分母，不按半分计分。"""
    st = collections.defaultdict(lambda: [0,0])
    for a, mv in games:
        rid = attr(a,'rule'); res = attr(a,'bresult')
        try: v = float(res) if res else None
        except ValueError: v = None
        if v not in (0, 0.5, 1) or not rid: continue
        st[rid][0] += (v == 0); st[rid][1] += 1
    print('%-4s %-24s %-6s %-8s %-8s %s' % ('id','名称','类别','局数','白胜率','CI'))
    for rid in sorted(st, key=lambda x:int(x)):
        white_wins, n = st[rid]
        if n < 100: continue
        nm, cat = rules.get(rid, ('?',0))
        p = white_wins/n
        print('%-4s %-24s %-6d %-8d %6.1f%%  ±%.2f' % (rid, nm[:22], cat, n, 100*p, ci95(p,n)))

def sec_195(ops, games):
    """§3.3.19e 前5手序列：只平移，不合并旋转或镜像。"""
    seq = collections.Counter(); tot = 0
    for a, mv in games:
        ms = parse_moves(mv)
        if len(ms) < 5: continue
        b1 = ms[0]
        seq[tuple((r-b1[0], c-b1[1]) for r,c in ms[:5])] += 1
        tot += 1
    print('前5手只平移，未合并旋转/镜像；>=5 手对局: %d  不同序列: %d' % (tot, len(seq)))
    acc = 0
    for i,(s,n) in enumerate(seq.most_common(), 1):
        acc += n
        if i in (1,12,50,100,500,1000):
            print('  前 %-5d 种覆盖 %.1f%%' % (i, 100*acc/tot))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', help='本地 RenjuNet RIF XML 文件')
    parser.add_argument('--section', choices=('all', '313', 'sym', '317', '195'), default='all')
    args = parser.parse_args()
    sec = args.section
    ops, rules, games = load(args.path)
    for name, fn in (('313', lambda: sec_313(ops, games)),
                     ('sym', lambda: sec_sym(ops, games)),
                     ('317', lambda: sec_317(ops, rules, games)),
                     ('195', lambda: sec_195(ops, games))):
        if sec in ('all', name):
            print('\n===== %s =====' % name)
            fn()
