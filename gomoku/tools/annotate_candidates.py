# -*- coding: utf-8 -*-
"""对弈标注辅助（查询）：输入着法序列 → 输出候选点列表（r, c, 深推分）。

用途：人机对弈时由对弈助手调用，把引擎候选列给人类评分；
评分经 save_annotation.py 存盘，用于监督训练候选点评分网络
（人类三档标注：best=最好 / second=次好 / bad=不能走）。

输入（stdin JSON）：
  {"moves": [{"r": int, "c": int, "player": 1|2}, ...], "player": 1|2}

输出（stdout JSON）：
  {
    "player": 1|2,
    "stone_count": int,
    "candidates": [
      {"r": int, "c": int, "score": float, "deep": bool}, ...
    ]
  }

deep=true  → 来自深推 ranked（真实分支分，可排序）
deep=false → 开局候选或兜底补充（score=0，仅作选项）
"""
import json
import sys

from gomoku.core.utils import empty_board, place, BLACK, WHITE
from gomoku.core.engine import Engine


def main():
    data = json.load(sys.stdin)
    moves = data.get('moves') or []
    player = int(data.get('player', WHITE))

    e = Engine()
    b = empty_board()
    for m in moves:
        place(b, int(m['r']), int(m['c']), int(m['player']))
        e.on_move(b, int(m['r']), int(m['c']), int(m['player']))
    stone_count = sum(1 for row in b for v in row if v != 0)

    # 开局（0/1 子）：固定候选（天元 / 天元邻域），不深推
    opening = e._opening_candidates(b)
    if opening is not None:
        out = [{'r': r, 'c': c, 'score': 0.0, 'deep': False} for (r, c) in opening]
        print(json.dumps({'player': player, 'stone_count': stone_count,
                          'candidates': out}, ensure_ascii=False))
        return

    res = e.analyze_turn(b, player)
    ranked = res['part4_result'].get('ranked') or []
    out = [{'r': x['r'], 'c': x['c'], 'score': float(x['score']), 'deep': True}
           for x in ranked]
    # 候选不足 2 个时补兜底，但两种局面不补：
    #   1. 唯一强制候选（score>=1e8：必堵/必胜点）——补出来的全是"走了被绝杀"的假候选；
    #   2. 无候选（ranked 为空）时兜底就是真实决策路径，照常补。
    if len(out) == 1 and out[0]['score'] >= 1e8:
        pass  # 强制唯一解：不补假候选
    elif len(out) < 2:
        fb = e.search._fallback(b, player=player)
        seen = {(x['r'], x['c']) for x in out}
        for (r, c) in fb:
            if len(out) >= 5:
                break
            if (r, c) not in seen:
                out.append({'r': r, 'c': c, 'score': 0.0, 'deep': False})
                seen.add((r, c))
    print(json.dumps({'player': player, 'stone_count': stone_count,
                      'candidates': out}, ensure_ascii=False))


if __name__ == '__main__':
    main()
