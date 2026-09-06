# -*- coding: utf-8 -*-
"""插件统一落子接口：Node 端通过子进程调用本脚本。

输入（stdin JSON）：
  {
    "moves": [{"r": int, "c": int, "player": 1|2}, ...],
    "player": 1|2   # 当前要落子的一方（黑=1 白=2）
  }

输出（stdout JSON）：
  {"row": int, "col": int}

特殊开局：
  - 黑第一手（moves 为空且 player=BLACK）→ 天元 (7,7)
  - 白第二手（moves 长度 1 且 player=WHITE）→ 取 engine 开局候选列表第一个
其余局面 → engine.analyze_turn 深度推演选点。
"""
import json
import os
import sys

from gomoku.core.utils import empty_board, place, BLACK, WHITE
from gomoku.core.engine import Engine
from gomoku.tools import ga_tune as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 注：标准包运行方式为 `python -m gomoku.adapter.ai_move`（cwd=插件根）


def load_ga_params():
    """加载 GA 最优参数并应用到正式引擎。
    默认【不加载】（实测两轮 GA 参数均过拟合寻优开局，真实中盘局面
    胜率低于默认占位参数：默认 67% vs GA 54-55%，24 真实开局评估）；
    设置环境变量 DSH_GOMOKU_GA=1 才启用（实验用）。
    找不到/损坏时静默回退默认参数。"""
    if os.environ.get('DSH_GOMOKU_GA') != '1':
        return
    path = os.path.join(ROOT, 'data', 'ga_best_params.json')
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        params = data.get('params', data)
        G.apply_params(G._fix_genome(params))
    except Exception:
        pass


def main():
    load_ga_params()
    data = json.load(sys.stdin)
    moves = data.get('moves') or []
    player = int(data.get('player', WHITE))

    e = Engine()
    b = empty_board()
    for m in moves:
        place(b, int(m['r']), int(m['c']), int(m['player']))
        e.on_move(b, int(m['r']), int(m['c']), int(m['player']))

    # 黑第一手：天元
    if player == BLACK and len(moves) == 0:
        print(json.dumps({'row': 7, 'col': 7}))
        return

    # 白第二手：开局候选列表取第一个
    if player == WHITE and len(moves) == 1:
        cand = e.compute_move(b)
        if isinstance(cand, list) and cand:
            r, c = cand[0]
            print(json.dumps({'row': r, 'col': c}))
            return

    res = e.analyze_turn(b, player)
    mv = res['part4_result']['move']
    if isinstance(mv, list):
        mv = mv[0] if mv else None
    if mv is None:
        # 兜底：天元或首个空位
        r, c = 7, 7
        if b[r][c] != 0:
            for rr in range(15):
                for cc in range(15):
                    if b[rr][cc] == 0:
                        r, c = rr, cc
                        break
                else:
                    continue
                break
        print(json.dumps({'row': r, 'col': c}))
        return
    print(json.dumps({'row': mv[0], 'col': mv[1]}))


if __name__ == '__main__':
    main()
