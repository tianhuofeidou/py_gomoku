# -*- coding: utf-8 -*-
"""机机对战模式：两个 AI 自动对弈。

用法：
  python vs_mode.py [--black-params ga_best_params.json] [--white-params ...]
                    [--t0 40] [--max-moves 60] [--no-dynamic] [--quiet] [--out game.json]

参数文件格式与 ga_tune.py 输出一致：{"params": {...}} 或直接是 params dict。
未指定参数文件时使用 search.py / deep_search.py 默认参数。
"""
import argparse
import json
import os

from python.tools import ga_tune as G
from python.core import search as S
from python.core.utils import empty_board, place, EMPTY, BLACK, WHITE
from python.core.engine import Engine


def load_params(path):
    """加载 GA 参数文件，返回规范化 genome。"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'params' in data:
        data = data['params']
    return G._fix_genome(data)


def print_board(b):
    cols = 'ABCDEFGHIJKLMNO'
    print('   ' + ' '.join(cols))
    for r in range(15):
        row = []
        for c in range(15):
            v = b[r][c]
            row.append('X' if v == BLACK else ('O' if v == WHITE else '.'))
        print('%2d %s' % (r + 1, ' '.join(row)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--black-params', default=None, help='黑方参数 JSON（ga_tune 输出格式）')
    ap.add_argument('--white-params', default=None, help='白方参数 JSON（ga_tune 输出格式）')
    ap.add_argument('--t0', type=float, default=None, help='深推温度；不填则用参数里的 deep_t0，默认 40')
    ap.add_argument('--max-moves', type=int, default=60, help='最大步数')
    ap.add_argument('--no-dynamic', action='store_true', help='禁用动态攻防配比（双方固定 3攻2防）')
    ap.add_argument('--quiet', action='store_true', help='不逐手打印棋盘')
    ap.add_argument('--out', default=None, help='保存棋谱 JSON 到文件')
    args = ap.parse_args()

    black_params = load_params(args.black_params) if args.black_params else G.BASELINE
    white_params = load_params(args.white_params) if args.white_params else G.BASELINE

    if args.no_dynamic:
        S.GEAR_ENABLED = False
    else:
        S.GEAR_ENABLED = True

    e = Engine()
    b = empty_board()
    moves = []
    turn = BLACK

    print('=== 机机对战 ===')
    print('黑: %s | 白: %s' % (
        os.path.basename(args.black_params) if args.black_params else '默认',
        os.path.basename(args.white_params) if args.white_params else '默认'))
    if not args.quiet:
        print_board(b)

    for step in range(args.max_moves):
        params = black_params if turn == BLACK else white_params
        G.apply_params(params)
        t0 = args.t0 if args.t0 is not None else params['deep_t0']
        G.DeepSearch.T0 = t0

        res = e.analyze_turn(b, turn)
        move = res['part4_result']['move']
        if move is None:
            break
        r, c = move
        if not (0 <= r < 15 and 0 <= c < 15) or b[r][c] != EMPTY:
            print('非法落子: %r' % (move,), file=sys.stderr)
            break

        place(b, r, c, turn)
        e.on_move(b, r, c, turn)
        moves.append({'step': step + 1, 'player': 'B' if turn == BLACK else 'W',
                      'row': r, 'col': c,
                      'coord': 'ABCDEFGHIJKLMNO'[c] + str(r + 1)})

        if not args.quiet:
            print('\n第 %d 手 %s：%s' % (
                step + 1, '黑' if turn == BLACK else '白',
                'ABCDEFGHIJKLMNO'[c] + str(r + 1)))
            print_board(b)

        if G.DeepSearch._check_win(b, r, c, turn):
            winner = 'B' if turn == BLACK else 'W'
            print('\n=== 结果：%s 胜（第 %d 手 %s） ===' % (
                '黑' if winner == 'B' else '白', step + 1,
                'ABCDEFGHIJKLMNO'[c] + str(r + 1)))
            if args.out:
                with open(args.out, 'w', encoding='utf-8') as f:
                    json.dump({'winner': winner, 'moves': moves,
                               'reason': 'five', 'max_moves': args.max_moves},
                              f, ensure_ascii=False, indent=2)
            return

        turn = 3 - turn

    print('\n=== 结果：和棋 / 步数上限（%d 手） ===' % len(moves))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'winner': None, 'moves': moves,
                       'reason': 'draw_or_max', 'max_moves': args.max_moves},
                      f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
