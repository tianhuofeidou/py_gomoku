# -*- coding: utf-8 -*-
"""五子棋 独立版 —— 命令行对战（终端也能玩）。

用法：
    python play_cli.py

人类执黑先手，AI 执白。输入坐标（列字母+行号，如 h8；或 行,列 如 7,7）。
命令：u=悔棋  q=退出  h=帮助
"""
import sys

from gomoku.game import BLACK, WHITE, EMPTY, COLUMNS
from gomoku.session import PlaySession

SIZE = 15


def print_board(b):
    print('   ' + ' '.join(COLUMNS))
    for r in range(SIZE):
        row = []
        for c in range(SIZE):
            v = b[r][c]
            row.append('X' if v == BLACK else ('O' if v == WHITE else '.'))
        print('%2d %s' % (r + 1, ' '.join(row)))


def parse_coord(tok):
    tok = tok.strip().lower()
    if not tok:
        return None
    # 形如 h8 / h08
    if tok[0] in COLUMNS.lower():
        c = COLUMNS.lower().index(tok[0])
        try:
            r = int(tok[1:]) - 1
        except ValueError:
            return None
        return r, c
    # 形如 7,7 或 7 7
    part = tok.replace('，', ',').replace(' ', ',')
    try:
        a, b = part.split(',')
        return int(a) - 1, int(b) - 1
    except Exception:
        return None


def main():
    s = PlaySession('cli')
    print('=== 五子棋 独立版 CLI：你执黑先手，AI 执白 ===')
    while True:
        g = s.game
        print_board(g.board())
        if g.winner is not None:
            if g.winner == 0:
                print('🤝 和棋')
            elif g.winner == g.ai_player():
                print('🎉 AI 获胜')
            else:
                print('😅 你获胜')
            print('输入 n 开新局，q 退出')
            tok = input('> ').strip().lower()
            if tok == 'q':
                break
            if tok == 'n':
                s.reset(mode='ai', human_player=BLACK)
                continue
            continue
        if not g.is_human_turn():
            # AI 轮
            print('AI 思考中…')
            out = s.ai_play()
            if out:
                (r, c), ranked, net = out
                print('AI 落子 %s%d [纯算法]' % (COLUMNS[c], r + 1))
            continue
        tok = input('你走（如 h8，u=悔棋，q=退出）> ').strip().lower()
        if tok == 'q':
            break
        if tok == 'u':
            s.undo('human')
            print('已悔棋')
            continue
        if tok == 'h':
            print('坐标格式：列字母+行号（如 h8），或 行,列（如 7,7）')
            continue
        coord = parse_coord(tok)
        if coord is None:
            print('无法识别：' + tok)
            continue
        r, c = coord
        if not (0 <= r < SIZE and 0 <= c < SIZE):
            print('越界')
            continue
        res = s.place(r, c, BLACK)
        if not res['ok']:
            print(res['error'])
            continue
        # AI 应手
        out = s.ai_play()
        if out:
            (ar, ac), ranked, net = out
            print('AI 落子 %s%d [纯算法]' % (COLUMNS[ac], ar + 1))


if __name__ == '__main__':
    main()
