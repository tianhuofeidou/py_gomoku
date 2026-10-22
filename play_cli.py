# -*- coding: utf-8 -*-
"""五子棋 独立版 —— 命令行对战（终端也能玩）。

用法：
    python play_cli.py

人类执黑先手，AI 执白。输入坐标（如 h8；或从 1 开始的行,列，8,8 与 h8 等价）。
命令：u=悔棋  q=退出  h=帮助

AI 每走一手会打印本次实际决策的来源、耗时、叶子评估状态和前五个候选（含五态）。
这些字段全部取自本次决策结果：入口只负责显示，不另算一套选点或叶子评估。
"""
import time

from gomoku import service
from gomoku.game import BLACK, WHITE, EMPTY, COLUMNS
from gomoku.session import PlaySession

SIZE = 15
CANDIDATE_LINES = 5


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


def fmt_score(score, opening=False):
    """候选分展示。

    1e9 是"不深推、直接采用"的约定值：开局谱着和战术直取都会拿到它，但两者的
    含义完全不同——谱着只是定式着法，不是必胜手，所以分开显示，避免玩家误读。
    """
    if score is None:
        return '未补算'
    if score > 1e8:
        return '谱着' if opening else '直取'
    return '%.0f' % score


def leaf_text(status):
    """叶子评估状态：只有真正触发叶子评分时才报叶子数。"""
    status = status or {}
    if status.get('mode') == 'not_triggered':
        return '未触发'
    return '纯算法 %d 个叶子' % status.get('total', 0)


def print_candidates(ranked, selected=None, opening=False, limit=CANDIDATE_LINES):
    """打印本次决策的候选（含五态）。ranked 已由核心按五态档位排好序。"""
    if not ranked:
        return
    if len(ranked) > limit:
        print('  候选（前 %d / 共 %d）：' % (limit, len(ranked)))
    else:
        print('  候选（%d）：' % len(ranked))
    for i, x in enumerate(ranked[:limit]):
        mark = '  ← 选中' if selected == (x['r'], x['c']) else ''
        name = x.get('state_name') or '无'
        tag = '' if name == '无' else '  ⚔%s' % name
        print('   %d. %s%d  分 %s%s%s' % (
            i + 1, COLUMNS[x['c']], x['r'] + 1,
            fmt_score(x.get('score'), opening), mark, tag))


def ai_turn(s):
    """AI 走一手，并把本次决策的来源、耗时、叶子状态和候选打到终端。"""
    t0 = time.perf_counter()
    out = s.ai_play()
    dt = time.perf_counter() - t0
    if not out:
        return False
    (r, c), ranked, net_used = out
    source = s.engine.opening_source
    label = service.decision_source_label(source)
    if net_used:      # 只在真实完成网络推理时才这样标
        label += '＋神经网络'
    # 开局直取：谱着不是必胜手，分数按"谱着"显示。
    opening = source in ('book', 'rules')
    print('AI 落子 %s%d' % (COLUMNS[c], r + 1))
    print('  决策：%s · 叶子：%s · 用时 %.1fs' % (
        label, leaf_text(s.engine.deep_search.leaf_eval_status()), dt))
    print_candidates(ranked, (r, c), opening=opening)
    return True


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
            ai_turn(s)
            continue
        tok = input('你走（如 h8，u=悔棋，q=退出）> ').strip().lower()
        if tok == 'q':
            break
        if tok == 'u':
            s.undo('human')
            print('已悔棋')
            continue
        if tok == 'h':
            print('坐标格式：列字母+行号（如 h8），或从 1 开始的行,列（8,8 与 h8 等价）')
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
        ai_turn(s)


if __name__ == '__main__':
    main()
