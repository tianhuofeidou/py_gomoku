# -*- coding: utf-8 -*-
"""棋谱复盘：记录有限搜索中的五态与已评估着法的风险。

为什么需要它：跨引擎对局的胜负只告诉我们"输了"，输在哪一手要靠复盘。本工具重放棋谱，
在指定一方的每个回合调用 `analyze_turn`，记录：

  - 该手各候选的五态档位（是否在本次搜索中已全部必败）；
  - 实际落子与引擎首选是否一致、落到什么档位；
  - 是否存在非必败候选，且实际着法已被评估为硬必败。

不在候选中的实际着法未被评估，保持未知；“无”和“存在必败”均非硬必败，
也不保证可守。全部候选必败仅针对本次实际展开的搜索树，不是全盘证明。

用法：
    python -m gomoku.tools.review --json <对战台导出的 json> [--side a] [--t0 160]

`--side a` 表示复盘对战台里 `--black` 指定的那个引擎（B 为对手）；
`--t0` 可压低温度加速（默认沿用引擎默认档 160，与实战一致）。
"""
import argparse
import json
import sys

from gomoku.core.engine import Engine
from gomoku.core.deep_search import DeepSearch
from gomoku.core.utils import BLACK, WHITE, empty_board, place
from gomoku.game import check_win

COLUMNS = 'ABCDEFGHIJKLMNO'


def cd(r, c):
    return '%s%d' % (COLUMNS[c], r + 1)


def our_color(game, side):
    """由对战台记录推断指定一方在该局的颜色。"""
    a_is_white = game.get('a_white')
    if a_is_white is None:
        a_is_white = bool(game.get('swap'))
    return (WHITE if a_is_white else BLACK) if side == 'a' else (BLACK if a_is_white else WHITE)


def review_game(game, side='a', start_ply=6):
    """复盘一局；返回 (rows, first_all_lost_ply, missed_save_ply)。"""
    color = our_color(game, side)
    moves = game['moves']
    engine = Engine()
    board = empty_board()
    rows = []
    first_all_lost = None
    missed_save = None
    for i, m in enumerate(moves):
        player = m['player']
        if player == color and i >= start_ply:
            res = engine.analyze_turn(board, color)
            ranked = res['part4_result'].get('ranked') or []
            states = [it.get('state') for it in ranked]
            names = [it.get('state_name') for it in ranked]
            top = ranked[0] if ranked else None
            played = (m['r'], m['c'])
            chosen = next((it for it in ranked if (it['r'], it['c']) == played), None)
            save = next((it for it in ranked if it.get('state') != DeepSearch.STATE_LOSE), None)
            all_lost = bool(ranked) and all((s == DeepSearch.STATE_LOSE) for s in states)
            if all_lost and first_all_lost is None:
                first_all_lost = i + 1
            if (missed_save is None and save is not None and chosen is not None
                    and chosen.get('state') == DeepSearch.STATE_LOSE):
                missed_save = (i + 1, cd(save['r'], save['c']), cd(*played))
            rows.append({
                'ply': i + 1, 'move': cd(*played),
                'top': cd(top['r'], top['c']) if top else '-',
                'top_state': top.get('state_name') if top else '-',
                'played_state': (chosen or {}).get('state_name', '未评估（不在候选）'),
                'cand_states': '/'.join(sorted(set(n or '-' for n in names))) or '-',
                'save': cd(save['r'], save['c']) if save else None,
            })
        place(board, m['r'], m['c'], player)
        engine.on_move(board, m['r'], m['c'], player)
        if check_win(board, m['r'], m['c'], player):
            break
    return rows, first_all_lost, missed_save


def main(argv=None):
    ap = argparse.ArgumentParser(description='棋谱复盘：定位败着')
    ap.add_argument('--json', required=True, help='对战台导出的 JSON（含 games[].moves）')
    ap.add_argument('--side', default='a', choices=('a', 'b'), help='复盘哪一方（a=--black 引擎）')
    ap.add_argument('--t0', type=float, default=None, help='覆盖 DeepSearch.T0（默认沿用引擎默认档）')
    ap.add_argument('--start-ply', type=int, default=6, help='从第几手开始复盘（跳过开局）')
    ap.add_argument('--summary', action='store_true', help='只打跨局汇总（每局明细不打）')
    args = ap.parse_args(argv)

    if args.t0 is not None:
        from gomoku.core.deep_search import DeepSearch
        DeepSearch.T0 = float(args.t0)

    with open(args.json, encoding='utf-8') as f:
        data = json.load(f)
    games = data.get('games') or []
    if not games:
        print('该 JSON 没有棋谱（需要 --json 由对战台生成）')
        return 1
    stats = []
    for g in games:
        moves = g.get('moves')
        if not moves or not isinstance(moves[0], dict) or 'player' not in moves[0]:
            print('第 %s 局：没有可复盘的棋谱（单开局模式也要用 --json 记录）' % g.get('game'))
            continue
        rows, first_lost, missed = review_game(g, side=args.side, start_ply=args.start_ply)
        stats.append({'winner': g.get('winner'), 'plies': len(moves),
                      'first_lost': first_lost, 'missed': missed})
        head = '第 %s 局（%s，%s 手）' % (
            g.get('game'), 'A方' if args.side == 'a' else 'B方', len(moves))
        if not args.summary:
            print('\n=== %s ===' % head)
            for row in rows:
                print('  %2d手 实际=%-4s 引擎首选=%-4s(%s)  实际档位=%-6s 候选档位=%s%s' % (
                    row['ply'], row['move'], row['top'], row['top_state'],
                    row['played_state'], row['cand_states'],
                    ('  非必败候选=' + row['save']) if row['save'] else ''))
            if first_lost:
                print('  → 第 %d 手首次出现全部候选必败（仅限本次搜索树）' % first_lost)
            if missed:
                print('  → 第 %d 手实际着法为硬必败，存在非必败候选 %s（实际走 %s；仅限本次搜索树）' % missed)
            if not first_lost and not missed:
                print('  → 本次复盘未发现上述情况；未评估着法保持未知')

    lost = [s['first_lost'] for s in stats if s['first_lost']]
    print('\n=== 汇总（%d 局）===' % len(stats))
    if lost:
        lost.sort()
        mid = lost[len(lost) // 2]
        print('  首次"全部候选必败"的手数：n=%d  最小 %d  中位 %d  最大 %d' % (
            len(lost), lost[0], mid, lost[-1]))
        early = sum(1 for p in lost if p <= 12)
        print('  其中 %d 局发生在第 12 手以内（开局阶段）' % early)
    else:
        print('  复核范围内没有出现"全部候选必败"')
    print('  按 verdict：' + ', '.join(
        '%s×%d' % (k, sum(1 for s in stats if s['winner'] == k))
        for k in ('B', 'W', 'draw') if any(s['winner'] == k for s in stats)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
