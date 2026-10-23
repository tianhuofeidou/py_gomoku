# -*- coding: utf-8 -*-
"""败局回归库的独立命令。

只用标准库；所有落子/判胜/选点都委托 ``gomoku.loss_book`` 与既有引擎，
本文件不复制任何规则。

子命令：
    list      查看待处理败局
    import    从对战台 JSON 导入失败棋局
    locate    定位疑似关键败手（或人工指定）
    retest    在"当前版本"上从附近节点重赛
    show      查看某局的复测明细
    archive   把已通过的案例移入归档
"""
import argparse
import json
import os
import sys

from gomoku import loss_book as lb


def _fmt_fp(fp):
    if not fp:
        return '-'
    eng = (fp.get('engine') or {}).get('digest') or '-'
    return '%s/%s' % (eng, fp.get('book') or '-')


def cmd_list(args):
    cases = lb.list_cases(status=args.status, kind=args.kind, store=None)
    if not cases:
        print('（没有符合条件的案例）')
        return 0
    print('%-10s %-6s %-8s %-6s %-8s %-16s %s' % (
        'id', 'AI色', '结局', '状态', '手数', '指纹(引擎/开局库)', '来源数'))
    for c in cases:
        print('%-10s %-6s %-8s %-6s %-8s %-16s %d' % (
            c['id'][:8], '黑' if c['aiColor'] == lp_black() else '白',
            c.get('kind'), c.get('status'), len(c.get('moves') or []),
            _fmt_fp(c.get('fingerprint')), len(c.get('sources') or [])))
    print('\n共 %d 例' % len(cases))
    return 0


def lp_black():
    from gomoku.game import BLACK
    return BLACK


def cmd_import(args):
    with open(args.json, encoding='utf-8') as f:
        data = json.load(f)
    games = data.get('games') or []
    if not games:
        print('该 JSON 没有 games[]，无法导入')
        return 1
    ai_side = args.ai_side            # 'a' 或 'b'
    added = merged = skipped = 0
    by_kind = {}
    for g in games:
        moves = g.get('moves')
        if not moves or not isinstance(moves[0], dict) or 'player' not in moves[0]:
            print('第 %s 局：没有逐手棋谱，跳过' % g.get('game'))
            skipped += 1
            continue
        a_white = g.get('a_white')
        if a_white is None:
            a_white = bool(g.get('swap'))
        from gomoku.game import BLACK, WHITE
        ai_color = (WHITE if a_white else BLACK) if ai_side == 'a' else (BLACK if a_white else WHITE)
        try:
            winner = lb.normalize_winner(g.get('winner'))
        except ValueError as e:
            print('第 %s 局：%s，跳过' % (g.get('game'), e))
            skipped += 1
            continue
        # 只导入"AI 输"的局：胜/和/未知一律跳过并计数
        if winner is None or winner == 0 or winner == ai_color:
            skipped += 1
            continue
        # 结局分类：对战台用 win_by 区分"超时胜""五连胜""异常"（match.py:539-549），
        # 超时与异常不能混进正常棋力失败，否则败因统计会失真。
        win_by = (g.get('win_by') or '').strip().lower()
        if win_by == 'timeout':
            kind = lb.KIND_TIMEOUT
        elif win_by == 'error':
            kind = lb.KIND_ABORTED
        else:
            # 'five' 是正常五连；win_by 缺失是旧格式，无法区分超时，
            # 按正常棋力失败处理，但把 win_by=None 记进 config 以便追溯。
            kind = lb.KIND_LOSS
        case, is_new = lb.add_case(
            moves, ai_color, winner=winner, kind=kind,
            source='match-import', source_detail=args.json,
            config={'a_side': ai_side, 'opening': g.get('opening'),
                    'win_by': win_by or None},
        )
        by_kind[kind] = by_kind.get(kind, 0) + 1
        added += int(is_new)
        merged += int(not is_new)
    print('导入完成：新增 %d 例，并入 %d 例，跳过 %d 局（非 AI 失败）' % (added, merged, skipped))
    if by_kind:
        print('  按结局分类：' + '、'.join('%s=%d' % (k, v) for k, v in sorted(by_kind.items())))
        if lb.KIND_TIMEOUT in by_kind or lb.KIND_ABORTED in by_kind:
            print('  注意：超时/异常与正常棋力失败已分开记录，'
                  '用 `list --kind loss` 只看正常失败。')
    return 0


def cmd_locate(args):
    cases = lb.list_cases(store=None)
    target = _pick(cases, args.id)
    if target is None:
        return 1
    if args.ply is not None:
        suspects, undet = [], None
        case = lb.set_suspects(target, suspects=suspects, undetermined=None,
                               manual_ply=args.ply)
        print('已人工指定：本方第 %d 手' % args.ply)
        print('  重赛起点：', [s['startPly'] for s in lb.retest_starts(case, manual_ply=args.ply)])
        return 0
    suspects, undet = lb.locate_suspects(target)
    case = lb.set_suspects(target, suspects=suspects, undetermined=undet)
    if suspects:
        print('疑似关键败手（共 %d 个）：' % len(suspects))
        for s in suspects:
            print('  第 %-3d 手 %-4s  %-14s  %s' % (s['ply'], s['move'], s['reason'], s['detail']))
        print('\n重赛起点（1 基手数）：', [x['startPly'] for x in lb.retest_starts(case)])
    else:
        print('无法明确定位，原因：%s' % undet)
        print('可用 --ply N 人工指定"本方第 N 次落子"后重赛。')
    return 0


def cmd_retest(args):
    cases = lb.list_cases(status='pending', store=None)
    if args.stale:
        cases = lb.list_cases(status='pending', fingerprint=lb.current_fingerprint(), store=None)
    if args.id:
        cases = [c for c in cases if c['id'].startswith(args.id)]
    if not cases:
        print('（没有需要复测的案例）')
        return 0
    print('将对 %d 例执行附近节点重赛（当前指纹 %s）\n' % (len(cases), _fmt_fp(lb.current_fingerprint())))
    agg = {}
    for c in cases:
        if not (c.get('suspects') or c.get('manualPly')):
            print('%s 未定位疑似败手，先跑 locate' % c['id'][:8])
            agg['skipped-no-suspect'] = agg.get('skipped-no-suspect', 0) + 1
            continue
        starts = lb.retest_starts(c, manual_ply=c.get('manualPly'))
        if not starts:
            print('%s 没有有效复测起点' % c['id'][:8])
            continue
        batch = lb.begin_retest_batch(c, starts)
        total = len(starts)
        tested = 0
        for st in starts:
            r = lb.retest_case(c, start_ply=st['startPly'],
                               manual_ply=c.get('manualPly'),
                               max_plies=args.max_plies,
                               batch_total=total, batch_id=batch['id'])
            agg[r['result']] = agg.get(r['result'], 0) + 1
            tested += 1
            print('  %s 起点%-4s → %-18s %s' % (
                c['id'][:8], r.get('startPly'), r['result'], (r.get('detail') or '')[:60]))
            if args.stop_after_first and r['result'] in lb.PASSING_RETESTS:
                if tested < total:
                    lb.force_pending(c, 'stop-after-first: 只测了 %d/%d 个起点' % (tested, total))
                    print('    已按 --stop-after-first 提前停止：本轮只测了 %d/%d 个起点，'
                          '状态保留 pending。' % (tested, total))
                break
    print('\n汇总：', agg)
    print('\n注意：「通过原败线回归」只表示这条旧败线在新版本上不再复现，'
          '不等于证明全局不败。')
    return 0


def cmd_show(args):
    cases = lb.list_cases(store=None)
    target = _pick(cases, args.id)
    if target is None:
        return 1
    c = target
    print('案例 %s  AI 执%s  结局=%s  状态=%s' % (
        c['id'][:8], '黑' if c['aiColor'] == lp_black() else '白', c.get('kind'), c.get('status')))
    print('手数=%d  来源=%d  指纹=%s' % (
        len(c.get('moves') or []), len(c.get('sources') or []), _fmt_fp(c.get('fingerprint'))))
    for s in c.get('sources') or []:
        print('  来源 %s: %s %s %s' % (s['id'][:8], s.get('source'),
                                     s.get('kind'), s.get('sourceDetail') or ''))
    if c.get('suspects'):
        print('\n疑似败手：')
        for s in c['suspects']:
            print('  第 %-3d 手 %-4s %s — %s' % (s['ply'], s['move'], s['reason'], s['detail']))
    elif c.get('suspectsUndetermined'):
        print('\n未定位：%s' % c['suspectsUndetermined'])
    if c.get('manualPly'):
        print('人工指定：本方第 %d 手' % c['manualPly'])
    if c.get('retests'):
        print('\n复测记录（%d 条）：' % len(c['retests']))
        for r in c['retests']:
            print('  起点%-4s %-18s %-6s %s' % (
                r.get('startPly'), r['result'], r.get('plies'), (r.get('detail') or '')[:60]))
    return 0


def cmd_archive(args):
    moved = lb.archive_passed()
    print('已移入归档：%d 例' % len(moved))
    for c in moved:
        print('  %s（保留原棋谱与 %d 条复测记录）' % (c['id'][:8], len(c.get('retests') or [])))
    if not moved:
        print('（没有状态为 passed 的案例；需先通过原败线回归）')
    return 0


def _pick(cases, id_prefix):
    if not cases:
        print('（库里没有案例）')
        return None
    if not id_prefix:
        return cases[0]
    hit = [c for c in cases if c['id'].startswith(id_prefix)]
    if not hit:
        print('找不到 id 以 %s 开头的案例' % id_prefix)
        return None
    return hit[0]


def main(argv=None):
    # Windows CI 的管道可能使用 cp1252；在 argparse 输出中文帮助前统一编码。
    # StringIO 等文本捕获流没有 reconfigure，也不经过字节编码，保持其接口。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(prog='loss_book', description='败局回归库')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('list', help='查看待处理败局')
    p.add_argument('--status', default=None, choices=('pending', 'passed'))
    p.add_argument('--kind', default=None)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser('import', help='从对战台 JSON 导入失败棋局')
    p.add_argument('--json', required=True)
    p.add_argument('--ai-side', default='a', choices=('a', 'b'))
    p.set_defaults(func=cmd_import)

    p = sub.add_parser('locate', help='定位疑似关键败手')
    p.add_argument('--id', default=None)
    p.add_argument('--ply', type=int, default=None, help='人工指定：本方第 N 次落子')
    p.set_defaults(func=cmd_locate)

    p = sub.add_parser('retest', help='在新版本上从附近节点重赛')
    p.add_argument('--id', default=None)
    p.add_argument('--max-plies', type=int, default=None)
    p.add_argument('--stale', action='store_true', help='只测指纹已变的案例')
    p.add_argument('--stop-after-first', action='store_true', default=False,
                   help='首个起点通过就停（不推荐：没测齐的批次不算通过，会退回 pending）')
    p.set_defaults(func=cmd_retest)

    p = sub.add_parser('show', help='查看逐局结果')
    p.add_argument('--id', default=None)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser('archive', help='把已通过的案例移入归档')
    p.set_defaults(func=cmd_archive)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
