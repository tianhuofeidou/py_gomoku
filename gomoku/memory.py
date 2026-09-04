# -*- coding: utf-8 -*-
"""全局对局记忆（独立版，复刻插件的 memory.js）。

跨会话共享：总战绩 totals / 败因类型 lossByType / 双边指纹 dualMem /
败局棋谱 badLines / 胜局棋谱 goodLines，含 8 变换对称归一化、
败因与五连线判定、memoryHint 文本提示，对局结束由 game.Game._archive_finished 调用 record_result。
"""
import json
import time

from gomoku import paths
from gomoku.game import SIZE, EMPTY, BLACK, WHITE, COLUMNS

AI_DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]

# 8 种对称变换（4 旋转 × 2 镜像）
SYMMS = [
    lambda r, c: (r, c),
    lambda r, c: (c, 14 - r),
    lambda r, c: (14 - r, 14 - c),
    lambda r, c: (14 - c, r),
    lambda r, c: (r, 14 - c),
    lambda r, c: (14 - r, c),
    lambda r, c: (c, r),
    lambda r, c: (14 - c, 14 - r),
]
SYMM_INV = [0, 3, 2, 1, 4, 5, 6, 7]

LOSS_TYPES = ['活四', '冲四', '跳四', '双活三', '其他']

EMPTY_GLOBAL_MEMORY = {
    'totals': {'wins': 0, 'losses': 0, 'draws': 0},
    'lossByType': {k: 0 for k in LOSS_TYPES},
    'dualMem': {},
    'badLines': [],
    'goodLines': [],
}

_global_memory = None


def _ai_in(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE


def _hidden_lines_of(b, player):
    """简易潜在线检测：全部 5 格窗口含 player 子 >= 2 视为潜在线。"""
    out = []
    for r in range(SIZE):
        for c in range(SIZE):
            for dr, dc in AI_DIRS:
                cells = []
                cnt = 0
                ok = True
                for k in range(5):
                    rr, cc = r + dr * k, c + dc * k
                    if not _ai_in(rr, cc):
                        ok = False
                        break
                    cells.append({'r': rr, 'c': cc, 'v': b[rr][cc]})
                    if b[rr][cc] == player:
                        cnt += 1
                if ok and cnt >= 2:
                    out.append({'cells': cells, 'cnt': cnt})
    return out


def parse_moves_key(s):
    arr = []
    if not s:
        return arr
    for part in s.split(';'):
        i = part.find(',')
        if i < 0:
            continue
        try:
            arr.append({'r': int(part[:i]), 'c': int(part[i + 1:])})
        except ValueError:
            continue
    return arr


def moves_key_of(arr):
    return ';'.join(str(m['r']) + ',' + str(m['c']) for m in arr)


def transform_moves(arr, f):
    return [{'r': f(m['r'], m['c'])[0], 'c': f(m['r'], m['c'])[1]} for m in arr]


def _moves_cmp(a, b):
    n = max(len(a), len(b))
    for i in range(n):
        if i >= len(a):
            return -1
        if i >= len(b):
            return 1
        if a[i]['r'] != b[i]['r']:
            return a[i]['r'] - b[i]['r']
        if a[i]['c'] != b[i]['c']:
            return a[i]['c'] - b[i]['c']
    return 0


def canon_key(opp_moves, ai_moves):
    best = None
    for f in SYMMS:
        o = transform_moves(opp_moves, f)
        a = transform_moves(ai_moves, f)
        if best is None or _moves_cmp(o, best[0]) < 0 or (
                _moves_cmp(o, best[0]) == 0 and _moves_cmp(a, best[1]) < 0):
            best = [o, a]
    return moves_key_of(best[0]) + '|' + moves_key_of(best[1])


def sym_prefix_match(hist_opp, cur_opp):
    if len(cur_opp) == 0 or len(cur_opp) > len(hist_opp):
        return False
    for f in SYMMS:
        t = transform_moves(cur_opp, f)
        ok = True
        for i in range(len(t)):
            if t[i]['r'] != hist_opp[i]['r'] or t[i]['c'] != hist_opp[i]['c']:
                ok = False
                break
        if ok:
            return True
    return False


def sym_dual_prefix_match(hist_opp, hist_ai, cur_opp, cur_ai):
    if len(cur_opp) > len(hist_opp) or len(cur_ai) > len(hist_ai):
        return -1
    for fi, f in enumerate(SYMMS):
        to = transform_moves(cur_opp, f)
        ta = transform_moves(cur_ai, f)
        ok = True
        for i in range(len(to)):
            if to[i]['r'] != hist_opp[i]['r'] or to[i]['c'] != hist_opp[i]['c']:
                ok = False
                break
        if ok:
            for i in range(len(ta)):
                if ta[i]['r'] != hist_ai[i]['r'] or ta[i]['c'] != hist_ai[i]['c']:
                    ok = False
                    break
        if ok:
            return fi
    return -1


def win_line_overlap(hist_win, cur_window_cells, player):
    my_sub = [{'r': x['r'], 'c': x['c']} for x in cur_window_cells if x['v'] == player]
    if len(my_sub) < 2:
        return 0
    best = 0
    for f in SYMMS:
        t = transform_moves(hist_win, f)
        n = 0
        for a in my_sub:
            for b in t:
                if a['r'] == b['r'] and a['c'] == b['c']:
                    n += 1
                    break
        if n > best:
            best = n
    return best


def loss_type_of(b, r, c, player):
    """败因判定：人类最后一手成五的杀法类型。传入的 b 会被临时清掉该点。"""
    b = [row[:] for row in b]
    b[r][c] = EMPTY
    best = None
    for dr, dc in AI_DIRS:
        count = 1
        gap = 0
        rr, cc = r + dr, c + dc
        while _ai_in(rr, cc) and b[rr][cc] == player:
            count += 1
            rr += dr
            cc += dc
        end1 = 1 if (_ai_in(rr, cc) and b[rr][cc] == EMPTY) else 0
        gr, gc = rr + dr, cc + dc
        if _ai_in(rr, cc) and b[rr][cc] == EMPTY and _ai_in(gr, gc) and b[gr][gc] == player:
            while _ai_in(gr, gc) and b[gr][gc] == player:
                gap += 1
                gr += dr
                gc += dc
        rr, cc = r - dr, c - dc
        while _ai_in(rr, cc) and b[rr][cc] == player:
            count += 1
            rr -= dr
            cc -= dc
        end2 = 1 if (_ai_in(rr, cc) and b[rr][cc] == EMPTY) else 0
        gr2, gc2 = rr - dr, cc - dc
        if _ai_in(rr, cc) and b[rr][cc] == EMPTY and _ai_in(gr2, gc2) and b[gr2][gc2] == player:
            while _ai_in(gr2, gc2) and b[gr2][gc2] == player:
                gap += 1
                gr2 -= dr
                gc2 -= dc
        if count + gap >= 5:
            best = {'count': count, 'gap': gap, 'end1': end1, 'end2': end2}
            break
    if best is None:
        return '其他'
    if best['gap'] > 0:
        return '跳四'
    if best['count'] >= 5:
        if best['end1'] and best['end2']:
            return '活四'
        if best['end1'] or best['end2']:
            return '冲四'
    return '其他'


def win_line_of(b, r, c, player):
    """五连线：player 最后一手 (r,c) 成五的那条线（至多 5 点）。"""
    for dr, dc in AI_DIRS:
        cells = [[r, c]]
        rr, cc = r + dr, c + dc
        while _ai_in(rr, cc) and b[rr][cc] == player:
            cells.append([rr, cc])
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while _ai_in(rr, cc) and b[rr][cc] == player:
            cells.insert(0, [rr, cc])
            rr -= dr
            cc -= dc
        if len(cells) >= 5:
            return [{'r': p[0], 'c': p[1]} for p in cells[:5]]
    return []


# ---------------------------------------------------------------------------
# 加载 / 保存
# ---------------------------------------------------------------------------
def global_memory():
    global _global_memory
    if _global_memory is None:
        fresh = {
            'totals': {'wins': 0, 'losses': 0, 'draws': 0},
            'lossByType': dict(EMPTY_GLOBAL_MEMORY['lossByType']),
            'dualMem': {},
            'badLines': [],
            'goodLines': [],
        }
        try:
            with open(paths.global_memory_file(), 'r', encoding='utf-8') as f:
                p = json.load(f)
            if isinstance(p, dict):
                if isinstance(p.get('totals'), dict):
                    fresh['totals'].update(p['totals'])
                if isinstance(p.get('lossByType'), dict):
                    fresh['lossByType'].update(p['lossByType'])
                if isinstance(p.get('dualMem'), dict):
                    fresh['dualMem'] = p['dualMem']
                if isinstance(p.get('badLines'), list):
                    fresh['badLines'] = p['badLines']
                if isinstance(p.get('goodLines'), list):
                    fresh['goodLines'] = p['goodLines']
        except Exception:
            pass
        _global_memory = fresh
        _migrate_dual_mem_keys()
        _seed_bad_lines_from_dual_mem()
        _seed_global_memory_from_games()
    return _global_memory


def save_global_memory():
    try:
        paths.ensure_base()
        with open(paths.global_memory_file(), 'w', encoding='utf-8') as f:
            json.dump(_global_memory, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _seed_global_memory_from_games():
    from gomoku import game as game_mod
    gm = _global_memory
    if gm['totals']['wins'] + gm['totals']['losses'] + gm['totals']['draws'] > 0 or len(gm['dualMem']) > 0:
        return
    changed = False
    for _, g in game_mod.games.items():
        if not g or not getattr(g, 'history', None):
            continue
        for h in g.history:
            if h and h.get('winner') == 0:
                gm['totals']['draws'] += 1
            elif h and h.get('winner') == WHITE:
                gm['totals']['wins'] += 1
            elif h and h.get('winner') == BLACK:
                gm['totals']['losses'] += 1
            changed = True
    if changed:
        save_global_memory()


def _migrate_dual_mem_keys():
    gm = _global_memory
    if not gm or not isinstance(gm.get('dualMem'), dict):
        return
    nd = {}
    changed = False
    for dk, rec in gm['dualMem'].items():
        sep = dk.find('|')
        opp_arr = parse_moves_key(dk) if sep < 0 else parse_moves_key(dk[:sep])
        ai_arr = [] if sep < 0 else parse_moves_key(dk[sep + 1:])
        nk = canon_key(opp_arr, ai_arr)
        if nk != dk:
            changed = True
        rec = rec or {}
        if nk not in nd:
            nd[nk] = {'wins': 0, 'losses': 0, 'draws': 0}
        nd[nk]['wins'] += rec.get('wins', 0)
        nd[nk]['losses'] += rec.get('losses', 0)
        nd[nk]['draws'] += rec.get('draws', 0)
    if changed:
        gm['dualMem'] = nd
        save_global_memory()


def _seed_bad_lines_from_dual_mem():
    gm = _global_memory
    if not gm or not isinstance(gm.get('dualMem'), dict) or (gm.get('badLines') and len(gm['badLines']) > 0):
        return
    lines = []
    for dk, rec in gm['dualMem'].items():
        if not rec or not rec.get('losses'):
            continue
        sep = dk.find('|')
        if sep < 0:
            continue
        opp_arr = parse_moves_key(dk[:sep])
        ai_arr = parse_moves_key(dk[sep + 1:])
        if not opp_arr or not ai_arr:
            continue
        lines.append({'opp': opp_arr, 'ai': ai_arr, 'killType': None, 'at': 0})
    if lines:
        gm['badLines'] = lines
        save_global_memory()


# ---------------------------------------------------------------------------
# 结算：对局结束写入记忆（AI 视角）
# ---------------------------------------------------------------------------
def record_result(g):
    if g is None or g.winner is None or g.mode != 'ai' or not getattr(g, 'moves', None):
        return
    hp = g.human_player_()
    ai = g.ai_player()
    b = g.board()
    gm = global_memory()
    at = int(time.time() * 1000)
    opp_moves = [{'r': m['r'], 'c': m['c']} for m in g.moves if m['player'] == hp]
    ai_moves = [{'r': m['r'], 'c': m['c']} for m in g.moves if m['player'] == ai]
    win = ai if g.winner == ai else hp
    last = None
    for i in range(len(g.moves) - 1, -1, -1):
        if g.moves[i]['player'] == win:
            last = g.moves[i]
            break
    if last is None:
        return
    kt = loss_type_of(b, last['r'], last['c'], win)
    wl = win_line_of(b, last['r'], last['c'], win)
    if g.winner == ai:
        gm['totals']['wins'] = gm['totals'].get('wins', 0) + 1
        gm['goodLines'].append({'opp': opp_moves, 'ai': ai_moves, 'killType': None, 'winLine': wl, 'at': at})
    elif g.winner == hp:
        gm['totals']['losses'] = gm['totals'].get('losses', 0) + 1
        gm['lossByType'][kt] = gm['lossByType'].get(kt, 0) + 1
        gm['badLines'].append({'opp': opp_moves, 'ai': ai_moves, 'killType': kt, 'winLine': wl, 'at': at})
    else:
        gm['totals']['draws'] = gm['totals'].get('draws', 0) + 1
    save_global_memory()


# ---------------------------------------------------------------------------
# 记忆提示（给大模型/界面的文本）
# ---------------------------------------------------------------------------
def memory_hint(g):
    gm = global_memory()
    lines = []
    t = gm['totals']
    if t['wins'] + t['losses'] + t['draws'] > 0:
        lines.append('📊 全局战绩：%s 胜 %s 负 %s 平' % (t['wins'], t['losses'], t['draws']))
    type_parts = [k + '×' + str(v) for k, v in gm['lossByType'].items() if v > 0]
    if type_parts:
        lines.append('💀 历史败因：' + '、'.join(type_parts))
    if gm['dualMem']:
        hp = g.human_player_()
        opp_moves = [{'r': m['r'], 'c': m['c']} for m in g.moves if m['player'] == hp][:4]
        if opp_moves:
            total_losses = 0
            total_wins = 0
            for dk in gm['dualMem']:
                sep = dk.find('|')
                if sep < 0:
                    continue
                if sym_prefix_match(parse_moves_key(dk[:sep]), opp_moves):
                    total_losses += gm['dualMem'][dk].get('losses', 0)
                    total_wins += gm['dualMem'][dk].get('wins', 0)
            if total_losses > 0:
                lines.append('⚠️ 以这个开局（对手%d手）你赢过我 %d 次，谨慎' % (len(opp_moves), total_losses))
            if total_wins > 0:
                lines.append('✅ 以这个开局我赢过你 %d 次' % total_wins)
            ai_moves = [{'r': m['r'], 'c': m['c']} for m in g.moves if m['player'] == g.ai_player()][:4]
            if ai_moves:
                sum_losses = 0
                sum_wins = 0
                for dk in gm['dualMem']:
                    sep = dk.find('|')
                    if sep < 0:
                        continue
                    if sym_dual_prefix_match(parse_moves_key(dk[:sep]), parse_moves_key(dk[sep + 1:]), opp_moves, ai_moves) >= 0:
                        sum_losses += gm['dualMem'][dk].get('losses', 0)
                        sum_wins += gm['dualMem'][dk].get('wins', 0)
                if sum_losses > 0 and sum_losses > sum_wins:
                    lines.append('♻️ 当前双边应对与历史败局相同（%d 次），建议换一种下法' % sum_losses)
                elif sum_wins > 0 and sum_wins >= sum_losses:
                    lines.append('✅ 当前双边应对历史战绩占优，可继续')
        # 整局棋谱记忆
        if gm.get('badLines') or gm.get('goodLines'):
            ai_all = [{'r': m['r'], 'c': m['c']} for m in g.moves if m['player'] == g.ai_player()]
            bad_hits = 0
            good_hits = 0
            kill_count = {}
            for bl in gm.get('badLines') or []:
                if not bl or not isinstance(bl.get('opp'), list) or not isinstance(bl.get('ai'), list):
                    continue
                if sym_dual_prefix_match(bl['opp'], bl['ai'], opp_moves, ai_all) >= 0:
                    bad_hits += 1
                    kt = bl.get('killType') or '其他'
                    kill_count[kt] = kill_count.get(kt, 0) + 1
            for gl in gm.get('goodLines') or []:
                if not gl or not isinstance(gl.get('opp'), list) or not isinstance(gl.get('ai'), list):
                    continue
                if sym_dual_prefix_match(gl['opp'], gl['ai'], opp_moves, ai_all) >= 0:
                    good_hits += 1
            if bad_hits > 0:
                ks = '、'.join(k + '×' + str(v) for k, v in kill_count.items())
                lines.append('💀 历史教训：这个结构你败过 %d 次（%s），历史着法已降权，建议换路' % (bad_hits, ks))
            if good_hits > 0:
                lines.append('✅ 历史经验：这个结构你赢过 %d 次，历史着法已升权，可沿用' % good_hits)
        # 潜在线预警
        if gm.get('badLines'):
            b_cur = g.board()
            warns = []
            for bl in gm['badLines']:
                if not bl or not isinstance(bl.get('winLine'), list) or len(bl['winLine']) < 5:
                    continue
                w_h = _hidden_lines_of(b_cur, WHITE)
                b_h = _hidden_lines_of(b_cur, BLACK)
                for hl in w_h:
                    if win_line_overlap(bl['winLine'], hl['cells'], WHITE) >= 2:
                        warns.append('白线 ' + ''.join(COLUMNS[x['c']] + str(x['r'] + 1) for x in hl['cells']) + '（%d子）' % hl['cnt'])
                for hl in b_h:
                    if win_line_overlap(bl['winLine'], hl['cells'], BLACK) >= 2:
                        warns.append('黑线 ' + ''.join(COLUMNS[x['c']] + str(x['r'] + 1) for x in hl['cells']) + '（%d子）' % hl['cnt'])
            uniq = []
            for w in warns:
                if w not in uniq:
                    uniq.append(w)
            if uniq:
                lines.append('🚨 潜在线预警：' + '；'.join(uniq[:3]) + ' 与你上次的杀线重合，慎防重蹈覆辙')
    return '\n'.join(lines)
