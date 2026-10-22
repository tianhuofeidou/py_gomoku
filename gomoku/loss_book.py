# -*- coding: utf-8 -*-
"""败局回归库：收录 AI 败局、定位疑似关键败手、在新版本上从附近节点重赛。

设计约束（来自项目规范）：

- **不另造选点或判胜**：重赛复用 ``service.ai_move``，判胜复用 ``game.check_win``，
  对称归一化复用 ``memory.canon_key``，复盘复用 ``tools.review.review_game``。
- **不把"最后一手/对方成五点/首次全候选必败"直接当败手**：定位只产出"疑似节点 + 理由"，
  定位不出时如实返回空列表并给出 ``undetermined`` 原因，允许人工指定手数。
- **五态只描述有限搜索**：任何结论都带 ``bounded=True``，不得表述为全局证明。
- **回放分叉不伪造**：旧谱落点被占、谱尽未终局、预算耗尽都记为"未完成"，
  只在真实胜局或合法终局和棋时才允许移入归档。
- **数据只写用户存档目录**：``paths.base_dir()/loss-book.json``，可用 ``GOMOKU_HOME`` 覆盖。
- 降级/跳过必须可见：所有异常路径都写 ``kind`` + ``detail``，不静默吞掉。
"""
import json
import os
import time
from uuid import uuid4

from gomoku import memory, paths
from gomoku.core.utils import empty_board, place
from gomoku.game import BLACK, WHITE, check_win

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

FILE_VERSION = 1
RULE_ID = 'freestyle15'          # 本项目正式规则：无禁手、长连算胜

# 结局分类：正常棋力失败与异常必须分开
KIND_LOSS = 'loss'               # 正常下完但输了
KIND_WIN = 'win'                 # 复核用：正常赢（不收录，仅测试可写入）
KIND_DRAW = 'draw'               # 合法终局和棋（满盘）
KIND_TIMEOUT = 'timeout'         # 超时
KIND_PROTOCOL = 'protocol-error'  # 协议异常
KIND_ILLEGAL = 'illegal-move'    # 非法落子
KIND_ABORTED = 'aborted'         # 截断/异常中断

NORMAL_KINDS = (KIND_LOSS,)
ABNORMAL_KINDS = (KIND_TIMEOUT, KIND_PROTOCOL, KIND_ILLEGAL, KIND_ABORTED)

# 重赛结论
RETEST_LOSS = 'loss'                       # 仍失败
RETEST_WIN = 'win'                         # 真实胜局
RETEST_DRAW = 'draw'                       # 合法终局和棋
RETEST_DIVERGED = 'diverged'               # 旧败线改变，待续测（旧谱落点已被占）
RETEST_SCRIPT_END = 'script-exhausted'     # 原棋谱用尽但没有终局
RETEST_BUDGET = 'budget-exhausted'         # 达到测试预算或手数上限
RETEST_ERROR = 'error'                     # 异常/非法落子/超时

# 只有这两种才能移出待处理库
PASSING_RETESTS = (RETEST_WIN, RETEST_DRAW)


# ---------------------------------------------------------------------------
# 存储
# ---------------------------------------------------------------------------

def store_file():
    return os.path.join(paths.base_dir(), 'loss-book.json')


def _empty_store():
    return {'version': FILE_VERSION, 'cases': []}


def load_store():
    """读库；文件不存在返回空库。损坏时抛错，不静默重建。"""
    fp = store_file()
    if not os.path.exists(fp):
        return _empty_store()
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('loss-book.json 结构异常：顶层不是对象')
    cases = data.get('cases')
    if not isinstance(cases, list):
        raise ValueError('loss-book.json 结构异常：cases 不是数组')
    data.setdefault('version', FILE_VERSION)
    return data


def save_store(data):
    """原子写入：先写 .tmp 再替换，避免半截文件。"""
    paths.ensure_base()
    fp = store_file()
    tmp = fp + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, fp)
    return fp


# ---------------------------------------------------------------------------
# 指纹
# ---------------------------------------------------------------------------

def _engine_fingerprint():
    """引擎代码指纹：只取"会改变选点"的源码文件，避免把无关文件算进来。

    用相对路径 + 内容哈希，保证跨机器可比较；读不到的文件记为缺失而不是跳过。
    """
    import hashlib
    root = os.path.dirname(os.path.abspath(__file__))
    files = [
        'core/pattern.py',
        'core/search.py',
        'core/deep_search.py',
        'core/engine.py',
        'core/opening_book.py',
        'core/opening_book_data.py',
        'core/utils.py',
    ]
    h = hashlib.sha256()
    missing = []
    for rel in files:
        fp = os.path.join(root, *rel.split('/'))
        if not os.path.exists(fp):
            missing.append(rel)
            continue
        h.update(rel.encode('utf-8'))
        with open(fp, 'rb') as f:
            h.update(f.read())
    return {'digest': h.hexdigest()[:16], 'missing': missing, 'files': len(files) - len(missing)}


def _book_fingerprint():
    """开局库数据指纹：库变了，开局阶段的选择就可能不同，复测要重跑。"""
    import hashlib
    root = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(root, 'core', 'opening_book_data.py')
    if not os.path.exists(fp):
        return None
    with open(fp, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def current_fingerprint():
    return {'engine': _engine_fingerprint(), 'book': _book_fingerprint(),
            'rule': RULE_ID, 'at': int(time.time() * 1000)}


# ---------------------------------------------------------------------------
# 棋谱 / 键
# ---------------------------------------------------------------------------

def normalize_moves(moves):
    """把各种来源的着法统一成 [{'r','c','player'}]；非法输入抛错。"""
    out = []
    for m in moves or []:
        if isinstance(m, dict):
            r, c, p = m.get('r'), m.get('c'), m.get('player')
        elif isinstance(m, (list, tuple)) and len(m) >= 3:
            r, c, p = m[0], m[1], m[2]
        else:
            raise ValueError('无法识别的着法：%r' % (m,))
        if not (isinstance(r, int) and isinstance(c, int) and p in (BLACK, WHITE)):
            raise ValueError('着法字段非法：%r' % (m,))
        out.append({'r': r, 'c': c, 'player': p})
    return out


def split_by_color(moves, ai_color):
    """拆成本方（AI）/ 对方两串，用于对称键。"""
    ours = [{'r': m['r'], 'c': m['c']} for m in moves if m['player'] == ai_color]
    theirs = [{'r': m['r'], 'c': m['c']} for m in moves if m['player'] != ai_color]
    return ours, theirs


def record_key(moves, ai_color):
    """对称归一化键：同一棋谱的旋转/镜像得到同一键。"""
    ours, theirs = split_by_color(moves, ai_color)
    return memory.canon_key(theirs, ours)


# ---------------------------------------------------------------------------
# 收录
# ---------------------------------------------------------------------------

def normalize_winner(winner):
    """把各种来源的胜负表示统一成 BLACK / WHITE / 0（和棋）/ None（未知）。

    对战台（``tools/match.py``）输出的是 ``'B'`` / ``'W'`` / ``'draw'`` / ``None``，
    而 ``Game`` 里用的是 ``BLACK``(=1) / ``WHITE``(=2) / 0。两者必须都认，
    否则会出现"胜局被当成败局"这类静默错误。
    """
    if winner is None:
        return None
    if isinstance(winner, str):
        key = winner.strip().lower()
        if key in ('b', 'black', '1'):
            return BLACK
        if key in ('w', 'white', '2'):
            return WHITE
        if key in ('draw', 'd', '0', 'tie'):
            return 0
        raise ValueError('无法识别的胜负表示：%r' % (winner,))
    if winner in (BLACK, WHITE, 0):
        return int(winner)
    raise ValueError('无法识别的胜负表示：%r' % (winner,))


def _result_kind(winner, ai_color):
    if winner is None:
        return KIND_ABORTED
    if winner == 0:
        return KIND_DRAW
    return KIND_WIN if winner == ai_color else KIND_LOSS


def add_case(moves, ai_color, *, winner=None, kind=None, source='live',
             source_detail=None, started_at=None, ended_at=None,
             config=None, fingerprint=None, note=None, store=None):
    """收录一局。

    - ``kind`` 未给时按胜负推断；异常结局（超时/协议/非法）必须显式传入。
    - 相同 ``recordKey``（含旋转镜像）的棋谱只保留一份 moves，
      但**每次都追加一条来源记录**，不丢历史。
    - 返回 ``(case, is_new)``；``case['sources'][-1]['id']`` 是本条来源 id，
      悔棋撤销时用它精确删除，不会误删同棋谱的其它失败。
    """
    ai_color = int(ai_color)
    if ai_color not in (BLACK, WHITE):
        raise ValueError('aiColor 必须是 BLACK/WHITE')
    moves = normalize_moves(moves)
    winner = normalize_winner(winner)
    kind = kind or _result_kind(winner, ai_color)
    if kind not in (KIND_LOSS, KIND_WIN, KIND_DRAW) + ABNORMAL_KINDS:
        raise ValueError('未知结局类型：%r' % (kind,))

    data = store if store is not None else load_store()
    key = record_key(moves, ai_color)
    now = int(time.time() * 1000)
    fp = fingerprint or current_fingerprint()

    src = {
        'id': uuid4().hex,
        'source': source,
        'sourceDetail': source_detail,
        'kind': kind,
        'winner': winner,
        'startedAt': started_at,
        'endedAt': ended_at if ended_at is not None else now,
        'addedAt': now,
        'fingerprint': fp,
        'config': dict(config or {}),
        'note': note,
    }

    for case in data['cases']:
        # kind 参与去重：异常结局（超时/协议/非法/截断）不能并入正常棋力失败。
        if (case.get('recordKey') == key and case.get('aiColor') == ai_color
                and case.get('kind') == kind):
            case['sources'].append(src)
            case['updatedAt'] = now
            save_store(data)
            return case, False

    case = {
        'id': uuid4().hex,
        'recordKey': key,
        'aiColor': ai_color,
        'rule': RULE_ID,
        'moves': moves,
        'winner': winner,
        'kind': kind,
        'fingerprint': fp,
        'config': dict(config or {}),
        'sources': [src],
        'suspects': [],
        'suspectsUndetermined': None,
        'manualPly': None,
        'retests': [],
        'status': 'pending',
        'createdAt': now,
        'updatedAt': now,
    }
    data['cases'].append(case)
    save_store(data)
    return case, True


def remove_source(source_id, store=None):
    """撤销一条来源记录（悔棋撤销终局时用）。

    只删这一条来源；来源清空后整案删除。**不会**因为棋谱相同而连带删别的失败记录。
    返回被删的 case（若整案被删）或 None。
    """
    data = store if store is not None else load_store()
    for i, case in enumerate(data['cases']):
        srcs = case.get('sources') or []
        hit = [s for s in srcs if s.get('id') == source_id]
        if not hit:
            continue
        case['sources'] = [s for s in srcs if s.get('id') != source_id]
        case['updatedAt'] = int(time.time() * 1000)
        removed_case = None
        if not case['sources']:
            removed_case = data['cases'].pop(i)
        save_store(data)
        return removed_case
    return None


# ---------------------------------------------------------------------------
# 疑似关键败手定位
# ---------------------------------------------------------------------------

def _our_ply_indices(moves, ai_color):
    """本方每次落子在 moves 里的下标（0 基）。"""
    return [i for i, m in enumerate(moves) if m['player'] == ai_color]


def locate_suspects(case, *, start_ply=6, max_suspects=5, engine_factory=None):
    """定位疑似关键败手。

    **不**把"最后一手""对方成五点""首次全候选必败"直接当败手，而是按以下优先级收集：

    1. ``missed-five``：漏掉直接成五，或漏掉必须堵的五；
    2. ``missed-save``：实际着法被判必败，而同时存在未判必败的其它候选；
    3. ``deterioration``：局面明显恶化之前的本方着法
       （用"实际着法档位"首次变成必败的前一个本方回合）。

    返回 ``(suspects, undetermined_reason)``。定位不到时 suspects 为 []，
    reason 说明为什么（例如局面为空、引擎不可用、全程未出现必败信号）。
    """
    moves = case.get('moves') or []
    ai_color = case.get('aiColor')
    if not moves or ai_color not in (BLACK, WHITE):
        return [], 'no-moves'
    if not any(m['player'] == ai_color for m in moves):
        return [], 'ai-never-moved'

    try:
        from gomoku.tools.review import review_game
    except Exception as e:            # 可见地报告，不静默
        return [], 'review-unavailable: %s' % e

    game = {'moves': moves, 'a_white': (ai_color == WHITE)}
    try:
        rows, first_all_lost, missed = review_game(game, side='a', start_ply=start_ply)
    except Exception as e:
        return [], 'review-failed: %s' % e

    by_ply = {r['ply']: r for r in rows}
    suspects = []

    # 优先级 1：漏五类
    from gomoku.core.deep_search import DeepSearch
    lose = DeepSearch.STATE_LOSE
    win = DeepSearch.STATE_WIN

    for ply, row in sorted(by_ply.items()):
        states = (row.get('cand_states') or '').split('/')
        if row.get('top_state') == '必胜' and row.get('played_state') != '必胜':
            suspects.append({
                'ply': ply, 'move': row['move'], 'reason': 'missed-five',
                'detail': '引擎首选为必胜着法（%s），实际走了 %s' % (row['top'], row['move']),
                'top': row['top'], 'playedState': row.get('played_state'),
            })

    # 优先级 2：漏掉可守的候选
    if missed:
        ply, save_move, played = missed
        suspects.append({
            'ply': ply, 'move': played, 'reason': 'missed-save',
            'detail': '实际着法判必败，但存在未判必败的候选 %s' % save_move,
            'save': save_move,
        })

    # 优先级 3：恶化之前的本方着法
    first_bad = None
    for ply, row in sorted(by_ply.items()):
        if row.get('played_state') == '必败':
            first_bad = ply
            break
    if first_bad is not None:
        plies = sorted(by_ply)
        idx = plies.index(first_bad)
        if idx > 0:
            prev = plies[idx - 1]
            if not any(s['ply'] == prev for s in suspects):
                suspects.append({
                    'ply': prev, 'move': by_ply[prev]['move'], 'reason': 'deterioration',
                    'detail': '下一手（第 %d 手）实际着法已判必败，本手是恶化前最后一次本方落子' % first_bad,
                })

    # 去重 + 限流，保留最先出现的（手数早的）
    seen = set()
    uniq = []
    for s in sorted(suspects, key=lambda x: x['ply']):
        if s['ply'] in seen:
            continue
        seen.add(s['ply'])
        uniq.append(s)
    uniq = uniq[:max_suspects]

    if uniq:
        return uniq, None
    if first_all_lost:
        return [], 'only-all-lost-at-%d' % first_all_lost
    return [], 'no-signal'


def set_suspects(case, suspects=None, undetermined=None, manual_ply=None, store=None):
    """把定位结果写回案例；允许人工指定手数（manual_ply）。"""
    data = store if store is not None else load_store()
    for c in data['cases']:
        if c.get('id') != case.get('id'):
            continue
        c['suspects'] = list(suspects or [])
        c['suspectsUndetermined'] = undetermined
        c['manualPly'] = manual_ply
        c['updatedAt'] = int(time.time() * 1000)
        save_store(data)
        return c
    return None


# ---------------------------------------------------------------------------
# 从关键节点附近重赛
# ---------------------------------------------------------------------------

def retest_starts(case, manual_ply=None):
    """给出重赛起点（1 基手数）：疑似败手、前一个本方回合、前两个本方回合。"""
    moves = case.get('moves') or []
    ai_color = case.get('aiColor')
    plies = _our_ply_indices(moves, ai_color)     # 0 基下标
    if not plies:
        return []
    if manual_ply is not None:
        # 人工指定的是"第几次本方落子"（1 基），换算成 moves 下标
        k = int(manual_ply)
        if not (1 <= k <= len(plies)):
            raise ValueError('人工手数超出范围：%d（本方共 %d 手）' % (k, len(plies)))
        anchor = plies[k - 1]
    else:
        sus = case.get('suspects') or []
        if not sus:
            return []
        # suspects 里的 ply 是 moves 的 1 基序号，换成下标
        anchor = int(sus[0]['ply']) - 1
        if anchor not in plies:
            # 兜底：取不超过 anchor 的最后一个本方手
            cand = [p for p in plies if p <= anchor]
            if not cand:
                return []
            anchor = cand[-1]
    idx = plies.index(anchor)
    outs = []
    for back in (0, 1, 2):
        j = idx - back
        if j < 0:
            continue
        outs.append({'startPly': plies[j] + 1, 'startIndex': plies[j], 'back': back})
    return outs


def _restore_prefix(moves, upto_index, engine_factory):
    """真实恢复前缀：逐手 place + engine.on_move，返回 (board, engine, error)。

    不做"只摆最后几颗子"的偷懒——每一步都过一遍状态同步。
    """
    engine = engine_factory()
    board = empty_board()
    for i in range(upto_index):
        m = moves[i]
        if board[m['r']][m['c']] != 0:
            return None, None, 'prefix-conflict-at-%d' % (i + 1)
        place(board, m['r'], m['c'], m['player'])
        engine.on_move(board, m['r'], m['c'], m['player'])
    return board, engine, None


def _terminal(board, last_move):
    """判定终局：成五返回胜者颜色；满盘无五连返回 0（合法和棋）；否则 None。

    只检查成五是不够的——本项目没有"禁手/超步"等早停规则，
    唯一可能的终局就是成五或下满。
    """
    if not last_move:
        return None
    if check_win(board, last_move['r'], last_move['c'], last_move['player']):
        return last_move['player']
    for row in board:
        for cell in row:
            if cell == 0:
                return None
    return 0


def retest_case(case, *, start_ply=None, manual_ply=None, max_plies=None, batch_total=None, batch_id=None,
                ai_move_fn=None, engine_factory=None, store=None,
                fingerprint=None, stop_after_first=True):
    """在"当前版本"上从某个起点重赛。

    - AI 每次轮到自己都重新选点（``ai_move_fn``，默认 ``service.ai_move``）；
    - 对手按原棋谱对应回合落子；
    - 旧谱落点被占 → ``diverged``；谱尽无终局 → ``script-exhausted``；
      预算/手数耗尽 → ``budget-exhausted``；异常 → ``error``。
    - **不**为了走完而跳过非法旧谱落点，也不编造更弱的对手。

    结果追加到存储中的 retests；CLI 为整轮传入 batch_id，独立调用会新建一轮。
    一轮要求的全部起点未测齐时保留 pending，旧轮结果只保留作历史。
    """
    # 只有需要真引擎时才 import service：mock 测试与"纯逻辑校验"不该被
    # 引擎模块的 import 失败拖累（例如开局库数据暂时不一致时）。
    moves = case.get('moves') or []
    ai_color = case.get('aiColor')
    if ai_move_fn is None or engine_factory is None:
        from gomoku import service
        if ai_move_fn is None:
            ai_move_fn = service.ai_move
        if engine_factory is None:
            engine_factory = service.create_engine

    starts = retest_starts(case, manual_ply=manual_ply)
    if not starts:
        return {'result': RETEST_ERROR, 'detail': 'no-start-point', 'startPly': None}
    chosen = starts[0]
    if start_ply is not None:
        hit = [s for s in starts if s['startPly'] == start_ply]
        if not hit:
            return {'result': RETEST_ERROR,
                    'detail': 'start-ply-not-in-candidates: %s' % [s['startPly'] for s in starts],
                    'startPly': start_ply}
        chosen = hit[0]

    if batch_id is None:
        batch = begin_retest_batch(case, starts, fingerprint=fingerprint, store=store)
        batch_id = batch['id']
    else:
        batch = case.get('retestBatch') or {}
        if batch.get('id') != batch_id:
            raise ValueError('复测轮次与案例不匹配')
    fingerprint = batch['fingerprint']

    start_index = chosen['startIndex']
    budget = max_plies if max_plies is not None else max(60, len(moves) * 2)

    board, engine, err = _restore_prefix(moves, start_index, engine_factory)
    if err:
        return _record_retest(case, {
            'startPly': chosen['startPly'], 'result': RETEST_ERROR, 'detail': err,
            'plies': 0, 'fingerprint': fingerprint or current_fingerprint(),
        }, store, batch_total=batch_total, batch_id=batch_id)

    # 从起点开始：先让"轮到的一方"走
    last = moves[start_index - 1] if start_index > 0 else None
    cur = (last['player'] % 2 + 1) if last else BLACK

    # 对手着法是独立序列：重赛里对手只按原谱"自己的回合"落子，
    # 不能直接用 moves 下标（那里面混着 AI 自己的着法）。
    opp_moves = [m for m in moves if m['player'] != ai_color]
    opp_idx = sum(1 for m in moves[:start_index] if m['player'] != ai_color)
    played = 0
    # 重赛的真实历史：前缀 + 重新走出的着法。给 ai_move 用的是这一串，
    # 不是原谱——否则开局库与重复局面判断会看到不存在的着法。
    played_moves = list(moves[:start_index])

    while True:
        term = _terminal(board, last)
        if term is not None:
            res = RETEST_DRAW if term == 0 else (RETEST_WIN if term == ai_color else RETEST_LOSS)
            return _record_retest(case, {
                'startPly': chosen['startPly'], 'result': res, 'detail': 'terminal',
                'winner': term, 'plies': played,
                'fingerprint': fingerprint or current_fingerprint(),
            }, store, batch_total=batch_total, batch_id=batch_id)

        # 最后一手可能恰好成五或填满棋盘，必须先判终局再检查预算。
        if played >= budget:
            break

        if cur == ai_color:
            try:
                mv, _ranked, _net = ai_move_fn(board, played_moves, cur, engine=engine)
            except Exception as e:
                return _record_retest(case, {
                    'startPly': chosen['startPly'], 'result': RETEST_ERROR,
                    'detail': 'ai-move-failed: %s' % e, 'plies': played,
                    'fingerprint': fingerprint or current_fingerprint(),
                }, store, batch_total=batch_total, batch_id=batch_id)
            if not mv:
                return _record_retest(case, {
                    'startPly': chosen['startPly'], 'result': RETEST_ERROR,
                    'detail': 'ai-move-empty', 'plies': played,
                    'fingerprint': fingerprint or current_fingerprint(),
                }, store, batch_total=batch_total, batch_id=batch_id)
            r, c = mv[0], mv[1]
            if board[r][c] != 0:
                return _record_retest(case, {
                    'startPly': chosen['startPly'], 'result': RETEST_ERROR,
                    'detail': 'ai-move-occupied: (%d,%d)' % (r, c), 'plies': played,
                    'fingerprint': fingerprint or current_fingerprint(),
                }, store, batch_total=batch_total, batch_id=batch_id)
        else:
            if opp_idx >= len(opp_moves):
                return _record_retest(case, {
                    'startPly': chosen['startPly'], 'result': RETEST_SCRIPT_END,
                    'detail': '对手旧谱已用尽，但未终局', 'plies': played,
                    'fingerprint': fingerprint or current_fingerprint(),
                }, store, batch_total=batch_total, batch_id=batch_id)
            r, c = opp_moves[opp_idx]['r'], opp_moves[opp_idx]['c']
            if board[r][c] != 0:
                return _record_retest(case, {
                    'startPly': chosen['startPly'], 'result': RETEST_DIVERGED,
                    'detail': '旧败线改变：对手第 %d 手旧谱落点 (%d,%d) 已被占' % (opp_idx + 1, r, c),
                    'plies': played, 'divergedAt': opp_idx + 1,
                    'fingerprint': fingerprint or current_fingerprint(),
                }, store, batch_total=batch_total, batch_id=batch_id)
            opp_idx += 1

        place(board, r, c, cur)
        engine.on_move(board, r, c, cur)
        played_moves.append({'r': r, 'c': c, 'player': cur})
        last = {'r': r, 'c': c, 'player': cur}
        cur = cur % 2 + 1
        played += 1

    return _record_retest(case, {
        'startPly': chosen['startPly'], 'result': RETEST_BUDGET,
        'detail': '达到手数上限 %d，未终局' % budget, 'plies': played,
        'fingerprint': fingerprint or current_fingerprint(),
    }, store, batch_total=batch_total, batch_id=batch_id)


def begin_retest_batch(case, starts, *, fingerprint=None, store=None):
    """在计算前持久化本轮起点集合和 pending；进程中断也不会沿用旧轮通过状态。"""
    expected = [s['startPly'] for s in starts]
    if not expected or any(type(p) is not int or p < 1 for p in expected) or len(set(expected)) != len(expected):
        raise ValueError('本轮起点必须是非空且互不重复的合法手数')
    batch = {'id': uuid4().hex, 'expectedStarts': expected,
             'fingerprint': fingerprint or current_fingerprint(), 'at': int(time.time() * 1000)}
    data = load_store() if store is None else store
    for c in data['cases']:
        if c.get('id') == case.get('id'):
            c['retestBatch'] = batch
            c['status'] = 'pending'
            c['updatedAt'] = batch['at']
            # 调用方可能持有另一次 load 的对象，CLI 后续起点需要相同的轮次身份。
            case['retestBatch'] = batch
            case['status'] = 'pending'
            save_store(data)
            return batch
    raise ValueError('找不到待复测案例')


def _batch_results(case, fingerprint):
    """取"同一版本批次"的复测结果。

    判定只看**当前指纹这一批**：旧版本上失败过，不该让新版本永远不能通过；
    反过来，旧版本赢过也不该让新版本蒙过。历史记录原样保留在 retests 里。
    """
    if not fingerprint:
        return list(case.get('retests') or [])
    out = []
    for r in case.get('retests') or []:
        rf = r.get('fingerprint')
        # 没有记录指纹的结果（直接调用 _record_retest 的场景）归入当前批次。
        if rf is None or _fp_matches(rf, fingerprint):
            out.append(r)
    return out


def evaluate_status(case, fingerprint=None, batch_total=None):
    """当前独立轮次必须覆盖所有不同起点且全部通过；旧轮和重复记录不能凑数。

    - 该批次**至少一条**且**全部**通过 → ``passed``；
    - 当前轮次按 expectedStarts 判断覆盖，不按累计记录条数判断；
    - 无轮次标记的旧记录沿用指纹筛选，但同样按不同起点计数；
    - 否则 ``pending``。
    """
    active = case.get('retestBatch')
    if active:
        batch = [r for r in case.get('retests', []) if r.get('batchId') == active['id']]
        expected = set(active['expectedStarts'])
        if ({r.get('startPly') for r in batch} != expected
                or any(not _fp_matches(r.get('fingerprint'), active['fingerprint']) for r in batch)):
            return 'pending'
    else:
        batch = _batch_results(case, fingerprint or current_fingerprint())
    if not batch:
        return 'pending'
    if batch_total is not None and len({r.get('startPly') for r in batch}) < int(batch_total):
        return 'pending'
    if all(r.get('result') in PASSING_RETESTS for r in batch):
        return 'passed'
    return 'pending'


def _record_retest(case, result, store, batch_total=None, batch_id=None):
    """保存结果，按当前轮次重算状态；迟到的旧轮结果只能追加历史。

    ``store`` 为 None 时自行读改写（CLI 直接调用不必关心事务）；
    传入时复用调用方的事务上下文，避免重复读写。
    ``batch_total`` 是本轮应测的起点数；给出时未测齐不会标 passed。
    """
    result.setdefault('at', int(time.time() * 1000))
    if batch_id is not None:
        result['batchId'] = batch_id
    own = store is None
    data = load_store() if own else store
    for c in data['cases']:
        if c.get('id') != case.get('id'):
            continue
        rets = c.setdefault('retests', [])
        rets.append(result)
        c['updatedAt'] = result['at']
        # 状态只由"本批次"决定，历史记录保留但不参与判定。
        c['status'] = evaluate_status(c, result.get('fingerprint'), batch_total=batch_total)
        if own:
            save_store(data)
        break
    return result


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------

def _fp_matches(a, b):
    """比较两个指纹是否"引擎与开局库都相同"。"""
    if not a or not b:
        return False
    eng_a = (a.get('engine') or {}).get('digest')
    eng_b = (b.get('engine') or {}).get('digest')
    return eng_a is not None and eng_a == eng_b and a.get('book') == b.get('book')


def last_tested_fingerprint(case):
    """案例最近一次复测所用的指纹。

    优先取最后一条 retest 的指纹；没有复测过就退回收录时的指纹。
    **不能用收录指纹判断"是否已测过当前版本"**——收录之后可能已经复测过多轮。
    """
    rets = case.get('retests') or []
    for r in reversed(rets):
        fp = r.get('fingerprint')
        if fp:
            return fp
    return case.get('fingerprint')


def list_cases(*, status=None, kind=None, fingerprint=None, store=None):
    """列出案例。``fingerprint`` 给定时只返回"最近一次复测的指纹与当前不同"的待测案例。"""
    data = store if store is not None else load_store()
    out = []
    for c in data['cases']:
        if status and c.get('status') != status:
            continue
        if kind and c.get('kind') != kind:
            continue
        if fingerprint and _fp_matches(last_tested_fingerprint(c), fingerprint):
            continue
        out.append(c)
    return out


def force_pending(case, reason, store=None):
    """把案例强制退回 pending，并记录原因。

    用于"本批没测齐所有起点却已经算出 passed"的情形——
    部分通过不能当作通过。
    """
    own = store is None
    data = load_store() if own else store
    out = None
    for c in data['cases']:
        if c.get('id') != case.get('id'):
            continue
        c['status'] = 'pending'
        c['updatedAt'] = int(time.time() * 1000)
        c['lastIncomplete'] = {'reason': reason, 'at': c['updatedAt']}
        if own:
            save_store(data)
        out = c
        break
    return out


def archive_passed(store=None):
    """把已通过的案例移到归档区（保留原棋谱与复测历史，不删除）。"""
    data = store if store is not None else load_store()
    moved = []
    keep = []
    for c in data['cases']:
        if c.get('status') == 'passed':
            c['archivedAt'] = int(time.time() * 1000)
            moved.append(c)
        else:
            keep.append(c)
    data['cases'] = keep
    data.setdefault('archive', []).extend(moved)
    if moved:
        save_store(data)
    return moved
