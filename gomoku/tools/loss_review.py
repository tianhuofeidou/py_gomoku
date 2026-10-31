"""离线败手诊断：规则事实、教师同层对照与逐档复核，不参与正式选棋。

Rapfi normal-v1 原始缓存沿用离线实验的身份与摘要格式；本模块不依赖调参工具。
M信号是有限搜索结论；普通分、未判败、单条PV均不构成全盘不败证明。
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import tomllib
from uuid import uuid4

from gomoku.core.utils import empty_board
from gomoku.game import check_win


class ReviewStop(RuntimeError):
    """预算、缓存缺项等可观察的未完成原因。"""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        with tmp.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def board_of(moves):
    """严格验证完整交替棋谱；独立盘面不绑定Search，不触碰调用方状态表。"""
    board = empty_board()
    ended = False
    for i, move in enumerate(moves):
        if (not isinstance(move, (list, tuple)) or len(move) != 3
                or any(type(x) is not int for x in move)):
            raise ValueError('棋谱必须为零基[r,c,player]整数数组')
        r, c, player = move
        if ended or not (0 <= r < 15 and 0 <= c < 15) or player != i % 2 + 1 or board[r][c]:
            raise ValueError('棋谱越界、重复、轮次错误或终局后继续落子')
        board[r][c] = player
        ended = check_win(board, r, c, player)
    return board, ended


def immediate_wins(board, player):
    """只在隔离盘面检查真实成五；finally恢复，不借用引擎的有限五态。"""
    wins = []
    for r in range(15):
        for c in range(15):
            if board[r][c] == 0:
                board[r][c] = player
                try:
                    if check_win(board, r, c, player):
                        wins.append([r, c])
                finally:
                    board[r][c] = 0
    return wins


def cd(move):
    return chr(65 + move[1]) + str(move[0] + 1)


def value(raw):
    if not isinstance(raw, str) or not re.fullmatch(r'[+-]?\d+|[+-]M\d+', raw):
        raise ValueError('教师评价格式非法')
    if 'M' in raw:
        return dict(kind='mate', raw=raw, utility=-1.0 if raw.startswith('-') else 1.0)
    return dict(kind='cp', raw=raw, utility=.9 * math.tanh(int(raw) / 600.0))


def parse_eval(messages):
    tokens = [m.group(1) for line in messages
              if (m := re.search(r'\bEval\s+([+-]?M\d+|[+-]?\d+)(?=\s|\||$)', line))]
    if not tokens:
        raise ValueError('教师缺少Eval；需要Rapfi normal输出')
    return value(tokens[-1])


def flip(evaluation):
    """子局面轮到对手，将零和标量评价转回落子方；不翻转本方引擎五态。"""
    raw = evaluation['raw']
    raw = (('-' if raw.startswith('+') else '+') + raw[1:]) if 'M' in raw else str(-int(raw))
    return dict(evaluation, raw=raw, utility=-evaluation['utility'])


def pv_info(label, moves):
    """仅取同分、同根着法的PV，拒绝重复、越界和终局后续着；不称强制证明。"""
    names = []
    for line in label['messages']:
        m = re.search(r'Time\s+\d+(?:\.\d+)?(?:ms|s)\s*\|\s*((?:[A-O](?:1[0-5]|[1-9])\s*)+)$', line)
        score = re.search(r'Eval\s+([^\s|]+)', line)
        if m and score and score.group(1) == label['evaluation']['raw']:
            proposed = re.findall(r'[A-O](?:1[0-5]|[1-9])', m.group(1))
            if proposed and proposed[0] == cd(label['move']):
                names = proposed
    if not names:
        return dict(valid=None, moves=[], detail='未提供匹配PV；不作变化证明')
    board, ended = board_of(moves)
    player = len(moves) % 2 + 1
    for name in names:
        r, c = int(name[1:]) - 1, ord(name[0]) - 65
        if ended or board[r][c]:
            return dict(valid=False, moves=[], detail='拒绝PV：重复/已占点或终局后续着')
        board[r][c] = player
        ended = check_win(board, r, c, player)
        player = 3 - player
    return dict(valid=True, moves=names, terminal=bool(ended), detail='仅核验单条PV合法性')


class RapfiReviewer:
    """每次查询独立进程、同预算对照；全任务最多4并发并共享时间/查询额度。"""
    def __init__(self, exe, cache_dir, *, seconds=600, workers=4, max_queries=2000,
                 cache_only=False):
        self.exe = Path(exe).resolve()
        if not self.exe.is_file():
            raise ValueError('教师可执行文件不存在')
        if not math.isfinite(seconds) or seconds <= 0 or workers not in range(1, 5) or max_queries < 1:
            raise ValueError('教师预算必须为正，workers为1至4')
        config = self.exe.parent / 'config.toml'
        parsed = tomllib.loads(config.read_text(encoding='utf-8')) if config.exists() else {}
        general = parsed.get('general', {})
        if (general.get('message_mode', 'normal') != 'normal'
                or general.get('coord_conversion_mode', 'none') != 'none'
                or parsed.get('database', {}).get('enable_by_default', False)):
            raise ValueError('教师须normal输出、none坐标转换且关闭数据库')
        files = [self.exe] + sorted(p for p in self.exe.parent.iterdir()
                                   if p.suffix.lower() in ('.toml', '.bin', '.lz4', '.nnue'))
        self.files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        self.cache = Path(cache_dir)
        self.deadline = time.monotonic() + seconds
        self.workers, self.max_queries, self.cache_only = workers, max_queries, cache_only
        self.queries = 0
        self._lock_path = None
        self._nonce = uuid4().hex
        self._guard = threading.Lock()

    def identity(self, nodes):
        milliseconds = {200000: 15000, 1000000: 60000, 4000000: 120000,
                        16000000: 240000}.get(nodes, max(15000, min(240000, nodes // 30)))
        return dict(adapter='rapfi-normal-v1', rule='freestyle15', threads=1,
                    nodes=nodes, milliseconds=milliseconds, files=self.files)

    def _path(self, moves, nodes):
        return self.cache / ('teacher-' + str(nodes)) / 'labels' / (digest([self.identity(nodes), moves]) + '.json')

    def __enter__(self):
        if not self.cache_only and self._lock_path is None:
            self.cache.mkdir(parents=True, exist_ok=True)
            path = self.cache / 'loss-review.lock'
            try:
                with path.open('x', encoding='utf-8') as stream:
                    json.dump(dict(pid=os.getpid(), nonce=self._nonce), stream)
            except FileExistsError:
                raise ReviewStop('缓存目录已有loss-review.lock；确认旧任务退出后处理，不自动抢锁') from None
            self._lock_path = path
        return self

    def close(self):
        if self._lock_path is not None:
            path, self._lock_path = self._lock_path, None
            if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('nonce') == self._nonce:
                path.unlink()

    def __exit__(self, *args):
        self.close()

    def read(self, moves, nodes):
        path = self._path(moves, nodes)
        if not path.exists():
            raise ReviewStop('教师缓存缺项：%s' % path.name)
        data = json.loads(path.read_text(encoding='utf-8'))
        checksum = data.pop('payload_sha256', None)
        if (checksum != digest(data) or data.get('identity') != self.identity(nodes)
                or data.get('key') != path.stem):
            raise ValueError('教师缓存身份或摘要损坏')
        board, ended = board_of(moves)
        move = data.get('move')
        if (ended or not isinstance(move, list) or len(move) != 2
                or any(type(x) is not int or not 0 <= x < 15 for x in move)
                or board[move[0]][move[1]]):
            raise ValueError('教师缓存着法非法')
        evaluation = data.get('evaluation', {})
        if evaluation.get('kind') == 'immediate-five':
            if move not in immediate_wins(board, len(moves) % 2 + 1):
                raise ValueError('缓存伪造直接成五')
            expected = dict(kind='immediate-five', raw='+M1', utility=1.0)
        else:
            expected = parse_eval(data.get('messages', []))
            replies = [s for s in data['messages'] if re.fullmatch(r'\d+,\d+', s)]
            if not replies or list(reversed(list(map(int, replies[-1].split(','))))) != move:
                raise ValueError('缓存坐标与协议回复不一致')
        if evaluation != expected:
            raise ValueError('缓存评价与原始输出不一致')
        return data

    def prefetch(self, histories, nodes):
        unique = {digest(h): h for h in histories}
        missing = []
        for h in unique.values():
            board, ended = board_of(h)
            if ended or len(h) >= 225:
                raise ValueError('不能请求终局着法')
            if self._path(h, nodes).exists():
                self.read(h, nodes)
            else:
                missing.append(h)
        if not missing:
            return
        if self.cache_only:
            raise ReviewStop('只读缓存缺少%d个查询；没有启动教师' % len(missing))
        self.__enter__()
        with self._guard:
            if self.queries + len(missing) > self.max_queries:
                raise ReviewStop('达到本轮共享教师查询额度')
            self.queries += len(missing)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(lambda h: self._query(h, nodes), missing))

    def _query(self, moves, nodes):
        if time.monotonic() >= self.deadline:
            raise ReviewStop('达到教师总时长预算')
        identity = self.identity(nodes)
        board, _ = board_of(moves)
        wins = immediate_wins(board, len(moves) % 2 + 1)
        key = digest([identity, moves])
        if wins:
            data = dict(key=key, identity=identity, move=wins[0], seconds=0.0,
                        evaluation=dict(kind='immediate-five', raw='+M1', utility=1.0), messages=[])
        else:
            data = self._native(moves, identity)
            data.update(key=key, identity=identity)
        data['payload_sha256'] = digest(data)
        write_json(self._path(moves, nodes), data)

    def _native(self, moves, identity):
        started = time.monotonic()
        until_time = min(self.deadline, started + identity['milliseconds'] / 1000 + 8)
        proc = subprocess.Popen([str(self.exe)], cwd=self.exe.parent, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding='utf-8', errors='replace',
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        incoming = queue.Queue()
        def reader():
            try:
                for line in proc.stdout:
                    incoming.put(line.strip())
            finally:
                incoming.put(None)
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        messages = []
        def send(text):
            proc.stdin.write(text + '\n')
            proc.stdin.flush()
        def receive(pattern):
            while True:
                remaining = until_time - time.monotonic()
                if remaining <= 0:
                    raise ReviewStop('教师查询超时/总预算耗尽')
                try:
                    line = incoming.get(timeout=remaining)
                except queue.Empty:
                    raise ReviewStop('教师查询超时/总预算耗尽') from None
                if line is None:
                    raise RuntimeError('教师提前退出：' + repr(messages[-3:]))
                messages.append(line)
                if line.startswith(('ERROR', 'UNKNOWN')) or len(messages) > 10000:
                    raise RuntimeError('教师协议错误：' + line[:200])
                if re.fullmatch(pattern, line):
                    return line
        try:
            send('START 15')
            receive('OK')
            for item in ('rule 0', 'thread_num 1', 'max_memory 268435456',
                         'max_node %d' % identity['nodes'],
                         'timeout_turn %d' % identity['milliseconds'], 'timeout_match 0'):
                send('INFO ' + item)
            send('BOARD')
            player = len(moves) % 2 + 1
            for r, c, color in moves:
                send('%d,%d,%d' % (c, r, 1 if color == player else 2))
            send('DONE')
            c, r = map(int, receive(r'\d+,\d+').split(','))
            board, _ = board_of(moves)
            if not (0 <= r < 15 and 0 <= c < 15) or board[r][c]:
                raise ValueError('教师返回非法着法')
            return dict(move=[r, c], evaluation=parse_eval(messages), messages=messages,
                        seconds=time.monotonic() - started)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            thread.join(timeout=2)
            proc.stdin.close()
            proc.stdout.close()


def input_key(case):
    return digest({k: case.get(k) for k in ('moves', 'aiColor', 'kind', 'rule', 'config', 'fingerprint')})


def _history(case):
    return [[m['r'], m['c'], m['player']] if isinstance(m, dict) else list(m)
            for m in case.get('moves', [])]


def _child(history, move):
    board, _ = board_of(history)
    r, c = move
    if not (0 <= r < 15 and 0 <= c < 15) or board[r][c]:
        raise ValueError('教师建议非法')
    player = len(history) % 2 + 1
    board[r][c] = player
    return None if check_win(board, r, c, player) or len(history) == 224 else history + [[r, c, player]]


def _branch(teacher, history, move, nodes):
    h = _child(history, move)
    if h is None:
        board, _ = board_of(history)
        board[move[0]][move[1]] = len(history) % 2 + 1
        won = check_win(board, *move, len(history) % 2 + 1)
        return dict(move=move, evaluation=dict(kind='exact-five' if won else 'exact-draw',
                    raw='+M0' if won else '0', utility=1.0 if won else 0.0),
                    cache_key=None, pv=dict(valid=True, moves=[]))
    label = teacher.read(h, nodes)
    return dict(move=move, evaluation=flip(label['evaluation']), cache_key=label['key'],
                seconds=label['seconds'], pv=pv_info(label, h), search=_search_info(label))


def _search_info(label):
    result = dict(requested_nodes=label['identity']['nodes'], depth=None, reported_nodes=None,
                  reported_seconds=None)
    for line in label['messages']:
        m = re.search(r'Depth\s+(\d+)-(\d+).*?Node\s+([\d.]+)([KM]?)\s*\|\s*Time\s+([\d.]+)(ms|s)', line)
        if m:
            result.update(depth=[int(m[1]), int(m[2])],
                          reported_nodes=float(m[3]) * {'': 1, 'K': 1000, 'M': 1000000}[m[4]],
                          reported_seconds=float(m[5]) * (.001 if m[6] == 'ms' else 1))
    return result


def _is_win(branch):
    return branch['evaluation']['raw'].startswith('+M')


def _is_loss(branch):
    return branch['evaluation']['raw'].startswith('-M')


def _category(actual, best):
    if _is_loss(actual) and _is_win(best):
        return 'teacher-win-to-loss'
    if _is_loss(actual) and _is_loss(best):
        return 'both-searched-losing'
    if _is_loss(actual):
        return 'defense-unknown'
    if _is_win(best) and not _is_win(actual):
        return 'possible-missed-win'
    return 'evaluation-only'


def _level(teacher, rows, history, nodes):
    roots = [history[:row['ply'] - 1] for row in rows]
    teacher.prefetch(roots, nodes)
    proposals = []
    for row, h in zip(rows, roots):
        root = teacher.read(h, nodes)
        alternatives = {tuple(root['move'])}
        if row['levels']:
            alternatives.update(tuple(b['move']) for b in row['levels'][-1]['alternatives'])
        alternatives = [list(m) for m in sorted(alternatives)]
        proposals.append((row, h, root, alternatives))
    children = [child for row, h, root, alternatives in proposals
                for move in [row['point']] + alternatives if (child := _child(h, move)) is not None]
    teacher.prefetch(children, nodes)
    for row, h, root, alternatives in proposals:
        actual = _branch(teacher, h, row['point'], nodes)
        branches = [_branch(teacher, h, m, nodes) for m in alternatives]
        best = max(branches, key=lambda b: b['evaluation']['utility'])
        row['levels'].append(dict(nodes=nodes, root_key=root['key'], root_move=root['move'],
            root_pv=pv_info(root, h),
            actual=actual, alternatives=branches, best=best, category=_category(actual, best),
            screening_gap=best['evaluation']['utility'] - actual['evaluation']['utility']))


def review_case(case, *, teacher=None, nodes=(200000, 1000000, 4000000),
                start_ply=None, shortlist=3, max_suspects=5):
    """完整扫描实际本方回合；只把规则事实和跨预算稳定胜/败分叉作为自动重赛锚点。

    start_ply保留旧API的“跳过前N手”含义；缺省只跳过config.opening_plies。
    没有该元数据时从第一手检查，不猜测固定开局长度。普通分候选留在rows中待复核。
    """
    report = dict(schema=1, backend='rapfi' if teacher else 'rules', status='complete',
                  input_key=input_key(case), source_fingerprint=copy.deepcopy(case.get('fingerprint')),
                  createdAt=int(time.time() * 1000), bounded=True, rows=[], suspects=[],
                  undetermined=None, warnings=[], budgets=list(nodes) if teacher else [])
    if not case.get('moves') or case.get('aiColor') not in (1, 2):
        report['undetermined'] = 'no-moves'
        return report
    try:
        history = _history(case)
        if not any(m[2] == case['aiColor'] for m in history):
            report['undetermined'] = 'ai-never-moved'
            return report
        board, ended = board_of(history)
        if case.get('kind', 'loss') != 'loss':
            report['undetermined'] = 'abnormal-or-not-loss'
            return report
        if not ended or history[-1][2] == case['aiColor']:
            report['undetermined'] = 'incomplete-or-not-ai-loss'
            return report
        skip = (case.get('config') or {}).get('opening_plies', 0) if start_ply is None else start_ply
        if type(skip) is not int or not 0 <= skip < len(history):
            raise ValueError('预置/跳过手数非法')
        if (not nodes or any(type(n) is not int or n < 1 for n in nodes)
                or tuple(sorted(set(nodes))) != tuple(nodes) or shortlist < 1 or max_suspects < 1):
            raise ValueError('节点预算须为严格递增正整数，shortlist/max_suspects须为正')
        report['opening_plies'] = skip
        report['notes'] = ([] if start_ply is not None or 'opening_plies' in (case.get('config') or {})
                           else ['未记录预置手数；从第一手扫描，可用--start-ply明确跳过预置手'])
    except (ValueError, TypeError, KeyError, IndexError) as error:
        report.update(status='invalid', undetermined='invalid-game: ' + str(error))
        return report
    ai, suspects = case['aiColor'], []
    for i, m in enumerate(history):
        if i < skip or m[2] != ai:
            continue
        before, _ = board_of(history[:i])
        own, opp = immediate_wins(before, ai), immediate_wins(before, 3 - ai)
        row = dict(ply=i + 1, ownPly=sum(x[2] == ai for x in history[:i + 1]),
                   point=m[:2], move=cd(m), own_wins=own, opponent_wins=opp, levels=[])
        report['rows'].append(row)
        reason, alternative = None, None
        if own and m[:2] not in own:
            reason, alternative = 'missed-win', own[0]
        elif not own and len(opp) == 1 and m[:2] != opp[0]:
            reason, alternative = 'missed-block', opp[0]
        row['already_unavoidable'] = not own and len(opp) >= 2
        if reason:
            row['rule_reason'] = reason
            suspects.append(dict(ply=i + 1, ownPly=row['ownPly'], move=row['move'], reason=reason,
                alternative=cd(alternative), point=alternative, confidence='rule-local', bounded=True,
                detail=('漏掉实际立即成五' if reason == 'missed-win' else '漏堵唯一立即成五点')
                       + '：' + cd(alternative) + '；不将堵后未知当作全盘可守'))
    active = [r for r in report['rows'] if not r.get('rule_reason') and not r['already_unavoidable']]
    if teacher and active:
        report['teacher_identities'] = {str(n): teacher.identity(n) for n in nodes}
        try:
            _level(teacher, active, history, nodes[0])
            selected = sorted([r for r in active if r['levels'][-1]['screening_gap'] >= .05],
                              key=lambda r: (-r['levels'][-1]['screening_gap'], -r['ply']))[:shortlist]
            for n in nodes[1:]:
                if selected:
                    _level(teacher, selected, history, n)
            for row in selected:
                if len(row['levels']) < 2:
                    continue
                last = row['levels'][-2:]
                winning = [{tuple(b['move']) for b in level['alternatives'] if _is_win(b)
                            and b['pv'].get('valid') is not False} for level in last]
                stable = winning[0] & winning[1]
                if (stable and all(_is_loss(level['actual']) and level['actual']['pv'].get('valid') is not False
                                   for level in last)):
                    alternative = list(sorted(stable)[0])
                    suspects.append(dict(ply=row['ply'], ownPly=row['ownPly'], move=row['move'],
                        reason='teacher-win-to-loss', alternative=cd(alternative), point=alternative,
                        confidence='teacher-bounded', bounded=True,
                        detail='实际线教师判败，替代%s教师判赢，连续两档预算稳定；有限搜索证据' % cd(alternative)))
        except ReviewStop as error:
            report.update(status='partial', undetermined='teacher-budget/cache: ' + str(error))
        except (OSError, RuntimeError, ValueError) as error:
            report.update(status='error', undetermined='teacher-error: ' + str(error))
        for row in active:
            for level in row['levels']:
                if level['root_pv'].get('valid') is False:
                    report['warnings'].append(dict(ply=row['ply'], nodes=level['nodes'],
                        cache_key=level['root_key'], detail=level['root_pv']['detail']))
                for branch in [level['actual']] + level['alternatives']:
                    if branch['pv'].get('valid') is False:
                        report['warnings'].append(dict(ply=row['ply'], nodes=level['nodes'],
                            cache_key=branch['cache_key'], detail=branch['pv']['detail']))
    # 硬战术优先；同档选择最近一次机会，max_suspects不会被更早的弱线索挤占。
    suspects.sort(key=lambda s: (s['confidence'] != 'rule-local', -s['ply']))
    report['suspects'] = suspects[:max_suspects]
    if not report['suspects'] and report['undetermined'] is None:
        report['undetermined'] = 'no-confirmed-signal' if teacher else 'no-rule-signal; teacher-not-used'
    return report
