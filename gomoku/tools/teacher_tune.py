# -*- coding: utf-8 -*-
"""纯算法离线教师调参：prepare / run / report / protocol。

教师适配当前 Rapfi normal 消息格式。每次查询启动独立进程，固定单线程、节点
和时间上限，防止置换表与查询顺序污染候选对照。不会下载引擎或加载拟合结果到
正式程序。所有标注、树和断点均落到用户指定实验目录。
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import subprocess
import sys
import threading
import time
import tomllib

from gomoku.core.opening_book import canonical_position
from gomoku.core.utils import empty_board
from gomoku.game import check_win
from gomoku.tools import tune_params as T

SCHEMA = 1
ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def source_fingerprint():
    paths = sorted((ROOT / 'core').glob('*.py'))
    paths += [ROOT / 'game.py', Path(__file__), Path(T.__file__)]
    return T.digest({p.relative_to(ROOT).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in paths
        if p.name != 'evaluate.py'})  # 不依赖本地被忽略的 NN 实验模块。


def board_of(moves):
    board = empty_board()
    ended = False
    for i, m in enumerate(moves):
        if (not isinstance(m, (list, tuple)) or len(m) != 3
                or any(type(v) is not int for v in m)):
            raise ValueError('moves 必须为零基 [r,c,player] 整数数组')
        r, c, player = m
        if ended or not (0 <= r < 15 and 0 <= c < 15) or player != i % 2 + 1 or board[r][c]:
            raise ValueError('棋谱越界、重复、轮次错误或在终局后继续落子')
        board[r][c] = player
        ended = check_win(board, r, c, player)
    return board, ended


def validate_positions(positions):
    if not isinstance(positions, list) or not positions:
        raise ValueError('没有局面')
    seen, source_games = {}, {}
    groups = {name: set() for name in ('train', 'validation', 'test')}
    for p in positions:
        board, ended = board_of(p['moves'])
        if (ended or len(p['moves']) >= 225 or type(p['player']) is not int
                or p['player'] != len(p['moves']) % 2 + 1):
            raise ValueError('拟合局面必须未终局且 player 与黑先轮次一致')
        split = p['split']
        if split not in groups or not isinstance(p['group'], str) or not p['group']:
            raise ValueError('split 或 group 无效')
        groups[split].add(p['group'])
        if p.get('source_game') is not None:
            old_split = source_games.setdefault(p['source_game'], split)
            if old_split != split:
                raise ValueError('同一原始棋局不能跨训练/验证/测试集')
        key = T.digest(canonical_position(board, p['player'])[0])
        if key in seen:
            raise ValueError('存在重复或旋转镜像等价局面')
        seen[key] = split
    if any(not v for v in groups.values()) or any(groups[a] & groups[b]
            for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('必须有互相隔离的 train / validation / test 棋局组')


def prepare(paths, limit=200, seed=1, min_ply=8, max_ply=60, stride=6):
    """按完整棋局/开局家族分组，等价局面连通的组整体划入同一数据集。"""
    records, sources = [], []
    for path in paths:
        raw = Path(path).read_bytes()
        sources.append(dict(sha256=hashlib.sha256(raw).hexdigest(), file=Path(path).name))
        data = json.loads(raw)
        for gi, game in enumerate(data.get('games', [])):
            moves = [[m['r'], m['c'], m['player']] for m in game['moves']]
            board_of(moves)  # 严格校验完整棋谱，不能把异常尾部静默截掉。
            opening = game.get('opening_moves')
            if opening:
                if opening != [[m[0], m[1]] for m in moves[:len(opening)]]:
                    raise ValueError('opening_moves 与实际棋谱前缀不一致')
                family_moves = [[r, c, i % 2 + 1] for i, (r, c) in enumerate(opening)]
            else:
                family_moves = moves[:min(6, len(moves))]
            fb, _ = board_of(family_moves)
            family = T.digest(canonical_position(fb, len(family_moves) % 2 + 1)[0])
            for ply in range(min_ply, min(max_ply + 1, len(moves))):
                # 每个取样区间同时保留黑白两手，不能 stride 为偶数就只取黑方。
                if (ply - min_ply) % stride >= 2:
                    continue
                history = moves[:ply]
                board, ended = board_of(history)
                if ended:
                    continue
                key = T.digest(canonical_position(board, ply % 2 + 1)[0])
                records.append(dict(id=key, moves=history, player=ply % 2 + 1,
                                    group=family, source_game=f'{len(sources)-1}:{gi}'))
    parent = {p['group']: p['group'] for p in records}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    seen = {}
    for p in records:
        if p['id'] in seen:
            a, b = find(p['group']), find(seen[p['id']]['group'])
            parent[max(a, b)] = min(a, b)
        else:
            seen[p['id']] = p
    buckets = {}
    for p in seen.values():
        p['group'] = find(p['group'])
        buckets.setdefault(p['group'], []).append(p)
    rng = random.Random(seed)
    families = sorted(buckets)
    rng.shuffle(families)
    if len(families) < 3 or limit < 3:
        raise ValueError('至少需要三个不等价的棋局/开局家族；请增加输入棋谱')
    # 轮流从各组取样，避免前面一盘占满样本额度。
    for group in families:
        rng.shuffle(buckets[group])
    selected = []
    while len(selected) < limit:
        added = 0
        for group in families:
            if buckets[group] and len(selected) < limit:
                selected.append(buckets[group].pop())
                added += 1
        if not added:
            break
    families = [g for g in families if any(p['group'] == g for p in selected)]
    count = max(1, round(len(families) * .2))
    assignment = {g: ('validation' if i < count else 'test' if i < count * 2 else 'train')
                  for i, g in enumerate(families)}
    for p in selected:
        p['split'] = assignment[p['group']]
    validate_positions(selected)
    return dict(schema=SCHEMA, seed=seed, sources=sources, positions=selected)


def parse_eval(lines):
    """只接受已观测的 normal Eval 格式；缺失/歧义不能伪造零分。"""
    token = None
    for line in lines:
        found = re.search(r'\bEval\s+([+-]?M\d+|[+-]?\d+)(?=\s|\||$)', line)
        if found:
            token = found.group(1)
    if token is None:
        raise ValueError('教师未输出可解析 Eval；请使用 Rapfi normal 消息模式')
    if 'M' in token:
        return dict(kind='mate', raw=token, utility=-1.0 if token.startswith('-') else 1.0)
    # 有限普通分压到 (-.9,.9)，确保有限搜索的 mate 标签始终另占最高/最低档。
    # 此数值是训练尺度，不宣称胜率；不把教师原始分直接当我方叶子目标。
    cp = int(token)
    return dict(kind='cp', raw=token, utility=.9 * math.tanh(cp / 600.0))


def immediate_wins(board, player):
    """仅识别下一子实际成五，复用正式判胜；模拟后保证棋盘恢复。"""
    result = []
    for r in range(15):
        for c in range(15):
            if board[r][c] == 0:
                board[r][c] = player
                try:
                    if check_win(board, r, c, player):
                        result.append([r, c])
                finally:
                    board[r][c] = 0
    return result


class BudgetStop(RuntimeError):
    pass


class RapfiTeacher:
    def __init__(self, exe, out, nodes=200000, milliseconds=2000, deadline=None, max_queries=5000):
        self.exe = Path(exe).resolve()
        if not self.exe.is_file():
            raise ValueError('教师可执行文件不存在')
        config = self.exe.parent / 'config.toml'
        if config.exists():
            parsed = tomllib.loads(config.read_text(encoding='utf-8'))
            general = parsed.get('general', {})
            if (general.get('message_mode', 'normal') != 'normal'
                    or general.get('coord_conversion_mode', 'none') != 'none'
                    or parsed.get('database', {}).get('enable_by_default', False)):
                raise ValueError('教师配置须 normal 输出、none 坐标转换且关闭数据库')
        self.cache = Path(out) / 'labels'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.deadline = deadline or float('inf')
        self.max_queries = max_queries
        self.queries = len(list(self.cache.glob('*.json')))
        files = [self.exe] + sorted(p for p in self.exe.parent.iterdir()
                                    if p.suffix.lower() in ('.toml', '.bin', '.lz4', '.nnue'))
        self.identity = dict(adapter='rapfi-normal-v1', rule='freestyle15', threads=1,
                             nodes=nodes, milliseconds=milliseconds,
                             files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

    def _query(self, moves):
        board, ended = board_of(moves)
        if ended or len(moves) == 225:
            raise ValueError('终局不能向教师继续请求落子')
        key = T.digest([self.identity, moves])
        path = self.cache / f'{key}.json'
        if path.exists():
            data = read_json(path)
            checksum = data.pop('payload_sha256', None)
            if (data.get('key') != key or data.get('identity') != self.identity
                    or checksum != T.digest(data)):
                raise ValueError('教师缓存指纹损坏')
            return data
        if time.monotonic() >= self.deadline or self.queries >= self.max_queries:
            raise BudgetStop('达到教师标注时间或查询预算；已完成查询可续用')
        # Rapfi 对直接成五可能只回坐标、不输出 Eval。这里用实际判胜事实提供
        # 标签，不能把“缺 Eval”一概当作零分或凭搜索五态猜胜负。
        player = len(moves) % 2 + 1
        wins = immediate_wins(board, player)
        if wins:
            data = dict(key=key, identity=self.identity, move=wins[0], seconds=0.0,
                        evaluation=dict(kind='immediate-five', raw='+M1', utility=1.0), messages=[])
            data['payload_sha256'] = T.digest(data)
            write_json(path, data)
            self.queries += 1
            return data
        start = time.monotonic()
        remaining = self.deadline - start
        per_query = min(remaining, self.identity['milliseconds'] / 1000 + 8)
        proc = subprocess.Popen([str(self.exe)], cwd=self.exe.parent,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace',
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        lines_queue = queue.Queue()

        def reader():
            try:
                for line in proc.stdout:
                    lines_queue.put(line.strip())
            finally:
                lines_queue.put(None)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        messages = []

        def send(line):
            proc.stdin.write(line + '\n')
            proc.stdin.flush()

        def until(pattern):
            end = start + per_query
            while True:
                try:
                    line = lines_queue.get(timeout=max(.001, end - time.monotonic()))
                except queue.Empty:
                    raise TimeoutError('教师查询超时') from None
                if line is None:
                    raise RuntimeError('教师提前退出: ' + repr(messages[-5:]))
                messages.append(line)
                if line.startswith(('ERROR', 'UNKNOWN')):
                    raise RuntimeError('教师协议错误: ' + line)
                if re.fullmatch(pattern, line):
                    return line
                if time.monotonic() >= end:
                    raise TimeoutError('教师持续输出但未完成查询')
        try:
            send('START 15')
            until('OK')
            for setting in ('rule 0', 'thread_num 1', 'max_memory 268435456',
                            f'max_node {self.identity["nodes"]}',
                            f'timeout_turn {self.identity["milliseconds"]}', 'timeout_match 0'):
                send('INFO ' + setting)
            send('BOARD')
            for r, c, color in moves:
                send(f'{c},{r},{1 if color == player else 2}')
            send('DONE')
            reply = until(r'\d+,\d+')
            c, r = map(int, reply.split(','))
            if not (0 <= r < 15 and 0 <= c < 15) or board[r][c]:
                raise ValueError('教师返回非法着法')
            value = parse_eval(messages)
            data = dict(key=key, identity=self.identity, move=[r, c], evaluation=value,
                        seconds=time.monotonic() - start, messages=messages)
            data['payload_sha256'] = T.digest(data)
            write_json(path, data)
            self.queries += 1
            return data
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            thread.join(timeout=2)
            proc.stdin.close()
            proc.stdout.close()

    def root(self, position):
        return self._query(position['moves'])

    def utility(self, position, move):
        board, _ = board_of(position['moves'])
        r, c = move
        if not (0 <= r < 15 and 0 <= c < 15) or board[r][c]:
            raise ValueError('待标注候选非法')
        player = position['player']
        board[r][c] = player
        if check_win(board, r, c, player):
            return 1.0
        if len(position['moves']) == 224:
            return 0.0
        result = self._query(position['moves'] + [[r, c, player]])
        return -result['evaluation']['utility']  # 子局面轮到对手，严格翻回根方视角。


def isolated_search(position, params, settings, record, deadline):
    payload = dict(position=position, params=params, t0=settings['t0'],
                   node_limit=settings['node_limit'], record=record)
    env = dict(os.environ, DSH_GOMOKU_GA='0', PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    timeout = min(settings['search_seconds'], deadline - time.monotonic())
    if timeout <= 0:
        raise BudgetStop('达到本轮总时长预算')
    try:
        result = subprocess.run([sys.executable, '-B', '-m', 'gomoku.tools.teacher_tune', '_search'],
                                input=json.dumps(payload), text=True, encoding='utf-8', env=env,
                                cwd=ROOT.parent, capture_output=True, timeout=timeout,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired:
        raise T.SearchLimit('单局面搜索超过时间上限') from None
    if result.returncode:
        if 'SearchLimit' in result.stderr:
            raise T.SearchLimit('单局面搜索超过节点上限')
        raise RuntimeError('搜索子进程失败: ' + result.stderr[-1800:])
    return json.loads(result.stdout)


class Experiment:
    def __init__(self, out, settings, positions, teacher, source, deadline):
        self.out, self.settings, self.positions = Path(out), settings, positions
        self.teacher, self.source, self.deadline = teacher, source, deadline

    def tree(self, position, params, record=True):
        # 叶子权重不参与树结构键；所有前段参数、历史、源码和预算都参与。
        effective = {k: v for k, v in params.items() if not record or k not in T.LEAF_KEYS}
        identity = [self.source, self.settings, position, effective, record]
        key = T.digest(identity)
        path = self.out / ('trees' if record else 'probes') / f'{key}.json'
        if path.exists():
            cached = read_json(path)
            if cached.get('key') != key or cached.get('payload_sha256') != T.digest(cached['result']):
                raise ValueError('树缓存指纹损坏')
            return cached['result']
        result = isolated_search(position, params, self.settings, record, self.deadline)
        write_json(path, dict(key=key, result=result, payload_sha256=T.digest(result)))
        return result

    def case(self, position, params):
        tree = self.tree(position, params)
        root = self.teacher.root(position)
        best = root['move']
        utility = {f'{row["r"]},{row["c"]}': self.teacher.utility(position, [row['r'], row['c']])
                   for row in tree['rows']}
        reference = self.teacher.utility(position, best)
        return dict(position=position, tree=tree, utilities=utility,
                    reference=reference, teacher_move=best)

    def cases(self, params, split):
        result = []
        selected = [p for p in self.positions if p['split'] == split]
        for i, position in enumerate(selected):
            result.append(self.case(position, params))
            if i % 10 == 0:
                print(f'{split}: {i+1}/{len(selected)}', flush=True)
        return result

    def check_compute(self, params, split, baseline):
        """先用正式路径筛掉超预算试参，再花教师额度生成新标签。"""
        nodes, seconds = 0, 0.0
        for position in self.positions:
            if position['split'] != split:
                continue
            probe = self.tree(position, params, record=False)
            nodes += probe['nodes']
            seconds += probe['seconds']
            # 开销非负，部分局面的累积已超全集限额即可提前拒绝。
            if (nodes > max(1, baseline['nodes']) * self.settings['cost_ratio']
                    or seconds > max(.01 * baseline['positions'], baseline['seconds']) * self.settings['time_ratio']):
                raise T.SearchLimit('正式路径试参累计计算量/时长超过同集基线额度')

    def metrics(self, cases, params):
        regrets, losses, costs, times = [], [], [], []
        absent = blocked = differing = 0
        for case in cases:
            rows = case['tree']['rows']
            selected = T.choose(rows, params)
            key = f'{selected["r"]},{selected["c"]}'
            regrets.append(max(0.0, case['reference'] - case['utilities'][key]))
            teacher_move = tuple(case['teacher_move'])
            in_pool = next((row for row in rows if (row['r'], row['c']) == teacher_move), None)
            absent += in_pool is None
            if in_pool:
                blocked += T.DeepSearch.STATE_ORDER[in_pool['state']] < T.DeepSearch.STATE_ORDER[selected['state']]
            probe = self.tree(case['position'], params, record=False)
            if probe['move'] != [selected['r'], selected['c']]:
                raise AssertionError('完整缓存与正式早停搜索选点不一致')
            # 对参与选点的行核对状态与分值，不能仅核对同一坐标。
            actual = next(r for r in probe['rows'] if [r['r'], r['c']] == probe['move'])
            if actual['state'] != selected['state'] or not math.isclose(
                    actual['score'], T.replay_row(selected, params)[0], rel_tol=1e-10, abs_tol=1e-7):
                raise AssertionError('缓存与正式路径的根状态或分值不一致')
            costs.append(probe['nodes'])
            times.append(probe['seconds'])
            differing += probe['move'] != case['teacher_move']
            values = [T.replay_row(row, params)[0] for row in rows]
            pair_loss = []
            top = max(T.DeepSearch.STATE_ORDER[row['state']] for row in rows)
            for i, a in enumerate(rows):
                for j, b in enumerate(rows):
                    if T.DeepSearch.STATE_ORDER[a['state']] != top or a['state'] != b['state']:
                        continue
                    gap = case['utilities'][f'{a["r"]},{a["c"]}'] - case['utilities'][f'{b["r"]},{b["c"]}']
                    if gap > self.settings['margin']:
                        z = (values[j] - values[i]) / 500
                        pair_loss.append(gap * (max(z, 0) + math.log1p(math.exp(-abs(z)))))
            losses.append(sum(pair_loss) / len(pair_loss) if pair_loss else 0)
        if not regrets:
            raise ValueError('此数据集没有可评估局面')
        regret, loss = sum(regrets) / len(regrets), sum(losses) / len(losses)
        return dict(positions=len(regrets), regret=regret, ranking_loss=loss,
                    objective=regret + .02 * loss, severe=sum(x >= .5 for x in regrets),
                    nodes=sum(costs), seconds=sum(times), p95_seconds=sorted(times)[math.ceil(.95 * len(times))-1],
                    teacher_move_absent=absent, teacher_move_state_blocked=blocked,
                    different_coordinate=differing)


def within_cost(metrics, baseline, settings):
    return (metrics['nodes'] <= max(1, baseline['nodes']) * settings['cost_ratio']
            and metrics['seconds'] <= max(.01 * baseline['positions'], baseline['seconds']) * settings['time_ratio'])


def write_report(out, checkpoint):
    data = checkpoint
    lines = ['# 离线教师调参实验', '', f'状态：{data["status"]}',
             f'已完成步骤：{data.get("next_step", 0)}', '',
             '参数只供实验入口显式加载；教师一致率不是棋力胜率。',
             '测试集保留未查询；验证集只在最后验收，不用于每次试参。', '',
             '| 数据集/参数 | 局面 | 教师差距均值 | 严重误选 | 搜索节点 | 搜索秒数 |',
             '|---|---:|---:|---:|---:|---:|']
    for name in ('baseline_train', 'candidate_train', 'baseline_validation', 'candidate_validation'):
        m = data.get(name)
        if m:
            lines.append(f'| {name} | {m["positions"]} | {m["regret"]:.6f} | {m["severe"]} | {m["nodes"]} | {m["seconds"]:.3f} |')
    lines += ['', '| 数据集/参数 | 教师着法未入候选 | 被五态档位压制 | 与教师坐标不同 |',
              '|---|---:|---:|---:|']
    for name in ('baseline_train', 'candidate_train', 'baseline_validation', 'candidate_validation'):
        m = data.get(name)
        if m:
            lines.append(f'| {name} | {m["teacher_move_absent"]} | {m["teacher_move_state_blocked"]} | {m["different_coordinate"]} |')
    lines += ['', '| 步骤 | 参数组 | 接受 | 可拟合局面/比较对 | 拒绝原因 |', '|---:|---|---|---|---|']
    for event in data.get('history', []):
        fit = event.get('fit')
        comparisons = f'{fit["trainable_positions"]}/{fit["pairs"]}' if fit else '—'
        reason = event.get('reason', '—' if event['accepted'] else '未改善目标或未满足资源/严重误选约束')
        lines.append(f'| {event["step"]+1} | {event["group"]} | {event["accepted"]} | {comparisons} | {reason} |')
    lines += ['', '“搜索节点”统计实际创建的落子上下文，含候选展开；不包含教师标注成本。',
              '时间为 Python 搜索段实测，不含子进程启动、缓存读取和标注；正式搜索探针使用原早停路径。',
              '固定初始温度并不保证相同计算量；每个候选参数都经过节点与搜索时长约束。',
              '被候选池排除或被五态排序压制的教师着法会单列，不能声称已由叶子拟合解决。',
              '', f'建议采用：{data.get("recommended", False)}',
              f'停止原因：{data.get("stop_reason", "无")}', '']
    path = Path(out) / 'report.md'
    with path.open('w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))


class RunLock:
    """操作系统文件锁；进程崩溃自动释放，不通过 PID 猜测锁是否失效。"""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.handle = None
        try:
            self.handle = self.path.open('a+b')
            self.handle.seek(0)
            if self.handle.read(1) == b'':
                self.handle.write(b'0')
                self.handle.flush()
            self.handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if self.handle is not None:
                self.handle.close()
            raise ValueError('另一个进程正在使用此实验目录') from None
        return self

    def __exit__(self, *args):
        self.handle.close()


def run_experiment(args):
    if sys.version_info[:2] != (3, 14):
        raise ValueError('实验必须使用 Python 3.14')
    out = Path(args.out).resolve()
    repo = ROOT.parent.resolve()
    local_experiments = repo / 'storages' / 'experiments'
    if out.is_relative_to(repo) and not out.is_relative_to(local_experiments):
        raise ValueError('项目内实验产物只能写入已忽略的 storages/experiments/；也可使用仓库外目录')
    out.mkdir(parents=True, exist_ok=True)
    with RunLock(out / 'run.lock'):
        return _run_locked(args, out)


def _run_locked(args, out):
    source = source_fingerprint()
    deadline = time.monotonic() + args.max_seconds
    teacher = RapfiTeacher(args.teacher, out, args.teacher_nodes, args.teacher_ms,
                           deadline, args.max_queries)
    groups = args.groups.split(',')
    if len(set(groups)) != len(groups) or any(g not in T.GROUPS for g in groups):
        raise ValueError('groups 应为 leaf,score,gear,search 的无重复子集')
    settings = dict(t0=args.t0, node_limit=args.node_limit, search_seconds=args.search_seconds,
                    margin=args.margin, cost_ratio=args.cost_ratio, time_ratio=args.time_ratio)
    if args.positions_file:
        dataset = read_json(args.positions_file)
    else:
        dataset = prepare(args.games, args.positions, args.seed, args.min_ply, args.max_ply, args.stride)
    validate_positions(dataset['positions'])
    identity = dict(schema=SCHEMA, source=source, dataset=T.digest(dataset), teacher=teacher.identity,
                    settings=settings, groups=groups, rounds=args.rounds, trials=args.trials,
                    leaf_steps=args.leaf_steps, seed=args.seed, python=sys.version)
    path = out / 'checkpoint.json'
    if path.exists():
        if not args.resume:
            raise ValueError('实验目录已有断点；请使用 --resume 或新目录')
        checkpoint = read_json(path)
        if checkpoint['identity'] != identity:
            raise ValueError('源码、数据、教师或实验配置已变；不能沿用此断点，请使用新目录')
        if checkpoint['status'] == 'complete':
            print('实验已完成；结果位于 ' + str(out), flush=True)
            return 0
    else:
        if args.resume:
            raise ValueError('--resume 指定的实验没有断点')
        base = T.defaults()
        checkpoint = dict(identity=identity, baseline=base, params=base, next_step=0,
                          status='running', history=[])
        write_json(out / 'positions.json', dataset)
        write_json(path, checkpoint)
    experiment = Experiment(out, settings, dataset['positions'], teacher, source, deadline)
    base = checkpoint['baseline']
    schedule = []
    for round_index in range(args.rounds):
        for group in groups:
            schedule.extend((round_index, group, trial) for trial in range(1 if group == 'leaf' else args.trials))
            if group != 'leaf' and 'leaf' in groups:
                schedule.append((round_index, 'leaf', -1))
    try:
        if 'baseline_train' not in checkpoint:
            cases = experiment.cases(base, 'train')
            checkpoint['baseline_train'] = experiment.metrics(cases, base)
            checkpoint['candidate_train'] = checkpoint['baseline_train']
            write_json(path, checkpoint)
        for step in range(checkpoint['next_step'], len(schedule)):
            if time.monotonic() >= deadline:
                raise BudgetStop('达到本轮总时长预算')
            round_index, group, trial = schedule[step]
            current = checkpoint['params']
            seed = args.seed + 1009 * (step + 1)
            fit = None
            if group == 'leaf':
                cases = experiment.cases(current, 'train')
                proposal, fit = T.fit_leaf(cases, current, base, args.leaf_steps, seed,
                                           args.margin, deadline=deadline)
            else:
                proposal = T.perturb(current, base, group, seed)
            T.validate(proposal, base)
            print(f'步骤 {step+1}/{len(schedule)} round={round_index+1} group={group}', flush=True)
            try:
                experiment.check_compute(proposal, 'train', checkpoint['baseline_train'])
                cases = experiment.cases(proposal, 'train')
                metrics = experiment.metrics(cases, proposal)
                accepted = (within_cost(metrics, checkpoint['baseline_train'], settings)
                            and metrics['severe'] <= checkpoint['candidate_train']['severe']
                            and metrics['objective'] < checkpoint['candidate_train']['objective'] - 1e-9)
                event = dict(step=step, group=group, accepted=accepted, metrics=metrics, fit=fit)
                if accepted:
                    checkpoint['params'] = proposal
                    checkpoint['candidate_train'] = metrics
            except T.SearchLimit as exc:
                event = dict(step=step, group=group, accepted=False, reason=str(exc))
            checkpoint['history'].append(event)
            checkpoint['next_step'] = step + 1
            checkpoint['status'] = 'running'
            checkpoint.pop('stop_reason', None)
            write_json(path, checkpoint)
            write_report(out, checkpoint)
        # 验证集只在所有训练试参结束后查询；test 从未交给教师或拟合器。
        base_val = experiment.cases(base, 'validation')
        cand_val = experiment.cases(checkpoint['params'], 'validation')
        checkpoint['baseline_validation'] = experiment.metrics(base_val, base)
        checkpoint['candidate_validation'] = experiment.metrics(cand_val, checkpoint['params'])
        b, c = checkpoint['baseline_validation'], checkpoint['candidate_validation']
        recommended = (checkpoint['candidate_train']['objective'] < checkpoint['baseline_train']['objective'] - 1e-9
                       and c['objective'] < b['objective'] - 1e-9
                       and c['regret'] <= b['regret'] + 1e-9 and c['severe'] <= b['severe']
                       and within_cost(c, b, settings))
        checkpoint.update(status='complete', recommended=recommended)
        checkpoint.pop('stop_reason', None)
        export = dict(schema=SCHEMA, source=source, t0=args.t0, baseline=base,
                      params=checkpoint['params'], recommended=recommended,
                      teacher=teacher.identity, experiment=T.digest(identity))
        write_json(out / 'candidate.json', export)
        write_json(out / 'best.json', dict(export, params=checkpoint['params'] if recommended else base))
        write_json(path, checkpoint)
        write_report(out, checkpoint)
        print(f'完成，建议采用={recommended}；请查看 {out / "report.md"}', flush=True)
        return 0
    except (BudgetStop, KeyboardInterrupt, T.SearchLimit, TimeoutError) as exc:
        checkpoint.update(status='paused', stop_reason=str(exc) or '用户中断')
        write_json(path, checkpoint)
        write_report(out, checkpoint)
        print(f'已保存断点：{checkpoint["stop_reason"]}；原命令加 --resume 可继续。', flush=True)
        return 2
    except (ValueError, RuntimeError, OSError, AssertionError) as exc:
        checkpoint.update(status='failed', stop_reason=f'{type(exc).__name__}: {exc}')
        write_json(path, checkpoint)
        write_report(out, checkpoint)
        raise


def evaluate_test(args):
    """只验收已冻结结果，不回写参数或训练断点。"""
    if sys.version_info[:2] != (3, 14):
        raise ValueError('实验必须使用 Python 3.14')
    out = Path(args.out).resolve()
    with RunLock(out / 'run.lock'):
        checkpoint = read_json(out / 'checkpoint.json')
        if checkpoint['status'] != 'complete':
            raise ValueError('请先完成训练及验证，再查询保留测试集')
        params_file = out / args.params
        data = read_json(params_file)
        source = source_fingerprint()
        if data['source'] != source or data['experiment'] != T.digest(checkpoint['identity']):
            raise ValueError('参数文件不属于此源码/实验')
        T.validate(data['params'], checkpoint['baseline'])
        deadline = time.monotonic() + args.max_seconds
        identity = checkpoint['identity']['teacher']
        teacher = RapfiTeacher(args.teacher, out, identity['nodes'], identity['milliseconds'],
                               deadline, args.max_queries)
        if teacher.identity != identity:
            raise ValueError('测试必须使用相同教师及配套文件')
        dataset = read_json(out / 'positions.json')
        if T.digest(dataset) != checkpoint['identity']['dataset']:
            raise ValueError('测试数据与原实验不一致')
        experiment = Experiment(out, checkpoint['identity']['settings'], dataset['positions'],
                                teacher, source, deadline)
        baseline, params = checkpoint['baseline'], data['params']
        a = experiment.metrics(experiment.cases(baseline, 'test'), baseline)
        b = experiment.metrics(experiment.cases(params, 'test'), params)
        result = dict(source=source, params=T.digest(data), baseline=a, candidate=b,
                      warning='测试集已查询；不能再用此集调参并声称它是未见测试集')
        path = out / ('test-' + T.digest(data)[:16] + '.json')
        write_json(path, result)
        print(path, flush=True)
        return 0


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('必须是有限正数')
    return result


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('必须是正整数')
    return result


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--games', action='append', help='match.py 导出的 JSON，可重复指定')
        p.add_argument('--positions', type=positive_int, default=200)
        p.add_argument('--seed', type=int, default=1)
        p.add_argument('--min-ply', type=positive_int, default=8)
        p.add_argument('--max-ply', type=positive_int, default=60)
        p.add_argument('--stride', type=positive_int, default=6)
        p.add_argument('--out', required=True)
        if name == 'run':
            p.add_argument('--positions-file', help='prepare 生成的固定分组数据')
            p.add_argument('--teacher', required=True, help='用户提供的 Rapfi 可执行文件')
            p.add_argument('--teacher-nodes', type=positive_int, default=200000)
            p.add_argument('--teacher-ms', type=positive_int, default=2000)
            p.add_argument('--max-queries', type=positive_int, default=5000)
            p.add_argument('--max-seconds', type=positive, default=7200)
            p.add_argument('--search-seconds', type=positive, default=15)
            p.add_argument('--node-limit', type=positive_int, default=50000)
            p.add_argument('--t0', type=positive, default=80)
            p.add_argument('--groups', default='leaf,score')
            p.add_argument('--rounds', type=positive_int, default=2)
            p.add_argument('--trials', type=positive_int, default=4)
            p.add_argument('--leaf-steps', type=positive_int, default=300)
            p.add_argument('--margin', type=positive, default=.08)
            p.add_argument('--cost-ratio', type=positive, default=1.1)
            p.add_argument('--time-ratio', type=positive, default=1.25)
            p.add_argument('--resume', action='store_true')
    report = sub.add_parser('report')
    report.add_argument('--out', required=True)
    evaluate = sub.add_parser('evaluate', help='冻结参数后对保留测试集验收')
    evaluate.add_argument('--out', required=True)
    evaluate.add_argument('--teacher', required=True)
    evaluate.add_argument('--params', default='best.json', choices=('best.json', 'candidate.json'))
    evaluate.add_argument('--max-seconds', type=positive, default=3600)
    evaluate.add_argument('--max-queries', type=positive_int, default=10000)
    protocol = sub.add_parser('protocol')
    protocol.add_argument('--params', required=True)
    sub.add_parser('_search', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.command == '_search':
            payload = json.load(sys.stdin)
            print(json.dumps(T.search_position(**payload), allow_nan=False))
        elif args.command in ('run', 'prepare'):
            if args.min_ply > args.max_ply:
                raise ValueError('min-ply 不能大于 max-ply')
            if not args.games and not getattr(args, 'positions_file', None):
                raise ValueError('必须提供 --games 或 --positions-file')
            if args.games and getattr(args, 'positions_file', None):
                raise ValueError('--games 和 --positions-file 二选一')
            if args.command == 'prepare':
                data = prepare(args.games, args.positions, args.seed, args.min_ply, args.max_ply, args.stride)
                if Path(args.out).exists():
                    raise ValueError('输出文件已存在，拒绝覆盖')
                write_json(args.out, data)
                print(f'已准备 {len(data["positions"])} 个局面')
            else:
                return run_experiment(args)
        elif args.command == 'report':
            write_report(args.out, read_json(Path(args.out) / 'checkpoint.json'))
            print(Path(args.out) / 'report.md')
        elif args.command == 'evaluate':
            return evaluate_test(args)
        else:
            data = read_json(args.params)
            if data['schema'] != SCHEMA or data['source'] != source_fingerprint():
                raise ValueError('参数与当前源码不匹配，请重新验证')
            T.validate(data['params'], data['baseline'])
            from gomoku import protocol as P
            os.environ['DSH_GOMOKU_GA'] = '0'
            print('实验参数已显式加载；此入口用于配对对战验收。', file=sys.stderr, flush=True)
            with T.parameter_scope(data['params'], data['t0']):
                return P.run()
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
