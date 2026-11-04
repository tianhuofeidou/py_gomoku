"""桌面搜索录像与独立调试进程；正式引擎只保留默认关闭的观察钩子。"""
import bisect
import copy
import json
import queue
import time
from pathlib import Path


class TraceStopped(Exception):
    """调试停止/限额：整个独立进程退出，不再使用被打断的模拟棋盘。"""


class TraceRecorder:
    """节点按对象分配ID，路径节点不因同盘面合并；事件序号是执行顺序。"""
    def __init__(self, send, checkpoint=None, max_events=200_000):
        self.send = send
        self.checkpoint = checkpoint or (lambda: None)
        self.max_events = max_events
        self.seq = 0
        self.round = 0
        self.real_moves = []
        self.started = time.monotonic()
        self.nodes = {}  # 持有引用，避免GC后id复用造成录像节点串线。

    def event(self, kind, node=None, **data):
        self.checkpoint()
        if self.seq >= self.max_events:
            raise TraceStopped('录像达到%d事件上限，搜索已停止，结果未完成' % self.max_events)
        self.seq += 1
        event = dict(seq=self.seq, kind=kind, node=node, round=self.round,
                     seconds=time.monotonic()-self.started, **copy.deepcopy(data))
        self.send(event)
        return event

    def identify(self, node, parent=None, move=None, origin='search'):
        if node is None:
            return None
        key = id(node)
        if key in self.nodes:
            return self.nodes[key][1]
        parent_id = self.identify(parent) if parent is not None else None
        node_id = 'n%d' % (len(self.nodes)+1)
        self.nodes[key] = (node, node_id)
        data = dict(parent=parent_id, move=move, origin=origin, mover=node.get('mover'))
        if parent is None:
            data['board'] = list(node['board'])
            data['base_moves'] = list(self.real_moves)
        self.event('node', node_id, **data)
        return node_id

    def begin(self, moves, player, t0):
        self.round += 1
        self.real_moves = list(moves)
        self.event('round', moves=moves, turn=player, t0=t0)

    def record(self, kind, node=None, **data):
        parent = data.pop('parent', None)
        previous = data.pop('previous', None)
        source = data.pop('source', None)
        if kind == 'reroot' and data.get('hit'):
            parent = previous
        node_id = self.identify(node, parent, data.get('move'))
        if previous is not None:
            data['previous'] = self.identify(previous)
        if source is not None:
            data['source'] = self.identify(source)
        self.event(kind, node_id, **data)
        if kind == 'cache_copy' and node is not None:
            # 复制的是完成子树，导入节点必须标缓存，不能假装刚逐个搜索过。
            pending = [(node, None, None)]
            while pending:
                item, owner, move = pending.pop()
                item_id = self.identify(item, owner, move, 'cache')
                value = item.get('move_result')
                if value is not None:
                    self.event('snapshot', item_id, **value,
                               score_player=item.get('main_player'), cached=True)
                for child_move, child in reversed(tuple(item['children'].items())):
                    pending.append((child, item, child_move))


def load_trace(path):
    """读取导出的搜索录像；不启动搜索，不修改原文件。

    返回 (TraceModel, partial, moves)。schema、事件序号、父子顺序和根棋盘
    由 TraceModel 与原有校验共同核验；坏文件直接抛 ValueError。
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('无法读取搜索录像：%s' % exc) from exc
    if not isinstance(data, dict) or data.get('schema') != 'gomoku-search-trace-v1':
        raise ValueError('不是 gomoku-search-trace-v1 搜索录像')
    events = data.get('events')
    if not isinstance(events, list) or not events:
        raise ValueError('搜索录像没有 events')
    model = TraceModel()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError('搜索录像事件不是对象')
        try:
            model.append(event)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('搜索录像事件无效：%s' % exc) from exc
    if not model.rounds:
        raise ValueError('搜索录像缺少 round 事件')
    first_round = model.events[model.rounds[0]]
    moves = first_round.get('moves')
    if not isinstance(moves, list):
        raise ValueError('搜索录像缺少真实着法')
    if not model.roots:
        raise ValueError('搜索录像缺少 root 事件')
    root_index, root_node = model.roots[-1]
    if root_node not in model.nodes:
        raise ValueError('root 事件指向不存在的节点')
    if model.nodes[root_node]['event'].get('parent') is not None:
        raise ValueError('root 事件必须指向无父节点')
    if model.nodes[root_node]['created'] >= root_index:
        raise ValueError('root 事件必须位于根节点事件之后')
    orphan_roots = [node_id for node_id, node in model.nodes.items()
                    if node['event'].get('parent') is None]
    if orphan_roots != [root_node]:
        raise ValueError('搜索录像必须只有一个无父节点的根')
    model.seek(len(model.events)-1)
    return model, bool(data.get('partial')), [tuple(m) for m in moves]


class TraceModel:
    """只存事件和索引。拖动时间轴时按节点自身事件复算，不借用未来状态。"""
    def __init__(self):
        self.events = []
        self.nodes = {}
        self.rounds = []
        self.roots = []
        self.cursor = -1
        self.max_depth = 0

    def append(self, event):
        if event['seq'] != len(self.events)+1:
            raise ValueError('搜索录像事件序号不连续')
        index = len(self.events)
        self.events.append(event)
        node_id = event.get('node')
        if event['kind'] == 'node':
            parent = event['parent']
            if parent is not None and parent not in self.nodes:
                raise ValueError('录像子节点早于父节点')
            depth = 0 if parent is None else self.nodes[parent]['depth'] + 1
            self.nodes[node_id] = dict(created=index, event=event, indices=[], children=[],
                                       depth=depth)
            if parent is not None:
                self.nodes[parent]['children'].append(node_id)
            self.max_depth = max(self.max_depth, depth)
        if node_id is not None:
            self.nodes[node_id]['indices'].append(index)
        if event['kind'] == 'round':
            self.rounds.append(index)
        if event['kind'] in ('root', 'reroot'):
            self.roots.append((index, node_id))

    def seek(self, index):
        self.cursor = max(-1, min(int(index), len(self.events)-1))

    def root(self):
        index = bisect.bisect_right(self.roots, (self.cursor, '\uffff'))-1
        return self.roots[index][1] if index >= 0 else None

    def at(self, node_id):
        node = self.nodes[node_id]
        if node['created'] > self.cursor:
            return None
        value = dict(node['event'])
        for index in node['indices']:
            if index > self.cursor:
                break
            event = self.events[index]
            value.update({k:v for k,v in event.items() if k not in ('seq', 'kind', 'round', 'seconds', 'node')})
            value['last_kind'] = event['kind']
        return value

    def board(self, node_id):
        chain = []
        while node_id is not None:
            data = self.nodes[node_id]['event']
            if 'board' in data:
                flat = data['board']
                board = [flat[r*15:(r+1)*15] for r in range(15)]
                break
            chain.append(data['move'])
            node_id = data['parent']
        else:
            raise ValueError('录像节点缺少根棋盘')
        for r, c, player in reversed(chain):
            if board[r][c] != 0:
                raise ValueError('录像路径重复落子')
            board[r][c] = player
        return board

    def moves(self, node_id):
        chain = []
        while node_id is not None:
            data = self.nodes[node_id]['event']
            if 'board' in data:
                return list(data['base_moves']) + list(reversed(chain))
            chain.append(data['move'])
            node_id = data['parent']
        raise ValueError('录像节点缺少原始着法')


def private_memory_bytes():
    """Windows私有内存；其他平台明确不可用，不把RSS伪装成私有内存。"""
    import sys
    if sys.platform != 'win32':
        return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
    data = Counters()
    data.cb = ctypes.sizeof(data)
    kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(data), data.cb):
        raise ctypes.WinError()
    return data.PrivateUsage


def debug_worker(commands, messages, running, stopping, moves):
    """单个独立进程持有引擎；成功分析后保留缓存，停止/异常后整个进程退出。"""
    import os
    import tempfile
    import traceback
    from gomoku.core.engine import Engine
    from gomoku.core.utils import empty_board
    from gomoku.game import check_win

    batch = []
    last_flush = time.monotonic()
    deadline = None
    memory_cap = None
    last_resource = 0.

    def flush():
        nonlocal batch, last_flush
        if batch:
            messages.put(('events', batch))
            batch = []
        last_flush = time.monotonic()

    def checkpoint():
        nonlocal last_resource
        while True:
            now = time.monotonic()
            if stopping.is_set():
                raise TraceStopped('用户停止搜索')
            if deadline is not None and now >= deadline:
                raise TraceStopped('达到调试墙钟上限（含暂停），未完成')
            if now-last_resource >= .15:
                used = private_memory_bytes()
                if memory_cap and used is not None and used > memory_cap:
                    raise TraceStopped('达到调试进程私有内存上限，未完成')
                last_resource = now
                if batch and now-last_flush >= .05:
                    flush()
            if running.wait(.05):
                return

    def send(event):
        batch.append(event)
        if len(batch) >= 64 or time.monotonic()-last_flush >= .05:
            flush()

    try:
        # 调试只用独立临时数据目录，不读写玩家的全局胜负记忆。
        with tempfile.TemporaryDirectory(prefix='gomoku-search-debug-') as home:
            os.environ['GOMOKU_HOME'] = home
            engine = Engine()
            board = empty_board()
            actual = []
            ended = False
            for r, c, player in moves:
                if ended or player != 1+len(actual)%2 or not (0 <= r < 15 and 0 <= c < 15) or board[r][c]:
                    raise ValueError('载入局面的着法顺序、坐标或终局不合法')
                board[r][c] = player
                engine.on_move(board, r, c, player)
                actual.append((r, c, player))
                ended = check_win(board, r, c, player)
            recorder = TraceRecorder(send, checkpoint)
            engine.deep_search._debug_observer = recorder
            messages.put(('ready', dict(moves=actual, ended=ended)))
            while not stopping.is_set():
                try:
                    command = commands.get(timeout=.2)
                except queue.Empty:
                    continue
                action = command['action']
                if action == 'analyze':
                    if ended or len(actual) == 225:
                        messages.put(('invalid', '终局没有可继续搜索的着法'))
                        continue
                    seconds = command['seconds']
                    deadline = time.monotonic()+seconds if seconds > 0 else None
                    memory_cap = command['memory_mib']*1024*1024
                    engine.deep_search.T0 = command['t0']
                    turn = 1+len(actual)%2
                    recorder.begin(actual, turn, engine.deep_search.T0)
                    result = engine.analyze_turn(board, turn)
                    part = result['part4_result']
                    recorder.record('complete', engine.deep_search._tree_root,
                        selected_move=part['move'], ranked=part['ranked'], result_type=part['type'],
                        opening_source=engine.opening_source, cache_stats=dict(engine.deep_search.cache_stats))
                    flush()
                    deadline = None
                    messages.put(('done', dict(move=part['move'], ranked=part['ranked'],
                        cache_stats=dict(engine.deep_search.cache_stats), moves=actual)))
                elif action == 'move':
                    r, c = command['point']
                    if ended or not (0 <= r < 15 and 0 <= c < 15) or board[r][c]:
                        messages.put(('invalid', '此处不能落子'))
                        continue
                    player = 1+len(actual)%2
                    board[r][c] = player
                    recorder.real_moves = actual + [(r, c, player)]
                    engine.on_move(board, r, c, player)
                    actual.append((r, c, player))
                    ended = check_win(board, r, c, player)
                    flush()
                    messages.put(('moved', dict(moves=actual, ended=ended)))
                else:
                    raise ValueError('未知调试命令：'+str(action))
    except TraceStopped as exc:
        flush()
        messages.put(('stopped', str(exc)))
    except BaseException:
        flush()
        messages.put(('error', traceback.format_exc()))
    finally:
        messages.put(('exited', None))
