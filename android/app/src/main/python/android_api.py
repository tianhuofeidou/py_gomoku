# -*- coding: utf-8 -*-
"""Android 端 Python 桥接层。

Kotlin 侧调用约定：
    init_app(home)          设置数据目录、导入引擎、恢复上次棋局
    new_game(human_black)   开新局，返回状态 JSON
    state()                 读取当前状态 JSON
    human_move(r, c)        人类落子，返回 {ok, error?, state}
    ai_move()               AI 落子，返回 {ok, move?, state}
    undo() / resign() / draw()
    memory()                返回记忆页数据 JSON

所有跨语言返回值都是 JSON 字符串，避免 Chaquopy 复杂对象转换。
数据目录通过 GOMOKU_HOME 注入 gomoku.paths，指向 App 私有 filesDir。
"""
import json
import os

_session = None
_game_mod = None
_service_mod = None
_memory_mod = None

BLACK = 1
WHITE = 2
EMPTY = 0
SIZE = 15

# 与桌面 play.py 的 TEMP_LEVELS 保持一致：玩家语言，不暴露温度概念。
# 移动端初始为“标准”（T0=160），可在 UI 切换，下一手生效。
GEAR_LEVELS = [
    ('轻快', 80, '约 0.3s/手'),
    ('标准', 160, '约 3s/手（复杂局面可达 17s）'),
    ('认真', 240, '约 15s/手（复杂局面可达 97s）'),
]
_DEFAULT_GEAR = 160.0


def _dump(obj):
    return json.dumps(obj, ensure_ascii=False)


class AndroidSession:
    """把 Game 状态机 + Engine 包装成移动端会话，并复用磁盘存档。"""

    SID = 'android-local'

    def __init__(self):
        self.engine = _service_mod.create_engine()
        # 使用实例属性而不是 DeepSearch.T0 类属性，避免未来多会话互相影响。
        self.engine.deep_search.T0 = _DEFAULT_GEAR
        # game_of 在无存档时会创建默认 Game，这里补一次落盘，之后每次
        # 落子/结算都由 Game.persist 保存到 App 私有目录。
        loaded = _game_mod.game_of(self.SID)
        loaded.sid = self.SID
        _game_mod.games[self.SID] = loaded
        _game_mod.persist()
        self.game = loaded
        self.engine.reset(self.game.moves)

    def reset(self, human_player=BLACK):
        # 与 PlaySession.reset 一致：保留已完成对局历史，重置棋盘与引擎。
        history = self.game.history[:]
        self.game = _game_mod.Game(human_player=human_player, mode='ai', ai_mode='engine')
        self.game.history = history
        self.game.sid = self.SID
        _game_mod.games[self.SID] = self.game
        _game_mod.persist()
        self.engine.reset()

    def place(self, r, c, player):
        res = self.game.place(r, c, player)
        if res.get('ok'):
            self.engine.on_move(self.game.board(), r, c, player)
        return res

    def ai_play(self):
        game = self.game
        if game.winner is not None:
            return None
        player = game.ai_player()
        if player != game.current_player():
            return None
        mv, _ranked, _net_used = _service_mod.ai_move(
            game.board(), game.moves, player, engine=self.engine)
        if mv is None:
            return None
        r, c = mv
        res = self.place(r, c, player)
        if not res.get('ok'):
            raise RuntimeError(res.get('error'))
        return (r, c)

    def undo(self):
        # 与 PlaySession 相同：Game 只回退 moves，引擎必须全盘重建，
        # 否则状态表残留幽灵子会污染后续候选和必杀判定。
        self.game.undo_human_move()
        self.engine.reset(self.game.moves)


def _result_of(winner, human):
    if winner is None:
        return None
    if winner == 0:
        return 'draw'
    if winner == human:
        return 'human_win'
    return 'ai_win'


def _summary():
    try:
        totals = _memory_mod.global_memory().get('totals') or {}
        wins = int(totals.get('wins', 0))
        losses = int(totals.get('losses', 0))
        draws = int(totals.get('draws', 0))
    except Exception:
        return '暂无历史战绩'
    total = wins + losses + draws
    if total <= 0:
        return '暂无历史战绩'
    # 全局记忆按 AI 视角记账：AI 败 = 人类胜，这里换算成人类战绩。
    my_wins = losses
    my_losses = wins
    rate = int(round(my_wins * 100.0 / total))
    return '你的战绩 %d胜 %d负 %d平 · 胜率 %d%%' % (my_wins, my_losses, draws, rate)


def _state():
    g = _session.game
    board = g.board()
    flat = []
    for r in range(SIZE):
        flat.extend(board[r])
    last = None
    if g.moves:
        m = g.moves[-1]
        last = {'r': m['r'], 'c': m['c'], 'player': m['player']}
    human = g.human_player_()
    current = g.current_player()
    return {
        'board': flat,
        'moves': g.moves,
        'moveCount': len(g.moves),
        'current': current,
        'human': human,
        'ai': g.ai_player(),
        'winner': g.winner,
        'over': g.winner is not None,
        'last': last,
        'canUndo': any(m['player'] == human for m in g.moves),
        'turn': 'human' if current == human else 'ai',
        'result': _result_of(g.winner, human),
        'summary': _summary(),
        'gear': float(getattr(_session.engine.deep_search, 'T0', _DEFAULT_GEAR)),
    }


def _memory():
    gm = _memory_mod.global_memory()
    g = _session.game
    recent = []
    for h in (g.history or [])[-12:][::-1]:
        recent.append({
            'result': _result_of(h.get('winner'), h.get('humanPlayer', BLACK)),
            'moves': len(h.get('moves') or []),
            'at': h.get('endedAt') or h.get('startedAt') or 0,
            'humanPlayer': h.get('humanPlayer', BLACK),
        })
    bad = []
    for bl in (gm.get('badLines') or [])[-12:][::-1]:
        bad.append({
            'killType': bl.get('killType') or '其他',
            'moves': len(bl.get('opp') or []) + len(bl.get('ai') or []),
            'at': bl.get('at') or 0,
        })
    good = []
    for gl in (gm.get('goodLines') or [])[-12:][::-1]:
        good.append({
            'killType': gl.get('killType') or '胜局',
            'moves': len(gl.get('opp') or []) + len(gl.get('ai') or []),
            'at': gl.get('at') or 0,
        })
    try:
        hint = _memory_mod.memory_hint(g) or ''
    except Exception:
        hint = ''
    ai_totals = gm.get('totals') or {}
    ai_wins = int(ai_totals.get('wins', 0))
    ai_losses = int(ai_totals.get('losses', 0))
    ai_draws = int(ai_totals.get('draws', 0))
    human_totals = {'wins': ai_losses, 'losses': ai_wins, 'draws': ai_draws}
    ai_loss_total = sum(int(v) for v in (gm.get('lossByType') or {}).values())
    hint_lines = [ln for ln in hint.split('\n') if ln and not ln.startswith('📊 全局战绩：')]
    if ai_wins + ai_losses + ai_draws > 0:
        hint_lines.insert(0, '📊 你的战绩：%d 胜 %d 负 %d 平' % (ai_losses, ai_wins, ai_draws))
    hint = '\n'.join(hint_lines)
    return {
        'totals': human_totals,
        'lossByType': gm.get('lossByType') or {},
        'aiLossTotal': ai_loss_total,
        'recent': recent,
        'badLines': bad,
        'goodLines': good,
        'hint': hint,
    }


# ---------------------------------------------------------------------------
# Kotlin 可调用入口
# ---------------------------------------------------------------------------
def init_app(home):
    global _session, _game_mod, _service_mod, _memory_mod
    os.environ['GOMOKU_HOME'] = home
    os.makedirs(home, exist_ok=True)
    # 延迟导入：确保 gomoku.paths 首次读取环境变量时 GOMOKU_HOME 已生效。
    from gomoku import game as game_mod
    from gomoku import service as service_mod
    from gomoku import memory as memory_mod
    _game_mod = game_mod
    _service_mod = service_mod
    _memory_mod = memory_mod
    _session = AndroidSession()
    return _dump({'ok': True, 'state': _state()})


def new_game(human_black):
    _session.reset(BLACK if human_black else WHITE)
    return _dump({'ok': True, 'state': _state()})


def state():
    return _dump(_state())


def human_move(r, c):
    human = _session.game.human_player_()
    res = _session.place(int(r), int(c), human)
    return _dump({'ok': bool(res.get('ok')), 'error': res.get('error'), 'state': _state()})


def ai_move():
    if _session is None:
        return _dump({'ok': False, 'error': '尚未初始化', 'state': None})
    if _session.game.winner is not None:
        return _dump({'ok': False, 'error': '棋局已结束', 'state': _state()})
    mv = _session.ai_play()
    return _dump({'ok': mv is not None, 'move': list(mv) if mv else None, 'state': _state()})


def undo():
    _session.undo()
    return _dump({'ok': True, 'state': _state()})


def resign():
    _session.game.exec('resign', 'human')
    return _dump({'ok': True, 'state': _state()})


def draw():
    _session.game.exec('draw', 'human')
    return _dump({'ok': True, 'state': _state()})


def memory():
    return _dump(_memory())


def gear_levels():
    """返回可选棋力档位（名称、T0、玩家可读说明）。"""
    return _dump([
        {'name': name, 't0': t0, 'note': note}
        for (name, t0, note) in GEAR_LEVELS
    ])


def set_gear(t0):
    """切换 AI 棋力档位；下一手生效，并返回最新状态。"""
    if _session is None:
        return _dump({'ok': False, 'error': '尚未初始化', 'state': None})
    try:
        value = float(t0)
    except (TypeError, ValueError):
        return _dump({'ok': False, 'error': '非法棋力档位', 'state': _state()})
    allowed = [item[1] for item in GEAR_LEVELS]
    if value not in allowed:
        return _dump({'ok': False, 'error': '不支持的棋力档位', 'state': _state()})
    _session.engine.deep_search.T0 = value
    return _dump({'ok': True, 'state': _state()})
