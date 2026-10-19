# -*- coding: utf-8 -*-
"""对战台：驱动两个 Gomocup 协议引擎互相对局（自建管理器，无图形界面、可批量）。

用途：把"棋力"变成可复现的数字。
  - 两个引擎都是 stdin/stdout 进程（本引擎用 `python -m gomoku.protocol`，
    外部引擎用它们的 pbrain-*.exe）；
  - 支持均衡开局（多条开局、黑白互换），避免确定性引擎永远走同一盘；
  - 判胜复用 `gomoku.game.check_win`（自由规则：连续 ≥5 即胜），不复制规则；
  - 输出每局胜负与汇总胜率，可落 JSON 供后续统计。

用法示例：
  python -m gomoku.tools.match --black "python -m gomoku.protocol" \
      --white "python -m gomoku.protocol" --games 2 --max-moves 40

管道由读取线程接收，主线程按绝对截止时间等待；超时判负并结束该局子进程，
避免无回复或持续 DEBUG 输出阻塞整批对局。
"""
import argparse
import json
import math
import os
import queue
import shlex
import subprocess
import sys
import threading
import time

from gomoku.core.utils import BLACK, WHITE, SIZE, EMPTY, empty_board, place
from gomoku.game import check_win

COLUMNS = 'ABCDEFGHIJKLMNO'

# 满盘 225 手；默认上限取 225，避免把长对局截成平局（连珠赛制是 200 手判和）
DEFAULT_MAX_MOVES = 225

# 官方赛制时限，单位毫秒：(单手, 整盘, 说明)。
# 出处：Gomocup 2026 公告原文——"Time per move/per match will be 5s/120s for fastgame,
# 300s/1000s for final leagues, and 30s/180s for the rest."
# 为什么必须带上整盘时限：只限单手时，慢引擎可以每手都贴着上限走、一盘花掉几百秒而**不受任何惩罚**，
# 等于单方面加强对手（实测 CARBON 一盘花 50~159 秒，我们只花 0.8 秒）。
# free 档仅用于内部调试，**不得**用它报棋力数字：它不是任何赛制口径。
RULESETS = {
    'standard': (30000, 180000, '30s/手 + 180s/盘（Gomocup 标准赛制）'),
    'fast': (5000, 120000, '5s/手 + 120s/盘（fastgame）'),
    'final': (300000, 1000000, '300s/手 + 1000s/盘（决赛联赛）'),
    'free': (30000, 0, '30s/手、整盘不限（内部调试，非赛制口径）'),
}


def cd(r, c):
    return '%s%d' % (COLUMNS[c], r + 1)


class Engine(object):
    """一个协议引擎子进程（管理器视角）。"""

    def __init__(self, name, command, timeout_turn=30000, timeout_match=0):
        self.name = name
        self.command = shlex.split(command, posix=False)
        self.timeout_turn = timeout_turn
        self.time_left = None
        self.timed_out = False
        self._closed = False
        self._lines = queue.Queue()
        self.last_reply = ''
        # Gomocup 约定：引擎可执行文件以 pbrain- 开头，且通常需要在自己的目录下工作
        # （数据文件、日志都放在那里）。python -m 这类命令不满足该前缀，保持继承 cwd。
        cwd = None
        exe = self.command[0] if self.command else ''
        if os.path.basename(exe).lower().startswith('pbrain-'):
            cwd = os.path.dirname(os.path.abspath(exe))
        self.proc = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1, cwd=cwd)
        self.last_ms = 0.0
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self):
        """阻塞读取只发生在后台；主线程用队列与截止时间控制等待。"""
        try:
            for line in self.proc.stdout:
                self._lines.put((time.perf_counter(), line))
        finally:
            self._lines.put((time.perf_counter(), None))

    def start_game(self, size, timeout_turn, timeout_match=0, max_memory=0, rule=0):
        """按协议开一局：START → 校验 OK → INFO 各项。

        rule：bitmask（0=五连以上即胜的 Freestyle，1=恰好五连，4=连珠，8=caro）。
        规范要求引擎在 START 后必须回 OK，否则管理器应停止；这里显式校验并记录。
        """
        self.timeout_turn = timeout_turn
        t0 = time.perf_counter()
        # 初始化不占单手预算，但必须有等待上限，且占用本方整盘时间。
        init_ms = timeout_turn if timeout_turn > 0 else 30000
        if timeout_match > 0:
            init_ms = min(init_ms, timeout_match)
        self.send('START %d' % size)
        try:
            reply = self.read_line(t0 + init_ms / 1000.0)
        except TimeoutError:
            self.last_reply = '<START 超时>'
            return False
        finally:
            self.last_ms = (time.perf_counter() - t0) * 1000.0
        self.last_reply = reply or '<无回复>'
        if not reply or not reply.upper().startswith('OK'):
            self.last_reply = reply or '<无回复>'
            return False
        self.send('INFO timeout_turn %d' % timeout_turn)
        self.send('INFO timeout_match %d' % timeout_match)
        self.send('INFO max_memory %d' % max_memory)
        self.send('INFO game_type 1')          # 1 = 对手是程序
        self.send('INFO rule %d' % rule)
        return True

    def send_time_left(self, ms):
        """告知剩余总时长。

        只在**设了总时长上限**（timeout_match > 0）时调用：规范要求此时每手前必须发 time_left，
        引擎据此分配单手预算。反向教训：不限总时长却发一个大值（如 2147483647），
        部分引擎会认为"时间管够"而把单手时间花到几百秒，等于单方面加强对手
        （实测 CARBON 单手想到 505 秒，我们 80 局全负）。
        """
        self.time_left = ms
        self.send('INFO time_left %d' % ms)

    def send(self, line):
        self.proc.stdin.write(line + '\n')
        self.proc.stdin.flush()

    def read_line(self, deadline=None):
        while True:
            try:
                if deadline is None:
                    received, line = self._lines.get()
                else:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise queue.Empty
                    received, line = self._lines.get(timeout=remaining)
                    if received > deadline:
                        raise queue.Empty
            except queue.Empty:
                self.timed_out = True
                raise TimeoutError('%s 等待回复超时' % self.name) from None
            if line is None:
                return None
            line = line.strip()
            if not line:
                continue
            head = line.split()[0].upper()
            if head in ('MESSAGE', 'DEBUG'):
                continue          # 协议允许引擎先发消息，跳过
            return line

    def ask_move(self, cmd, payload=None):
        """发命令并取回坐标行；payload 为 BOARD 的石子行 ([X, Y, field], ...)。

        返回 (r, c, 耗时毫秒) 或 (None, None, 耗时毫秒)；超时抛 TimeoutError。
        """
        t0 = time.perf_counter()
        limits = [self.timeout_turn] if self.timeout_turn > 0 else []
        if self.time_left is not None:
            limits.append(max(0, self.time_left))
        deadline = t0 + min(limits) / 1000.0 if limits else None
        self.send(cmd)
        if payload is not None:
            for (x, y, field) in payload:
                self.send('%d,%d,%d' % (x, y, field))
            self.send('DONE')
        try:
            reply = self.read_line(deadline)
        finally:
            self.last_ms = (time.perf_counter() - t0) * 1000.0
        self.last_reply = reply or '<无回复>'
        if not reply:
            return None, None, self.last_ms
        head = reply.split()[0].upper()
        if head in ('ERROR', 'UNKNOWN'):
            return None, None, self.last_ms
        try:
            xs, ys = reply.split(',')[:2]
            return int(ys), int(xs), self.last_ms      # 协议 (X,Y) → 内核 (r,c)
        except ValueError:
            return None, None, self.last_ms

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.timed_out and self.proc.poll() is None:
            self.proc.kill()
        try:
            self.send('END')
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
        self._reader.join(timeout=1)
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass


def play_one(black_cmd, white_cmd, opening, max_moves, timeout_turn, verbose=False,
             timeout_match=0):
    """跑一局；返回 (winner, moves, log)。winner: 'B'/'W'/'draw'/None(异常)。

    异常（无回复／落子非法）一律返回 None 并写入 log：这类局不算胜负，由调用方单列，
    避免把工具问题混进棋力统计。

    时间口径参照 Piskvork（source/game.cpp），**每方一台独立的钟**：
      - 各引擎只累计自己的思考时间（`players[player].time += p->time`），每局清零；
      - `time_left` = 本方整盘预算 − 本方已用（`timeGame*1000 - time - timeInit`），
        不是"全局挂钟剩余"。用全局挂钟会把对手的思考时间和协议开销算到本方头上。
      - 任一限制超出即判负：单手超时，或本方累计耗时超过整盘预算；
        等待预算取单手上限和本方剩余整盘预算的较小值。初始化也计入整盘时间。
    反向教训：整盘不限（timeout_match=0）时若仍发一个大值 time_left，引擎会以为时间管够——
    实测 CARBON 单手想到 505 秒，等于单方面加强对手。这也是"必须按赛制给整盘预算"的原因。
    """
    engines = []
    board = empty_board()
    moves = []
    log = []
    try:
        b = Engine('black', black_cmd, timeout_turn=timeout_turn)
        engines.append(b)
        w = Engine('white', white_cmd, timeout_turn=timeout_turn)
        engines.append(w)
        for e in (b, w):
            if not e.start_game(SIZE, timeout_turn, timeout_match=timeout_match):
                log.append('%s 未按协议回 OK 于 START：%s' % (e.name, e.last_reply[:120]))
                return None, moves, log
        turn = BLACK
        for (r, c) in opening:                     # 开局子：内核 (r, c) 顺序，黑先
            place(board, r, c, turn)
            moves.append({'r': r, 'c': c, 'player': turn})
            turn = WHITE if turn == BLACK else BLACK

        started = {BLACK: False, WHITE: False}     # 该引擎是否已经拿到过盘面
        spent = {BLACK: b.last_ms, WHITE: w.last_ms}
        while len(moves) < max_moves:
            eng = b if turn == BLACK else w
            payload = None
            if started[turn]:
                # 引擎盘中：只需告诉它对手刚下在哪
                cmd = 'TURN %d,%d' % (moves[-1]['c'], moves[-1]['r'])
            elif not moves:
                cmd = 'BEGIN'
            elif len(moves) == 1:
                cmd = 'TURN %d,%d' % (moves[-1]['c'], moves[-1]['r'])
            else:
                # 有自动开局：按 Piskvork 的做法用 BOARD 发**全部**石子（按落子顺序），
                # field 1=己方 / 2=对方。只发最后一手会让引擎看不见前面的开局子，
                # 从而落在它自己不知道的已有子上（实测会产生大量"落子非法"假局）。
                cmd = 'BOARD'
                payload = [(m['c'], m['r'], 1 if m['player'] == turn else 2)
                           for m in moves]
            if timeout_match > 0:
                # 只报**本方**剩余时间：Piskvork 的 time_left 是各引擎自己的钟
                eng.send_time_left(max(0, int(timeout_match - spent[turn])))
            try:
                r, c, ms = eng.ask_move(cmd, payload)
            except TimeoutError:
                log.append('%s 超时判负（等待回复达到单手或整盘剩余时限）' % eng.name)
                return ('W' if turn == BLACK else 'B'), moves, log
            started[turn] = True
            spent[turn] += ms
            over_turn = timeout_turn > 0 and ms > timeout_turn
            over_match = timeout_match > 0 and spent[turn] > timeout_match
            if over_turn or over_match:
                log.append('%s 超时判负（单手 %.0fms / 上限 %dms，本方整盘已用 %.0fms / 上限 %dms）'
                           % (eng.name, ms, timeout_turn, spent[turn], timeout_match))
                return ('W' if turn == BLACK else 'B'), moves, log
            if r is None:
                log.append('%s 无合法回复: %s（%.0fms）' % (eng.name, eng.last_reply[:120], ms))
                return None, moves, log
            if not (0 <= r < SIZE and 0 <= c < SIZE) or board[r][c] != EMPTY:
                log.append('%s 落子非法 (%d,%d)' % (eng.name, r, c))
                return None, moves, log
            place(board, r, c, turn)
            rec = {'r': r, 'c': c, 'player': turn, 'engine': eng.name, 'ms': round(ms, 1)}
            moves.append(rec)
            if verbose:
                print('  %s %s (%s) %.0fms' % (cd(r, c), '黑' if turn == BLACK else '白',
                                               eng.name, ms), flush=True)
            if check_win(board, r, c, turn):
                return ('B' if turn == BLACK else 'W'), moves, log
            turn = WHITE if turn == BLACK else BLACK
        return 'draw', moves, log
    finally:
        for eng in engines:
            eng.close()


def official_openings_path():
    """官方开局集路径（gomoku/data/official_openings.json）。"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'official_openings.json')


def load_official_openings(limit=0, offset=0):
    """载入官方（Piskvork/Gomocup）均衡开局集，返回 [(r, c), ...] 列表（黑先）。

    数据已由 `gomoku/data/official_openings.json` 从「以天元为原点的偏移」换算为
    15×15 的绝对坐标；超出棋盘的条目在原文件里单列，不在此返回。
    为什么用它：锦标赛由管理器自动摆 3/5 手开局，保证同一对 AI 多盘不重样；
    用同一套开局做 A/B，我们的数字才与 Gomocup 口径一致。

    offset/limit 用来**分批跑**：整批一次跑完要几十分钟，中途断电、重启或误杀进程
    会让已跑完的对局全部作废（本项目真发生过两次）。按 offset 切片后，每批各自落盘，
    代价只是多起几次进程。
    """
    path = official_openings_path()
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    openings = [[(int(r), int(c)) for (r, c) in moves] for moves in data['openings']]
    if offset:
        openings = openings[offset:]
    return openings[:limit] if limit and limit < len(openings) else openings


def gen_openings(count, seed=1, stones=3, radius=3):
    """生成 count 条确定性开局（黑先中心 + 后续随机贴近中心），按 8 对称去重。

    用途：确定性引擎自对弈永远走同一盘，必须靠开局集制造样本多样性。
    A/B 对比时**不需要"均衡"开局**——同一开局黑白各打一遍即可抵消先手优势；
    均衡开局只在"绝对排名"场景才必要。
    """
    import random
    syms = [
        lambda r, c: (r, c), lambda r, c: (c, 14 - r), lambda r, c: (14 - r, 14 - c),
        lambda r, c: (14 - c, r), lambda r, c: (r, 14 - c), lambda r, c: (14 - r, c),
        lambda r, c: (c, r), lambda r, c: (14 - c, 14 - r),
    ]
    rng = random.Random(seed)
    seen = set()
    out = []
    guard = 0
    while len(out) < count and guard < count * 200:
        guard += 1
        moves = [(7, 7)]
        while len(moves) < stones:
            r = rng.randint(7 - radius, 7 + radius)
            c = rng.randint(7 - radius, 7 + radius)
            if (r, c) in moves:
                continue
            moves.append((r, c))
        key = tuple(sorted(moves))
        canon = min(tuple(sorted(f(r, c) for (r, c) in moves)) for f in syms)
        if canon in seen:
            continue
        seen.add(canon)
        out.append(moves)
    return out


def _worker(job):
    """多进程 worker：跑一局并返回结果（Windows spawn 下必须可 pickle）。"""
    black_cmd, white_cmd, opening, max_moves, timeout_turn, timeout_match = job
    winner, moves, log = play_one(black_cmd, white_cmd, opening, max_moves, timeout_turn,
                                  timeout_match=timeout_match)
    return winner, moves, log


def _elo_delta(p):
    """胜率 → Elo 差（>0 表示 A 更强）。p 取极值时截断，避免对数发散。"""
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / p - 1.0)


def _wilson(wins, total, z=1.96):
    """胜率的 Wilson 95% 置信区间（小样本也稳）。"""
    if total == 0:
        return 0.0, 0.0
    p = wins / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / d
    return max(0.0, center - half), min(1.0, center + half)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Gomocup 协议对战台')
    ap.add_argument('--black', required=True, help='执黑引擎命令行')
    ap.add_argument('--white', required=True, help='执白引擎命令行')
    ap.add_argument('--games', type=int, default=1)
    ap.add_argument('--swap', action='store_true', help='黑白互换（奇数局交换）')
    ap.add_argument('--max-moves', type=int, default=DEFAULT_MAX_MOVES,
                    help='手数上限，达到即判和（默认 %d = 满盘；连珠赛制 200 手判和）' % DEFAULT_MAX_MOVES)
    ap.add_argument('--ruleset', default='standard', choices=tuple(RULESETS),
                    help='官方赛制时限预设（默认 standard）：'
                         + '；'.join('%s=%s' % (k, v[2]) for k, v in RULESETS.items()))
    ap.add_argument('--timeout-turn', type=int, default=None, help='单手时限（毫秒），给定时覆盖 --ruleset')
    ap.add_argument('--timeout-match', type=int, default=None, help='整盘时限（毫秒，0=不限），给定时覆盖 --ruleset')
    ap.add_argument('--opening', default='', help='开局，如 "7,7;7,8"（协议坐标 X,Y，分号分隔）')
    ap.add_argument('--openings', type=int, default=0,
                    help='>0 进入批量模式：生成 N 条开局，每条让 A 黑白各打一遍（用于统计胜率）')
    ap.add_argument('--opening-set', default='official', choices=('official', 'random'),
                    help='批量模式的开局来源：official=官方均衡开局集（默认，与锦标赛口径一致）')
    ap.add_argument('--seed', type=int, default=1, help='批量模式的开局生成种子（确定性）')
    ap.add_argument('--opening-stones', type=int, default=3, help='批量模式每条开局的石子数')
    ap.add_argument('--parallel', type=int, default=1, help='批量模式并行进程数')
    ap.add_argument('--opening-offset', type=int, default=0,
                    help='官方开局集起始偏移，配合 --openings 分批跑（每批独立落盘）')
    ap.add_argument('--json', default='', help='结果落盘路径')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args(argv)

    _turn, _match, _label = RULESETS[args.ruleset]          # 未显式指定时用赛制预设
    args.timeout_turn = _turn if args.timeout_turn is None else args.timeout_turn
    args.timeout_match = _match if args.timeout_match is None else args.timeout_match
    print('时限：%s → 单手 %dms / 整盘 %dms%s' % (
        args.ruleset, args.timeout_turn, args.timeout_match,
        '' if args.timeout_match else '（整盘不限，仅限内部调试，勿用于棋力结论）'), flush=True)

    opening = []
    for item in args.opening.split(';'):
        item = item.strip()
        if item:
            xs, ys = item.split(',')
            opening.append((int(ys), int(xs)))     # 命令行给协议 (X,Y)，内部统一 (r,c)

    if args.openings > 0:
        return run_bulk(args)

    results = []
    tally = {'B': 0, 'W': 0, 'draw': 0, 'err': 0}
    for i in range(args.games):
        swap = args.swap and (i % 2 == 1)
        black_cmd, white_cmd = (args.white, args.black) if swap else (args.black, args.white)
        t0 = time.perf_counter()
        winner, moves, log = play_one(black_cmd, white_cmd, opening,
                                      args.max_moves, args.timeout_turn, args.verbose,
                                      timeout_match=args.timeout_match)
        dt = time.perf_counter() - t0
        tally[winner if winner in ('B', 'W') else ('draw' if winner == 'draw' else 'err')] += 1
        print('第 %d 局: %s  手数=%d  用时=%.1fs%s' % (
            i + 1, {'B': '黑胜', 'W': '白胜', 'draw': '平局'}.get(winner, '异常'),
            len(moves), dt, ('  黑=' + ('白方' if swap else '黑方')) if swap else ''), flush=True)
        for line in log:
            print('    ' + line, flush=True)
        results.append({'game': i + 1, 'swap': swap, 'winner': winner,
                        'moves': moves, 'seconds': round(dt, 2), 'notes': log})

    total = args.games
    print('\n汇总: 共 %d 局  黑胜 %d / 白胜 %d / 平 %d / 异常 %d' % (
        total, tally['B'], tally['W'], tally['draw'], tally['err']), flush=True)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({'args': vars(args), 'tally': tally, 'games': results},
                      f, ensure_ascii=False, indent=1)
        print('已写入 ' + args.json, flush=True)
    return 0


def run_bulk(args):
    """批量模式：N 条开局 × 黑白各一遍，统计 A（--black 指定的引擎）的胜率与置信区间。"""
    from multiprocessing import Pool

    openings = gen_openings(args.openings, seed=args.seed, stones=args.opening_stones)
    if args.opening_set == 'official':
        try:
            openings = load_official_openings(limit=args.openings, offset=args.opening_offset)
            print('开局来源：官方均衡开局集（gomoku/data/official_openings.json），'
                  '第 %d ~ %d 条，共 %d 条' % (
                      args.opening_offset, args.opening_offset + len(openings) - 1, len(openings)),
                  flush=True)
        except Exception as exc:                     # 缺文件/损坏时回退随机开局，并明确报告
            print('官方开局集不可用（%s），回退随机开局' % exc, flush=True)
            openings = gen_openings(args.openings, seed=args.seed, stones=args.opening_stones)
    else:
        if args.opening_offset:
            print('注意：随机开局集不支持 --opening-offset，偏移被忽略', flush=True)
        print('开局来源：随机生成（seed=%d，%d 石子）' % (args.seed, args.opening_stones), flush=True)
    jobs, meta = [], []
    for idx, op in enumerate(openings):
        jobs.append((args.black, args.white, op, args.max_moves,
                     args.timeout_turn, args.timeout_match))
        meta.append({'opening': args.opening_offset + idx, 'a_white': False, 'opening_moves': op})
        jobs.append((args.white, args.black, op, args.max_moves,
                     args.timeout_turn, args.timeout_match))
        meta.append({'opening': args.opening_offset + idx, 'a_white': True, 'opening_moves': op})

    t0 = time.perf_counter()
    outs = []

    def _flush_partial():
        """每打完一局就把已完成的部分写盘。

        为什么不是只在最后写：整批跑完要几十分钟，本项目已经被主机重启杀掉过两次整批，
        结果一个字节都没留下。写 `<json>.partial` 后最坏只丢当前这一局。
        """
        if not args.json:
            return
        with open(args.json + '.partial', 'w', encoding='utf-8') as f:
            json.dump({'args': vars(args), 'partial': True, 'done': len(outs),
                       'total': len(jobs), 'elapsed': round(time.perf_counter() - t0, 1),
                       'results': [dict(m, winner=w, plies=len(mv), moves=mv, notes=lg)
                                   for m, (w, mv, lg) in zip(meta, outs)]},
                      f, ensure_ascii=False, indent=1)

    if args.parallel > 1:
        with Pool(args.parallel) as pool:
            for out in pool.imap(_worker, jobs):     # imap 保序，便于逐局落盘
                outs.append(out)
                _flush_partial()
    else:
        for j in jobs:
            outs.append(_worker(j))
            _flush_partial()
    elapsed = time.perf_counter() - t0

    a_wins = a_losses = draws = errs = 0
    by_five = by_timeout = lost_timeout = 0
    games = []
    for m, (winner, moves, log) in zip(meta, outs):
        if winner == 'B':
            a_win = not m['a_white']
        elif winner == 'W':
            a_win = m['a_white']
        else:
            a_win = None
        # 赢法分开记：超时胜是赛制给的，不能和"五连赢下来"混为一谈
        timed_out = any('超时判负' in n for n in log)
        if timed_out:
            win_by = 'timeout'
            if a_win:
                by_timeout += 1
            elif a_win is False:
                lost_timeout += 1
        elif winner in ('B', 'W'):
            win_by = 'five'
            if a_win:
                by_five += 1
        else:
            win_by = winner or 'error'
        if a_win is True:
            a_wins += 1
        elif a_win is False:
            a_losses += 1
        elif winner == 'draw':
            draws += 1
        else:
            errs += 1
        games.append(dict(m, winner=winner, a_win=a_win, win_by=win_by, plies=len(moves),
                          moves=moves, notes=log))
        if args.verbose:
            print('  开局%d A执%s → %s（%d 手，%s）%s' % (
                m['opening'], '白' if m['a_white'] else '黑',
                'A胜' if a_win else ('A负' if a_win is False else winner),
                len(moves), win_by, ('  ' + '; '.join(log)) if log else ''), flush=True)

    total = a_wins + a_losses + draws
    lo, hi = _wilson(a_wins + 0.5 * draws, total) if total else (0.0, 0.0)
    print('\n=== 批量结果：A（--black 引擎）===', flush=True)
    print('开局 %d 条 × 黑白各一遍 = %d 局（异常 %d）  总用时 %.1fs' % (
        len(openings), total, errs, elapsed), flush=True)
    print('A 胜 %d / 负 %d / 平 %d   胜率(含半分平) %.1f%%  95%%CI [%.1f%%, %.1f%%]' % (
        a_wins, a_losses, draws, 100 * (a_wins + 0.5 * draws) / total if total else 0,
        100 * lo, 100 * hi), flush=True)
    print('赢法拆解：五连拿下 %d 局；对手超时送 %d 局；我方超时丢 %d 局' % (
        by_five, by_timeout, lost_timeout), flush=True)
    if total:
        score = (a_wins + 0.5 * draws) / total
        print('A 相对 B 的 Elo 差估计 %.0f（区间 %.0f ~ %.0f）' % (
            _elo_delta(score), _elo_delta(lo), _elo_delta(hi)), flush=True)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({'args': vars(args), 'summary': {
                'openings': len(openings), 'games': total, 'a_wins': a_wins,
                'a_losses': a_losses, 'draws': draws, 'errs': errs,
                'winrate': (a_wins + 0.5 * draws) / total if total else 0,
                'win_by': {'five': by_five, 'opp_timeout': by_timeout,
                           'our_timeout': lost_timeout},
                'ci95': [lo, hi], 'seconds': round(elapsed, 1)},
                'games': games}, f, ensure_ascii=False, indent=1)
        print('已写入 ' + args.json, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
