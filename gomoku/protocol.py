# -*- coding: utf-8 -*-
"""Gomocup / Piskvork 协议适配层（stdin / stdout）。

用途：把本引擎接到 Gomocup 生态里——既能在 Piskvork 管理器里和现成引擎对战，
也能被 `gomoku/tools/match.py` 这类自建管理器批量驱动，从而把"棋力"变成可测的数字。

规范要点（照抄外部标准，勿凭感觉改）：
  - 坐标 **0 基**；`TURN/BOARD` 里 **X 是列、Y 是行**；
  - 一行一条命令；管理器用 CR LF 结尾，本层两种都接受；
  - **每次输出必须 flush**，否则管理器会认为引擎卡死；
  - 必须处理 START / BEGIN / INFO / BOARD / TURN / END；未知命令回 `UNKNOWN`；
  - 可执行文件名必须以 `pbrain-` 开头，否则管理器按旧的文件协议对待；
  - 除协议内容外**不得向 stdout 输出任何东西**（诊断一律走 stderr）。
    规范要求棋盘 ≤20 时才被接受：本引擎是 15×15 定长，因此 `START 20` 回 `ERROR`，
    管理器可以再试其它尺寸。

坐标映射：协议 (X, Y) → 内核零基 (r, c) = (Y, X)。
五子棋在竖直镜像下同构，管理器与引擎各自维持一致视角即可，映射方向不影响力；
若将来要严格对齐外部开局库，这里是一行的事。

时间：Gomocup 时限"不比 30 秒/手、3 分钟/局更严"。本层**不做迭代加深**（不改引擎内部逻辑），
只记录 INFO 里的时限供上层观察；实测 T0=160 中盘 avg 3.08s / 最慢 17.2s，余量足够。
"""
import sys

from gomoku import service
from gomoku.core.utils import BLACK, WHITE, SIZE, empty_board, place

ABOUT = 'name="QifengXiaojing", version="1.6.0", author="tianhuofeidou", country="CHN"'
SUPPORTED_SIZES = (SIZE,)


def _write(text):
    """向管理器输出一行并 flush（规范硬要求）。"""
    sys.stdout.write(text + '\n')
    sys.stdout.flush()


class Brain(object):
    """一局对局的协议状态机：维护棋盘与着法序列，选点委托 service.ai_move。"""

    def __init__(self):
        self.engine = None
        self.board = None
        self.moves = []            # [{'r': 'c', 'player'}]，与 service.ai_move 约定一致
        self.my_color = None       # 本局我方颜色；None = 尚未确定
        self.timeout_turn = 30000  # INFO timeout_turn（毫秒）
        self.timeout_match = 0     # INFO timeout_match（毫秒，0=不限）
        self.max_memory = 0

    # ---------- 棋盘与状态 ----------

    def reset(self):
        """新对局：重建引擎与棋盘（引擎状态必须与棋盘同步重置）。"""
        self.engine = service.create_engine()
        self.board = empty_board()
        self.moves = []
        self.my_color = None

    def _sync_move(self, r, c, player):
        place(self.board, r, c, player)
        self.engine.on_move(self.board, r, c, player)
        self.moves.append({'r': r, 'c': c, 'player': player})

    def _turn_color(self):
        """按着法数推断轮次（黑先）。"""
        return BLACK if len(self.moves) % 2 == 0 else WHITE

    def _opponent_color(self):
        return WHITE if self.my_color == BLACK else BLACK

    # ---------- 选点 ----------

    def _pick(self):
        """轮到我方：返回协议坐标字符串，落子并同步引擎状态。"""
        if self.my_color is None:
            self.my_color = self._turn_color()
        player = self.my_color
        if player not in (BLACK, WHITE) or self._turn_color() != player:
            # 局面轮次与我方颜色不符：按规范回 ERROR，绝不抛异常打死进程
            return None
        mv, _ranked, _net = service.ai_move(self.board, self.moves, player,
                                           engine=self.engine)
        if mv is None:
            return None
        r, c = mv
        if not (0 <= r < SIZE and 0 <= c < SIZE) or self.board[r][c] != 0:
            return None
        self._sync_move(r, c, player)
        return '%d,%d' % (c, r)          # 内核 (r, c) → 协议 (X, Y) = (c, r)

    # ---------- 命令处理 ----------

    def handle(self, line):
        """处理一行命令，返回要发送的回复文本（None = 不回复）。"""
        line = line.strip()
        if not line:
            return None
        parts = line.split()
        cmd = parts[0].upper()
        args = parts[1:]

        if cmd == 'START':
            size = int(args[0]) if args else SIZE
            if size not in SUPPORTED_SIZES:
                return 'ERROR unsupported board size %d (this brain is 15x15)' % size
            self.reset()
            return 'OK'

        if cmd == 'RESTART':
            self.reset()
            return 'OK'

        if cmd == 'INFO':
            self._handle_info(args)
            return None

        if cmd == 'BEGIN':
            self.my_color = BLACK
            mv = self._pick()
            return mv if mv else 'ERROR no legal move'

        if cmd == 'TURN':
            x, y = self._parse_coords(args[0] if args else '')
            self._sync_move(y, x, self._turn_color())   # 协议 (X,Y) → 内核 (r,c)
            mv = self._pick()
            return mv if mv else 'ERROR no legal move'

        if cmd == 'BOARD':
            return None      # 盘面由 _read_board 续读，见 run()

        if cmd == 'TAKEBACK':
            # 管理器要求撤销一手（人类对局用）。内核不支持局部回退，按规范回 UNKNOWN。
            return 'UNKNOWN takeback not supported'

        if cmd == 'ABOUT':
            return ABOUT

        if cmd == 'END':
            return '__END__'

        return 'UNKNOWN %s' % cmd

    def _handle_info(self, args):
        if len(args) < 2:
            return
        key, value = args[0].lower(), args[1]
        try:
            num = int(value)
        except ValueError:
            num = None
        if key == 'timeout_turn' and num is not None:
            self.timeout_turn = num
        elif key == 'timeout_match' and num is not None:
            self.timeout_match = num
        elif key == 'max_memory' and num is not None:
            self.max_memory = num

    @staticmethod
    def _parse_coords(text):
        xs, ys = text.split(',')[:2]
        return int(xs), int(ys)

    def read_board(self, stream):
        """读 BOARD 后续行直到 DONE，按 field 1=己方 / 2=对方建盘面。

        自由五子棋允许乱序。BOARD 随后要求本方落子，因此双方子数相同则本方黑，
        对方多一子则本方白。按颜色交替重建以保持内核轮次一致；每种颜色内部保留
        输入顺序。有序输入不变，乱序输入不能恢复真实历史，只能重建等价盘面。
        """
        self.reset()
        raw = []
        invalid = False
        while True:
            line = stream.readline()
            if not line:
                return None
            line = line.strip()
            if line.upper() == 'DONE':
                break
            if not line:
                continue
            try:
                x, y, field = map(int, line.split(','))
            except ValueError:
                invalid = True
                continue  # 仍消费到 DONE，避免后续命令错位
            if not (0 <= x < SIZE and 0 <= y < SIZE) or field not in (1, 2):
                invalid = True
            raw.append((x, y, field))
        if invalid or len({(x, y) for x, y, _ in raw}) != len(raw):
            return 'ERROR invalid BOARD'
        own = [(x, y) for x, y, field in raw if field == 1]
        opp = [(x, y) for x, y, field in raw if field == 2]
        if len(own) == len(opp):
            self.my_color = BLACK
            black, white = own, opp
        elif len(opp) == len(own) + 1:
            self.my_color = WHITE
            black, white = opp, own
        else:
            return 'ERROR invalid BOARD stone counts'
        for i, (x, y) in enumerate(black):
            self._sync_move(y, x, BLACK)
            if i < len(white):
                x, y = white[i]
                self._sync_move(y, x, WHITE)
        mv = self._pick()
        # 轮到对方时应回 ERROR（管理器据此判负/结束），不能静默不回——那会让对方等到超时
        return mv if mv else 'ERROR unexpected turn'


def run(stream_in=None, stream_out=None):
    """主循环：读命令 → 回复。返回 0（正常 END 退出）。"""
    stream_in = stream_in or sys.stdin
    brain = Brain()
    while True:
        line = stream_in.readline()
        if not line:
            return 0
        stripped = line.strip()
        if not stripped:
            continue
        cmd = stripped.split()[0].upper()
        if cmd == 'BOARD':
            reply = brain.read_board(stream_in)
        else:
            reply = brain.handle(stripped)
        if reply == '__END__':
            return 0
        if reply:
            _write(reply)


def main(argv=None):
    """协议入口。

    `--t0` 覆盖推演温度（`DeepSearch.T0`）：对战台批量自对弈时可降到 40 加速，
    默认沿用引擎默认档（160），与界面/正式对局一致。用 parse_known_args，
    避免管理器传入的额外参数把引擎打死。
    """
    import argparse
    ap = argparse.ArgumentParser(description='Gomocup 协议入口（stdin/stdout）')
    ap.add_argument('--t0', type=float, default=None,
                    help='覆盖 DeepSearch.T0（默认 160；批量对战可用 40 加速）')
    args, _unknown = ap.parse_known_args(argv)
    if args.t0 is not None:
        from gomoku.core.deep_search import DeepSearch
        DeepSearch.T0 = float(args.t0)
    return run()


if __name__ == '__main__':
    sys.exit(main())
