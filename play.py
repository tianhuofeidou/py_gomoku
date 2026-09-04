# -*- coding: utf-8 -*-
"""五子棋 独立版 —— tkinter 游玩界面（零第三方依赖）。

用法：
    python play.py

功能（对照插件 lib/ui + server 还原）：
  - 三种模式：人机(ai) / 人人(pvp) / 机机(vs)
  - 人机可换边：人类执黑先手 或 人类执白后手
  - 引擎（纯算法 + 可选决策网络）自动落子，并展示推荐全候选
  - 悔棋 / 认输 / 求和 / 新对局
  - 对局结束立即可见；胜负自动写入全局记忆（totals/lossByType/badLines/goodLines）
"""
import tkinter as tk
from tkinter import ttk

from gomoku.game import Game, SIZE, EMPTY, BLACK, WHITE, COLUMNS
from gomoku import service, memory
from gomoku.session import PlaySession

MODE_LABEL = {'ai': '人机', 'pvp': '人人', 'vs': '机机'}


class GomokuApp:
    CELL = 34
    MARGIN = 26
    SIZE = 15

    def __init__(self, root):
        self.root = root
        self.root.title('五子棋 · 独立版（gomoku）')
        self.session = PlaySession('local')
        self.last_move = None
        self.ai_thinking = False
        self._build_ui()
        self._new_game(mode='ai', human_player=BLACK)

    # ---------- UI ----------
    def _build_ui(self):
        pad = 10
        panel = ttk.Frame(self.root, padding=pad)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(panel, text='模式').grid(row=0, column=0, sticky='w', pady=(0, 2))
        self.mode_var = tk.StringVar(value='ai')
        for i, (label, val) in enumerate([('人机', 'ai'), ('人人', 'pvp'), ('机机', 'vs')]):
            ttk.Radiobutton(panel, text=label, value=val, variable=self.mode_var,
                            command=self._on_mode_change).grid(row=1 + i, column=0, sticky='w')

        ttk.Label(panel, text='人类执子').grid(row=5, column=0, sticky='w', pady=(10, 2))
        self.hcolor_var = tk.StringVar(value='black')
        for i, (label, val) in enumerate([('黑先手', 'black'), ('白后手', 'white')]):
            ttk.Radiobutton(panel, text=label, value=val, variable=self.hcolor_var,
                            command=self._on_mode_change).grid(row=6 + i, column=0, sticky='w')

        ttk.Separator(panel, orient='horizontal').grid(row=9, column=0, sticky='ew', pady=8)
        for i, (label, cmd) in enumerate([('新对局', self._new_game_btn), ('悔棋', self._undo),
                                          ('认输', self._resign), ('求和', self._draw),
                                          ('机机走一步', self._vs_step), ('查看记忆', self._show_memory)]):
            ttk.Button(panel, text=label, command=cmd).grid(row=10 + i, column=0, sticky='ew', pady=2)

        self.turn_label = ttk.Label(panel, text='', justify=tk.LEFT, wraplength=210)
        self.turn_label.grid(row=17, column=0, sticky='w', pady=8)

        self.info = tk.Text(self.root, width=38, height=26, state=tk.DISABLED)
        self.info.pack(side=tk.LEFT, fill=tk.Y)

        w = self.MARGIN * 2 + self.CELL * (self.SIZE - 1)
        self.canvas = tk.Canvas(self.root, width=w, height=w, bg='#e9c786', highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind('<Button-1>', self._on_click)

    def _cell_xy(self, r, c):
        return self.MARGIN + c * self.CELL, self.MARGIN + r * self.CELL

    # ---------- 绘制 ----------
    def _draw_board(self, board):
        cv = self.canvas
        cv.delete('all')
        right = self.MARGIN + (self.SIZE - 1) * self.CELL
        for i in range(self.SIZE):
            x, y = self._cell_xy(i, 0)
            cv.create_line(x, self.MARGIN, x, right)
            cv.create_line(self.MARGIN, y, right, y)
        for i in range(self.SIZE):
            x, _ = self._cell_xy(0, i)
            _, y = self._cell_xy(i, 0)
            cv.create_text(self.MARGIN - 12, y, text=str(i + 1), font=('Arial', 8), fill='#555')
            cv.create_text(x, self.MARGIN - 12, text=COLUMNS[i], font=('Arial', 8), fill='#555')
        for (r, c) in [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]:
            x, y = self._cell_xy(r, c)
            cv.create_oval(x - 3, y - 3, x + 3, y + 3, fill='#333', outline='')
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                v = board[r][c]
                if v == EMPTY:
                    continue
                x, y = self._cell_xy(r, c)
                color = '#111' if v == BLACK else '#f5f5f5'
                cv.create_oval(x - self.CELL * 0.42, y - self.CELL * 0.42,
                               x + self.CELL * 0.42, y + self.CELL * 0.42,
                               fill=color, outline='#333')
        if self.last_move:
            r, c = self.last_move
            if 0 <= r < self.SIZE and 0 <= c < self.SIZE and board[r][c] != EMPTY:
                x, y = self._cell_xy(r, c)
                cv.create_oval(x - 4, y - 4, x + 4, y + 4, fill='#e33', outline='')

    # ---------- 交互 ----------
    def _on_click(self, ev):
        if self.session.game.mode == 'vs' or self.ai_thinking:
            return
        c = round((ev.x - self.MARGIN) / self.CELL)
        r = round((ev.y - self.MARGIN) / self.CELL)
        if not (0 <= r < self.SIZE and 0 <= c < self.SIZE):
            return
        game = self.session.game
        if game.winner is not None:
            return
        if game.mode == 'ai' and not game.is_human_turn():
            return
        player = game.current_player() if game.mode == 'pvp' else game.human_player_()
        res = self.session.place(r, c, player)
        if not res['ok']:
            self._log(res['error'])
            return
        self.last_move = (r, c)
        self._after_move(ai_followup=True)

    def _vs_step(self):
        game = self.session.game
        if game.mode != 'vs' or game.winner is not None:
            return
        self._run_ai(lambda: self._do_ai_move(game.current_player()))

    def _after_move(self, ai_followup):
        game = self.session.game
        if game.winner is not None:
            self._log(self._end_text())
            self._refresh()
            return
        if ai_followup and game.mode == 'ai' and game.current_player() == game.ai_player():
            self._run_ai(lambda: self._do_ai_move(game.ai_player()))
        self._refresh()

    def _do_ai_move(self, player):
        game = self.session.game
        if game.winner is not None:
            return
        try:
            out = self.session.ai_play(player)
            if not out:
                return
            (r, c), ranked, net_used = out
            self.last_move = (r, c)
            self._log('AI(%s) 落子 %s%d%s' % ('黑' if player == BLACK else '白',
                                              COLUMNS[c], r + 1,
                                              '  [决策网络]' if net_used else '  [纯算法]'))
            if ranked and len(ranked) > 1:
                self._log('候选：' + ' · '.join(self._rank_text(x) for x in ranked[:5]))
        except Exception as e:
            self._log('AI 计算失败：' + str(e))
        self._refresh()

    @staticmethod
    def _rank_text(x):
        return '%s%d(%.0f)' % (COLUMNS[x['c']], x['r'] + 1, x['score'])

    def _run_ai(self, fn):
        self.ai_thinking = True
        self.turn_label['text'] = '思考中…'
        self.root.update_idletasks()
        try:
            fn()
        finally:
            self.ai_thinking = False

    # ---------- 控制 ----------
    def _new_game(self, mode=None, human_player=None):
        hp = BLACK if (human_player or 'black') == 'black' else WHITE
        mode = mode or self.mode_var.get()
        self.session.reset(mode=mode, human_player=hp)
        self.last_move = None
        self._log('新对局 · %s · 人类执%s%s' % (MODE_LABEL[mode], '黑' if hp == BLACK else '白',
                                              '' if mode != 'ai' else '，AI=' + ('白' if hp == BLACK else '黑')))
        game = self.session.game
        if game.mode == 'ai' and game.current_player() == game.ai_player():
            self._run_ai(lambda: self._do_ai_move(game.ai_player()))
        self._refresh()

    def _new_game_btn(self):
        self._new_game()

    def _undo(self):
        game = self.session.game
        if game.mode == 'pvp':
            self.session.undo('last')
        else:
            self.session.undo('human')
        self.last_move = (game.moves[-1]['r'], game.moves[-1]['c']) if game.moves else None
        self._log('悔棋')
        self._refresh()

    def _resign(self):
        game = self.session.game
        if game.winner is not None:
            return
        game.exec('resign', 'human')
        self._log(self._end_text())
        self._refresh()

    def _draw(self):
        game = self.session.game
        if game.winner is not None:
            return
        game.exec('draw', 'human')
        self._log(self._end_text())
        self._refresh()

    def _on_mode_change(self):
        hp = BLACK if self.hcolor_var.get() == 'black' else WHITE
        self._new_game(mode=self.mode_var.get(), human_player=hp)

    # ---------- 刷新 ----------
    def _log(self, text):
        self.info['state'] = tk.NORMAL
        self.info.insert(tk.END, text + '\n')
        self.info.see(tk.END)
        self.info['state'] = tk.DISABLED

    def _end_text(self):
        g = self.session.game
        if g.winner == 0:
            return '🤝 和棋'
        if g.mode == 'ai':
            return '🎉 AI 获胜' if g.winner == g.ai_player() else '😅 人类获胜'
        return ('黑方' if g.winner == BLACK else '白方') + '获胜'

    def _refresh(self):
        g = self.session.game
        self._draw_board(g.board())
        if g.winner is not None:
            self.turn_label['text'] = self._end_text() + '（点「新对局」重来）'
        elif g.mode == 'ai':
            who = '你' if g.is_human_turn() else 'AI'
            self.turn_label['text'] = '轮到 %s · %s' % (who, '黑' if g.current_player() == BLACK else '白')
        else:
            self.turn_label['text'] = '轮到 %s' % ('黑' if g.current_player() == BLACK else '白')
        # 记忆提示（人机）
        if g.mode == 'ai':
            try:
                hint = memory.memory_hint(g)
                if hint:
                    self._log('━━ 记忆 ━━\n' + hint)
            except Exception:
                pass


    def _show_memory(self):
        g = self.session.game
        try:
            hint = memory.memory_hint(g) if g.mode == 'ai' else '记忆只在人机对局中累计。'
            self._log('━━ 记忆 ━━\n' + (hint or '（暂无记忆）'))
        except Exception as e:
            self._log('记忆读取失败：' + str(e))

def main():
    root = tk.Tk()
    GomokuApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
