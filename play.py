# -*- coding: utf-8 -*-
"""五子棋 独立版 —— tkinter 游玩界面（零第三方依赖）。

用法：
    python play.py

功能（对照插件 lib/ui + server 还原）：
  - 三种模式：人机(ai) / 人人(pvp) / 机机(vs)
  - 人机可换边：人类执黑先手 或 人类执白后手
  - 引擎（纯算法）自动落子，并展示推荐全候选
  - 悔棋 / 认输 / 求和 / 新对局
  - 对局结束立即可见；胜负自动写入全局记忆（totals/lossByType/badLines/goodLines）
"""
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

from gomoku.game import Game, SIZE, EMPTY, BLACK, WHITE, COLUMNS
from gomoku import service, memory
from gomoku.session import PlaySession
from gomoku.core.deep_search import DeepSearch

MODE_LABEL = {'ai': '人机', 'pvp': '人人', 'vs': '机机'}
APP_TITLE = '棋逢小鲸 v1.2.1'


def _resource_path(relative_path):
    """返回源码运行或 PyInstaller 打包后的资源绝对路径。"""
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, relative_path)

# AI 棋力档位：(名称, 温度预算 T0, 说明)——玩家语言，不暴露温度概念
# 本机实测（24 个 KataGo 中盘开局，纯算法，含剪枝）：
#   80→avg 0.17s(最慢 0.7s) / 160→avg 2.2s(最慢 12s) / 240→avg 10s(最慢 38s)。
TEMP_LEVELS = [
    ('轻快', 80, '约 0.2s/手'),
    ('标准', 160, '约 2s/手（复杂局面可达 12s）'),
    ('认真', 240, '约 10s/手（复杂局面可达 38s）'),
]


class GomokuApp:
    CELL = 34
    MARGIN = 26
    SIZE = 15
    # 5 态颜色：金色=硬必胜，绿色=存在必胜，橙色=存在必败，暗红=必败。
    STATE_TAGS = {
        '必胜': 'state_win',
        '存在必胜': 'state_exists_win',
        '无': 'state_none',
        '存在必败': 'state_exists_lose',
        '必败': 'state_lose',
    }

    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        try:
            self.root.iconbitmap(_resource_path(os.path.join('assets', 'icons', 'gomoku-ds.ico')))
        except tk.TclError:
            # 图标缺失不影响源码调试或正常对弈。
            pass
        self.root.geometry('1120x780')      # 初始窗口；棋盘随窗口自适应缩放
        self.root.minsize(720, 560)
        self.session = PlaySession('local')
        # 默认纯算法（神经网络已全局停用）
        self.session.engine.use_decision_net = False
        self.session.engine.deep_search.use_nn = False
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

        # AI 棋力档位（温度银行预算映射；越高推演越深越慢，下一手生效）
        ttk.Label(panel, text='AI 棋力').grid(row=16, column=0, sticky='w', pady=(10, 2))
        cur = DeepSearch.T0
        self.temp_var = tk.IntVar(value=min(TEMP_LEVELS, key=lambda x: abs(x[1] - cur))[1])
        for i, (name, t0, note) in enumerate(TEMP_LEVELS):
            ttk.Radiobutton(panel, text='%s（%s）' % (name, note), value=t0,
                            variable=self.temp_var,
                            command=self._on_temp_change).grid(row=17 + i, column=0, sticky='w')

        ttk.Separator(panel, orient='horizontal').grid(row=20, column=0, sticky='ew', pady=(8, 5))
        self.leaf_eval_var = tk.StringVar(value='叶子评估：尚未触发')
        self.decision_eval_var = tk.StringVar(value='决策选点：纯算法')
        ttk.Label(panel, textvariable=self.leaf_eval_var, justify=tk.LEFT,
                  wraplength=210).grid(row=21, column=0, sticky='w')
        ttk.Label(panel, textvariable=self.decision_eval_var, justify=tk.LEFT,
                  wraplength=210).grid(row=22, column=0, sticky='w')

        self.turn_label = ttk.Label(panel, text='', justify=tk.LEFT, wraplength=210)
        self.turn_label.grid(row=23, column=0, sticky='w', pady=8)

        # 宽度按最长候选行预留，保证“序号 + 坐标 + 分数 + 选中/必胜标记”单行显示。
        self.info = tk.Text(self.root, width=50, height=26, state=tk.DISABLED)
        self.info.pack(side=tk.LEFT, fill=tk.Y)
        self.info.tag_configure('state_win', foreground='#FFB300')
        self.info.tag_configure('state_exists_win', foreground='#2E7D32')
        self.info.tag_configure('state_none', foreground='#9E9E9E')
        self.info.tag_configure('state_exists_lose', foreground='#EF6C00')
        self.info.tag_configure('state_lose', foreground='#C62828')

        # 棋盘画布：填满剩余空间，格子尺寸随窗口动态计算（全屏自动放大）
        self.canvas = tk.Canvas(self.root, bg='#f2ecdf', highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<Configure>', lambda _e: self._refresh())

    def _grid(self):
        """动态棋盘参数：(margin_x, margin_y, cell)。
        格子尺寸让棋盘铺满画布短边（15 列线 = 14 间隔 + 两侧坐标字空间），
        横竖分别居中；全屏/放大窗口自动变大，仅保留最小边距。
        画布尚未布局（winfo=1）时回退到固定 34。"""
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return self.MARGIN, self.MARGIN, self.CELL
        short = min(w, h)
        label = max(14, min(34, short // 32))          # 行号/列号预留空间
        cell = max(12, (short - 2 * label) // (self.SIZE - 1))
        mx = max(2, (w - cell * (self.SIZE - 1)) // 2)
        my = max(2, (h - cell * (self.SIZE - 1)) // 2)
        return mx, my, cell

    def _cell_xy(self, r, c):
        mx, my, cell = self._grid()
        return mx + c * cell, my + r * cell

    # ---------- 绘制 ----------
    def _draw_board(self, board):
        cv = self.canvas
        cv.delete('all')
        mx, my, cell = self._grid()
        right = mx + (self.SIZE - 1) * cell
        bottom = my + (self.SIZE - 1) * cell
        label_font = max(8, min(14, cell // 3))
        # ① 棋盘底板 + 边界（木色板、深色粗边框）——棋盘是一个整体
        pad = max(4, cell // 6)
        cv.create_rectangle(mx - pad, my - pad, right + pad, bottom + pad,
                            fill='#e2b876', outline='#7a4a1e', width=3)
        # ② 网格线
        for i in range(self.SIZE):
            x, _ = self._cell_xy(0, i)      # 第 i 条竖线：x = mx + i*cell
            _, y = self._cell_xy(i, 0)      # 第 i 条横线：y = my + i*cell
            cv.create_line(x, my, x, bottom, fill='#7a4a1e')
            cv.create_line(mx, y, right, y, fill='#7a4a1e')
        # ③ 行号/列号（棋盘外侧）
        for i in range(self.SIZE):
            x, _ = self._cell_xy(0, i)
            _, y = self._cell_xy(i, 0)
            cv.create_text(max(2, mx - 14), y, text=str(i + 1),
                           font=('Arial', label_font), fill='#8a6a3a')
            cv.create_text(x, max(2, my - 14), text=COLUMNS[i],
                           font=('Arial', label_font), fill='#8a6a3a')
        # ④ 星位
        star = max(2, cell // 9)
        for (r, c) in [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]:
            x, y = self._cell_xy(r, c)
            cv.create_oval(x - star, y - star, x + star, y + star, fill='#333', outline='')
        # ⑤ 棋子
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                v = board[r][c]
                if v == EMPTY:
                    continue
                x, y = self._cell_xy(r, c)
                color = '#111' if v == BLACK else '#f5f5f5'
                rad = cell * 0.42
                cv.create_oval(x - rad, y - rad, x + rad, y + rad,
                               fill=color, outline='#333')
        # ⑥ 上一步标记
        if self.last_move:
            r, c = self.last_move
            if 0 <= r < self.SIZE and 0 <= c < self.SIZE and board[r][c] != EMPTY:
                x, y = self._cell_xy(r, c)
                mark = max(3, cell // 8)
                cv.create_oval(x - mark, y - mark, x + mark, y + mark, fill='#e33', outline='')

    # ---------- 交互 ----------
    def _on_click(self, ev):
        if self.session.game.mode == 'vs' or self.ai_thinking:
            return
        mx, my, cell = self._grid()
        c = round((ev.x - mx) / cell)
        r = round((ev.y - my) / cell)
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
        who = '你' if game.mode == 'ai' else ('黑' if player == BLACK else '白')
        self._log('【%s】落子 %s%d' % (who, COLUMNS[c], r + 1))
        self._refresh()          # 先画上人类落子，棋盘立即更新
        self._after_move(ai_followup=True)

    def _vs_step(self):
        game = self.session.game
        if self.ai_thinking or game.mode != 'vs' or game.winner is not None:
            return
        self._run_ai(lambda: self._ai_calc(game.current_player()))

    def _after_move(self, ai_followup):
        game = self.session.game
        if game.winner is not None:
            self._log(self._end_text())
            self._refresh()
            return
        if ai_followup and game.mode == 'ai' and game.current_player() == game.ai_player():
            self._run_ai(lambda: self._ai_calc(game.ai_player()))
        self._refresh()

    def _ai_calc(self, player):
        """AI 计算（在子线程执行，纯计算不碰 tkinter）。
        返回 (player, (r,c), ranked, net_used, leaf_status, 耗时秒)；无棋可走返回 None。"""
        game = self.session.game
        if game.winner is not None:
            return None
        t0 = time.perf_counter()
        out = self.session.ai_play(player)
        if not out:
            return None
        (r, c), ranked, net_used = out
        leaf_status = self.session.engine.deep_search.leaf_eval_status()
        return player, (r, c), ranked, net_used, leaf_status, time.perf_counter() - t0

    @staticmethod
    def _fmt_score(s):
        """分支分格式化：1e9 量级 = 必胜/必防直取；其余保留整数。"""
        return '直取' if s > 1e8 else ('%.0f' % s)

    def _ai_apply(self, calc):
        """AI 计算结果上屏（主线程）：落子标记 + 结构化候选打分。"""
        player, (r, c), ranked, net_used, leaf_status, dt = calc
        self.last_move = (r, c)
        who = '黑' if player == BLACK else '白'
        mode = leaf_status.get('mode')
        total = leaf_status.get('total', 0)
        if mode == 'not_triggered':
            leaf_text = '未触发（无需叶子估值）'
            leaf_tag = ' [叶子未触发]'
        else:
            leaf_text = '纯算法（%d 个叶子）' % total
            leaf_tag = ' [纯算法 %d]' % total
        self.leaf_eval_var.set('叶子评估：' + leaf_text)
        self.decision_eval_var.set('决策选点：纯算法')
        tag = leaf_tag
        self._log('【%s】AI 思考 %.1fs → 落子 %s%d%s' % (who, dt, COLUMNS[c], r + 1, tag))
        if ranked and len(ranked) > 1:
            entries = []
            for i, x in enumerate(ranked[:5]):
                mark = '   ← 选中' if (x['r'], x['c']) == (r, c) else ''
                state_name = x.get('state_name') or '无'
                tag = '' if state_name == '无' else '  ⚔%s' % state_name
                text = '  %d. %s%d  分 %s%s%s' % (i + 1, COLUMNS[x['c']], x['r'] + 1,
                                                  self._fmt_score(x['score']), mark, tag)
                entries.append((text, self.STATE_TAGS.get(state_name, 'state_none')))
            self._log_candidates(entries)
        if ranked and all(x.get('state') == -2 for x in ranked):
            self._log_tagged('⚠ 全部候选均发现败势分支，局面高危', 'state_exists_lose')

    def _run_ai(self, fn):
        """后台线程跑 AI（fn 为纯计算），主线程保持响应不冻结；
        完成后回到主线程画盘收尾。思考期间点棋/按钮被 ai_thinking 保护拦下；
        状态栏实时显示已思考秒数（0.1s 刷新），落子后日志记录总耗时。"""
        if self.ai_thinking:
            return
        self.ai_thinking = True
        self._ai_start = time.perf_counter()
        self.turn_label['text'] = '🤔 思考中… 0.0s'
        self.root.update_idletasks()
        box = {}

        def worker():
            try:
                box['ok'] = fn()
            except Exception as e:
                box['err'] = e

        def poll():
            if 'err' in box:
                self.ai_thinking = False
                self._log('AI 计算失败：' + str(box['err']))
                self._refresh()
                return
            if 'ok' in box:
                self.ai_thinking = False
                calc = box['ok']
                if calc:
                    self._ai_apply(calc)
                self._refresh()
                return
            # 实时刷新思考秒数（不创建新线程，仅更新 label）
            self.turn_label['text'] = '🤔 思考中… %.1fs' % (time.perf_counter() - self._ai_start)
            self.root.after(50, poll)

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, poll)

    # ---------- 控制 ----------
    def _human_color(self, human_player=None):
        """解析人类执子：接受字符串 'black'/'white' 或整数 BLACK/WHITE；
        不传时尊重界面单选（hcolor_var）。
        （旧写法把整数 BLACK(1) 与字符串 'black' 比较 → 恒判成 WHITE，
         导致启动默认变成 AI 先手落子，棋盘开局非空。）"""
        if human_player is None:
            human_player = self.hcolor_var.get()
        if isinstance(human_player, str):
            return BLACK if human_player == 'black' else WHITE
        return BLACK if human_player == BLACK else WHITE

    def _new_game(self, mode=None, human_player=None):
        if self.ai_thinking:
            return
        hp = self._human_color(human_player)
        mode = mode or self.mode_var.get()
        self.session.reset(mode=mode, human_player=hp)
        self.last_move = None
        self.leaf_eval_var.set('叶子评估：尚未触发')
        self.decision_eval_var.set('决策选点：纯算法')
        self._log('新对局 · %s · 人类执%s%s' % (MODE_LABEL[mode], '黑' if hp == BLACK else '白',
                                              '' if mode != 'ai' else '，AI=' + ('白' if hp == BLACK else '黑')))
        game = self.session.game
        if game.mode == 'ai' and game.current_player() == game.ai_player():
            self._run_ai(lambda: self._ai_calc(game.ai_player()))
        self._refresh()

    def _new_game_btn(self):
        self._new_game()

    def _undo(self):
        if self.ai_thinking:
            return
        game = self.session.game
        if game.mode == 'pvp':
            self.session.undo('last')
        else:
            self.session.undo('human')
        self.last_move = (game.moves[-1]['r'], game.moves[-1]['c']) if game.moves else None
        self._log('悔棋')
        self._refresh()

    def _resign(self):
        if self.ai_thinking:
            return
        game = self.session.game
        if game.winner is not None:
            return
        game.exec('resign', 'human')
        self._log(self._end_text())
        self._refresh()

    def _draw(self):
        if self.ai_thinking:
            return
        game = self.session.game
        if game.winner is not None:
            return
        game.exec('draw', 'human')
        self._log(self._end_text())
        self._refresh()

    def _on_mode_change(self):
        hp = BLACK if self.hcolor_var.get() == 'black' else WHITE
        self._new_game(mode=self.mode_var.get(), human_player=hp)

    # ---------- AI 棋力档位 ----------
    def _on_temp_change(self):
        """档位切换：更新温度银行预算（下一手生效）。"""
        DeepSearch.T0 = float(self.temp_var.get())

    # ---------- 刷新 ----------
    def _log(self, text):
        self.info['state'] = tk.NORMAL
        self.info.insert(tk.END, text + '\n')
        self.info.see(tk.END)
        self.info['state'] = tk.DISABLED

    def _log_tagged(self, text, tag_name):
        """带颜色写入一条日志（tag_name 为 Text tag）。"""
        self.info['state'] = tk.NORMAL
        self.info.insert(tk.END, text + '\n', tag_name)
        self.info.see(tk.END)
        self.info['state'] = tk.DISABLED

    def _log_candidates(self, entries):
        """候选日志：按 5 态颜色写入；entries = [(text, tag_name), ...]。"""
        self.info['state'] = tk.NORMAL
        self.info.insert(tk.END, '候选打分：\n')
        for text, tag_name in entries:
            self.info.insert(tk.END, text + '\n', tag_name)
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
            self.turn_label['text'] = '🎯 轮到 %s · %s' % (who, '黑' if g.current_player() == BLACK else '白')
        else:
            self.turn_label['text'] = '🎯 轮到 %s' % ('黑' if g.current_player() == BLACK else '白')


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
