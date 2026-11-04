"""EXE/源码共用的复盘内搜索树调试面板，Tk主线程绘制，独立进程搜索。"""
import json
import multiprocessing as mp
import queue
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .search_trace import TraceModel, debug_worker, load_trace


KIND_LABELS = dict(node='建立节点', root='搜索根', reroot='真实落子换根', enter='模拟落子',
    candidates='查询候选', branches='保留分支', temperature='实际扣温', resume='叶子恢复温度',
    result='评分/五态回传', leaf='叶子评分', cutoff='预算结束', cache_hit='缓存复用',
    cache_visit='沿旧树访问', cache_copy='复制完成子树', snapshot='导入缓存结果',
    state_cut='硬态早停', defense_end='复合防守无解', round='开始一轮', complete='本轮完成')
STATE_NAMES = {1:'必胜', 2:'存在必胜', 0:'无', -2:'存在必败', -1:'必败'}


def coord(move):
    return '-' if move is None else chr(65+move[1])+str(move[0]+1)


def trace_export_directory():
    """源码相对项目根；打包后相对EXE，不能写入PyInstaller临时解包目录。"""
    base = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[2]
    return base / 'storages' / 'reviews' / '搜索树导出'


class SearchTreePanel(ttk.Frame):
    """嵌入棋谱复盘页签；连续真实落子保留本面板自己的缓存。"""
    def __init__(self, parent, moves, board_style, t0=160):
        super().__init__(parent)
        self.style = board_style
        self.SIZE, self.CELL, self.MARGIN = board_style.SIZE, board_style.CELL, board_style.MARGIN
        self.model = TraceModel()
        self.actual_moves = [tuple(m[:3]) for m in moves]
        self.initial_moves = list(self.actual_moves)
        self.selected = None
        self.display_actual = True
        self.last_move, self.analysis_hints = None, []
        self.process = None
        self.busy = False
        self.ready = False
        self.ended = False
        self.closing = False
        self.close_started = None
        self.last_ranked = []
        self.zoom = 1.0
        self.collapsed = set()
        self.follow = tk.BooleanVar(value=True)
        self.playing = False
        self.paused_search = False
        self.updating = False
        self._pending_load = None
        self.t0 = tk.StringVar(value=str(t0))
        self.seconds = tk.StringVar(value='120')
        self.memory = tk.StringVar(value='512')
        self.depth = tk.IntVar(value=5)
        self.speed = tk.StringVar(value='100')
        self.status = tk.StringVar(value='正在建立独立调试棋局…')
        self.counts = tk.StringVar(value='尚无搜索录像')
        self.round_var = tk.StringVar(value='')
        self._build_ui()
        self._start_worker(self.actual_moves)
        self.bind('<Destroy>', self._destroyed, add='+')
        self._poll_job = self.after(80, self._poll)

    def _build_ui(self):
        controls = ttk.Frame(self, padding=8)
        controls.pack(fill=tk.X)
        self.search_button = ttk.Button(controls, text='搜索当前局面', command=self._search)
        self.search_button.pack(side=tk.LEFT)
        self.pause_button = ttk.Button(controls, text='暂停搜索', command=self._pause_search)
        self.pause_button.pack(side=tk.LEFT, padx=4)
        self.stop_button = ttk.Button(controls, text='停止搜索', command=self._stop)
        self.stop_button.pack(side=tk.LEFT)
        for label, var, width in [('T0', self.t0, 6), ('墙钟上限秒（0不限）', self.seconds, 6),
                                  ('内存MiB', self.memory, 6)]:
            ttk.Label(controls, text=label).pack(side=tk.LEFT, padx=(10, 2))
            ttk.Entry(controls, textvariable=var, width=width).pack(side=tk.LEFT)
        ttk.Button(controls, text='读取录像JSON', command=self._load_recording).pack(side=tk.RIGHT, padx=4)
        ttk.Button(controls, text='导出录像JSON', command=self._export).pack(side=tk.RIGHT)
        note = ('独立调试，不改主棋局或胜负记忆；首次重建不导入主局旧树。'
                '在右侧当前棋盘连续落子后再次搜索，可观察换根与续温。录像上限20万事件。')
        ttk.Label(self, text=note, padding=(10, 0), wraplength=1400).pack(fill=tk.X)
        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        tree_area = ttk.Frame(body)
        board_area = ttk.Frame(body)
        body.add(tree_area, weight=3)
        body.add(board_area, weight=2)
        tools = ttk.Frame(tree_area)
        tools.pack(fill=tk.X)
        ttk.Checkbutton(tools, text='实时跟随', variable=self.follow, command=self._render).pack(side=tk.LEFT)
        ttk.Label(tools, text='显示深度').pack(side=tk.LEFT, padx=(10, 0))
        self.depth_spin = ttk.Spinbox(tools, from_=1, to=1, width=4, textvariable=self.depth,
                                      state='readonly', command=self._render)
        self.depth_spin.pack(side=tk.LEFT)
        ttk.Button(tools, text='−', width=2, command=lambda:self._zoom_tree(1/1.15)).pack(side=tk.LEFT)
        ttk.Button(tools, text='+', width=2, command=lambda:self._zoom_tree(1.15)).pack(side=tk.LEFT)
        ttk.Label(tools, text='灰旧/蓝新/黄当前 · 双击折叠').pack(side=tk.LEFT, padx=8)
        box = ttk.Frame(tree_area)
        box.pack(fill=tk.BOTH, expand=True)
        self.tree = tk.Canvas(box, bg='#f8fafc', highlightthickness=0)
        vertical = ttk.Scrollbar(box, orient=tk.VERTICAL, command=self.tree.yview)
        horizontal = ttk.Scrollbar(box, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        vertical.pack(side=tk.RIGHT, fill=tk.Y)
        horizontal.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind('<MouseWheel>', lambda e: self.tree.yview_scroll(-int(e.delta/120), 'units'))
        self.tree.bind('<Button-1>', self._tree_click)
        self.tree.bind('<Double-Button-1>', self._toggle_node)
        self.tree.bind('<Control-MouseWheel>', lambda e:self._zoom_tree(1.15 if e.delta>0 else 1/1.15))
        ttk.Label(tree_area, textvariable=self.counts).pack(fill=tk.X)
        self.canvas = tk.Canvas(board_area, bg='#f2ecdf', highlightthickness=0, height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', lambda e: self._draw_position())
        self.canvas.bind('<Button-1>', self._board_click)
        board_buttons = ttk.Frame(board_area)
        board_buttons.pack(fill=tk.X, pady=4)
        ttk.Button(board_buttons, text='看当前调试棋盘', command=self._show_actual).pack(side=tk.LEFT)
        ttk.Button(board_buttons, text='落推荐点', command=self._play_recommended).pack(side=tk.LEFT, padx=4)
        ttk.Button(board_buttons, text='以所选节点重建', command=self._use_node).pack(side=tk.LEFT)
        ttk.Button(board_buttons, text='回到载入局面', command=self._reset).pack(side=tk.LEFT, padx=4)
        self.detail = tk.Text(board_area, height=12, width=53, wrap='word', state=tk.DISABLED)
        self.detail.pack(fill=tk.BOTH)
        timeline = ttk.Frame(self, padding=(8, 0))
        timeline.pack(fill=tk.X)
        self.slider = ttk.Scale(timeline, from_=0, to=1, command=self._seek)
        self.slider.pack(fill=tk.X)
        buttons = ttk.Frame(timeline)
        buttons.pack(fill=tk.X)
        for label, func in [('首事件', lambda:self._jump(0)), ('上事件', lambda:self._jump(self.model.cursor-1)),
                            ('下事件', lambda:self._jump(self.model.cursor+1)), ('播放/暂停录像', self._playback),
                            ('下一叶子/回传', self._next_result), ('最新', self._latest)]:
            ttk.Button(buttons, text=label, command=func).pack(side=tk.LEFT, padx=2)
        ttk.Combobox(buttons, values=['1', '10', '100', '1000'], textvariable=self.speed,
                     state='readonly', width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(buttons, text='事件/帧').pack(side=tk.LEFT)
        self.round_box = ttk.Combobox(buttons, textvariable=self.round_var, state='readonly', width=8)
        self.round_box.pack(side=tk.LEFT, padx=8)
        self.round_box.bind('<<ComboboxSelected>>', self._select_round)
        ttk.Label(self, textvariable=self.status, padding=8, wraplength=1400).pack(fill=tk.X)

    def _start_worker(self, moves):
        ctx = mp.get_context('spawn')
        self.commands, self.messages = ctx.Queue(), ctx.Queue()
        self.running, self.stopping = ctx.Event(), ctx.Event()
        self.running.set()
        self.process = ctx.Process(target=debug_worker,
            args=(self.commands, self.messages, self.running, self.stopping, list(moves)), daemon=True)
        self.process.start()
        self.ready = False
        self.busy = False
        self._buttons()

    def _buttons(self):
        alive = self.process is not None and self.process.is_alive()
        self.search_button.configure(state='normal' if self.ready and not self.busy and not self.ended else 'disabled')
        self.pause_button.configure(state='normal' if self.busy and alive else 'disabled',
                                    text='继续搜索' if self.paused_search else '暂停搜索')
        self.stop_button.configure(state='normal' if alive else 'disabled')

    def _search(self):
        if not self.ready or self.busy:
            return
        try:
            t0, seconds, memory = float(self.t0.get()), float(self.seconds.get()), int(self.memory.get())
            if not (0 < t0 <= 10000 and 0 <= seconds <= 86400 and 128 <= memory <= 8192):
                raise ValueError('T0须在(0,10000]；墙钟0～86400秒；私有内存128～8192MiB')
        except ValueError as exc:
            messagebox.showerror('搜索设置', str(exc), parent=self)
            return
        self.busy, self.paused_search = True, False
        self.running.set()
        self.follow.set(True)
        self.commands.put(dict(action='analyze', t0=t0, seconds=seconds, memory_mib=memory))
        self.status.set('搜索中：进程%d；可暂停计算，也可独立暂停录像回看。' % self.process.pid)
        self._buttons()

    def _pause_search(self):
        if self.busy:
            self.paused_search = not self.paused_search
            self.running.clear() if self.paused_search else self.running.set()
            self._buttons()
            self.status.set('计算已请求暂停，墙钟上限仍计时。' if self.paused_search else '继续计算。')

    def _stop(self):
        if self.process is not None and self.process.is_alive():
            self.stopping.set()
            self.running.set()
            self.close_started = time.monotonic()
            self.ready = False
            self.status.set('正在停止本调试进程；保留已收到的部分录像，中断缓存不再使用。')
            self._buttons()

    def _poll(self):
        changed = False
        try:
            until = time.monotonic()+.035
            while time.monotonic() < until:
                kind, payload = self.messages.get_nowait()
                changed = True
                if kind == 'events':
                    for event in payload:
                        self.model.append(event)
                elif kind in ('ready', 'moved'):
                    self.actual_moves = [tuple(m) for m in payload['moves']]
                    self.ended = payload['ended']
                    self.ready = True
                    self.last_ranked = []
                    self.display_actual = True
                    if kind == 'moved':
                        self.follow.set(False)
                        self.model.seek(len(self.model.events)-1)
                    self.status.set('当前%d手，轮到%s；可搜索或点击棋盘落子。' %
                                    (len(self.actual_moves), '黑' if len(self.actual_moves)%2 == 0 else '白'))
                elif kind == 'done':
                    self.busy = False
                    self.last_ranked = payload['ranked']
                    stats = payload['cache_stats']
                    self.status.set('完成，推荐%s；子树命中%d、叶子命中%d、续温%d。可真实落子后再次搜索。' %
                        (coord(payload['move']), stats['subtree_hits'], stats['leaf_hits'], stats['frontier_resumes']))
                elif kind in ('error', 'stopped'):
                    self.busy = self.ready = False
                    self.status.set(('错误：' if kind == 'error' else '未完成：') + str(payload))
                    self._show_text(str(payload))
                elif kind == 'invalid':
                    self.busy = False
                    self.ready = True
                    self.status.set(str(payload))
                elif kind == 'exited':
                    self.busy = self.ready = False
        except queue.Empty:
            pass  # 非阻塞队列暂时为空是正常轮询，不是吞掉计算异常。
        if self.process is not None and not self.process.is_alive():
            self.process.join(timeout=0)
            if self.ready or self.busy:
                self.status.set('调试进程已退出，退出码%s；可重建局面。' % self.process.exitcode)
            self.ready = self.busy = False
        if self.close_started is not None and self.process is not None and self.process.is_alive():
            if time.monotonic()-self.close_started > 2:
                self.process.terminate()
                self.process.join(timeout=1)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=1)
                self.status.set('本调试进程已强制停止，退出码%s；录像仅覆盖收到的事件。' % self.process.exitcode)
        if self.closing and (self.process is None or not self.process.is_alive()):
            self.destroy()
            return
        if self._pending_load is not None and (self.process is None or
                                                not self.process.is_alive()):
            path = self._pending_load
            self._pending_load = None
            self._load_trace_path(path)
        if self.follow.get() and self.model.events:
            self.model.seek(len(self.model.events)-1)
            if changed:
                node = next((e['node'] for e in reversed(self.model.events) if e.get('node')), None)
                self.selected = node
                self.display_actual = False
        elif self.playing and self.model.events:
            self.model.seek(self.model.cursor+int(self.speed.get()))
            current = self.model.events[self.model.cursor].get('node')
            if current is not None:
                self.selected = current
                self.display_actual = False
            changed = True
            if self.model.cursor == len(self.model.events)-1:
                self.playing = False
        if changed:
            self._render()
        self._buttons()
        self._poll_job = self.after(80, self._poll)

    def _jump(self, index):
        self.follow.set(False)
        self.playing = False
        self.model.seek(index)
        if self.model.cursor >= 0:
            current = next((e['node'] for e in reversed(self.model.events[:self.model.cursor+1]) if e.get('node')), None)
            self.selected = current
            self.display_actual = False
        self._render()

    def _seek(self, value):
        if not self.updating:
            self._jump(round(float(value)))

    def _playback(self):
        self.follow.set(False)
        self.playing = not self.playing
        if self.playing and self.model.cursor >= len(self.model.events)-1:
            self.model.seek(0)

    def _latest(self):
        self.follow.set(True)
        self.model.seek(len(self.model.events)-1)
        self._render()

    def _next_result(self):
        for index in range(self.model.cursor+1, len(self.model.events)):
            if self.model.events[index]['kind'] in ('leaf', 'cutoff', 'result', 'resume', 'state_cut', 'defense_end'):
                self._jump(index)
                return

    def _select_round(self, _event):
        index = self.round_box.current()
        if 0 <= index < len(self.model.rounds):
            self._jump(self.model.rounds[index])

    def _render(self):
        if not self.winfo_exists():
            return
        self.tree.delete('all')
        self.updating = True
        self.slider.configure(to=max(1, len(self.model.events)-1))
        self.slider.set(max(0, self.model.cursor))
        self.updating = False
        self.round_box.configure(values=['第%d轮' % (i+1) for i in range(len(self.model.rounds))])
        max_depth = max(1, int(getattr(self.model, 'max_depth', 0)))
        self.depth_spin.configure(to=max_depth, from_=1)
        if self.depth.get() > max_depth:
            self.depth.set(max_depth)
        if self.depth.get() < 1:
            self.depth.set(1)
        root = self.model.root()
        event = self.model.events[self.model.cursor] if self.model.cursor >= 0 else {}
        active = event.get('node')
        if active is None and self.follow.get():
            active = self.selected
        path = set()
        walk = self.selected
        while walk in self.model.nodes:
            path.add(walk)
            walk = self.model.nodes[walk]['event']['parent']
        pending = [(root, 0, None)] if root else []
        drawn, positions = 0, {}
        remaining_path = set(path)
        depth_limit = max(1, min(max_depth, self.depth.get()))
        while pending and drawn < 700:
            node_id, depth, parent = pending.pop()
            value = self.model.at(node_id)
            if value is None:
                continue
            remaining_path.discard(node_id)
            if drawn >= 700-len(remaining_path) and node_id not in path:
                continue  # 为实际当前路径保留位置，不让早期大子树淹没它。
            x, y = depth*174+12, drawn*33+12
            positions[node_id] = (x, y)
            if parent in positions:
                px, py = positions[parent]
                self.tree.create_line(px+152, py+12, x-6, py+12, x-6, y+12, x, y+12, fill='#a1aab5')
            first = self.model.nodes[node_id]['event']
            old = first['round'] < event.get('round', 0) or first.get('origin') == 'cache'
            color = '#facc15' if node_id == active else ('#e2e8f0' if old else '#dbeafe')
            outline = '#0f766e' if node_id == self.selected else '#94a3b8'
            move = first.get('move')
            label = ('黑' if move and move[2] == 1 else '白') + coord(move) if move else '根局面'
            if 'state' in value:
                label += ' '+STATE_NAMES.get(value['state'], '?')
            if value.get('score') is not None:
                label += ' %.1f' % value['score']
            tag = 'node:'+node_id
            self.tree.create_rectangle(x, y, x+152, y+25, fill=color, outline=outline, width=2, tags=tag)
            self.tree.create_text(x+6, y+12, anchor='w', text=label, font=('TkDefaultFont', 9), tags=tag)
            drawn += 1
            children = [n for n in self.model.nodes[node_id]['children'] if self.model.nodes[n]['created'] <= self.model.cursor]
            if node_id not in self.collapsed and (depth < depth_limit or node_id in path):
                pending.extend((child, depth+1, node_id) for child in reversed(children))
            elif children:
                self.tree.create_text(x+148, y+12, anchor='e', text='+%d' % len(children), tags=tag)
        width = max(700, max((v[0] for v in positions.values()), default=0)+174)
        height = max(100, drawn*33+30)
        self.tree.scale('all', 0, 0, self.zoom, self.zoom)
        for item in self.tree.find_all():
            if self.tree.type(item) == 'text':
                self.tree.itemconfigure(item, font=('TkDefaultFont', max(6, round(9*self.zoom))))
        self.tree.configure(scrollregion=(0, 0, width*self.zoom, height*self.zoom))
        if self.follow.get() and active in positions:
            x, y = positions[active]
            self.tree.yview_moveto(max(0, (y*self.zoom-150)/(height*self.zoom)))
            self.tree.xview_moveto(max(0, (x*self.zoom-150)/(width*self.zoom)))
        self.counts.set('事件 %d/%d · 当前可见%d节点（绘制最多700，深层可选路径展开）· %s' %
            (max(0, self.model.cursor+1), len(self.model.events), drawn, KIND_LABELS.get(event.get('kind'), '待搜索')))
        self._draw_position()
        if not self.display_actual and self.selected in self.model.nodes:
            value = self.model.at(self.selected)
            if value:
                self._show_node(self.selected, value)

    def _tree_click(self, event):
        items = self.tree.find_overlapping(self.tree.canvasx(event.x), self.tree.canvasy(event.y),
                                           self.tree.canvasx(event.x)+1, self.tree.canvasy(event.y)+1)
        for item in reversed(items):
            for tag in self.tree.gettags(item):
                if tag.startswith('node:'):
                    self.selected = tag[5:]
                    self.follow.set(False)
                    self.display_actual = False
                    node = self.model.nodes.get(self.selected)
                    if node is not None:
                        # 单击深层节点时先扩到该层，避免它仍在显示深度之外而看似消失。
                        depth = int(node.get('depth', 0))
                        if depth > self.depth.get():
                            self.depth.set(min(depth, max(1, int(getattr(self.model, 'max_depth', 0)))))
                    self._render()
                    return

    def _zoom_tree(self, factor):
        self.zoom = min(2.5, max(.5, self.zoom*factor))
        self._render()
        return 'break'

    def _toggle_node(self, event):
        self._tree_click(event)
        if self.selected is not None:
            if self.selected in self.collapsed:
                self.collapsed.remove(self.selected)
            else:
                self.collapsed.add(self.selected)
            self._render()

    def _grid(self):
        return self.style._grid(self)

    def _cell_xy(self, r, c):
        return self.style._cell_xy(self, r, c)

    def _draw_position(self):
        board = [[0]*15 for _ in range(15)]
        self.last_move = None
        points = []
        if not self.display_actual and self.selected in self.model.nodes and self.model.at(self.selected):
            board = self.model.board(self.selected)
            value = self.model.at(self.selected)
            move = self.model.nodes[self.selected]['event'].get('move')
            self.last_move = tuple(move[:2]) if move else None
            points = value.get('points', [])
        else:
            for r, c, player in self.actual_moves:
                board[r][c] = player
            self.last_move = self.actual_moves[-1][:2] if self.actual_moves else None
        self.style._draw_board(self, board)
        _mx, _my, cell = self._grid()
        for index, (r, c, weight) in enumerate(points, 1):
            if board[r][c] == 0:
                x, y = self._cell_xy(r, c)
                rad = cell*.28
                self.canvas.create_oval(x-rad, y-rad, x+rad, y+rad, fill='#fde68a', outline='#b45309')
                self.canvas.create_text(x, y, text=str(index), font=('Arial', max(8, int(cell/3))))

    def _show_text(self, text):
        self.detail.configure(state=tk.NORMAL)
        self.detail.delete('1.0', tk.END)
        self.detail.insert(tk.END, text)
        self.detail.configure(state=tk.DISABLED)

    def _show_node(self, node_id, value):
        first = self.model.nodes[node_id]['event']
        move = first.get('move')
        lines = ['节点%s · %s · 创建于第%d轮' % (node_id, coord(move), first['round'])]
        mover = move[2] if move else value.get('mover')
        if mover in (1, 2):
            lines.append('状态视角：%s%s' % ('该手落子方' if move else '根局面上一落子方',
                                          '黑' if mover == 1 else '白'))
        if 'score_player' in value:
            lines.append('评分视角：%s' % ('黑' if value['score_player'] == 1 else '白'))
        labels = dict(reason='候选来源', turn='接下来行棋方', weight='原权重', score='分数',
            complete='分数是否完整', temperature_before='扣温前', temperature='当前记录温度',
            cost='本步实际扣温', tactical='落子前战术身份', tactical_count='累计战术步',
            leaf_end_reason='结束原因', old_temperature='旧叶子温度', round_t0='本轮T0',
            blocked='续推是否被拦截', old_depth='旧深度', old_score='旧叶子分', path_weight='路径权重',
            cached='是否缓存', before_move='落子前相位', origin='来源')
        if 'state' in value:
            lines.append('五态：'+STATE_NAMES.get(value['state'], '?'))
        for key, label in labels.items():
            if key in value:
                lines.append('%s：%s' % (label, value[key]))
        if value.get('points'):
            lines.append('候选（原顺序/权重）：'+ '、'.join('%s:%g' % (coord(p),p[2]) for p in value['points']))
        lines.append('最后事件：'+KIND_LABELS.get(value.get('last_kind'), ''))
        lines.append('静态评分探针不作为树节点；显示其所在叶子的评分。')
        self._show_text('\n'.join(lines))

    def _show_actual(self):
        self.follow.set(False)
        self.display_actual = True
        self._draw_position()
        self._show_text('当前调试棋盘：%d手，轮到%s。点击空位实际落子；成功分析后的树会按落子命中换根。' %
                        (len(self.actual_moves), '黑' if len(self.actual_moves)%2 == 0 else '白'))

    def _board_click(self, event):
        if not self.display_actual:
            self.status.set('这是历史搜索节点；先点“看当前调试棋盘”落子，或点“以所选节点重建”。')
            return
        if not self.ready or self.busy or self.ended:
            return
        mx, my, cell = self._grid()
        r, c = round((event.y-my)/cell), round((event.x-mx)/cell)
        if 0 <= r < 15 and 0 <= c < 15:
            self.ready = False
            self.commands.put(dict(action='move', point=(r, c)))
            self._buttons()

    def _play_recommended(self):
        if self.ready and not self.busy and self.last_ranked:
            row = self.last_ranked[0]
            self.ready = False
            self.commands.put(dict(action='move', point=(row['r'], row['c'])))
            self._buttons()

    def _reset_to(self, moves):
        if self.busy:
            self.status.set('请先停止搜索并等进程退出，再重建局面。')
            return
        if self.process is not None and self.process.is_alive():
            self.status.set('请先点击“停止搜索”，待进程退出后再重建；旧录像可先导出。')
            return
        self.model = TraceModel()
        self.selected = None
        self.collapsed.clear()
        self.actual_moves = [tuple(m) for m in moves]
        self.display_actual = True
        self.close_started = None
        self.follow.set(True)
        self._start_worker(self.actual_moves)
        self._render()

    def _reset(self):
        self._reset_to(self.initial_moves)

    def _use_node(self):
        if self.selected in self.model.nodes and self.model.at(self.selected):
            self._reset_to(self.model.moves(self.selected))

    def _load_recording(self):
        """选择已导出的录像；若调试进程仍在，先停止再加载，避免事件串流。"""
        try:
            directory = trace_export_directory()
        except OSError:
            directory = None
        name = filedialog.askopenfilename(parent=self, title='读取搜索录像',
            initialdir=str(directory) if directory else None,
            filetypes=[('搜索录像', '*.json'), ('所有文件', '*.*')])
        if not name:
            return
        self.load_recording(name)

    def load_recording(self, path):
        """公开入口：可停止当前调试进程后加载指定录像。"""
        if self.process is not None and self.process.is_alive():
            self._pending_load = path
            self.status.set('正在停止当前调试进程，随后读取录像…')
            self._stop()
            return
        self._load_trace_path(path)

    def _load_trace_path(self, path):
        try:
            model, partial, moves = load_trace(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror('读取录像失败', str(exc), parent=self)
            self.status.set('读取录像失败：'+str(exc))
            return
        self.model = model
        self.actual_moves = [tuple(m) for m in moves]
        self.initial_moves = list(self.actual_moves)
        self.ready = False
        self.busy = False
        self.ended = False
        self.last_ranked = []
        self.display_actual = False
        self.follow.set(False)
        self.playing = False
        self.collapsed.clear()
        self.model.seek(len(self.model.events)-1)
        self.selected = next((e.get('node') for e in reversed(self.model.events)
                              if e.get('node')), None)
        self.status.set('已读取录像：%d事件%s；只读回放，未重新搜索。' %
                        (len(self.model.events), '（未完成）' if partial else ''))
        self._render()
        self._buttons()

    def _export(self):
        if not self.model.events:
            return
        try:
            directory = trace_export_directory()
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror('导出目录不可用', str(exc), parent=self)
            return
        name = filedialog.asksaveasfilename(parent=self, title='导出搜索录像', defaultextension='.json',
                                          initialdir=str(directory),
                                          initialfile=datetime.now().strftime('搜索树-%Y%m%d-%H%M%S.json'),
                                          filetypes=[('搜索录像', '*.json')])
        if name:
            try:
                with open(name, 'w', encoding='utf-8') as stream:
                    json.dump(dict(schema='gomoku-search-trace-v1', events=self.model.events,
                                   partial=self.busy or not self.ready), stream, ensure_ascii=False)
            except OSError as exc:
                messagebox.showerror('导出失败', str(exc), parent=self)
            else:
                self.status.set('录像已导出：'+name)

    def _close(self):
        self.closing = True
        self._stop()

    def _destroyed(self, event):
        if event.widget is not self or self.process is None:
            return
        if self._poll_job is not None:
            self.after_cancel(self._poll_job)
            self._poll_job = None
        # 复盘窗口关闭或面板替换时收走本进程，不涉及其他Python任务。
        self.stopping.set()
        self.running.set()
        self.process.join(timeout=.3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1)
        self.commands.close()
        self.messages.close()
