# -*- coding: utf-8 -*-
"""桌面手动复盘：读取 JSON 棋谱、回放和试走；不建立 Game 或写入对局记忆。"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from gomoku.game import SIZE, BLACK, WHITE, EMPTY, COLUMNS, check_win
from gomoku import paths


def _load_last_directory():
    """界面偏好独立于棋谱及胜负记忆；缺文件表示首次使用。"""
    path = Path(paths.base_dir()) / 'replay-settings.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(data, dict) or data.get('schema') != 1
            or not isinstance(data.get('last_directory'), str) or not data['last_directory']):
        raise ValueError('复盘目录设置格式错误')
    return data['last_directory']


def _save_last_directory(directory):
    """只在成功读到棋谱后记住目录，原子替换避免写到一半损坏旧设置。"""
    folder = Path(paths.base_dir())
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'replay-settings.json'
    content = dict(schema=1, last_directory=str(Path(directory).resolve()))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=folder,
                                         prefix='replay-settings-', suffix='.tmp', delete=False) as stream:
            temp_path = Path(stream.name)
            json.dump(content, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(temp_path, target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


@dataclass(frozen=True)
class ReplayRecord:
    """不可变原谱；显示视角取记录元数据，不从胜者反推我方颜色。"""
    identity: str
    source: str
    opponent: str
    our_color: str
    result: str
    reason: str
    moves: tuple

    def label(self):
        color = {'black': '我方黑', 'white': '我方白'}.get(self.our_color, '执子未知')
        result = {'win': '胜', 'loss': '负', 'draw': '和'}.get(self.result, '结果未知')
        return '%s · %s · %s · %d手 · %s' % (
            self.opponent or '对局', color, result, len(self.moves), self.identity)


def _validated_moves(raw):
    board = [[EMPTY] * SIZE for _ in range(SIZE)]
    moves = []
    ended = False
    if not isinstance(raw, list) or not raw or len(raw) > SIZE * SIZE:
        raise ValueError('着法列表为空或超过棋盘容量')
    for i, m in enumerate(raw):
        if ended:
            raise ValueError('第%d手：成五后仍有落子' % (i + 1))
        if not isinstance(m, dict):
            raise ValueError('第%d手：须使用 r/c/player 字段' % (i + 1))
        r, c, player = m.get('r'), m.get('c'), m.get('player')
        if any(type(v) is not int for v in (r, c, player)):
            raise ValueError('第%d手：行列与执子必须是整数' % (i + 1))
        if not (0 <= r < SIZE and 0 <= c < SIZE):
            raise ValueError('第%d手：坐标越界' % (i + 1))
        if player != (BLACK if i % 2 == 0 else WHITE):
            raise ValueError('第%d手：黑白顺序错误' % (i + 1))
        if board[r][c] != EMPTY:
            raise ValueError('第%d手：重复占点' % (i + 1))
        ms = m.get('ms')
        if ms is not None and (type(ms) not in (int, float) or not 0 <= ms < float('inf')):
            raise ValueError('第%d手：耗时非法' % (i + 1))
        moves.append((r, c, player, ms))
        board[r][c] = player
        ended = check_win(board, r, c, player)
    return tuple(moves)


def read_records(path):
    """兼容评测数据集、defeats、对战台 games 和单局存档；逐局报告坏记录。"""
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError('棋谱文件超过32MB，请选择拆分后的记录')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        if isinstance(data.get('games'), list):
            rows = data['games']
        elif isinstance(data.get('moves'), list):
            rows = [data]
        elif isinstance(data.get('game'), dict):
            rows = [data['game']]
        else:
            return [], []
    else:
        return [], []
    meta = {}
    sidecar = Path(str(path) + '.meta.json')
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text(encoding='utf-8-sig'))
    records, errors = [], []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or 'moves' not in row:
            continue
        # 联通性局不是本次正式评测，目录数据集默认排除，单局文件仍能手动打开。
        if row.get('scored') is False:
            continue
        try:
            moves = _validated_moves(row['moves'])
            color = row.get('our_color', meta.get('our_color', ''))
            if not color and type(row.get('a_white')) is bool:
                color = 'white' if row['a_white'] else 'black'
            result = row.get('result', '')
            if result not in ('win', 'loss', 'draw'):
                winner = row.get('winner')
                if winner == 'draw' or type(winner) is int and winner == 0:
                    result = 'draw'
                elif color and winner in ('B', 'W'):
                    result = 'win' if (winner == 'B') == (color == 'black') else 'loss'
                else:
                    result = ''
            identity = str(row.get('id') or meta.get('game_id') or '%s#%d' % (path.name, i + 1))
            records.append(ReplayRecord(identity, str(path),
                str(row.get('opponent_label') or row.get('opponent') or meta.get('opponent') or ''),
                color, result, str(row.get('reason') or row.get('win_by') or ''), moves))
        except (ValueError, TypeError) as exc:
            errors.append('%s 第%d局：%s' % (path.name, i + 1, exc))
    return records, errors


def scan_directory(directory):
    """只读选定目录及直接 results 子目录，避开运行缓存与模型目录。

    有统一 dataset.json 时读取该文件和“重赛_”补充棋谱，避免重复导入 defeats/逐局结果。
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('请选择有效的对局记录目录')
    if (directory / 'dataset.json').is_file():
        files = sorted(directory.glob('重赛_*.json')) + [directory / 'dataset.json']
    else:
        files = sorted(directory.glob('*.json'))
        if (directory / 'results').is_dir():
            files += sorted((directory / 'results').glob('*.json'))
    records, errors = [], []
    for path in files:
        if path.name.endswith('.meta.json'):
            continue
        try:
            found, bad = read_records(path)
            records.extend(found)
            errors.extend(bad)
        except (OSError, ValueError, TypeError) as exc:
            errors.append('%s：%s' % (path.name, exc))
    return records, errors


class ReplayPosition:
    """回放游标与临时分支独立于 Game/Engine，前后移动不触发终局归档。"""
    def __init__(self, record):
        self.record = record
        self.ply = 0
        self.branch_base = None
        self.branch = []
        self.cursor = 0

    def moves(self):
        if self.branch_base is None:
            return self.record.moves[:self.ply]
        return self.record.moves[:self.branch_base] + tuple(self.branch[:self.cursor])

    def board(self):
        board = [[EMPTY] * SIZE for _ in range(SIZE)]
        for r, c, player, _ms in self.moves():
            board[r][c] = player
        return board

    def seek(self, ply):
        if self.branch_base is None:
            self.ply = max(0, min(len(self.record.moves), int(ply)))
        else:
            self.cursor = max(0, min(len(self.branch), int(ply) - self.branch_base))

    def toggle_branch(self):
        if self.branch_base is None:
            self.branch_base = self.ply
        else:
            self.ply = self.branch_base
            self.branch_base = None
            self.branch = []
            self.cursor = 0

    def place(self, r, c):
        if self.branch_base is None:
            raise ValueError('请先开启试走分支')
        if type(r) is not int or type(c) is not int or not (0 <= r < SIZE and 0 <= c < SIZE):
            raise ValueError('坐标越界')
        board, moves = self.board(), self.moves()
        if moves and check_win(board, *moves[-1][:3]):
            raise ValueError('该局面已成五，请后退到成五之前再试走')
        if board[r][c] != EMPTY:
            raise ValueError('该位置已有棋子')
        self.branch = self.branch[:self.cursor]
        self.branch.append((r, c, BLACK if len(moves) % 2 == 0 else WHITE, None))
        self.cursor += 1


class ReplayWindow(tk.Toplevel):
    """EXE 内的手动复盘窗口，棋盘外观复用主界面绘制方法。"""
    def __init__(self, parent, board_style):
        super().__init__(parent)
        self.title('棋谱复盘 · 棋逢小鲸')
        self.geometry('1380x860')
        self.minsize(1060, 650)
        self.style = board_style
        self.SIZE, self.CELL, self.MARGIN = board_style.SIZE, board_style.CELL, board_style.MARGIN
        self.analysis_hints = []
        self.last_move = None
        self.records, self.visible = [], []
        self.last_directory = None
        self.position = None
        self.directory = tk.StringVar(value='请选择包含 JSON 对局记录的目录')
        self.filter_var = tk.StringVar(value='全部')
        self.status = tk.StringVar(value='选择目录后可切换棋局、逐手回放和手动试走。')
        self.note_map = {}
        self.note_key = None
        self._updating = False
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)
        ttk.Button(top, text='选择记录目录', command=self._choose_directory).pack(side=tk.LEFT)
        ttk.Button(top, text='打开棋谱文件', command=self._choose_file).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text='载入当前手到搜索树', command=self._open_search_debug).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text='读取搜索树JSON', command=self._open_search_trace).pack(side=tk.LEFT, padx=8)
        ttk.Label(top, textvariable=self.directory).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.pages = ttk.Notebook(self)
        self.pages.pack(fill=tk.BOTH, expand=True)
        self.replay_page = ttk.Frame(self.pages)
        self.search_page = ttk.Frame(self.pages)
        self.pages.add(self.replay_page, text='棋谱')
        self.pages.add(self.search_page, text='搜索树')
        self.debug_panel = None
        self.debug_origin = tk.StringVar(value='还未载入：在棋谱页选择棋局和手数，再点“载入当前手到搜索树”。')
        ttk.Label(self.search_page, textvariable=self.debug_origin, padding=8).pack(fill=tk.X)
        self.search_body = ttk.Frame(self.search_page)
        self.search_body.pack(fill=tk.BOTH, expand=True)
        self.search_hint = ttk.Label(self.search_body,
            text='可读取原有棋谱目录/JSON。原谱当前手和试走分支都能直接载入搜索树，无需重新输入局面。',
            padding=20, wraplength=1000)
        self.search_hint.pack(anchor='nw')
        body = ttk.Frame(self.replay_page)
        body.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(body, padding=10, width=350)
        left.pack(side=tk.LEFT, fill=tk.Y)
        self.filter_box = ttk.Combobox(left, values=['全部', '败局', '胜局', '和棋'],
                                      textvariable=self.filter_var, state='readonly', width=14)
        self.filter_box.pack(anchor='w')
        self.filter_box.bind('<<ComboboxSelected>>', lambda _e: self._filter())
        self.listbox = tk.Listbox(left, width=42, height=15, exportselection=False)
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=8)
        self.listbox.bind('<<ListboxSelect>>', self._select)
        ttk.Label(left, text='人工备注（对应原谱当前手；分支时对应起点）').pack(anchor='w')
        self.note = tk.Text(left, width=34, height=6, wrap='word')
        self.note.pack(fill=tk.X, pady=6)
        ttk.Button(left, text='导出人工备注与分支', command=self._export_notes).pack(fill=tk.X)
        self.detail = tk.Text(left, width=34, height=7, wrap='word', state=tk.DISABLED)
        self.detail.pack(fill=tk.X, pady=8)
        right = ttk.Frame(body, padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(right, bg='#f2ecdf', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', lambda _e: self._refresh())
        self.canvas.bind('<Button-1>', self._click)
        self.slider = ttk.Scale(right, from_=0, to=1, command=self._slide)
        self.slider.pack(fill=tk.X, pady=8)
        controls = ttk.Frame(right)
        controls.pack(fill=tk.X)
        for label, delta in [('首手前', 'first'), ('← 后退', -1), ('前进 →', 1), ('终局', 'last')]:
            ttk.Button(controls, text=label, command=lambda d=delta: self._step(d)).pack(side=tk.LEFT, padx=2)
        self.branch_button = ttk.Button(controls, text='试走分支', command=self._toggle_branch)
        self.branch_button.pack(side=tk.LEFT, padx=6)
        ttk.Label(self, textvariable=self.status, padding=10, wraplength=1100).pack(fill=tk.X)
        self.bind('<Left>', lambda e: self._key_step(e, -1))
        self.bind('<Right>', lambda e: self._key_step(e, 1))
        self.protocol('WM_DELETE_WINDOW', self._close)
        self.after_idle(self._restore_directory)

    def _open_search_debug(self):
        if self.position is None:
            self.status.set('请先选择棋谱和手数。')
            return
        moves = [tuple(m[:3]) for m in self.position.moves()]
        self.pages.select(self.search_page)
        panel = self.debug_panel
        if panel is not None and panel.winfo_exists():
            if panel.initial_moves == moves:
                self.status.set('当前手已经载入，保留已有搜索树和续温会话。')
                return
            if panel.busy:
                self.status.set('搜索树仍在计算；请先在搜索树页停止，待进程退出后再载入另一局面。')
                return
            panel.destroy()
        if self.search_hint is not None:
            self.search_hint.destroy()
            self.search_hint = None
        from .search_tree_gui import SearchTreePanel
        self.debug_panel = SearchTreePanel(self.search_body, moves, self.style)
        self.debug_panel.pack(fill=tk.BOTH, expand=True)
        kind = '原谱' if self.position.branch_base is None else '试走分支'
        self.debug_origin.set('调试来源：%s · %s第%d手；切回棋谱选另一手后，可再次点“载入当前手到搜索树”。' %
                              (self.position.record.identity, kind, len(moves)))
        self.status.set('局面已载入本窗口的搜索树页，点击“搜索当前局面”开始。')

    def _open_search_trace(self):
        """不依赖棋谱：直接读取导出的搜索录像并创建搜索树面板。"""
        from .search_tree_gui import SearchTreePanel, trace_export_directory
        try:
            directory = trace_export_directory()
        except OSError:
            directory = None
        path = filedialog.askopenfilename(parent=self, title='读取搜索树JSON',
            initialdir=str(directory) if directory else None,
            filetypes=[('搜索录像', '*.json'), ('所有文件', '*.*')])
        if not path:
            return
        self.pages.select(self.search_page)
        panel = self.debug_panel
        if panel is None or not panel.winfo_exists():
            if self.search_hint is not None:
                self.search_hint.destroy()
                self.search_hint = None
            panel = SearchTreePanel(self.search_body, [], self.style)
            panel.pack(fill=tk.BOTH, expand=True)
            self.debug_panel = panel
        panel.load_recording(path)
        self.debug_origin.set('搜索录像：%s' % path)
        self.status.set('正在读取搜索树录像。')

    def _grid(self):
        return self.style._grid(self)

    def _cell_xy(self, r, c):
        return self.style._cell_xy(self, r, c)

    def _choose_directory(self):
        initial = self.last_directory if self.last_directory and Path(self.last_directory).is_dir() else None
        folder = filedialog.askdirectory(parent=self, title='选择五子棋对局记录目录', initialdir=initial)
        if folder:
            self.load_path(folder)

    def _choose_file(self):
        initial = self.last_directory if self.last_directory and Path(self.last_directory).is_dir() else None
        path = filedialog.askopenfilename(parent=self, title='打开五子棋棋谱',
                                         filetypes=[('JSON 棋谱', '*.json')], initialdir=initial)
        if path:
            self.load_path(path)

    def _restore_directory(self):
        try:
            self.last_directory = _load_last_directory()
        except (OSError, ValueError) as exc:
            self.status.set('无法读取上次复盘目录，请重新选择：' + str(exc))
            return
        if not self.last_directory:
            return
        if not Path(self.last_directory).is_dir():
            self.directory.set(self.last_directory)
            self.status.set('上次的记录目录已不存在，请重新选择；未删除旧设置。')
            return
        self.load_path(self.last_directory, remember=False)

    def load_path(self, path, *, remember=True):
        self._save_note()
        try:
            records, errors = scan_directory(path) if Path(path).is_dir() else read_records(path)
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror('读取失败', str(exc), parent=self)
            return
        self.records = records
        self.directory.set(str(path))
        self.position, self.note_key = None, None
        self._filter()
        self.status.set('读取 %d 盘；%d 条读取错误。原始棋谱只读，备注需导出保存。' % (len(records), len(errors)))
        if records:
            selected = Path(path)
            self.last_directory = str((selected if selected.is_dir() else selected.parent).resolve())
            if remember:
                try:
                    _save_last_directory(self.last_directory)
                except OSError as exc:
                    self.status.set('已载入棋谱，但无法记住目录：' + str(exc))
        if errors:
            messagebox.showwarning('部分记录未加载', '\n'.join(errors[:12]), parent=self)

    def _filter(self):
        self._save_note()
        wanted = {'败局': 'loss', '胜局': 'win', '和棋': 'draw'}.get(self.filter_var.get())
        self.visible = [r for r in self.records if wanted is None or r.result == wanted]
        self.listbox.delete(0, tk.END)
        for record in self.visible:
            self.listbox.insert(tk.END, record.label())
        self.position, self.note_key = None, None
        self.note.delete('1.0', tk.END)
        if self.visible:
            self.listbox.selection_set(0)
            self._select()
        else:
            self._refresh()

    def _select(self, _event=None):
        selected = self.listbox.curselection()
        if not selected:
            return
        self._save_note()
        self.position = ReplayPosition(self.visible[selected[0]])
        self._load_note()
        self._refresh()

    def _anchor(self):
        p = self.position
        return p.ply if p.branch_base is None else p.branch_base

    def _save_note(self):
        if self.note_key is None:
            return
        text = self.note.get('1.0', 'end-1c')
        previous = self.note_map.get(self.note_key, {})
        variation = previous.get('variation', [])
        if self.position and self.position.branch_base is not None:
            variation = [list(m[:3]) for m in self.position.branch[:self.position.cursor]]
        if text.strip() or variation:
            self.note_map[self.note_key] = dict(note=text, variation=variation)
        else:
            self.note_map.pop(self.note_key, None)

    def _load_note(self):
        self.note.delete('1.0', tk.END)
        self.note_key = (self.position.record.source, self.position.record.identity, self._anchor())
        self.note.insert('1.0', self.note_map.get(self.note_key, {}).get('note', ''))

    def _step(self, delta):
        if self.position is None:
            return
        self._save_note()
        p = self.position
        n = len(p.moves())
        end = len(p.record.moves) if p.branch_base is None else p.branch_base + len(p.branch)
        p.seek(0 if delta == 'first' else end if delta == 'last' else n + delta)
        self._load_note()
        self._refresh()

    def _key_step(self, event, delta):
        if self.pages.select() != str(self.replay_page):
            return
        if isinstance(event.widget, (tk.Text, tk.Entry, ttk.Entry, ttk.Combobox, tk.Listbox)):
            return
        self._step(delta)
        return 'break'

    def _slide(self, value):
        if self.position is None or self._updating:
            return
        self._save_note()
        self.position.seek(round(float(value)))
        self._load_note()
        self._refresh()

    def _toggle_branch(self):
        if self.position:
            self._save_note()
            self.position.toggle_branch()
            self._load_note()
            self._refresh()

    def _click(self, event):
        if self.position is None or self.position.branch_base is None:
            return
        mx, my, cell = self._grid()
        r, c = round((event.y - my) / cell), round((event.x - mx) / cell)
        try:
            self.position.place(r, c)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self._save_note()
        self._refresh()

    def _refresh(self):
        board = self.position.board() if self.position else [[EMPTY] * SIZE for _ in range(SIZE)]
        moves = self.position.moves() if self.position else ()
        self.last_move = moves[-1][:2] if moves else None
        self.style._draw_board(self, board)
        _mx, _my, cell = self._grid()
        for i, (r, c, player, _ms) in enumerate(moves):
            x, y = self._cell_xy(r, c)
            self.canvas.create_text(x, y, text=str(i + 1),
                fill='#fff' if player == BLACK else '#111', font=('Arial', max(8, cell // 3)))
        if self.position is None:
            return
        p = self.position
        in_branch = p.branch_base is not None
        self.branch_button.configure(text='返回原谱' if in_branch else '试走分支')
        self._updating = True
        try:
            self.slider.configure(from_=p.branch_base if in_branch else 0,
                                  to=p.branch_base + len(p.branch) if in_branch else len(p.record.moves))
            self.slider.set(len(moves))
        finally:
            self._updating = False
        last = moves[-1] if moves else None
        line = ('手动分支' if in_branch else '原谱') + ' 第 %d 手' % len(moves)
        if last:
            line += ' · %s %s%d' % ('黑' if last[2] == BLACK else '白', COLUMNS[last[1]], last[0] + 1)
            if last[3] is not None:
                line += ' · %.3f 秒' % (last[3] / 1000)
        if not in_branch and p.ply < len(p.record.moves):
            r, c, player, _ms = p.record.moves[p.ply]
            line += ' · 下一手 %s %s%d' % ('黑' if player == BLACK else '白', COLUMNS[c], r + 1)
        self.status.set(line + '；备注对应原谱第%d手。' % self._anchor())
        self.detail.configure(state=tk.NORMAL)
        self.detail.delete('1.0', tk.END)
        self.detail.insert('1.0', p.record.label() + '\n来源：' + p.record.source
            + '\n终局原因：' + (p.record.reason or '未记录')
            + '\n试走仅是人工变化，不产生必胜/必败结论。')
        self.detail.configure(state=tk.DISABLED)

    def _export_notes(self):
        self._save_note()
        path = filedialog.asksaveasfilename(parent=self, title='导出人工复盘记录',
            defaultextension='.json', initialfile='manual-review-notes.json',
            filetypes=[('JSON', '*.json')])
        if not path:
            return False
        source_paths = {Path(r.source).resolve() for r in self.records}
        source_paths.update(Path(k[0]).resolve() for k in self.note_map)
        if Path(path).resolve() in source_paths:
            messagebox.showerror('不能覆盖原谱', '请另选文件保存人工备注。', parent=self)
            return False
        data = dict(schema='gomoku-desktop-manual-review-v1', annotations=[
            dict(source=k[0], game_id=k[1], ply=k[2], **v) for k, v in self.note_map.items()])
        try:
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        except OSError as exc:
            messagebox.showerror('保存失败', str(exc), parent=self)
            return False
        self.status.set('人工备注与试走分支已导出：' + path)
        return True

    def _close(self):
        self._save_note()
        if self.note_map:
            save = messagebox.askyesnocancel('保存复盘', '关闭前是否导出人工备注与试走分支？', parent=self)
            if save is None or save and not self._export_notes():
                return
        self.destroy()
