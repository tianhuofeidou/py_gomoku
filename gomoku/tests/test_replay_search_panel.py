"""复盘导入→选手数/分支→同窗搜索树，窗口和进程均隔离。"""
import json
import os
from pathlib import Path
import tempfile
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from play import GomokuApp
from gomoku.tests.test_search_trace import MOVES
from gomoku.tools.replay_gui import ReplayWindow
from gomoku.tools.search_tree_gui import SearchTreePanel


class ReplaySearchPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
        except tk.TclError as exc:
            raise unittest.SkipTest('需要可用Tk显示环境：'+str(exc))
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gomoku-replay-panel-test-')
        self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, GOMOKU_HOME=self.temp.name)
        env.start()
        self.addCleanup(env.stop)
        self.errors = []
        self.root.report_callback_exception = lambda typ, value, tb: self.errors.append(repr(value))
        self.window = ReplayWindow(self.root, GomokuApp)
        self.window.withdraw()
        self.root.update()
        path = Path(self.temp.name)/'game.json'
        raw = MOVES+[(5,5,1),(4,4,2)]
        path.write_text(json.dumps(dict(id='导入测试', moves=[dict(r=r,c=c,player=p) for r,c,p in raw])), encoding='utf-8')
        self.window.load_path(path, remember=False)

    def tearDown(self):
        panel = self.window.debug_panel
        process = panel.process if panel is not None and panel.winfo_exists() else None
        self.window.destroy()
        self.root.update()
        if process is not None:
            self.assertFalse(process.is_alive(), '关闭复盘窗口须收走自己的进程')
        self.assertFalse(self.errors, self.errors)

    def pump(self, predicate, timeout=10):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            self.root.update()
            self.assertFalse(self.errors, self.errors)
            if predicate():
                return
            time.sleep(.01)
        self.fail('等待复盘调试状态超时：'+self.window.status.get())

    def load_six(self):
        self.window.position.seek(6)
        self.window._open_search_debug()
        panel = self.window.debug_panel
        self.pump(lambda:panel.ready)
        return panel

    def test_selected_prefix_embeds_without_another_toplevel(self):
        before = {str(w) for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)}
        panel = self.load_six()
        self.assertIsInstance(panel, SearchTreePanel)
        self.assertIs(panel.winfo_toplevel(), self.window)
        self.assertEqual(panel.initial_moves, MOVES)
        self.assertEqual(self.window.pages.select(), str(self.window.search_page))
        self.assertEqual(before, {str(w) for w in self.root.winfo_children() if isinstance(w,tk.Toplevel)})
        self.assertIn('第6手', self.window.debug_origin.get())

    def test_tab_switch_and_same_position_keep_existing_trace_and_process(self):
        panel = self.load_six()
        panel.t0.set('40')
        panel.seconds.set('5')
        panel._search()
        self.pump(lambda:not panel.busy and bool(panel.last_ranked))
        count, process = len(panel.model.events), panel.process
        self.assertGreater(count,0)
        self.window.pages.select(self.window.replay_page)
        self.window._open_search_debug()
        self.assertIs(self.window.debug_panel,panel)
        self.assertIs(panel.process,process)
        self.assertEqual(len(panel.model.events),count)
        panel._zoom_tree(1.15)
        self.assertGreater(panel.zoom,1.)
        self.assertTrue(panel.tree.find_all())

    def test_loading_another_hand_replaces_idle_panel_and_stops_old_process(self):
        first = self.load_six()
        old = first.process
        self.window.position.seek(4)
        self.window._open_search_debug()
        new = self.window.debug_panel
        self.pump(lambda:new.ready)
        self.assertIsNot(new,first)
        self.assertEqual(new.initial_moves,MOVES[:4])
        self.assertFalse(old.is_alive())

    def test_branch_is_loaded_instead_of_original_next_move(self):
        self.window.position.seek(4)
        self.window.position.toggle_branch()
        self.window.position.place(5,6)
        self.window._open_search_debug()
        panel = self.window.debug_panel
        self.pump(lambda:panel.ready)
        self.assertEqual(panel.initial_moves,MOVES[:4]+[(5,6,1)])
        self.assertIn('试走分支',self.window.debug_origin.get())

    def test_debug_tab_arrow_keys_do_not_change_replay_ply(self):
        self.load_six()
        self.window._key_step(SimpleNamespace(widget=self.window.canvas),-1)
        self.assertEqual(self.window.position.ply,6)
        self.window.pages.select(self.window.replay_page)
        self.window._key_step(SimpleNamespace(widget=self.window.canvas),-1)
        self.assertEqual(self.window.position.ply,5)


if __name__ == '__main__':
    unittest.main()
