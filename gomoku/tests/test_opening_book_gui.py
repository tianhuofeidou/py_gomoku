# -*- coding: utf-8 -*-
"""隐藏 Tk 窗口验证棋谱来源和叶子统计；无图形环境时明确跳过。"""
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from gomoku.core.opening_book import OpeningBook
from gomoku.tests.test_opening_book import record, setup_position


class OpeningBookGuiTest(unittest.TestCase):
    def test_analysis_displays_book_off_book_algorithm_and_searched_candidates(self):
        try:
            import tkinter as tk
        except ImportError:
            self.skipTest('当前 Python 未提供 tkinter')
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest('当前环境没有可用的 Tk 图形显示')
        root.withdraw()
        self.addCleanup(root.destroy)
        from gomoku import game, memory
        from play import GomokuApp
        with tempfile.TemporaryDirectory(prefix='gomoku-opening-gui-') as folder:
            with (patch.dict(os.environ, {'GOMOKU_HOME': folder}),
                  patch.object(game, 'games', {}), patch.object(memory, '_global_memory', None)):
                app = GomokuApp(root)
                app.session.reset(mode='pvp')
                app._analysis_apply(app._analysis_calc(1))
                self.assertEqual(app.decision_eval_var.get(), '辅助决策：棋谱开局')
                self.assertIn('未触发', app.leaf_eval_var.get())
                self.assertEqual(app.session.game.moves, [])
                self.assertEqual(app.analysis_hints[0]['state_name'], '无')

                app.session.engine.opening_book = OpeningBook(())
                app._analysis_apply(app._analysis_calc(1))
                self.assertEqual(app.decision_eval_var.get(), '辅助决策：纯算法')
                self.assertIn('未触发', app.leaf_eval_var.get())

                engine, board, moves = setup_position(6)
                engine.opening_book = OpeningBook([record(board, 1, (0, 0))])
                app.session.engine = engine
                app.session.game.moves = moves
                app._analysis_apply(app._analysis_calc(1))
                self.assertEqual(app.decision_eval_var.get(), '辅助决策：棋谱候选＋纯算法')
                self.assertIn('纯算法', app.leaf_eval_var.get())
                self.assertEqual(app.session.game.moves, moves)

                app.session.reset(mode='pvp')
                app.session.engine.opening_book = OpeningBook()
                main_thread = threading.get_ident()
                worker_threads, apply_threads = [], []
                def calculate():
                    worker_threads.append(threading.get_ident())
                    return app._analysis_calc(1)
                def apply_result(calc):
                    apply_threads.append(threading.get_ident())
                    app._analysis_apply(calc)
                app._run_ai(calculate, apply=apply_result)
                deadline = time.monotonic() + 3
                while app.ai_thinking and time.monotonic() < deadline:
                    root.update()
                    time.sleep(.01)
                self.assertFalse(app.ai_thinking)
                self.assertNotEqual(worker_threads, [main_thread])
                self.assertEqual(apply_threads, [main_thread])
                self.assertEqual(app.decision_eval_var.get(), '辅助决策：棋谱开局')


if __name__ == '__main__':
    unittest.main()
