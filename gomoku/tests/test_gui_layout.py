"""有图形显示环境时检查实际 Tk 窗口的棋盘边界；无显示环境明确跳过。"""
import os
import tempfile
import unittest
from unittest.mock import patch


class GuiLayoutTest(unittest.TestCase):
    def test_board_fits_minimum_default_and_large_windows(self):
        try:
            import tkinter as tk
        except ImportError:
            self.skipTest('当前 Python 未提供 tkinter')
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest('当前环境没有可用的图形显示')
        self.addCleanup(root.destroy)
        root.withdraw()
        root.attributes('-alpha', 0.0)
        from gomoku import game, memory
        from play import GomokuApp
        with tempfile.TemporaryDirectory(prefix='gomoku-layout-') as folder:
            with (patch.dict(os.environ, {'GOMOKU_HOME': folder}),
                  patch.object(game, 'games', {}),
                  patch.object(memory, '_global_memory', None)):
                app = GomokuApp(root)
                root.deiconify()
                minimum = root.minsize()
                for width, height in (minimum, (1120, 780), (1400, 900)):
                    with self.subTest(width=width, height=height):
                        root.geometry('%dx%d' % (width, height))
                        root.update()
                        mx, my, cell = app._grid()
                        self.assertGreaterEqual(cell, 12)
                        self.assertGreaterEqual(mx, 14, '必须保留左侧坐标空间')
                        self.assertGreaterEqual(my, 14, '必须保留上方坐标空间')
                        self.assertLessEqual(mx + 14 * cell + 4, app.canvas.winfo_width())
                        self.assertLessEqual(my + 14 * cell + 4, app.canvas.winfo_height())


if __name__ == '__main__':
    unittest.main()
