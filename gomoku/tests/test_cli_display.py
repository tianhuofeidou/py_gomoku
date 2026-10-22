# -*- coding: utf-8 -*-
"""命令行前端的显示一致性。

旧实现把 AI 落子行硬编码成 `[纯算法]`，开局走谱着时会说错话。这里锁定三条：
1. 决策来源文案由 Engine.opening_source 决定，入口不再自己写死；
2. 开局谱着的 1e9 显示为"谱着"，不冒充必胜直取；
3. 叶子状态未触发时不报叶子数。
"""
import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import play_cli
from gomoku import game, memory, service
from gomoku.game import BLACK
from gomoku.session import PlaySession


class CliDisplayTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='gomoku-cli-')
        self.addCleanup(tmp.cleanup)
        for patcher in (
                patch.dict(os.environ, {'GOMOKU_HOME': tmp.name}),
                patch.object(game, 'games', {}),
                patch.object(memory, '_global_memory', None)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_source_label_mapping(self):
        self.assertEqual(service.decision_source_label('book'), '棋谱开局')
        self.assertEqual(service.decision_source_label('rules'), '谱外开局规则')
        self.assertEqual(service.decision_source_label('candidate'), '棋谱候选＋纯算法')
        self.assertEqual(service.decision_source_label(None), '纯算法')
        self.assertEqual(service.decision_source_label('unknown'), '纯算法')

    def test_book_score_is_not_reported_as_direct_capture(self):
        """谱着是定式着法，不是必胜手。"""
        self.assertEqual(play_cli.fmt_score(1e9, opening=True), '谱着')
        self.assertEqual(play_cli.fmt_score(1e9, opening=False), '直取')
        self.assertEqual(play_cli.fmt_score(37.4), '37')
        self.assertEqual(play_cli.fmt_score(None), '未补算')

    def test_leaf_text_only_reports_triggered_evals(self):
        self.assertEqual(play_cli.leaf_text({'mode': 'not_triggered', 'total': 0}), '未触发')
        self.assertEqual(play_cli.leaf_text({'mode': 'heuristic', 'total': 12}),
                         '纯算法 12 个叶子')

    def test_ai_turn_prints_real_source_instead_of_hardcoded_algorithm(self):
        s = PlaySession('cli-test')
        self.assertTrue(s.place(7, 7, BLACK)['ok'])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = play_cli.ai_turn(s)
        self.assertTrue(ok)
        out = buf.getvalue()
        self.assertIn('决策：', out)
        self.assertIn('叶子：', out)
        self.assertIn('候选', out)
        self.assertNotIn('[纯算法]', out, '不得再用硬编码的[纯算法]冒充决策来源')
        self.assertEqual(s.engine.opening_source, 'book',
                         '天元黑一后的相邻白二应命中可靠推荐')
        self.assertIn(service.decision_source_label(s.engine.opening_source), out)


if __name__ == '__main__':
    unittest.main()
