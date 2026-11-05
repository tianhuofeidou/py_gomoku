"""旗舰默认与GUI入口一致；不启动窗口或运行长搜索。"""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK, WHITE, empty_board


class FlagshipDefaultsTests(unittest.TestCase):
    def test_flagship_configuration_keeps_tactical_and_continuation_policy(self):
        d = Engine().deep_search
        self.assertEqual((d.T0, d.OPENING_COLD_T0, d.MIDDLE_COLD_T0), (160., 160., 120.))
        self.assertEqual((d.DT_M, d.MIN_SEARCH_PLIES), (200., 3))
        self.assertEqual((d.CAND_N_ORDINARY, d.TACTICAL_BATCH_SIZE), (4, 3))
        self.assertEqual((d.TACTICAL_FREE_PLIES, d.TACTICAL_STAGE_PLIES, d.TACTICAL_COSTS),
                         (8, 8, (2., 4., 8.)))
        self.assertIsNone(d.TACTICAL_EXTENSION_PLIES)
        self.assertEqual((d.CONTINUE_BASE, d.CONTINUE_HALF_DEPTH), (64., 4.))

    def test_engine_uses_flagship_opening_and_middle_budget_for_both_colors(self):
        for player in (BLACK, WHITE):
            engine = Engine()
            d = engine.deep_search
            budgets = []
            with patch.object(d, 'rank_candidates', side_effect=lambda *a: budgets.append(d._initial_temperature())):
                engine._rank_search(empty_board(), {(7, 7): 25.}, player, False)
                engine._rank_search(empty_board(), {(7, 7): 25.}, player, False)
            self.assertEqual(budgets, [160., 120.])
            self.assertIsNone(d._cold_temperature)
            self.assertEqual(d.T0, 160.)

    def test_gui_no_longer_has_temperature_presets_or_overrides(self):
        source = (Path(__file__).resolve().parents[2] / 'play.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        constants = [node.value for node in ast.walk(tree)
                     if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        self.assertIn('AI：旗舰', constants)
        for name in ('轻快', '标准', '认真', '约 0.3s/手'):
            self.assertNotIn(name, constants)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertNotIn('TEMP_LEVELS', names)
        self.assertNotIn('temp_var', attrs)
        self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name == '_on_temp_change'
                             for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr == 'T0'
                             and isinstance(node.ctx, ast.Store) for node in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
